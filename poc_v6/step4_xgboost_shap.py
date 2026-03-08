"""
ST545 POC v6 — Step 4: Lagged Features + Probability Calibration
===================================================================
Changes from v5:
  1. Added 1-3 day sentiment lagging (sent_mean, news_count)
  2. Implemented Probability Calibration (CalibratedClassifierCV)
  3. Dynamic scale_pos_weight for XGBoost to handle minor imbalance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

TICKERS = ['NVDA', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'LMT', 'NEM', 'AAPL', 'META', 'JPM']
CACHE_PATH = 'dataset/sentiment_cache.csv'

# ══════════════════════════════════════════════════════
#  1. Load & Build Daily Dataset
# ══════════════════════════════════════════════════════
cache = pd.read_csv(CACHE_PATH)
cache['Date'] = pd.to_datetime(cache['Date'])
print(f"--- Loaded sentiment cache: {len(cache)} articles ---")

all_daily = []
for ticker in TICKERS:
    market = pd.read_csv(f"dataset/{ticker}_market.csv")
    market['Date'] = pd.to_datetime(market['Date'])
    market = market.sort_values('Date')

    market['Next_Close'] = market['Close'].shift(-1)
    market['Price_Label'] = (market['Next_Close'] > market['Close']).astype(int)
    market = market.dropna(subset=['Price_Label'])
    market['Price_Label'] = market['Price_Label'].astype(int)

    market['volume_pct_chg'] = market['Volume'].pct_change()
    market['pe_chg'] = market['PE_Ratio'].diff()
    market['MA5'] = market['Close'].rolling(5).mean()
    market['MA10'] = market['Close'].rolling(10).mean()
    market['ma5_ratio'] = market['Close'] / market['MA5']
    market['ma10_ratio'] = market['Close'] / market['MA10']
    market['momentum_5d'] = market['Close'].pct_change(5)
    market['volatility_5d'] = market['Close'].pct_change().rolling(5).std()

    ticker_news = cache[cache['Ticker'] == ticker].copy()
    daily_sent = ticker_news.groupby('Date')['Sentiment_Score'].agg(
        sent_mean='mean', sent_std='std', sent_max='max', sent_min='min',
        news_count='count'
    ).reset_index()
    daily_sent['sent_std'] = daily_sent['sent_std'].fillna(0)

    # 🚀 v6: Add 1-3 day lags for sentiment features
    for lag in [1, 2, 3]:
        daily_sent[f'sent_mean_lag{lag}'] = daily_sent['sent_mean'].shift(lag)
        daily_sent[f'news_count_lag{lag}'] = daily_sent['news_count'].shift(lag)

    for pub in ticker_news['Publisher'].unique():
        pub_col = f"pub_{pub}_sent"
        pub_daily = ticker_news[ticker_news['Publisher'] == pub].groupby('Date')['Sentiment_Score'].mean()
        pub_daily = pub_daily.reset_index().rename(columns={'Sentiment_Score': pub_col})
        daily_sent = pd.merge(daily_sent, pub_daily, on='Date', how='left')

    market_cols = ['Date', 'Ticker', 'PE_Ratio', 'Volume', 'Price_Label',
                   'volume_pct_chg', 'pe_chg', 'ma5_ratio', 'ma10_ratio',
                   'momentum_5d', 'volatility_5d']
    daily = pd.merge(market[market_cols], daily_sent, on='Date', how='inner')
    all_daily.append(daily)

df = pd.concat(all_daily, ignore_index=True)
df = df.dropna(subset=['Price_Label']).sort_values('Date').reset_index(drop=True)
pub_sent_cols = [c for c in df.columns if c.startswith('pub_') and c.endswith('_sent')]
df[pub_sent_cols] = df[pub_sent_cols].fillna(0)

# Fill lag NaNs (from the start of each ticker's series)
lag_cols = [c for c in df.columns if '_lag' in c]
df[lag_cols] = df[lag_cols].fillna(0)

df = df.dropna().reset_index(drop=True)

print(f"--- Daily-level dataset: {len(df)} rows ({df['Ticker'].nunique()} tickers) ---")
print(f"--- Date range: {df['Date'].min().date()} to {df['Date'].max().date()} ---")

# ══════════════════════════════════════════════════════
#  2. Feature Matrix
# ══════════════════════════════════════════════════════
MARKET_COLS = ['PE_Ratio', 'Volume', 'volume_pct_chg', 'pe_chg',
               'ma5_ratio', 'ma10_ratio', 'momentum_5d', 'volatility_5d']
SENT_AGG_COLS = ['sent_mean', 'sent_std', 'sent_max', 'sent_min', 'news_count']
FEATURE_COLS = SENT_AGG_COLS + lag_cols + pub_sent_cols + MARKET_COLS

X_raw = df[FEATURE_COLS].values
y = df['Price_Label'].values

print(f"\n--- Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"--- Samples: {len(X_raw)}, Label balance: {y.mean():.2%} positive ---")

# ══════════════════════════════════════════════════════
#  3. XGBoost Hyperparameter Tuning (GridSearchCV)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Phase 1: XGBoost Hyperparameter Tuning (GridSearchCV + inner TSS)")
print("=" * 70)

xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [3, 5, 10],
}
print(f"Grid size: {np.prod([len(v) for v in xgb_param_grid.values()])} combinations")

inner_tscv = TimeSeriesSplit(n_splits=3)
scaler_tune = StandardScaler()
X_scaled_tune = scaler_tune.fit_transform(X_raw)

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    ),
    param_grid=xgb_param_grid,
    cv=inner_tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
xgb_grid.fit(X_scaled_tune, y)
best_xgb_params = xgb_grid.best_params_
print(f"\nBest XGBoost params: {best_xgb_params}")
print(f"Best inner CV AUC:  {xgb_grid.best_score_:.4f}")

# ══════════════════════════════════════════════════════
#  4. MLP Architecture Tuning
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Phase 2: MLP Hyperparameter Tuning (GridSearchCV + inner TSS)")
print("=" * 70)

mlp_param_grid = {
    'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01],
}
print(f"Grid size: {np.prod([len(v) for v in mlp_param_grid.values()])} combinations")

mlp_grid = GridSearchCV(
    MLPClassifier(
        activation='relu', solver='adam',
        max_iter=500, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20,
        random_state=42
    ),
    param_grid=mlp_param_grid,
    cv=inner_tscv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=0
)
mlp_grid.fit(X_scaled_tune, y)
best_mlp_params = mlp_grid.best_params_
print(f"\nBest MLP params: {best_mlp_params}")
print(f"Best inner CV AUC: {mlp_grid.best_score_:.4f}")

# ══════════════════════════════════════════════════════
#  5. Outer Evaluation: Calibrated Models × TimeSeriesSplit 5-fold
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Phase 3: Model Comparison (outer TimeSeriesSplit 5-fold + Calibration)")
print("=" * 70)

outer_tscv = TimeSeriesSplit(n_splits=5)

def get_calibrated_models(X_train, y_train):
    # Calculate scale_pos_weight for XGBoost
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    spw = neg_count / pos_count if pos_count > 0 else 1.0

    models = {
        'LogReg': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost (tuned)': xgb.XGBClassifier(
            **best_xgb_params,
            scale_pos_weight=spw,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=15,
            random_state=42
        ),
        'MLP (tuned)': MLPClassifier(
            **best_mlp_params,
            activation='relu', solver='adam',
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=42
        ),
    }

    calibrated_models = {}
    for name, model in models.items():
        # Use sigmoid calibration for all models to standardize probability outputs
        calibrated_models[name] = CalibratedClassifierCV(model, cv=3, method='sigmoid')
    
    return calibrated_models

results = {name: {'acc': [], 'auc': [], 'f1': []} for name in ['LogReg', 'XGBoost (tuned)', 'RandomForest', 'MLP (tuned)']}
majority_accs = []

for fold, (train_idx, test_idx) in enumerate(outer_tscv.split(X_raw)):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw[train_idx])
    X_test = scaler.transform(X_raw[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    maj = np.bincount(y_train).argmax()
    majority_accs.append(accuracy_score(y_test, np.full(len(y_test), maj)))

    cal_models = get_calibrated_models(X_train, y_train)
    fold_str = f"  Fold {fold+1}:"
    for name, model in cal_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        f1 = f1_score(y_test, pred)
        results[name]['acc'].append(acc)
        results[name]['auc'].append(auc)
        results[name]['f1'].append(f1)
        fold_str += f"  {name}={acc:.3f}/{auc:.3f}"
    print(fold_str)

maj_avg = np.mean(majority_accs)
print(f"\n{'Model (Calibrated)':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}")
print("-" * 48)
for name, res in results.items():
    avg_acc = np.mean(res['acc'])
    avg_auc = np.mean(res['auc'])
    avg_f1 = np.mean(res['f1'])
    print(f"{name:<20} {avg_acc:>8.4f} {avg_auc:>8.4f} {avg_f1:>8.4f}")
print(f"{'Majority Vote':<20} {maj_avg:>8.4f}")

# ══════════════════════════════════════════════════════
#  6. SHAP Analysis (tuned XGBoost)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Phase 4: SHAP Analysis (tuned XGBoost)")
print("=" * 70)

scaler_final = StandardScaler()
X_final = scaler_final.fit_transform(X_raw)

# Final model for SHAP (use the booster directly for TreeExplainer compatibility)
pos_count = np.sum(y == 1)
neg_count = np.sum(y == 0)
final_spw = neg_count / pos_count if pos_count > 0 else 1.0

final_xgb = xgb.XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=final_spw,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss'
)
final_xgb.fit(X_final, y, verbose=False)

explainer = shap.TreeExplainer(final_xgb)
shap_values = explainer.shap_values(X_final)

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_final, feature_names=FEATURE_COLS, show=False)
plt.title('SHAP Feature Importance (Tuned XGBoost + SPW, v6 Lags)')
plt.tight_layout()
plt.savefig('poc/result/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_final, feature_names=FEATURE_COLS, plot_type='bar', show=False)
plt.title('Mean |SHAP| Feature Importance (v6 Lags)')
plt.tight_layout()
plt.savefig('poc/result/shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)

top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
for rank, idx in enumerate(top3_idx):
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(idx, shap_values, X_final, feature_names=FEATURE_COLS, show=False)
    plt.title(f'SHAP Dependence: {FEATURE_COLS[idx]}')
    plt.tight_layout()
    plt.savefig(f'poc/result/shap_dependence_top{rank+1}.png', dpi=150)
    plt.close()

# ══════════════════════════════════════════════════════
#  7. Feature Group Importance
# ══════════════════════════════════════════════════════
sent_features = SENT_AGG_COLS + lag_cols + pub_sent_cols
market_features = MARKET_COLS
sent_shap = sum(mean_abs_shap[FEATURE_COLS.index(f)] for f in sent_features)
market_shap = sum(mean_abs_shap[FEATURE_COLS.index(f)] for f in market_features)
total_shap = sent_shap + market_shap

print(f"\nFeature group importance (mean |SHAP|):")
print(f"  Sentiment features: {sent_shap:.4f} ({sent_shap/total_shap*100:.1f}%)")
print(f"  Market features:    {market_shap:.4f} ({market_shap/total_shap*100:.1f}%)")

# ══════════════════════════════════════════════════════
#  8. MLP Training Curve Visualization (Final Model)
# ══════════════════════════════════════════════════════
print("\n--- MLP Training Curve ---")
mlp_viz = MLPClassifier(
    **best_mlp_params,
    activation='relu', solver='adam',
    max_iter=500, early_stopping=True,
    validation_fraction=0.15, n_iter_no_change=20,
    random_state=42
)
mlp_viz.fit(X_final, y)
print(f"  MLP converged in {mlp_viz.n_iter_} iterations")
print(f"  Final training loss: {mlp_viz.loss_:.6f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mlp_viz.loss_curve_, label='Training Loss', color='steelblue')
if hasattr(mlp_viz, 'validation_scores_') and mlp_viz.validation_scores_ is not None:
    ax.plot(mlp_viz.validation_scores_, label='Validation Score', color='darkorange')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss / Score')
ax.set_title(f'MLP Training Curve (architecture={best_mlp_params["hidden_layer_sizes"]})')
ax.legend()
plt.tight_layout()
plt.savefig('poc/result/mlp_training_curve.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════
#  9. Save Results
# ══════════════════════════════════════════════════════
with open('poc/result/step4_results.txt', 'w') as f:
    f.write("ST545 POC v6 Step 4 Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Tickers: {TICKERS}\n")
    f.write(f"Dataset: {len(df)} daily-level samples, {df['Ticker'].nunique()} tickers\n")
    f.write(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    f.write(f"Features: {len(FEATURE_COLS)} (including 3-day sentiment lags)\n")
    f.write(f"Validation: TimeSeriesSplit (5 outer folds, 3 inner folds for tuning)\n")
    f.write(f"Calibration: CalibratedClassifierCV (sigmoid) applied to all models\n\n")

    f.write("--- Hyperparameter Tuning Results (on raw features) ---\n")
    f.write(f"XGBoost best params: {best_xgb_params}\n")
    f.write(f"XGBoost best inner CV AUC: {xgb_grid.best_score_:.4f}\n")
    f.write(f"MLP best params: {best_mlp_params}\n")
    f.write(f"MLP best inner CV AUC: {mlp_grid.best_score_:.4f}\n\n")

    f.write("--- Model Comparison (outer 5-fold, Calibrated) ---\n")
    f.write(f"{'Model':<20} {'Acc':>8} {'AUC':>8} {'F1':>8}\n")
    f.write("-" * 48 + "\n")
    for name, res in results.items():
        f.write(f"{name:<20} {np.mean(res['acc']):>8.4f} {np.mean(res['auc']):>8.4f} {np.mean(res['f1']):>8.4f}\n")
    f.write(f"{'Majority Vote':<20} {maj_avg:>8.4f}\n\n")

    f.write("--- Per-fold results (Calibrated) ---\n")
    for fold_i in range(5):
        f.write(f"  Fold {fold_i+1}: ")
        for name in results:
            f.write(f"{name} Acc={results[name]['acc'][fold_i]:.4f} AUC={results[name]['auc'][fold_i]:.4f} | ")
        f.write(f"Majority={majority_accs[fold_i]:.4f}\n")

    f.write(f"\n--- Feature group importance (tuned XGBoost SHAP) ---\n")
    f.write(f"  Sentiment (incl lags): {sent_shap:.4f} ({sent_shap/total_shap*100:.1f}%)\n")
    f.write(f"  Market:                {market_shap:.4f} ({market_shap/total_shap*100:.1f}%)\n\n")

    f.write("--- Top SHAP features ---\n")
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    for idx in sorted_idx:
        f.write(f"  {FEATURE_COLS[idx]:30s}  {mean_abs_shap[idx]:.6f}\n")

    f.write(f"\n--- MLP details ---\n")
    f.write(f"  Architecture: {best_mlp_params['hidden_layer_sizes']}\n")
    f.write(f"  Converged in: {mlp_viz.n_iter_} iterations\n")
    f.write(f"  Final loss: {mlp_viz.loss_:.6f}\n")

print("\n[+] POC v6 Step 4 complete. Results in poc/result/")
