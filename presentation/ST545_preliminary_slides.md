# ST545 Preliminary Presentation Slides

For the April 1, 2026 mid-check presentation.
Recommended length: 7 slides, about 35-45 seconds per slide.

## Slide 1. Title

**Predictive Modeling of Equity Trends via Data-Driven Media Weighting and Nonlinear Interaction Analysis**

- Jacky Luo
- ST 545 Modern Statistical Learning
- Preliminary project presentation
- Goal: predict next-day stock direction by combining market data, fundamentals, and financial news

## Slide 2. Motivation and Research Questions

- Can publisher-specific media weighting identify which news sources carry the most predictive signal?
- Does FinBERT provide better financial text representations than traditional TF-IDF?
- Do nonlinear interactions between sentiment and market variables improve prediction quality?
- Can SHAP help us interpret how much signal comes from sentiment features versus technical features?

## Slide 3. Data Sources and Modeling Target

- Data sources:
  - Alpaca: daily OHLCV bars and Benzinga news
  - Finnhub: PE fundamentals and additional company news sources
- Universe: 10 U.S. tickers
  - NVDA, GOOGL, MSFT, AMZN, TSLA, LMT, NEM, AAPL, META, JPM
- Current cache:
  - 46,461 news articles from 2025-03-07 to 2026-03-06
  - 2,390 daily ticker-date rows in the final merged dataset from 2025-03-20 to 2026-03-06
- Prediction target:
  - use features at day `t` to classify whether price goes up at day `t+1`

Suggested visual:
- `poc/result/step1_2/publisher_distribution.png`

## Slide 4. Current Modeling Pipeline

- Step 1: benchmark TF-IDF vs. FinBERT on ticker-level news classification
- Step 2: learn ticker-specific publisher weights with Lasso or Ridge interaction terms
- Step 3: compress FinBERT embeddings with PCA and train tuned XGBoost models
- Step 4: use SHAP to measure sentiment-versus-market contribution
- Step 5: latest v13 hybrid model combines:
  - market features
  - weighted sentiment statistics
  - DQS gating for noisy news days
  - PCA-reduced FinBERT features
  - top 10 Lasso-selected keywords

## Slide 5. Preliminary Results

- FinBERT is slightly stronger than TF-IDF after tuning
  - Step 4 mean AUC gain over TF-IDF: `+0.0063`
- Sentiment-related features remain important
  - mean SHAP sentiment attribution: `80.16%`
- Best saved hybrid results so far:
  - v12 `LMT` hybrid AUC = `0.6036`
  - v13 `GOOGL` hybrid AUC = `0.5971`
- Strongest standalone feature family right now:
  - keywords-only mean AUC = `0.5564`

Suggested visual:
- `poc/result/step4/representation_comparison_xgb.png`

## Slide 6. What We Have Learned So Far

- The main bottleneck is feature fusion, not signal extraction
- In v13, mean hybrid AUC is `0.5362`
  - market-only mean AUC = `0.5431`
  - keywords-only mean AUC = `0.5564`
- Hybrid beats every single-feature baseline for only one ticker in v13
  - `GOOGL`
- Model class matters
  - for `LMT`, ablation gives `MLP = 0.6782`, `RF = 0.5923`, `XGBoost = 0.5653`
- Current interpretation:
  - gating helps some high-volume tech names
  - deeper fusion may be better handled by neural models than tree ensembles

Suggested visual:
- `poc/result/step4/shap_summary_LMT.png`

## Slide 7. Expected Deliverables and Next Steps

- Final deliverables:
  - cleaned multimodal daily dataset
  - reproducible end-to-end Python pipeline
  - comparative model evaluation with SHAP and ablation analysis
  - final report and final presentation
  - optional lightweight results browser or dashboard if time permits
- Next steps before the final submission:
  - test a modular expert strategy by ticker type
  - improve hybrid fusion so it consistently beats keyword-only and market-only baselines
  - package results and artifacts into a submission-ready workflow

Closing line:
- The project already shows measurable news signal, but the final challenge is turning that signal into a stable hybrid model.
