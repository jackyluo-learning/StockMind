# ST545 Preliminary Presentation Speaker Notes

Target length: about 5 minutes total.
Suggested pacing: 35-45 seconds per slide.

## Slide 1. Title

This project studies whether we can predict next-day stock direction more effectively by combining structured market data with unstructured financial news. The main idea is not only to classify price movement, but also to learn which publishers matter and whether their interaction with fundamentals creates usable predictive signal.

## Slide 2. Motivation and Research Questions

I organized the project around three questions. First, instead of manually trusting certain news outlets, can we learn publisher importance directly from data? Second, does FinBERT actually improve over TF-IDF for this financial text setting? Third, if we combine sentiment with market variables in a nonlinear model, do we gain predictive power, and can SHAP still tell us where the signal is coming from?

## Slide 3. Data Sources and Modeling Target

The dataset is built from Alpaca and Finnhub. Alpaca provides daily bars and Benzinga news, while Finnhub adds PE fundamentals and additional news sources such as Yahoo and SeekingAlpha. In the current cache, I have more than forty-six thousand news articles across ten major U.S. tickers, and after daily merging I have about twenty-three hundred ticker-date observations. The target is a simple and interpretable classification problem: use information available on day t to predict whether the stock moves up on day t plus 1.

## Slide 4. Current Modeling Pipeline

The pipeline has evolved in stages. I first benchmark TF-IDF and FinBERT, then estimate ticker-specific publisher weights using regularized interaction terms. After that, I reduce FinBERT embeddings with PCA, train tuned XGBoost models, and use SHAP for interpretation. The latest version, v13, adds DQS gating to down-weight noisy news days and combines market features, weighted sentiment summaries, reduced semantic features, and top keyword features into one hybrid model.

## Slide 5. Preliminary Results

The preliminary results are mixed but informative. FinBERT is only slightly better than TF-IDF on average after tuning, so representation alone is not the whole story. At the same time, SHAP shows that sentiment-related features account for about eighty percent of model importance on average, so news is clearly carrying signal. The best saved hybrid performance is around 0.60 AUC, and the strongest standalone feature family right now is the keyword-based component.

## Slide 6. What We Have Learned So Far

The main lesson is that extracting signal is easier than fusing signal. In the current v13 results, the full hybrid model is not yet consistently better than the market-only or keyword-only baselines. However, the ablation study on LMT shows that model choice matters a lot: an MLP reaches 0.6782 AUC and clearly beats the tree-based versions. So my current interpretation is that gating helps some tickers, but the final fusion step may need a more flexible model or a modular expert strategy.

## Slide 7. Expected Deliverables and Next Steps

For the final submission, I expect to deliver a cleaned multimodal dataset, a reproducible pipeline, comparative evaluation with interpretation, and the final written report. If time allows, I also want to package the results into a lightweight browser or dashboard. The main next step is to improve the fusion strategy so the hybrid model consistently outperforms the simpler baselines, while keeping the system interpretable and reproducible.

## Optional Shorter Version

If I need to cut time during class, I would shorten Slide 4 and Slide 6 first, and keep the data slide and results slide unchanged.
