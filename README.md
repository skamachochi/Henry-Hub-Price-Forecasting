# Henry Hub Gas Price Forecasting (XGBoost + PCA)

Forecast Henry Hub natural gas prices over the next **~3 months** using a machine-learning pipeline that combines **macro/market indicators + climate variables**, **PCA-based dimensionality reduction**, and an **XGBoost regression** model trained on lagged features. 

## Project overview
Natural gas prices are volatile and influenced by multiple drivers (market behavior, broader economic activity, sentiment, and weather-related effects). This project builds a supervised model to learn those relationships and produce short-horizon forecasts, with an emphasis on capturing overall trends and major fluctuations.
## Method summary
Pipeline (high level):
1. **Handle missing data** with forward-fill (and fill first-row gaps using the next available value).
2. **Standardize features** (excluding the target).
3. **Apply PCA** to retain **~90%** of the variance and reduce redundancy.
4. **Create lagged features** (tested `n_lags = 1, 3, 5`) to inject temporal dependence.
5. Train **XGBoost Regressor** and evaluate on a time-based split.

## Data
- **Target:** *Natural Gas US Henry Hub Gas* price series. 
- **Inputs:** a diverse set of indicators, including market and consumer confidence signals and climate-related variables. 
- **Missing values:** addressed via forward filling. 

## Train/test split
Time-series split (no shuffling): training data up to **Aug 2020**, test data from **Sep 2020 onward**. 

## Model
**XGBoost Regression** was chosen for strong performance on structured/tabular data, non-linear modeling capacity, and built-in regularization (useful for noisy energy-market data). 

## Results
Three lag configurations were tested. Best overall performance was achieved with **`n_lags = 1`**: 
- **R² (test):** 0.839  
- **RMSE:** 0.752  
- **MAPE:** 15.36%

The model captures general trends and many major fluctuations (2020–2024), but can deviate during extreme events (notably the **2022 price spike**) and tends to show larger errors at higher price levels. 

## Limitations
Key observed issues: 
- Underperformance during extreme price spikes (tail events)
- Increasing residual variance at higher predicted values (heteroscedasticity)
- Potential overfitting indicated by the train vs. test R² gap

## Suggested improvements
- Add regime/volatility features (e.g., volatility indices, storage shocks, event flags)
- Try quantile regression / conformal prediction for uncertainty bounds
- Compare against time-series baselines (ARIMA/Prophet) and sequence models
- More robust handling of extreme events (e.g., custom loss, reweighting)

