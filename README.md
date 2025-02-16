# investment-portfolio-analysis
Project from the course Laboratory of Data Analytics for Investment

This project aims to identify the best factors for portfolio construction, optimize risk management, and evaluate portfolio strategies through various risk metrics and performance measures. The following steps were implemented:
- Select the best factors from the 5 Fama-French factors (market, size, value, profitability, investment) and some additional factors (momentum, industry, short-term and long-term reversal factors) with the Lasso Regression evaluating betas
- Comparison between the linear regression with the 5 FF factors and the one with the selected factors 
- Calculate the variance-covariance matrices to understand the relationships between factors
- Evaluate the risk using the volatility, Value-at-Risk, and Expected Shortfall and compare between the performance in-sample and out-of-sample
- Construct 6 portfolio strategies such as Equal-Weight, Equal-Weight Dynamic, Minimum Variance, Mean-Variance, Market-Cap Weighted, and Risk parity
- Evaluate the portfolios using the risk metrics such Annualized Return, Annualized Volatility, Drawdown, Maximum Drawdown, Efficiency and VaR
- Conduct a cross-portfolio analysis through factor analysis to assess the portfolio's sensitivity to systematic risk factors and residual analysis to measure the idiosyncratic risk

## Files
- `factor_analysis.ipynb` – Factor selection and regression analysis
- `portfolio_construction.py` – Portfolio optimization models
- `risk_evaluation.py` – Risk assessment metrics

## Technologies Used
- **Python**: pandas, NumPy, scikit-learn, statsmodels
- **Data Visualization**: Matplotlib, Seaborn
- **Portfolio Optimization**: scipy.optimize, CVXPY
