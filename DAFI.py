from formulas import (load_data, clean_csv, load_data_web, load_data_indu, regression, lasso_reg,
                      load_daily, clean_return_data, clean_cap_data, annualized_total_return,
                      annualized_volatility, efficiency, drawdown, maximum_drawdown, value_at_risk,
                      expected_shortfall, split_dataset, VAR, evolve_weights)
import pandas as pd
from pandas import read_csv
import statsmodels.api as sm
import numpy as np
import cvxpy as cp
from cvxpy import psd_wrap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# Load necessary datasets

# SPX dataset
path = "data/SPX/daily_return_mat.csv"
spx = load_daily(path)

# Fama French 5 factors
path1 = "data/SX5E/ff_5_factors_returns_mat.csv"
ff5 = load_data(path1)

# Dataset 1: spx + Fama French
dataset1 = ff5.join([spx], how="inner")

# Momentum
path_mom = "data/mom.csv"
cleaned_path = clean_csv(path_mom)
mom = load_data_web(cleaned_path)

# Short Term Reversal Factor
path_strf = "data/strf.csv"
cleaned_path1 = clean_csv(path_strf)
strf = load_data_web(cleaned_path1)

# Long Term Reversal Factor
path_ltrf = "data/ltrf.csv"
cleaned_path2 = clean_csv(path_ltrf)
ltrf = load_data_web(cleaned_path2)

# Industry
path_indu = "data/industry.csv"
cleaned_path3 = clean_csv(path_indu)
industry = load_data_indu(cleaned_path3)

# Datset 2: all factors + daily returns
full = ff5.join([mom,strf,ltrf,industry,spx], how="inner")
print(full.info())

# Linear Regression and Lasso Regression - to select the best factors
regression(full,"daily_return")
lasso_reg(full,"daily_return")

# Dataset 3: SPX + selected factors
dataset3 = ff5.join([industry,mom,spx])
dataset3 = dataset3.drop(["SMB", "HML", "CMA", "Industry_Factor_3", "Industry_Factor_4", "Industry_Factor_5",], axis=1)
dataset3 = dataset3.dropna()

# Comparative regressions
regression(dataset1,"daily_return")
regression(dataset3,"daily_return")

#########################################
# Factors Analysis

# Var-covar Matrix
factors_dataset1 = dataset1[['Mkt-RF','SMB','HML', 'RMW','CMA']]
factors_dataset3 = dataset3[['Mkt-RF', 'RMW', 'Industry_Factor_1', 'Industry_Factor_2', 'MOM ']]
cov_matrix_dataset1 = factors_dataset1.cov()
cov_matrix_dataset3 = factors_dataset3.cov()
print("Matrice Var-Covar Dataset1:\n", cov_matrix_dataset1)
print("Matrice Var-Covar Dataset3:\n", cov_matrix_dataset3)

# Plot covariance matrix for dataset1
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_dataset1, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Covariance Matrix - Dataset1")
plt.show()
# Plot covariance matrix for dataset3
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_dataset3, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Covariance Matrix - Dataset3")
plt.show()

# Regression - beta ex-ante vs beta ex-post
X1 = factors_dataset1.values
y1 = dataset1['daily_return'].values
split_point_1 = int(len(dataset1) * 0.8) # Division in data Ex Ante (80%) and Ex Post (100%)
X1_ex_ante = X1[:split_point_1]
y1_ex_ante = y1[:split_point_1]

reg1_ex_ante = LinearRegression().fit(X1_ex_ante, y1_ex_ante)
beta_ex_ante_dataset1 = reg1_ex_ante.coef_
reg1_ex_post = LinearRegression().fit(X1, y1)
beta_ex_post_dataset1 = reg1_ex_post.coef_

X3 = factors_dataset3.values
y3 = dataset3['daily_return'].values
split_point_3 = int(len(dataset3) * 0.8)
X3_ex_ante = X3[:split_point_3]
y3_ex_ante = y3[:split_point_3]

reg3_ex_ante = LinearRegression().fit(X3_ex_ante, y3_ex_ante)
beta_ex_ante_dataset3 = reg3_ex_ante.coef_
reg3_ex_post = LinearRegression().fit(X3, y3)
beta_ex_post_dataset3 = reg3_ex_post.coef_

print("Beta Ex Ante (Dataset1):", beta_ex_ante_dataset1)
print("Beta Ex Post (Dataset1):", beta_ex_post_dataset1)
print("Beta Ex Ante (Dataset3):", beta_ex_ante_dataset3)
print("Beta Ex Post (Dataset3):", beta_ex_post_dataset3)

# Risk Metrics
returns = dataset3['daily_return']
in_sample_returns, out_of_sample_returns = split_dataset(returns, in_sample_ratio=0.8)

# Risk Metrics in-sample (80%)
in_sample_volatility = annualized_volatility(in_sample_returns)
in_sample_var95 = VAR(in_sample_returns, alpha=0.95)
in_sample_es95 = expected_shortfall(in_sample_returns, alpha=0.95)

# Risk Metrics out-of-sample (20%)
out_of_sample_volatility = annualized_volatility(out_of_sample_returns)
out_of_sample_var95 = VAR(out_of_sample_returns, alpha=0.95)
out_of_sample_es95 = expected_shortfall(out_of_sample_returns, alpha=0.95)

print("Risk Metrics In-Sample:")
print("Annualized Volatility:", in_sample_volatility)
print("VaR 95%:", in_sample_var95)
print("Expected Shortfall 95%:", in_sample_es95)

print("\nRisk Metrics Out-of-Sample:")
print("Annualized Volatility:", out_of_sample_volatility)
print("VaR 95%:", out_of_sample_var95)
print("Expected Shortfall 95%:", out_of_sample_es95)

# Comparison between in-sample and out-of-sample
print("\nComparison between In-Sample and Out-of-Sample:")
print(f"Volatility: In-Sample = {in_sample_volatility:.4f}, Out-of-Sample = {out_of_sample_volatility:.4f}")
print(f"VaR 95%: In-Sample = {in_sample_var95:.4f}, Out-of-Sample = {out_of_sample_var95:.4f}")
print(f"ES 95%: In-Sample = {in_sample_es95:.4f}, Out-of-Sample = {out_of_sample_es95:.4f}")

# Bar plot for risk metrics
risk_metrics_df = pd.DataFrame({
    'Metric': ['Volatility', 'VaR 95%', 'ES 95%', 'Volatility', 'VaR 95%', 'ES 95%'],
    'Period': ['In-Sample', 'In-Sample', 'In-Sample', 'Out-of-Sample', 'Out-of-Sample', 'Out-of-Sample'],
    'Value': [in_sample_volatility, in_sample_var95, in_sample_es95, out_of_sample_volatility, out_of_sample_var95, out_of_sample_es95]
})
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', hue='Period', data=risk_metrics_df, palette='tab10')
plt.title("Comparison of Risk Metrics (In-Sample vs Out-of-Sample)")
plt.ylabel("Value")
plt.xlabel("Metric")
plt.show()

# Betas
beta_ex_ante_df = pd.DataFrame({
    'Factors': factors_dataset1.columns,
    'Fama-French Ex Ante': beta_ex_ante_dataset1,
    'New Factors Ex Ante': beta_ex_ante_dataset3
})

beta_ex_post_df = pd.DataFrame({
    'Factors': factors_dataset1.columns,
    'Fama-French Ex Post': beta_ex_post_dataset1,
    'New Factors Ex Post': beta_ex_post_dataset3
})
plt.figure(figsize=(10, 6))
plt.plot(beta_ex_ante_df['Factors'], beta_ex_ante_df['Fama-French Ex Ante'], label='Fama-French Ex Ante', marker='o', color='red')
plt.plot(beta_ex_ante_df['Factors'], beta_ex_ante_df['New Factors Ex Ante'], label='New Factors Ex Ante', marker='o', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.title('Comparison of Beta Ex Ante between Fama-French and New Factors')
plt.xlabel('Factors')
plt.ylabel('Beta')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(beta_ex_post_df['Factors'], beta_ex_post_df['Fama-French Ex Post'], label='Fama-French Ex Post', marker='o', color='red')
plt.plot(beta_ex_post_df['Factors'], beta_ex_post_df['New Factors Ex Post'], label='New Factors Ex Post', marker='o', color='green')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend()
plt.title('Comparison of Beta Ex Post between Fama-French and New Factors')
plt.xlabel('Factors')
plt.ylabel('Beta')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_dataset1, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Var-Covar Matrix - Dataset1')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_dataset3, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Var-Covar Matrix - Dataset3')
plt.show()

#########################################
# Portfolio Construction

# Load the basket composition dataset
comp_df = pd.read_csv("data/SPX/BasketModel_compositionMat.csv", index_col=0)
comp_df.index = pd.to_datetime(comp_df.index.astype(str), format="%Y%m%d")
comp_df.rename_axis("date", inplace=True)
comp_df = comp_df[comp_df.index.weekday < 5]    # Remove weekends
comp_df = comp_df.sort_index()

# Load the returns dataset
return_df = clean_return_data("data/SPX/daily_return_mat.csv")
return_df = return_df[return_df.index.weekday < 5]    # Remove weekends
return_df = return_df.sort_index()

# Ensure both datasets have the same dates and assets
comp_df = comp_df.loc[return_df.index]
common_assets = comp_df.columns.intersection(return_df.columns)
return_df = return_df[common_assets]
comp_df = comp_df[common_assets]

# 1. Equal-Weight Portfolio
# Count number of active stocks per day (stocks with "1" in composition)
num_active_stocks = comp_df.sum(axis=1)
# Compute equal-weighted portfolio returns
equal_weight_returns = (return_df * comp_df).sum(axis=1) / num_active_stocks
equal_weight_portfolio = pd.DataFrame(equal_weight_returns, columns=['Equal-Weight Return'])
equal_weight_cum_performance = (equal_weight_portfolio + 1).cumprod()

# 2. Equal-Weight Dynamic Portfolio
num_active_stocks = comp_df.sum(axis=1).values  # Number of active stocks per day
weights = comp_df.iloc[0] / num_active_stocks[0]  # Equal weights at t=0
weights_dynamic = [weights.copy()]  # Store initial weights
dynamic_equal_weight_returns = []
# Iterate over time, updating weights dynamically
for i in range(1, len(return_df)):
    weights = evolve_weights(weights, return_df.iloc[i].values)  # Update weights
    weights_dynamic.append(weights.copy())  # Store updated weights
    portfolio_return = np.dot(weights, return_df.iloc[i].values)
    dynamic_equal_weight_returns.append(portfolio_return)
weights_dynamic_df = pd.DataFrame(weights_dynamic, index=return_df.index, columns=return_df.columns)
dynamic_equal_weight_portfolio = pd.DataFrame(dynamic_equal_weight_returns, index=return_df.index[1:], columns=['Equal-Weight Dynamic'])

# 3. Minimum Variance Portfolio
cov_matrix = return_df.cov()
cov_matrix = psd_wrap(cov_matrix)   # Wrap covariance matrix to enforce PSD property
# Define optimization problem
n = return_df.shape[1]
w = cp.Variable(n)
objective = cp.Minimize(cp.quad_form(w, cov_matrix))  # Minimize variance
constraints = [cp.sum(w) == 1, w >= 0]  # Fully invested, no shorting
# Solve optimization
problem = cp.Problem(objective, constraints)
problem.solve()
min_var_weights = w.value
# Compute portfolio returns
min_var_portfolio = (return_df * min_var_weights).sum(axis=1)
min_var_portfolio = pd.DataFrame(min_var_portfolio, columns=['Minimum Variance Return'])

# 4. Mean-Variance (Markowitz) Portfolio
expected_returns = return_df.mean().values
# Define optimization problem
objective = cp.Maximize(expected_returns @ w - 0.5 * cp.quad_form(w, cov_matrix))  # Maximize return-risk tradeoff
problem = cp.Problem(objective, constraints)
problem.solve()
mean_var_weights = w.value
# Compute portfolio returns
mean_var_portfolio = (return_df * mean_var_weights).sum(axis=1)
mean_var_portfolio = pd.DataFrame(mean_var_portfolio, columns=['Mean-Variance Return'])

# 5. Market-Cap Weighted Portfolio
# Load market capitalization data
cap_df = clean_cap_data("data/SPX/capitalization_mat.csv")
cap_df = cap_df.sort_index()
# Ensure both datasets have the same dates and assets
cap_df = cap_df.loc[return_df.index]
common_assets = cap_df.columns.intersection(return_df.columns)
cap_df = cap_df[common_assets]
return_df_2 = return_df[common_assets]
weights = cap_df.div(cap_df.sum(axis=1), axis=0)  # Compute market-cap weights
# Compute portfolio returns
market_cap_portfolio = (return_df_2 * weights).sum(axis=1)
market_cap_portfolio = pd.DataFrame(market_cap_portfolio, columns=['Market-Cap Weighted Return'])

# 6. Risk Parity Portfolio
# Compute asset volatility (standard deviation of returns)
asset_volatility = return_df.std()
risk_parity_weights = 1 / asset_volatility   # Compute risk-based weights (inverse of volatility)
risk_parity_weights /= risk_parity_weights.sum()   # Normalize weights to sum to 1
# Compute risk parity weighted portfolio returns
risk_parity_returns = (return_df * risk_parity_weights).sum(axis=1)
risk_parity_portfolio = pd.DataFrame(risk_parity_returns, columns=['Risk Parity Return'])

#########################################
# Compare the performance of the 6 portfolios

# Combine all portfolio returns into one DataFrame
portfolios_df = pd.concat([
    equal_weight_portfolio,
    dynamic_equal_weight_portfolio,
    min_var_portfolio,
    mean_var_portfolio,
    market_cap_portfolio,
    risk_parity_portfolio
], axis=1)
portfolios_df.columns = ["Equal-Weight","Dynamic", "Min Variance", "Mean-Variance", "Market-Cap", "Risk Parity"]
portfolios_df = portfolios_df.loc["2003-03-18":"2018-12-31"]
portfolios_df = portfolios_df.dropna()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(portfolios_df.columns):
    portfolios_df[col].plot(ax=axes[i], title=col, linewidth=2, color='cornflowerblue')
    axes[i].set_xlabel("Date")
    axes[i].set_ylabel("Portfolio Value")
    axes[i].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Metrics
dd = drawdown(portfolios_df)
dd.plot(linewidth=0.8)
plt.title("Drawdown of Portfolios Over Time")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend(loc="lower right", title="Portfolios")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

performance_metrics = pd.DataFrame({
    "Annualized Return": annualized_total_return(portfolios_df),
    "Annualized Volatility": annualized_volatility(portfolios_df),
    "Efficiency": efficiency(portfolios_df),
    "Max Drawdown": maximum_drawdown(portfolios_df),
    "VaR (95%)": value_at_risk(portfolios_df, alpha=0.05),
})
print(performance_metrics)

metrics = ["Annualized Return", "Annualized Volatility", "Efficiency", "Max Drawdown", "VaR (95%)"]
for metric in metrics:
    plt.figure(figsize=(10, 8))
    ax = performance_metrics[metric].sort_values(ascending=True).plot(kind="bar", color='cornflowerblue', edgecolor='black')
    plt.title(f"{metric} Across Portfolios")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(performance_metrics[metric].sort_values(ascending=True)):
        ax.text(i, v + 0.01 * max(performance_metrics[metric]), f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    plt.show()

#########################################
# Cross-portfolio analysis

# Factor regression for each portfolio
factors_dataset3 = sm.add_constant(factors_dataset3)   # Ensure factors dataset has an intercept for regression
portfolios_df, factors_dataset3 = portfolios_df.align(factors_dataset3, join='inner', axis=0)   # Align index
betas = {}
residuals = {}
r_squared = {}
for portfolio in portfolios_df.columns:
    X = factors_dataset3
    y = portfolios_df[portfolio]
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)  # Predicted returns
    residuals[portfolio] = y - y_pred
    betas[portfolio] = reg.coef_  # Store the betas
    r_squared[portfolio] = reg.score(X, y)

# Betas
betas_df = pd.DataFrame(betas, index=factors_dataset3.columns)
print(f"Betas:")
print(betas_df)
betas_df_without_constant = betas_df.drop('const', axis=0)   # Exclude the intercept
# Plot betas for each portfolio
betas_df_without_constant.plot(kind='bar', figsize=(10, 6), edgecolor='black')
plt.title('Factor Exposures (Betas) for Each Portfolio')
plt.ylabel('Beta')
plt.xlabel('Factors')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Residuals
residuals_df = pd.DataFrame(residuals)
cov_matrix_residuals = residuals_df.cov()
print(cov_matrix_residuals)
# Plot the covariance matrix of residuals using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cov_matrix_residuals, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Covariance Matrix of Residuals")
plt.xlabel('Portfolios')
plt.xticks(rotation=25)
plt.ylabel('Portfolios')
plt.show()

# Correlation between portfolios based on factor exposures
factor_correlation = betas_df.corr()
print(factor_correlation)
# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(factor_correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Portfolios Based on Factor Exposure")
plt.xticks(rotation=25)
plt.show()

# R-squared
r_squared_df = pd.DataFrame(list(r_squared.items()), columns=['Portfolio', 'R-squared'])
# Print the R-squared values
print(f"R-squared:")
print(r_squared_df)

# Risk evaluation
# Historical Risk: Standard deviation of portfolio returns
historical_risk = portfolios_df.std()
# Factorial Risk: Standard deviation of residuals (unexplained risk)
factorial_risk = residuals_df.std()
# Robust Risk: Median Absolute Deviation (MAD)
def mad(x):
    return np.median(np.abs(x - np.median(x)))
robust_risk = residuals_df.apply(mad)
# Combine all risk metrics
risk_df = pd.DataFrame({
    'Historical Risk': historical_risk,
    'Factorial Risk': factorial_risk,
    'Robust Risk': robust_risk
})
print(risk_df)
# Bar plot for risk comparison
risk_df.plot(kind='bar', figsize=(8, 6))
plt.title('Comparison of Risk Metrics Across Portfolios')
plt.ylabel('Risk')
plt.xlabel('Portfolios')
plt.xticks(rotation=25, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()