import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# To load data and transform it in csv
def load_data(path):
    df = pd.read_csv(path, index_col=0, dtype={0:"int64"})
    df.rename_axis("date", inplace=True)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df = df[~df.index.weekday.isin([5, 6])] #remove weekends
    return df

# To handle extra spaces in mom csv
def clean_csv(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    # Remove extra spaces after commas (both for negative and positive values)
    cleaned_lines = [line.replace(' ,', ',').replace(', ', ',').replace('   ', ' ').replace('  ', ' ') for line in lines]
    # Save the cleaned data to a new CSV file
    cleaned_path = "cleaned_" + path.split('/')[-1]  # Save the cleaned file with a new name
    with open(cleaned_path, 'w') as file:
        file.writelines(cleaned_lines)
    return cleaned_path

def load_data_web(path):
    df = pd.read_csv(path, index_col=0, dtype={0: "object"})
    df.rename_axis("date", inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors='coerce')
    df.sort_index(ascending=False, inplace=True)
    df /= 100
    return df

def load_data_indu(path):
    df = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
    df.rename_axis("date", inplace=True)
    df.index = pd.to_datetime(df.index, format="%Y%m%d", errors='coerce')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[~df.index.duplicated(keep="first")]
    df.fillna(df.mean(), inplace=True)
    df.sort_index(ascending=False, inplace=True)
    df /= 100
    df.rename(columns=lambda x: f"Industry_{x.strip()}", inplace=True)
    pca = PCA(n_components = 5)
    indu_factors = pca.fit_transform(df)
    df_pca = pd.DataFrame(indu_factors, index=df.index, columns=[f"Industry_Factor_{i+1}" for i in range(5)])
    return df_pca

def regression(dataset, target):
    # Splitting dataset
    X = dataset.drop(columns=[target])
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # Performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.10f}")
    # OLS Regression for beta coefficients & p-values
    X_train_sm = sm.add_constant(X_train_scaled)  # Add constant for intercept
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    results_df = pd.DataFrame({
        "Feature": ["Intercept"] + list(X.columns),
        "Beta Coefficient": model_sm.params,
        "P-Value": model_sm.pvalues
    })
    results_df["Significant (<0.05)?"] = results_df["P-Value"].apply(lambda p: "Yes" if p < 0.10 else "No")
    print("\nRegression Results:")
    print(results_df)
    # Residual Analysis
    residuals = y_test - y_pred
    residuals_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Residual": residuals})
    print("\nSum of Residuals:", sum(residuals_df["Residual"]))
    print("\nResidual Analysis:")
    print(residuals_df.describe())
    # Residuals Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.5, color="blue")
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

def lasso_reg(dataset, target):
    X = dataset.drop(columns=[target])
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lasso = LassoCV(cv=5,alphas=np.logspace(-4,1,50)).fit(X_train_scaled,y_train)
    y_pred = lasso.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_test)
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.10f}")
    print(f"Best Alpha (λ): {lasso.alpha_:.6f}")
    lasso_results = pd.DataFrame({
        "Feature": X.columns,
        "Lasso Coefficient": lasso.coef_
    })
    lasso_results["Selected"] = lasso_results["Lasso Coefficient"].apply(lambda coef: "Yes" if coef != 0 else "No")
    lasso_results_sorted = lasso_results[lasso_results["Selected"] == "Yes"].copy()
    lasso_results_sorted["Abs Coefficient"] = lasso_results_sorted["Lasso Coefficient"].abs()
    lasso_results_sorted = lasso_results_sorted.sort_values(by="Abs Coefficient", ascending=False).drop(columns=["Abs Coefficient"])
    print("\nLasso Regression Results (Selected Features Only, Sorted by Absolute Value):")
    print(lasso_results_sorted)

# To load data and transform it in csv
def load_daily(path):
    df = pd.read_csv(path, index_col=0, dtype={0:"int64"})
    df.rename_axis("date", inplace=True)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.fillna(0, inplace=True)
    df = df.mean(axis=1).round(18)
    df = df.to_frame(name="daily_return")
    return df

def clean_return_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.rename_axis("date", inplace=True)
    df = df[df.index.weekday < 5]    # Remove weekends
    # Drop rows with too many NaNs or zeros
    nan_threshold = 0.5 * df.shape[1]
    zero_threshold = 0.5 * df.shape[1]
    dates_with_too_many_nans = df.isnull().sum(axis=1) > nan_threshold
    dates_with_too_many_zeros = (df == 0).sum(axis=1) > zero_threshold
    dates_to_drop = df.index[dates_with_too_many_nans | dates_with_too_many_zeros]
    df = df.drop(index=dates_to_drop)
    # Drop columns with more than 80% missing data and fill remaining with forward and backward fill
    column_threshold = 0.8 * len(df)
    df = df.dropna(axis=1, thresh=column_threshold)
    df = df.ffill().bfill()
    return df

def clean_cap_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.rename_axis("date", inplace=True)
    df = df[df.index.weekday < 5]
    # Drop columns with more than 80% missing data and fill remaining with forward and backward fill
    column_threshold = 0.8 * len(df)
    df = df.dropna(axis=1, thresh=column_threshold)
    df = df.ffill().bfill()
    # Drop rows with more than 50% missing data
    nan_threshold = 0.5 * len(df.columns)
    dates_to_drop = df.index[df.isnull().sum(axis=1) > nan_threshold]
    df = df.drop(index=dates_to_drop)
    return df

ANNUALIZATION_CONSTANT = 261.

def VAR(returns, alpha=0.95):
    centered_returns = returns - returns.mean()
    sorted_returns = sorted(centered_returns)
    n = len(sorted_returns)
    i_alpha = round(n * (1 - alpha))
    var = -sorted_returns[i_alpha]
    return var * (ANNUALIZATION_CONSTANT ** 0.5)

def expected_shortfall(returns, alpha=0.95):
    var = VAR(returns, alpha) / (ANNUALIZATION_CONSTANT ** 0.5)
    below_var = returns[returns <= -var]
    es = -below_var.mean()
    return es * (ANNUALIZATION_CONSTANT ** 0.5)

# Dataset in in-sample e out-of-sample
def split_dataset(dataset, in_sample_ratio=0.8):
    split_point = int(len(dataset) * in_sample_ratio)
    in_sample = dataset.iloc[:split_point]
    out_of_sample = dataset.iloc[split_point:]
    return in_sample, out_of_sample

def annualized_total_return(df: pd.DataFrame) -> pd.Series:
    cumulative_return = (1 + df).cumprod()
    total_ret = cumulative_return.iloc[-1, :] / cumulative_return.iloc[0, :]
    n_days = df.shape[0]
    return total_ret ** (ANNUALIZATION_CONSTANT / n_days) - 1.

def annualized_volatility(df: pd.DataFrame) -> pd.Series:
    return df.std(axis=0) * (ANNUALIZATION_CONSTANT ** 0.5)

def efficiency(df: pd.DataFrame) -> pd.Series:
    avg_ret = df.mean(axis=0) * ANNUALIZATION_CONSTANT
    vol = df.std(axis=0) * (ANNUALIZATION_CONSTANT ** 0.5)
    return avg_ret / vol

def drawdown(df: pd.DataFrame) -> pd.DataFrame:
    cumulative_return = (1 + df).cumprod()
    return cumulative_return / cumulative_return.cummax() - 1

def maximum_drawdown(df: pd.DataFrame) -> pd.Series:
    dd = drawdown(df)
    return dd.min(axis=0)

def value_at_risk(df: pd.DataFrame, alpha=0.05) -> pd.Series:
    centered_returns = df - df.mean(axis=0)  # Center the returns
    var_values = []
    for p in df.columns:
        sorted_returns = np.sort(centered_returns[p].values)  # Sorted from worst to best
        i_alpha = int(round(len(sorted_returns) * alpha))
        var_values.append(-sorted_returns[i_alpha])
    return pd.Series(var_values, index=df.columns) * (ANNUALIZATION_CONSTANT ** 0.5)

def evolve_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    # Updates portfolio weights dynamically based on realized returns.
    returns = np.nan_to_num(returns, nan=0.)   # Handle NaNs
    w_tilde = weights * (returns + 1.)  # Compute new weights pre-normalization
    w_divisor = ((weights * returns).sum() + 1)  # Normalization factor
    return w_tilde / w_divisor