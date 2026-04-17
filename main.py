"""
===============================================================================
                VAR MODEL FROM SCRATCH - US AGGREGATE DEMAND ANALYSIS
===============================================================================
This script implements a full Vector Autoregression (VAR) model without using
any pre-built VAR or OLS libraries. It:
  1. Fetches US macroeconomic data from FRED.
  2. Preprocesses data (missing value handling, stationarity tests, transformations).
  3. Selects optimal lag order using AIC/BIC (implemented from scratch).
  4. Estimates the VAR model equation‑by‑equation using OLS (numpy only).
  5. Produces multi‑step forecasts (8 quarters ahead).
  6. Computes impulse response functions (IRF) via Cholesky decomposition.
  7. Plots forecasts and IRFs.

Author: Andrey Tolkushkin
Date:   2026-03-30
===============================================================================
"""

import os
import platform
if "OPENBLAS_CORETYPE" not in os.environ:
    os.environ["OPENBLAS_CORETYPE"] = "Haswell" if platform.machine() == "x86_64" else "ARMV8"

import numpy as np
import numpy.testing as npt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import chi2, norm, probplot  # diagnostics; MacKinnon ADF; Q-Q plots
from pandas_datareader import data as pdr
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')


# =============================================================================
# 0. SETUP
# =============================================================================

@dataclass
class Settings:
    fred_api_key =  "https://api.stlouisfed.org/fred/series/search?api_key=03511afc746e1fb32a12790f9ba8155b&search_text=canada"
    start_date = '1960-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    max_lags = 8
    prediction_steps = 8
   
settings = Settings()
    
    
# =============================================================================
# 1. DATA RETRIEVAL FROM FRED
# =============================================================================
# FRED API key is stored in the environment variable FRED_API_KEY
# This is optional as FRED database allows access to data for limited searches without api key
FRED_API_KEY = settings.fred_api_key    

# Define series IDs and variable names (coursework aggregate demand specification)
SERIES = {
    'GDPC1':    'Реальный ВВП',
    'PCE':      'Личное потребление',
    'GPDI':     'Валовые частные инвестиции',
    'GCE':      'Государственное потребление и инвестиции',
    'EXPGSC1':  'Реальный экспорт',
    'GDPDEF':   'Дефлятор ВВП',
    'FEDFUNDS': 'Ставка по федеральным фондам',
}

variables = ['GDPC1', 'PCE', 'GPDI', 'GCE', 'EXPGSC1', 'GDPDEF', 'FEDFUNDS']
log_vars = ['GDPC1', 'PCE', 'GPDI', 'GCE', 'EXPGSC1', 'GDPDEF']
# Check that all 7 variables are present in the SERIES dictionary
assert len(variables) == 7

def fetch_fred_data(series_ids, start_date=settings.start_date, end_date=None):
    """
    Fetch multiple FRED time series using pandas_datareader.
    Returns a pandas DataFrame with columns as series IDs.
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    data = pd.DataFrame()
    for sid in series_ids:
        try:
            # Use FRED API with your key
            if FRED_API_KEY:
                series = pdr.DataReader(sid, 'fred', start_date, end_date, api_key=FRED_API_KEY)
            else:
                series = pdr.DataReader(sid, 'fred', start_date, end_date)
            data[sid] = series[sid]
            print(f"+ Retrieved {sid}: {SERIES.get(sid, sid)}")
        except Exception as e:
            print(f"- Failed to fetch {sid}: {e}")
    
    return data

# Fetch the data
print("\n" + "="*80)
print("STEP 1: Fetching data from FRED...")
print("="*80)
df_raw = fetch_fred_data(list(SERIES.keys()))
if df_raw.empty:
    raise RuntimeError(
        "No FRED series were retrieved. Check internet access and FRED API key configuration."
    )
# Assert real export and gdp deflator are present in the needed spots
assert 'EXPGSC1' in df_raw.columns and 'GDPDEF' in df_raw.columns
print(f"\nVAR variables (coursework AD): {variables}")
print(f"Data shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min().date()} to {df_raw.index.max().date()}")

# =============================================================================
# 2. PREPROCESSING: ALIGN FREQUENCIES, HANDLE MISSING VALUES, TRANSFORMATIONS
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Preprocessing data...")
print("="*80)

# Convert to quarterly frequency (any higher-frequency series, e.g. FEDFUNDS, are averaged within quarter)
df_quarterly = df_raw.resample('Q').mean()

# Handle any remaining missing values: forward fill then backward fill 
# (replace any nan data with placeholder data from the previous lag)
df_quarterly.ffill(inplace=True)
df_quarterly.bfill(inplace=True)

print("df_quaterly", df_quarterly.head())

print("Missing values after cleaning:", df_quarterly.isnull().sum().sum())

data_raw = df_quarterly[variables].copy()

# Convert data to logarithmic scale
for var in log_vars:
    data_raw[var] = np.log(data_raw[var])

assert data_raw['GDPDEF'].mean() < 10, "GDPDEF should be log scale (mean << raw index level)"
print(f"\nTransformations applied: natural log for {log_vars}")
print("FEDFUNDS (interest rate) kept in levels.")

# =============================================================================
# 3. STATIONARITY TEST (ADF) AND DIFFERENCING
# =============================================================================
def _mackinnon_crit_constant_n1(nobs):
    """
    MacKinnon response‑surface critical values, ADF with constant (no trend), N=1.
    Uses 2010 coefficient update (same response‑surface form as MacKinnon 1994; see
    MacKinnon 2010 Working Paper 1227 / statsmodels.tsa.adfvalues).
    Returns [cv_1%, cv_5%, cv_10%] for the τ statistic (t‑ratio on y_{t-1}).
    """
    # Rows: 1%, 5%, 10%; cols: c0, c1/T, c2/T^2, c3/T^3
    tau_c_2010_n1 = np.array([
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8903, -4.234, -40.040],
        [-2.56677, -1.5384, -2.809, 0.0],
    ])
    if not np.isfinite(nobs) or nobs <= 0:
        return tau_c_2010_n1[:, 0]
    inv_t = 1.0 / float(nobs)
    out = np.empty(3)
    for i in range(3):
        coef_desc = tau_c_2010_n1[i, ::-1]  # np.polyval expects highest power first
        out[i] = np.polyval(coef_desc, inv_t)
    return out


def _mackinnon_p_constant_n1(tau_stat):
    """
    MacKinnon approximate p‑value for τ (ADF t‑stat on y_{t-1}), regression with
    constant, N=1 (MacKinnon 1994; polynomial tables as in statsmodels tsa/adfvalues).
    """
    tau_star_c = -1.61
    tau_min_c = -18.83
    tau_max_c = 2.74
    small_scaling = np.array([1.0, 1.0, 1e-2])
    tau_c_smallp = np.array([2.1659, 1.4412, 3.8269]) * small_scaling
    large_scaling = np.array([1.0, 1e-1, 1e-1, 1e-2])
    tau_c_largep = np.array([1.7339, 9.3202, -1.2745, -1.0368]) * large_scaling
    if tau_stat > tau_max_c:
        return 1.0
    if tau_stat < tau_min_c:
        return 0.0
    if tau_stat <= tau_star_c:
        coef = tau_c_smallp[::-1]
    else:
        coef = tau_c_largep[::-1]
    return float(norm.cdf(np.polyval(coef, tau_stat)))


def adf_test(series, max_lags=None, significance_level=0.05):
    """
    Augmented Dickey‑Fuller test from scratch.
    This step is needed to determine if the series is stationary or not.
    Uses OLS estimation of the ADF regression with lagged differences.
    Returns (test_statistic, p_value, is_stationary).
    p_value: MacKinnon (1994) response‑surface approximation (constant, N=1).
    is_stationary: reject unit root at 5% if τ < MacKinnon 5% critical value
    (one‑sided left tail; significance_level kept for API compatibility only).
    """
    y_full = series.dropna().values
    T = len(y_full)
    if T < 10:
        return (np.nan, np.nan, False)

    # Schwert (1989) lag length rule: int(12 * (T/100)^(1/4))
    if max_lags is None:
        max_lags = int(12 * (T / 100.0) ** (1.0 / 4.0))

    y = y_full
    dy = np.diff(y)  # length T-1
    max_lags = max(1, min(max_lags, len(dy) - 2))

    # ADF regression:
    # Δy_t = c + gamma*y_{t-1} + Σ_{i=1..p} alpha_i*Δy_{t-i} + e_t
    dy_target = dy[max_lags:]
    y_lag1 = y[max_lags:-1]
    lagged_terms = [dy[max_lags - i : -i] for i in range(1, max_lags + 1)]

    X_mat = np.column_stack([np.ones(len(dy_target)), y_lag1] + lagged_terms)
    y_target = dy_target
    T_adf = len(y_target)

    try:
        beta = np.linalg.lstsq(X_mat, y_target, rcond=None)[0]
        resid = y_target - X_mat @ beta
        dof = max(1, T_adf - X_mat.shape[1])
        sigma2 = np.sum(resid**2) / dof
        xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
        se = np.sqrt(sigma2 * np.diag(xtx_inv))
        tau_stat = beta[1] / se[1]  # τ statistic on y_{t-1} (MacKinnon tables)
    except Exception:
        return (np.nan, np.nan, False)

    crit_1_5_10 = _mackinnon_crit_constant_n1(T_adf)
    p_value = _mackinnon_p_constant_n1(tau_stat)
    # One‑sided test H0: unit root; reject (stationary) if τ < cv_5%
    is_stationary = tau_stat < crit_1_5_10[1]
    # Decision per coursework: 5% MacKinnon critical value (significance_level unused).

    return (tau_stat, p_value, is_stationary)

# Test each series
print("\nAugmented Dickey‑Fuller (constant): MacKinnon p‑value; stationary if τ < 5% critical value:")
stationarity_results = {}
adf_tau_by_var = {}
for var in variables:
    stat, pval, is_stat = adf_test(data_raw[var])
    adf_tau_by_var[var] = stat
    stationarity_results[var] = is_stat
    print(f"  {var:12s} : t-stat = {stat:8.4f}, p-value = {pval:.4f} -> {'Stationary' if is_stat else 'Non‑stationary'}")

# A1 — MacKinnon critical values vs τ (same T_adf construction as adf_test; coursework table)
print("\nADF vs MacKinnon critical values (+ = τ < CV -> reject unit root at that level):")
_hdr = (
    f"{'Variable':<10} | {'t-stat':>8} | {'CV 1%':>8} | {'CV 5%':>8} | {'CV 10%':>8} | "
    f"{'1%':^4} | {'5%':^4} | {'10%':^4} | Verdict"
)
print(_hdr)
print("-" * len(_hdr))
# Clean the data of any NaN values and calculate the t-statistic and print the verdict
# The verdict is based on the t-statistic and the MacKinnon critical values
# If the t-statistic is less than the MacKinnon critical value, the series is stationary
# If the t-statistic is greater than the MacKinnon critical value, the series is non-stationary
# The verdict is printed in the table
# The table is printed in a format that is easy to read
for var in variables:
    tau = adf_tau_by_var[var]
    y_full = data_raw[var].dropna().values
    if len(y_full) < 10 or not np.isfinite(tau):
        print(f"{var:<10} | {'n/a':>8} | {'n/a':>8} | {'n/a':>8} | {'n/a':>8} |  —  |  —  |  —  | n/a")
        continue
    Tlen = len(y_full)
    ml = int(12 * (Tlen / 100.0) ** (1.0 / 4.0))
    dy = np.diff(y_full)
    ml = max(1, min(ml, len(dy) - 2))
    T_adf = len(dy[ml:])
    cv1, cv5, cv10 = _mackinnon_crit_constant_n1(T_adf)
    c1 = "+" if tau < cv1 else "-"
    c5 = "+" if tau < cv5 else "-"
    c10 = "+" if tau < cv10 else "-"
    verdict = "Stationary" if stationarity_results[var] else "Non-stationary"
    print(
        f"{var:<10} | {tau:8.4f} | {cv1:8.4f} | {cv5:8.4f} | {cv10:8.4f} | "
        f" {c1:^3} | {c5:^3} | {c10:^3} | {verdict}"
    )

# Apply differencing to non‑stationary series
data_diff = data_raw.copy()
for var in variables:
    if not stationarity_results[var]:
        data_diff[var] = data_raw[var].diff()
        print(f"  Applied 1st difference to {var}")

# Drop initial NaN from differencing
data_diff.dropna(inplace=True)
print(f"\nFinal dataset shape after differencing: {data_diff.shape}")

# =============================================================================
# 3B. ADF ON FIRST DIFFERENCES (confirm stationarity after differencing)
# =============================================================================
print("\nAugmented Dickey‑Fuller on FIRST DIFFERENCES: confirming stationarity:")
adf_diff_results = {}
adf_diff_tau = {}
for var in variables:
    stat, pval, is_stat = adf_test(data_diff[var], max_lags=4)
    adf_diff_results[var] = is_stat
    adf_diff_tau[var] = stat
    print(f"  {var:12s} : t-stat = {stat:8.4f}, p-value = {pval:.4f} -> {'Stationary' if is_stat else 'Non‑stationary'}")
 
print("\nADF (first differences) vs MacKinnon critical values:")
_hdr2 = (
    f"{'Variable':<10} | {'t-stat':>8} | {'CV 1%':>8} | {'CV 5%':>8} | {'CV 10%':>8} | "
    f"{'1%':^4} | {'5%':^4} | {'10%':^4} | Verdict"
)
print(_hdr2)
print("-" * len(_hdr2))
for var in variables:
    tau = adf_diff_tau[var]
    y_full = data_diff[var].dropna().values
    if len(y_full) < 10 or not np.isfinite(tau):
        print(f"{var:<10} | {'n/a':>8} | {'n/a':>8} | {'n/a':>8} | {'n/a':>8} |  —  |  —  |  —  | n/a")
        continue
    Tlen = len(y_full)
    ml = int(12 * (Tlen / 100.0) ** (1.0 / 4.0))
    dy = np.diff(y_full)
    ml = max(1, min(ml, len(dy) - 2))
    T_adf = len(dy[ml:])
    cv1, cv5, cv10 = _mackinnon_crit_constant_n1(T_adf)
    c1 = "+" if tau < cv1 else "-"
    c5 = "+" if tau < cv5 else "-"
    c10 = "+" if tau < cv10 else "-"
    verdict = "Stationary" if adf_diff_results[var] else "Non-stationary"
    print(
        f"{var:<10} | {tau:8.4f} | {cv1:8.4f} | {cv5:8.4f} | {cv10:8.4f} | "
        f" {c1:^3} | {c5:^3} | {c10:^3} | {verdict}"
    )

# Запустите отдельно для диагностики
problem_vars = ['PCE', 'GCE', 'GDPDEF']
print("ADF на ВТОРЫХ разностях для проблемных переменных:")
for var in problem_vars:
    diff2 = data_diff[var].diff().dropna()
    stat, pval, is_stat = adf_test(diff2)
    print(f"  {var:12s} : t-stat = {stat:8.4f}, p-value = {pval:.4f} -> {'Stationary' if is_stat else 'Non-stationary'}")

print("\nADF с фиксированным числом лагов = 4 (квартальные данные, стандарт):")
for var in problem_vars:
    stat, pval, is_stat = adf_test(data_diff[var], max_lags=4)
    print(f"  {var:12s} : t-stat = {stat:8.4f}, p-value = {pval:.4f} -> {'Stationary' if is_stat else 'Non-stationary'}")
# =============================================================================
# 4. LAG ORDER SELECTION USING AIC / BIC (from scratch)
# =============================================================================
def compute_aic_bic(y, max_lags=settings.max_lags):
    """
    VAR(p) information criteria per coursework notation (Zivot-style IC for VAR).
    T_eff = effective sample size = T - p.
    n = number of endogenous variables (= k).
    m = n * p (parameters per equation attributable to lags, coursework definition).
    AIC(p)  = ln|Σ(p)| + (2/T_eff) * m^2
    BIC(p)  = ln|Σ(p)| + (ln(T_eff)/T_eff) * m^2
    HQ(p)   = ln|Σ(p)| + (2*ln(ln(T_eff))/T_eff) * m^2
    """
    T, k = y.shape
    n = k
    aic_vals = {}
    bic_vals = {}
    hq_vals = {}

    for p in range(1, max_lags + 1):
        # Build Y (dependent) and X (lagged regressors)
        Y = y[p:, :]                     # shape (T-p, k)
        X = np.array([y[t-p:t, :].flatten() for t in range(p, T)])   # rows: all lags
        X = np.column_stack([np.ones(X.shape[0]), X])   # add constant

        # OLS equation by equation
        residuals = np.zeros((T-p, k))
        for j in range(k):
            beta = np.linalg.lstsq(X, Y[:, j], rcond=None)[0]
            residuals[:, j] = Y[:, j] - X @ beta

        # Residual covariance matrix (ML / Lütkepohl-style divisor T_eff)
        T_eff = T - p
        sigma_u = (residuals.T @ residuals) / T_eff
        _, log_det_sigma = np.linalg.slogdet(sigma_u)

        m = n * p  # coursework: parameters per equation scaling term
        # Zivot — VAR information criteria (penalty on ln|Σ|)
        aic_vals[p] = log_det_sigma + (2.0 / T_eff) * (m ** 2)
        bic_vals[p] = log_det_sigma + (np.log(T_eff) / T_eff) * (m ** 2)
        hq_vals[p] = log_det_sigma + (2.0 * np.log(np.log(T_eff)) / T_eff) * (m ** 2)

    return aic_vals, bic_vals, hq_vals

print("\n" + "="*80)
print("STEP 3: Selecting optimal lag order using AIC, BIC, and HQ...")
print("="*80)
   
aic_dict, bic_dict, hq_dict = compute_aic_bic(data_diff.values, max_lags=settings.max_lags)

print("Lag order selection criteria:")
print("  Lag    AIC        BIC        HQ")
for p in range(1, settings.max_lags+1):
    print(f"  {p:2d}   {aic_dict[p]:8.4f}   {bic_dict[p]:8.4f}   {hq_dict[p]:8.4f}")

# Choose optimal lag based on AIC (or BIC)
p_opt = min(aic_dict, key=aic_dict.get)
p_opt_bic = min(bic_dict, key=bic_dict.get)
p_opt_hq = min(hq_dict, key=hq_dict.get)
print(f"\nOptimal lag order: p = {p_opt} (AIC) | p = {p_opt_bic} (BIC) | p = {p_opt_hq} (HQ)")
print(f"Optimal lag: AIC={p_opt}, BIC={p_opt_bic}, HQ={p_opt_hq}")
print(f"Using p = {p_opt} for subsequent analysis.")
p = p_opt

# =============================================================================
# 5. VAR MODEL ESTIMATION USING OLS (from scratch)
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Estimating VAR model via OLS...")
print("="*80)

y = data_diff.values
T, k = y.shape

# Build Y and X matrices for lag p
Y = y[p:, :]                     # (T-p, k)
X = []
for t in range(p, T):
    row = y[t-p:t, :].flatten()  # all lagged values for time t
    X.append(row)
X = np.array(X)
X = np.column_stack([np.ones(X.shape[0]), X])   # add constant

# OLS: estimate coefficients for each equation
# betas has shape (1 + p*k, k): row 0 = intercepts; rows 1..p*k align with
# stacked regressors [y_{t-p}', ..., y_{t-1}']' (oldest lag first in X).
betas = np.zeros((X.shape[1], k))
assert betas.shape == (1 + p * k, k)
residuals = np.zeros((Y.shape[0], k))
T_eff_est = T - p
n_par = X.shape[1]
ols_sigma = np.zeros(k)
ols_r2 = np.zeros(k)

for j in range(k):
    # Estimate the coefficients for each equation using OLS
    betas[:, j] = np.linalg.lstsq(X, Y[:, j], rcond=None)[0]
    # Calculate the predicted values
    y_hat = X @ betas[:, j]
    # Calculate the residuals
    residuals[:, j] = Y[:, j] - y_hat
    # Calculate the sum of squares of the residuals
    sse = float(np.sum(residuals[:, j] ** 2))
    # Calculate the sum of squares of the total variation
    sst = float(np.sum((Y[:, j] - np.mean(Y[:, j])) ** 2))
    # Calculate the R-squared
    ols_r2[j] = 1.0 - sse / sst if sst > 0 else np.nan
    ols_sigma[j] = np.sqrt(sse / max(1, T_eff_est - n_par))

# Residual covariance matrix
sigma_u = (residuals.T @ residuals) / T_eff_est

print(f"VAR({p}) model estimated successfully.")
print(f"Coefficient matrix shape: {betas.shape}")
print(f"Residual covariance matrix shape: {sigma_u.shape}")
for j, vn in enumerate(variables):
    print(f"  Equation {vn}: residual SE = {ols_sigma[j]:.6f}, R^2 = {ols_r2[j]:.6f}")

# Extract coefficient matrices for each lag (for later use)
# X row blocks (after constant): block i = y_{t-p+i} => maps to phi_{p-i} in Y_t = c + Σ phi_l Y_{t-l}.
coeffs = {}
coeffs['const'] = betas[0, :]
for ell in range(1, p + 1):
    # Calculate the block index
    i_block = p - ell
    # Extract the coefficients for the lag
    coeffs[f'lag_{ell}'] = betas[1 + i_block * k : 1 + (i_block + 1) * k, :]

# Test for autocorrelation in the residuals
def test_autocorrelation(residuals, lags=10, p=1):
    """
    Multivariate Portmanteau (Ljung–Box style) on VAR residuals — coursework diagnostics.
    For each equation j, sample ACF r_h at lag h; pooled Q uses all equations.
    Q = T*(T+2) * Σ_j Σ_{h=1}^{m} r_{j,h}^2 / (T-h), compared to Chi²(lags - p*k).
    """
    E = np.asarray(residuals, dtype=float)
    T_eff, k_dim = E.shape
    m = int(lags)
    acf_by_equation = {}
    Q_sum = 0.0
    for j in range(k_dim):
        e = E[:, j]
        denom = np.sum(e ** 2)
        rh = []
        for h in range(1, m + 1):
            if denom <= 0:
                rh.append(0.0)
            else:
                num = float(np.sum(e[h:] * e[:-h]))
                rh.append(num / denom)
        acf_by_equation[j] = rh
        for h in range(1, m + 1):
            r = rh[h - 1]
            Q_sum += (r ** 2) / (T_eff - h)
    Q_stat = T_eff * (T_eff + 2.0) * Q_sum
    df_chi = max(1, int(m - p * k_dim))
    p_value = float(chi2.sf(Q_stat, df_chi))
    is_autocorrelated = p_value < 0.05
    return {
        "Q_stat": float(Q_stat),
        "p_value": p_value,
        "is_autocorrelated": is_autocorrelated,
        "acf_by_equation": acf_by_equation,
    }


def test_heteroskedasticity(residuals, lags=5):
    """
    Coursework: ACF of squared residuals Е²_{j,t}; flag |ACF| > 2/sqrt(T) at any lag 1..lags.
    """
    E = np.asarray(residuals, dtype=float)
    T_eff, k_dim = E.shape
    m = int(lags)
    threshold = 2.0 / np.sqrt(T_eff)
    squared_acf = {}
    heteroskedastic_equations = []
    for j in range(k_dim):
        v = E[:, j] ** 2
        denom = np.sum(v ** 2)  # use SUM (e^2)^2 for ACF of squared series (coursework-style)
        acf_l = []
        for h in range(1, m + 1):
            if denom <= 0:
                acf_l.append(0.0)
            else:
                num = float(np.sum(v[h:] * v[:-h]))
                acf_l.append(num / denom)
        squared_acf[j] = acf_l
        if any(abs(a) > threshold for a in acf_l):
            heteroskedastic_equations.append(j)
    return {
        "heteroskedastic_equations": heteroskedastic_equations,
        "squared_acf": squared_acf,
        "threshold": float(threshold),
    }


def test_normality(residuals):
    """
    Jarque–Bera per equation (coursework): S, excess K, JB = T/6*(S² + K²/4) ~ Chi²(2).
    """
    E = np.asarray(residuals, dtype=float)
    T_eff, k_dim = E.shape
    jb_stats, p_values, skewness, excess_kurtosis = [], [], [], []
    non_normal_equations = []
    for j in range(k_dim):
        e = E[:, j]
        m2 = float(np.mean(e ** 2))
        if m2 <= 1e-20:
            S_j = 0.0
            K_j = -3.0
        else:
            S_j = float(np.mean(e ** 3) / (m2 ** 1.5))
            K_j = float(np.mean(e ** 4) / (m2 ** 2) - 3.0)
        jb = (T_eff / 6.0) * (S_j ** 2 + (K_j ** 2) / 4.0)
        pv = float(chi2.sf(jb, 2))
        jb_stats.append(float(jb))
        p_values.append(pv)
        skewness.append(S_j)
        excess_kurtosis.append(K_j)
        if pv < 0.05:
            non_normal_equations.append(j)
    return {
        "jb_stats": jb_stats,
        "p_values": p_values,
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "non_normal_equations": non_normal_equations,
    }


def run_all_diagnostics(residuals, p, k, lags_auto=10, lags_het=5):
    """Run autocorrelation, heteroskedasticity, and normality tests; coursework summary."""
    auto = test_autocorrelation(residuals, lags=lags_auto, p=p)
    het = test_heteroskedasticity(residuals, lags=lags_het)
    norm = test_normality(residuals)
    summary = {
        "autocorrelation": "FAIL" if auto["is_autocorrelated"] else "PASS",
        "heteroskedasticity": "FAIL" if len(het["heteroskedastic_equations"]) > 0 else "PASS",
        "normality": "FAIL" if len(norm["non_normal_equations"]) > 0 else "PASS",
    }
    return {
        "autocorrelation": auto,
        "heteroskedasticity": het,
        "normality": norm,
        "summary": summary,
    }


diag_results = run_all_diagnostics(residuals, p, k, lags_auto=10, lags_het=5)

# =============================================================================
# 6. MULTI‑STEP FORECASTING (8 quarters ahead)
# =============================================================================
print("\n" + "="*80)
print("STEP 5: Generating forecasts (8 quarters ahead)...")
print("="*80)

def forecast_var(coeffs, y, p, steps=settings.prediction_steps):
    """
    Iteratively forecast a VAR(p) model for 'steps' periods ahead.
    """
    forecasts = []
    current = y[-p:].copy()   # last p observations
    
    for _ in range(steps):
        pred = coeffs['const'].copy()
        # Phi_l multiplies Y_{t-l}; current[-1] is Y_{t-1}, current[-2] is Y_{t-2}, ...
        for ell in range(1, p + 1):
            lag_obs = current[-ell]
            pred += lag_obs @ coeffs[f'lag_{ell}']
        
        forecasts.append(pred)
        # Update current: shift and add new prediction
        current = np.vstack([current[1:], pred])
    
    return np.array(forecasts)

# forecast_var returns h-step-ahead predictions in the SAME units as y (data_diff):
# - series left in levels in data_diff are already predicted levels (here: log levels);
# - series that were first-differenced are predicted *changes* and must be integrated.
forecasts_hat = forecast_var(coeffs, y, p, steps=8)

last_obs_levels = data_raw.iloc[-1]
forecasts_levels = np.zeros_like(forecasts_hat)
for j, var in enumerate(variables):
    if stationarity_results[var]:
        forecasts_levels[:, j] = forecasts_hat[:, j]
    else:
        last = last_obs_levels[var]
        forecasts_levels[:, j] = last + np.cumsum(forecasts_hat[:, j])

# Map to plotting units: exponentiate where we use logs in data_raw
forecasts_final = forecasts_levels.copy()
for i, var in enumerate(variables):
    if var in log_vars:
        forecasts_final[:, i] = np.exp(forecasts_levels[:, i])
    # FEDFUNDS remains in levels (no log transform)

print("Forecasts for 8 quarters ahead:")
for i, var in enumerate(variables):
    print(f"  {var:12s} : {forecasts_final[-1, i]:8.2f} (last forecast)")

# =============================================================================
# 7. IMPULSE RESPONSE FUNCTIONS (IRF) from scratch
# =============================================================================
print("\n" + "="*80)
print("STEP 6: Computing Impulse Response Functions (IRF)...")
print("="*80)

def impulse_response(coeffs, sigma_u, p, horizon=20):
    """
    Compute orthogonalized impulse responses using Cholesky decomposition.
    Returns an array of shape (horizon, k, k): response of variable j to shock in variable i.
    """
    k = sigma_u.shape[0]
    # Cholesky decomposition of residual covariance (ordering as in variables list)
    P = np.linalg.cholesky(sigma_u)   # lower triangular

    # Companion form: Z_t = F Z_{t-1}, first block of Z is Y_t; top row of F is [Phi_1|...|Phi_p].
    # coeffs['lag_l'][m,j] = OLS coeff of regressor m on equation j -> Phi_l[j,m] = coeffs['lag_l'][m,j].
    phi_blocks = [coeffs[f'lag_{ell}'].T for ell in range(1, p + 1)]
    top = np.hstack(phi_blocks)
    assert top.shape == (k, k * p), "Companion top block must be (k, k*p) stacking Phi_1..Phi_p."
    for ell, Phi in enumerate(phi_blocks, start=1):
        assert Phi.shape == (k, k), f"Phi_{ell} must be (k,k), got {Phi.shape}."

    comp = np.zeros((k * p, k * p))
    comp[:k, :k * p] = top
    if p > 1:
        comp[k:, :k*(p-1)] = np.eye(k*(p-1))
    
    # Initialize impulse responses
    irf = np.zeros((horizon, k, k))
    # Shock vector e_i * Cholesky factor (standard deviation shock)
    for i in range(k):
        shock = P @ np.eye(k)[:, i]   # one standard deviation shock to variable i
        # Initial impact (period 0)
        irf[0, :, i] = shock
        # State vector (first k elements are the variables)
        state = np.zeros(k*p)
        state[:k] = shock
        # Propagate through companion matrix
        for h in range(1, horizon):
            state = comp @ state
            irf[h, :, i] = state[:k]
    
    return irf

IRF_HORIZON = 20
irf = impulse_response(coeffs, sigma_u, p, horizon=IRF_HORIZON)
print(f"IRF computed for {IRF_HORIZON} periods, shape = {irf.shape}")

# Lower Cholesky factor of SUM_u (same as inside impulse_response); needed for structural nu and HD
chol_P = np.linalg.cholesky(sigma_u)


def historical_decomposition(irf_array, residuals, chol_P):
    """
    Historical decomposition from the VMA / Wold representation (coursework).

    Reduced-form innovations e_t relate to structural shocks via e_t = P nu_t (P lower Cholesky of SUM_u),
    so nu_t = P^{-1} e_t. With orthogonalized IRF theta^s_{ij} = sig_s[i,j] (response of variable i to shock j
    at lag s), the contribution of shock j to variable i at sample time t is

        HD_{i,j}(t) = SUM_{s=0}^{min(t, H-1)} theta^s_{ij} nu_{j, t-s}

    (finite history; H = number of IRF horizons computed).
    Returns hd of shape (T_eff, k, k) with hd[t, i, j] as above.
    """
    R = np.asarray(residuals, dtype=float)
    T_eff, k = R.shape
    irf = np.asarray(irf_array, dtype=float)
    H, k1, k2 = irf.shape
    assert k1 == k2 == k, "irf_array must be (H, k, k) matching residuals (T_eff, k)"
    P = np.asarray(chol_P, dtype=float)
    assert P.shape == (k, k), "chol_P must be (k, k) lower triangular"

    # Structural shocks: nu_t = P^{-1} e_t  (columns e_t are residuals.T)
    eta = np.linalg.solve(P, R.T).T  # (T_eff, k), stable vs explicit inv

    hd = np.zeros((T_eff, k, k))
    # For each (i,j): causal convolution (theta * nu_j)_t = SUM_s theta^s_{ij} nu_{j,t-s}
    for j in range(k):
        for i in range(k):
            hd[:, i, j] = np.convolve(irf[:, i, j], eta[:, j], mode="full")[:T_eff]

    row_sum = hd.sum(axis=2)
    # Identity: SUM_j HD_{i,j}(t) = SUM_{s=0}^{min(t,H-1)} (theta_s nu_{t-s})_i  (truncated Wold / MA)
    ma_trunc = np.zeros((T_eff, k))
    for t in range(T_eff):
        smax = min(t + 1, H)
        for s in range(smax):
            ma_trunc[t] += irf[s] @ eta[t - s]
    assert np.allclose(row_sum, ma_trunc, rtol=0, atol=1e-9), "HD row sums must match truncated MA"

    # Coursework check vs demeaned reduced-form residual (often differs: residual is one e_t only)
    eps_dm = R - R.mean(axis=0, keepdims=True)
    diff = np.abs(row_sum - eps_dm)
    bad = diff > 1e-4
    if np.any(bad):
        n_bad = int(np.sum(bad))
        max_bad = float(diff.max())
        print(
            f"[historical_decomposition] |sum_j HD_ij(t) - demeaned e_it| > 1e-4 "
            f"for {n_bad} (t,i) pairs (max {max_bad:.6e}); typical under finite H."
        )

    return hd


def hd_plot_data(hd_array, variables, dates):
    """
    Package historical decomposition for plotting: one DataFrame per endogenous variable.

    Columns are structural shock contributions ('shock_<name>') plus a zero 'baseline' reference.
    Index aligns with the residual sample dates (Wold / MA timing).
    """
    hd = np.asarray(hd_array, dtype=float)
    T_eff, k, k2 = hd.shape
    assert k == k2 == len(variables), "hd_array (T_eff, k, k) must match len(variables)"
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    assert len(dates) == T_eff, "dates length must equal T_eff"

    out = {}
    for i, vname in enumerate(variables):
        cols = {f"shock_{s}": hd[:, i, j] for j, s in enumerate(variables)}
        df = pd.DataFrame(cols, index=dates)
        df["baseline"] = 0.0
        out[vname] = df
    return out


hd_array = historical_decomposition(irf, residuals, chol_P)
residual_dates = data_diff.index[p : p + residuals.shape[0]]
hd_dfs = hd_plot_data(hd_array, variables, residual_dates)
print(
    f"\nHistorical decomposition array shape {hd_array.shape} "
    f"(T_eff={residuals.shape[0]}); hd_plot_data keys: {list(hd_dfs.keys())}"
)


def fevd(irf_array, horizon):
    """
    Forecast error variance decomposition (FEVD) from orthogonalized IRFs.

    Coursework formula (sigma^2_{nu_j}=1 under normalized Cholesky structural shocks):
      FEVD_{i,j}(h) =
        [SUM_{s=0}^{h-1} (theta^s_{ij})²] / [SUM_{k=1}^{n} SUM_{s=0}^{h-1} (theta^s_{ik})^2]

    irf_array[s, i, j] = theta^s_{ij} (response of variable i to shock j at horizon s).
    Returns fevd[h, i, j] = FEVD at horizon (h+1), i.e. cumulative sums s = 0..h.
    """
    H = int(horizon)
    irf_full = np.asarray(irf_array, dtype=float)
    assert irf_full.shape[0] >= H, "irf_array first dimension must be >= horizon"
    irf_h = irf_full[:H]
    S, k, k2 = irf_h.shape
    assert k == k2, "irf_array must be (S, k, k)"
    assert S == H

    # sigma²_{nu_j} = 1 -> omit variance weights; use cumulative squared IRFs
    theta2 = irf_h ** 2
    cum_theta2 = np.cumsum(theta2, axis=0)  # (H, k, k): num_{i,j}(h) for h = 0..H-1

    den = cum_theta2.sum(axis=2, keepdims=True)  # SUM_j cum_theta2[h,i,j]
    fevd_out = cum_theta2 / den

    assert np.allclose(fevd_out.sum(axis=2), 1.0, atol=1e-6), "Each fevd[h,i,:] must sum to 1"

    # Coursework check: horizon 1 (h=0) ⇒ FEVD_{i,i}(1) = 1 i only if row i of Θ_0
    # loads on a single shock (true for all i iff SUM_u is diagonal so P is diagonal).
    row_support = (np.abs(irf_h[0]) > 1e-14).sum(axis=1)
    if np.all(row_support == 1):
        npt.assert_allclose(
            np.diagonal(fevd_out[0]), np.ones(k), rtol=0, atol=1e-6
        )
    else:
        # Recursive Cholesky: first variable’s impact row uses only shock 0
        npt.assert_allclose(fevd_out[0, 0, 0], 1.0, rtol=0, atol=1e-6)

    return fevd_out


def fevd_table(fevd_array, variables, horizon_steps=None):
    """
    Tabulate FEVD as percentages at selected forecast horizons (1-based steps).
    Rows: MultiIndex (horizon, variable); columns: shock names.
    """
    if horizon_steps is None:
        horizon_steps = [4, 8, 12, 20]
    variables = list(variables)
    k = len(variables)
    rows_idx = []
    data = []
    for h in horizon_steps:
        idx = h - 1
        if idx < 0 or idx >= fevd_array.shape[0]:
            continue
        block = np.round(fevd_array[idx] * 100.0, 2)
        for i, v in enumerate(variables):
            rows_idx.append((h, v))
            data.append(block[i, :])
    index = pd.MultiIndex.from_tuples(rows_idx, names=["horizon", "variable"])
    return pd.DataFrame(data, index=index, columns=variables)


fevd_arr = fevd(irf, IRF_HORIZON)

# =============================================================================
# 8. PLOTTING: FORECASTS AND IMPULSE RESPONSES
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Generating plots...")
print("="*80)

# ---- Plot 1: Historical data + forecasts ----
_n = len(SERIES)
_nc = 4
_nr = int(np.ceil(_n / _nc))
fig1, _axes1 = plt.subplots(_nr, _nc, figsize=(4 * _nc, 3.5 * _nr))
_axes1f = np.ravel(np.atleast_2d(_axes1))

for idx, (var, label) in enumerate(SERIES.items()):
    ax = _axes1f[idx]
    # Historical in same units as forecast (levels); data_raw uses logs for log_vars
    s = data_raw[var]
    if var in log_vars:
        hist = np.exp(s)
    else:
        hist = s
    ax.plot(hist.index, hist, label='Исторический', color='blue')
    # Forecasts
    forecast_idx = pd.date_range(start=hist.index[-1], periods=9, freq='Q')[1:]
    ax.plot(forecast_idx, forecasts_final[:, idx], 'ro--', label='Прогноз')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
for _j in range(_n, len(_axes1f)):
    _axes1f[_j].set_visible(False)

plt.suptitle(
    'Совокупный Спрос США: ВВП, C, I, G, NX, Дефлятор ВВП, Федеральные Фонды — Исторические & 8‑Q Прогнозы',
    fontsize=13,
)
plt.tight_layout()
plt.savefig('var_forecasts.png', dpi=150)
plt.show()

# ---- Plot 1b: Zoomed-in view around forecast ----
zoom_hist_points = 3  # last 2 historical values + most recent actual
fig1z, _axes1z = plt.subplots(_nr, _nc, figsize=(4 * _nc, 3.5 * _nr))
_axes1zf = np.ravel(np.atleast_2d(_axes1z))

for idx, (var, label) in enumerate(SERIES.items()):
    ax = _axes1zf[idx]

    # Historical in same units as forecast (levels); data_raw uses logs for log_vars
    s = data_raw[var]
    if var in log_vars:
        hist = np.exp(s)
    else:
        hist = s

    forecast_idx = pd.date_range(start=hist.index[-1], periods=9, freq='Q')[1:]
    hist_zoom = hist.iloc[-zoom_hist_points:]

    # Visual marker: where forecasts start
    ax.axvline(hist.index[-1], color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.plot(hist_zoom.index, hist_zoom.values, label='Исторический (приближение)', color='blue')
    ax.plot(forecast_idx, forecasts_final[:, idx], 'ro--', label='Прогноз')

    # Make x-axis labels readable: show a small subset of quarter ticks.
    all_ticks = pd.Index(list(hist_zoom.index) + list(forecast_idx))
    all_ticks = all_ticks[~all_ticks.duplicated()].sort_values()
    max_ticks = 6
    if len(all_ticks) > max_ticks:
        step = int(np.ceil(len(all_ticks) / max_ticks))
        tick_idxs = list(range(0, len(all_ticks), step))
        if tick_idxs[-1] != len(all_ticks) - 1:
            tick_idxs.append(len(all_ticks) - 1)
        ticks = all_ticks[tick_idxs]
    else:
        ticks = all_ticks

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t.year}Q{t.quarter}" for t in ticks], rotation=45, ha='right', fontsize=8)

    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
for _j in range(_n, len(_axes1zf)):
    _axes1zf[_j].set_visible(False)

plt.suptitle(
    'Приближение: Компоненты Совокупного Спроса (включая Дефлятор ВВП) — хвост + 8‑Q прогнозы',
    fontsize=13,
)
plt.tight_layout()
plt.savefig('var_forecasts_zoom.png', dpi=150)
plt.show()

# ---- Plot 2: Impulse Response Functions ----
# We'll show only responses to selected shocks to avoid clutter.
# Here we plot the effect of a shock to Real GDP (GDPC1) on all variables.
shock_var = 'GDPC1'
shock_idx = variables.index(shock_var)

fig2, _axes2 = plt.subplots(_nr, _nc, figsize=(4 * _nc, 3.5 * _nr))
_axes2f = np.ravel(np.atleast_2d(_axes2))

for i, (resp_var, label) in enumerate(SERIES.items()):
    ax = _axes2f[i]
    ax.plot(range(IRF_HORIZON), irf[:, i, shock_idx], 'b-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title(f'Ответ {label}')
    ax.set_xlabel('Горизонт (четверти)')
    ax.set_ylabel('Импульсивный отклик')
    ax.grid(True, alpha=0.3)
for _j in range(_n, len(_axes2f)):
    _axes2f[_j].set_visible(False)

plt.suptitle(f'Импульсивная переходная функция: Шоки на {SERIES[shock_var]}', fontsize=14)
plt.tight_layout()
plt.savefig('var_irf.png', dpi=150)
plt.show()

# ---- Additional: IRF for a monetary policy shock (FEDFUNDS) ----
shock_var2 = 'FEDFUNDS'
shock_idx2 = variables.index(shock_var2)

fig3, _axes3 = plt.subplots(_nr, _nc, figsize=(4 * _nc, 3.5 * _nr))
_axes3f = np.ravel(np.atleast_2d(_axes3))

for i, (resp_var, label) in enumerate(SERIES.items()):
    ax = _axes3f[i]
    ax.plot(range(IRF_HORIZON), irf[:, i, shock_idx2], 'g-', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title(f'Ответ {label}')
    ax.set_xlabel('Горизонт (четверти)')
    ax.set_ylabel('Импульсивный отклик')
    ax.grid(True, alpha=0.3)
for _j in range(_n, len(_axes3f)):
    _axes3f[_j].set_visible(False)

plt.suptitle(f'Импульсивная переходная функция: Шоки на {SERIES[shock_var2]}', fontsize=14)
plt.tight_layout()
plt.savefig('var_irf_monetary.png', dpi=150)
plt.show()

# =============================================================================
# B1 — Lag order selection criteria (AIC, BIC, HQ)
# =============================================================================
_lags_axis = np.arange(1, settings.max_lags + 1)
fig_lag, ax_lag = plt.subplots(1, 1, figsize=(9, 5))
_aic_y = np.array([aic_dict[pp] for pp in range(1, settings.max_lags + 1)])
_bic_y = np.array([bic_dict[pp] for pp in range(1, settings.max_lags + 1)])
_hq_y = np.array([hq_dict[pp] for pp in range(1, settings.max_lags + 1)])
ax_lag.plot(_lags_axis, _aic_y, "o-", color="C0", label="AIC", linewidth=1.5)
ax_lag.plot(_lags_axis, _bic_y, "s-", color="C1", label="BIC", linewidth=1.5)
ax_lag.plot(_lags_axis, _hq_y, "^-", color="C2", label="HQ", linewidth=1.5)
ax_lag.axvline(p_opt, color="gray", linestyle="--", linewidth=1.5, label=f"AIC-оптимальный p={p_opt}")
ax_lag.scatter(int(np.argmin(_aic_y)) + 1, np.min(_aic_y), color="C0", s=80, zorder=5, marker="o")
ax_lag.scatter(int(np.argmin(_bic_y)) + 1, np.min(_bic_y), color="C1", s=80, zorder=5, marker="s")
ax_lag.scatter(int(np.argmin(_hq_y)) + 1, np.min(_hq_y), color="C2", s=80, zorder=5, marker="^")
ax_lag.set_xlabel("Порядок лагов p")
ax_lag.set_ylabel("Значение информационного критерия")
ax_lag.set_title("Критерии выбора порядка лага")
ax_lag.legend(loc="best")
ax_lag.grid(True, alpha=0.3)
ax_lag.set_xticks(_lags_axis)
plt.tight_layout()
plt.savefig("var_lag_criteria.png", dpi=150)
plt.show()

# =============================================================================
# B2 — FEVD stacked bars by variable (horizons 1, 2, 4, 8, 12, 20)
# =============================================================================
print("\nFEVD table [%] at horizons 4, 8, 20 (all variables):")
print(fevd_table(fevd_arr, variables, horizon_steps=[4, 8, 20]).to_string())
_fevd_steps = [1, 2, 4, 8, 12, 20]
_fevd_idx = [h - 1 for h in _fevd_steps if h - 1 < fevd_arr.shape[0]]
_fevd_steps = [h for h in _fevd_steps if h - 1 < fevd_arr.shape[0]]
_kdim = len(variables)
_tab10 = plt.cm.tab10(np.linspace(0, 0.9, _kdim))
for _vi, _v in enumerate(variables):
    _fig_f, _ax_f = plt.subplots(figsize=(10, 5))
    _i = variables.index(_v)
    _xpos = np.arange(len(_fevd_idx))
    _bottom = np.zeros(len(_fevd_idx))
    for _j in range(_kdim):
        _hgt = fevd_arr[np.array(_fevd_idx), _i, _j] * 100.0
        _ax_f.bar(
            _xpos,
            _hgt,
            bottom=_bottom,
            label=variables[_j],
            color=_tab10[_j],
            width=0.65,
        )
        _bottom = _bottom + _hgt
    _ax_f.set_xticks(_xpos)
    _ax_f.set_xticklabels([str(h) for h in _fevd_steps])
    _ax_f.set_xlabel("Горизонт прогноза (четверти)")
    _ax_f.set_ylabel("Доля дисперсии ошибки прогноза (%)")
    _ax_f.set_ylim(0, 100)
    _ax_f.set_title(f"Декомпозиция дисперсии ошибки прогноза: {SERIES[_v]}")
    _ax_f.legend(title="Шоки на", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _ax_f.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"var_fevd_{_v}.png", dpi=150)
    plt.show()

# =============================================================================
# B3 — Historical decomposition (GDPC1, FEDFUNDS only)
# =============================================================================
_hd_plot = hd_array
_resid_dm = residuals - np.mean(residuals, axis=0, keepdims=True)
_hd_dates = pd.DatetimeIndex(pd.to_datetime(data_diff.index[p : p + residuals.shape[0]]))
for _vkey in ["GDPC1", "FEDFUNDS"]:
    _ii = variables.index(_vkey)
    _fig_h, _ax_h = plt.subplots(figsize=(12, 5))
    _layers = [_hd_plot[:, _ii, _j] for _j in range(_kdim)]
    _ax_h.stackplot(
        _hd_dates,
        *_layers,
        labels=[f"shock {variables[_j]}" for _j in range(_kdim)],
        colors=_tab10,
        alpha=0.85,
    )
    _ax_h.plot(
        _hd_dates,
        _resid_dm[:, _ii],
        color="black",
        linewidth=2.5,
        label="Униженный остаточный",
    )
    _ax_h.set_title(f"Историческое разложение: {SERIES[_vkey]}")
    _ax_h.set_xlabel("Дата")
    _ax_h.set_ylabel("Вклад (в тех же единицах, что и остатки VAR)")
    _ax_h.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    _ax_h.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"var_hd_{_vkey}.png", dpi=150)
    plt.show()

# =============================================================================
# B4 — Residual diagnostics figure (ACF, squared ACF, Q-Q)
# =============================================================================
diag_results = run_all_diagnostics(residuals, p, k, lags_auto=10, lags_het=5)
print("\n=== DIAGNOSTIC SUMMARY ===")
print(f"  autocorrelation     : {diag_results['summary']['autocorrelation']}")
print(f"  heteroskedasticity  : {diag_results['summary']['heteroskedasticity']}")
print(f"  normality           : {diag_results['summary']['normality']}")

_nlags_plot = 20
_Teff = residuals.shape[0]
_ci = 1.96 / np.sqrt(_Teff)


def _acf_like_test(e, nlags):
    e = np.asarray(e, dtype=float)
    d = float(np.sum(e ** 2))
    out = [1.0 if d > 0 else 0.0]
    for _h in range(1, nlags + 1):
        out.append(float(np.sum(e[_h:] * e[:-_h]) / d) if d > 0 else 0.0)
    return np.array(out)


def _acf_sq_like_test(e, nlags):
    w = np.asarray(e, dtype=float) ** 2
    d = float(np.sum(w ** 2))
    out = [1.0 if d > 0 else 0.0]
    for _h in range(1, nlags + 1):
        out.append(float(np.sum(w[_h:] * w[:-_h]) / d) if d > 0 else 0.0)
    return np.array(out)


fig_d, axes_d = plt.subplots(3, _kdim, figsize=(3.2 * _kdim, 10), squeeze=False)
_lags_x = np.arange(0, _nlags_plot + 1)
for _j in range(_kdim):
    _ej = residuals[:, _j]
    _rho = _acf_like_test(_ej, _nlags_plot)
    _ax = axes_d[0, _j]
    _ax.bar(_lags_x, _rho, color="steelblue", width=0.7)
    _ax.axhline(_ci, color="red", linestyle="--", linewidth=1)
    _ax.axhline(-_ci, color="red", linestyle="--", linewidth=1)
    _ax.axhline(0, color="black", linewidth=0.5)
    _ax.set_title(f"ACF ε — {variables[_j]}")
    _ax.set_xlabel("Лаг")
    _ax.set_ylabel("ρ(h)")
    _ax.set_xticks(_lags_x[::2])

    _rho2 = _acf_sq_like_test(_ej, _nlags_plot)
    _ax2 = axes_d[1, _j]
    _ax2.bar(_lags_x, _rho2, color="darkorange", width=0.7)
    _ax2.axhline(_ci, color="red", linestyle="--", linewidth=1)
    _ax2.axhline(-_ci, color="red", linestyle="--", linewidth=1)
    _ax2.axhline(0, color="black", linewidth=0.5)
    _ax2.set_title(f"ACF ε² — {variables[_j]}")
    _ax2.set_xlabel("Лаг")
    _ax2.set_ylabel("Коэффициент Автокорреляции")
    _ax2.set_xticks(_lags_x[::2])

    probplot(_ej, dist="norm", plot=axes_d[2, _j])
    axes_d[2, _j].set_title(f"Q-Q — {variables[_j]}")
    axes_d[2, _j].grid(True, alpha=0.3)
    axes_d[2, _j].set_xlabel("Теоретические Квантили")
    axes_d[2, _j].set_ylabel("Упорядоченные Значения")

plt.suptitle("Диагностика остаточных значений VAR", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("var_diagnostics.png", dpi=150)
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE.")
print("=" * 80)
print("Saved plots:")
for fname in [
    "var_forecasts.png",
    "var_forecasts_zoom.png",
    "var_irf.png",
    "var_irf_monetary.png",
    "var_lag_criteria.png",
    *[f"var_fevd_{v}.png" for v in variables],
    *[f"var_hd_{v}.png" for v in ["GDPC1", "FEDFUNDS"]],
    "var_diagnostics.png",
]:
    print(f"  {fname}")

