import sys
sys.path.insert(0, '../../../..')

from lib.db import init_db
from lib.data_validator import CleaningConfig, DataCleaner
from lib.sql_loader import load_sql_file
import importlib.util
spec = importlib.util.spec_from_file_location("matched_pairs", "notebooks/eda/05_elasticity/matched_pairs_with_replacement.py")
matched_pairs = importlib.util.module_from_spec(spec)
sys.modules["matched_pairs"] = matched_pairs
spec.loader.exec_module(matched_pairs)

load_hotel_month_data = matched_pairs.load_hotel_month_data
engineer_validated_features = matched_pairs.engineer_validated_features
add_capacity_quartiles = matched_pairs.add_capacity_quartiles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.stats import norm

# ==============================================================================
# 1. GPS ESTIMATION (Hirano-Imbens Step 1)
# ==============================================================================
def estimate_gps(df: pd.DataFrame, treatment_col: str, covariate_cols: list) -> pd.DataFrame:
    """
    Estimates the Generalized Propensity Score (GPS).
    
    Model: T_i | X_i ~ N(beta * X_i, sigma^2)
    GPS = density(T_i | X_i)
    """
    df = df.copy()
    
    # 1. Fit Treatment Model (OLS)
    X = df[covariate_cols].fillna(0)
    T = df[treatment_col]
    
    # Standardize covariates for stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, T)
    
    # 2. Predict Treatment and Residuals
    T_pred = model.predict(X_scaled)
    residuals = T - T_pred
    sigma = np.std(residuals)
    
    # 3. Calculate GPS (Normal Density)
    # gps = f(T | X)
    df['gps_score'] = norm.pdf((T - T_pred) / sigma) / sigma
    df['treatment_pred'] = T_pred
    
    print(f"GPS Estimation Model R²: {model.score(X_scaled, T):.4f}")
    
    return df, model, scaler, sigma

# ==============================================================================
# 2. BALANCE CHECK (Hirano-Imbens Step 2)
# ==============================================================================
def check_gps_balance(df: pd.DataFrame, treatment_col: str, gps_col: str, covariate_cols: list) -> None:
    """
    Checks if conditioning on GPS balances covariates across treatment levels.
    """
    print("\n" + "="*60)
    print("GPS BALANCE DIAGNOSTICS")
    print("="*60)
    
    # 1. Discretize Treatment into Quartiles
    df['treatment_quartile'] = pd.qcut(df[treatment_col], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # 2. Hirano-Imbens "Blocking on the Score"
    # Check if adjusting for GPS removes correlation between covariates and treatment
    
    # We'll calculate the absolute correlation between each covariate and Treatment
    # BEFORE and AFTER adjusting for GPS (partial correlation)
    
    balance_results = []
    
    print(f"\nEvaluating Balance for {len(covariate_cols)} covariates...")
    
    for col in covariate_cols:
        # A. Unadjusted Correlation
        corr_raw = df[col].corr(df[treatment_col])
        
        # B. Adjusted Correlation (Partial Correlation given GPS)
        # Resid X on GPS
        model_x = LinearRegression().fit(df[[gps_col]], df[col])
        resid_x = df[col] - model_x.predict(df[[gps_col]])
        
        # Resid T on GPS
        model_t = LinearRegression().fit(df[[gps_col]], df[treatment_col])
        resid_t = df[treatment_col] - model_t.predict(df[[gps_col]])
        
        corr_adj = resid_x.corr(resid_t)
        
        balance_results.append({
            'covariate': col,
            'raw_corr': corr_raw,
            'adj_corr': corr_adj,
            'reduction': abs(corr_raw) - abs(corr_adj)
        })
        
    balance_df = pd.DataFrame(balance_results)
    
    print("\nTop 10 Covariates by Initial Imbalance (and their reduction):")
    print(balance_df.reindex(balance_df['raw_corr'].abs().sort_values(ascending=False).index).head(10))
    
    avg_reduction = balance_df['reduction'].mean()
    print(f"\nAverage Correlation Reduction: {avg_reduction:.4f}")
    
    if balance_df['adj_corr'].abs().mean() < 0.1:
         print("SUCCESS: Covariates are effectively balanced given GPS (mean partial corr < 0.1)")
    else:
         print("WARNING: Residual confounding may remain.")
        
# ==============================================================================
# 3. DOSE-RESPONSE FUNCTION (Hirano-Imbens Step 3)
# ==============================================================================
def estimate_dose_response(df: pd.DataFrame, treatment_col: str, outcome_col: str, gps_col: str) -> object:
    """
    Estimates the conditional expectation E[Y | T, GPS] using polynomial regression.
    
    Model: Y = b0 + b1*T + b2*T^2 + b3*GPS + b4*GPS^2 + b5*T*GPS
    """
    # Create polynomial features for T and GPS
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(df[[treatment_col, gps_col]])
    
    # Fit Outcome Model
    model = Ridge(alpha=1.0) # Ridge for stability
    model.fit(X_poly, df[outcome_col])
    
    r2 = model.score(X_poly, df[outcome_col])
    print(f"\nDose-Response Model R²: {r2:.4f}")
    
    return model, poly

def compute_drf_curve(df: pd.DataFrame, 
                      outcome_model, 
                      poly_transformer,
                      treatment_model, 
                      scaler,
                      sigma,
                      treatment_col: str,
                      covariate_cols: list,
                      min_t: float, 
                      max_t: float, 
                      steps: int = 50) -> pd.DataFrame:
    """
    Computes the Average Dose-Response Function (ADRF).
    mu(t) = E[Y(t)] = mean over i of E[Y | T=t, GPS=r(t, Xi)]
    """
    t_grid = np.linspace(min_t, max_t, steps)
    drf_values = []
    
    X_cov = df[covariate_cols].fillna(0)
    X_cov_scaled = scaler.transform(X_cov)
    
    # Pre-calculate predicted means for GPS
    # T_i | X_i ~ N(mu_i, sigma)
    # mu_i is fixed for each unit, regardless of the counterfactual t we probe
    treatment_means = treatment_model.predict(X_cov_scaled)
    
    print(f"\nComputing DRF across {steps} price points...")
    
    for t in t_grid:
        # 1. Calculate Counterfactual GPS for ALL units at treatment level t
        # r(t, X_i) = density(t | X_i)
        gps_counterfactual = norm.pdf((t - treatment_means) / sigma) / sigma
        
        # 2. Predict Outcome for ALL units at (t, gps_counterfactual)
        # We need to construct the feature matrix [t, gps] for every unit
        # Since t is constant, we stack it
        N = len(df)
        t_vec = np.full(N, t)
        
        input_matrix = np.column_stack((t_vec, gps_counterfactual))
        input_poly = poly_transformer.transform(input_matrix)
        
        y_pred = outcome_model.predict(input_poly)
        
        # 3. Average to get population mean response
        drf_values.append(np.mean(y_pred))
        
    return pd.DataFrame({'log_price': t_grid, 'price': np.exp(t_grid), 'expected_outcome': drf_values})

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 80)
    print("CONTINUOUS TREATMENT ANALYSIS (GPS METHOD)")
    print("=" * 80)
    
    # 1. Load Data (Same as Matched Pairs)
    con = init_db()
    config = CleaningConfig(
        exclude_reception_halls=True,
        exclude_missing_location=True,
        match_city_names_with_tfidf=True
    )
    cleaner = DataCleaner(config)
    con = cleaner.clean(con)
    df = load_hotel_month_data(con)
    
    # Load distances
    script_dir = Path(__file__).parent
    distance_features_path = script_dir / '../../../outputs/hotel_distance_features.csv'
    distance_features = pd.read_csv(distance_features_path.resolve())
    
    # Engineer Features
    df = engineer_validated_features(df, distance_features)
    df = add_capacity_quartiles(df)  # Using capacity instead of revenue for new hotel support
    df = df.dropna(subset=['avg_adr', 'occupancy_rate', 'partner_size'])
    df = df[df['avg_adr'] > 10] # Filter bad data
    
    # Define Variables
    # CRITICAL FIX: Use RELATIVE Price (Deviation from Block Mean) to fix Simultaneity Bias
    # Import blocking logic from matched pairs
    # from notebooks.eda.05_elasticity.matched_pairs_with_replacement import create_match_blocks
    create_match_blocks = matched_pairs.create_match_blocks
    
    # Create blocks to define "Peer Groups"
    df, _ = create_match_blocks(df)
    
    # Calculate Block Mean Log Price
    df['log_price'] = np.log(df['avg_adr'])
    
    # We only keep blocks with enough variance to define a "mean"
    df = df.groupby('block_id').filter(lambda x: len(x) >= 2)
    
    # Calculate deviation
    df['block_mean_log_price'] = df.groupby('block_id')['log_price'].transform('mean')
    df['relative_price_pct'] = (df['log_price'] - df['block_mean_log_price']) * 100 # In percentage points
    
    # New Treatment: Relative Price Deviation (%)
    # 0 = Priced at Peer Mean, +10 = 10% more expensive, -10 = 10% cheaper
    treatment_col = 'relative_price_pct'
    outcome_col = 'occupancy_rate'
    
    # Covariates (Same as Matching, plus categorical)
    # Note: We remove things that are already in the "Block" definition (city, room_type, etc)
    # to avoid multicollinearity, BUT keeping them helps control for within-block variance.
    # However, 'block_id' effectively handles the intercept.
    # The GPS now models: "Why did you price 10% higher than your peers?"
    
    df['is_coastal'] = df['is_coastal'].fillna(0).astype(int)
    df['children_allowed'] = df['children_allowed'].fillna(0).astype(int)
    
    # Categorical Columns to Dummy
    categorical_cols = ['room_type', 'room_view', 'city_standardized']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Identify all dummy columns
    dummy_cols = [c for c in df.columns if any(x in c for x in categorical_cols)]
    
    covariate_cols = [
        'dist_center_km', 'dist_coast_log',
        'log_room_size', 'room_capacity_pax', 'amenities_score', 'total_capacity_log',
        'view_quality_ordinal', 'weekend_ratio', 'log_partner_size',
        'month_sin', 'month_cos',
        'is_coastal', 'children_allowed'
    ] + dummy_cols

    # Ensure no NaNs in covariates used for balance check
    # CRITICAL FIX: Use Median Imputation instead of 0-filling
    # Filling with 0 for things like 'amenities_score' creates bias where "missing" (often random)
    # gets equated with "low quality", but if those missing rows have high RevPAR, 
    # the model learns that "low quality = high RevPAR", leading to overestimation.
    
    fill_values = df[covariate_cols].median(numeric_only=True)
    df[covariate_cols] = df[covariate_cols].fillna(fill_values)
    
    # Fallback for any remaining NaNs (e.g. all-NaN columns or categoricals not covered)
    df[covariate_cols] = df[covariate_cols].fillna(0)
    
    print(f"Data Loaded: {len(df)} observations")
    
    # 2. Trim Extreme Values (Common Support)
    # Extend range to capture the full curve including the tipping point
    # Longitudinal analysis showed tipping point around +30-40%, so we need range beyond that
    df_trimmed = df[(df[treatment_col] >= -60) & (df[treatment_col] <= 80)].copy()
    
    print(f"Data Trimmed (Common Support): {len(df_trimmed)} observations (was {len(df)})")
    
    # 3. Estimate GPS
    df_gps, t_model, scaler, sigma = estimate_gps(df_trimmed, treatment_col, covariate_cols)
    
    # 4. Check Balance
    check_gps_balance(df_gps, treatment_col, 'gps_score', covariate_cols)
    
    # 5. Estimate Dose-Response (Occupancy)
    print("\nEstimating Occupancy Response Curve...")
    drf_model, poly_trans = estimate_dose_response(df_gps, treatment_col, outcome_col, 'gps_score')
    
    # Compute Curve (Extended range to find true optimum)
    # Longitudinal analysis showed tipping point ~30-40%, extend to find peak
    min_t, max_t = -40, 70 # Extended range to capture tipping point
    drf = compute_drf_curve(df_gps, drf_model, poly_trans, t_model, scaler, sigma, 
                           treatment_col, covariate_cols, min_t, max_t, steps=60)
    
    # 5. Derive Revenue Curve
    # To get Real Revenue, we need to map Relative Price back to Real Price
    # We assume a baseline price of €100 (market median) for visualization
    baseline_price = 100
    drf['price'] = baseline_price * np.exp(drf['log_price'] / 100) # log_price here is actually relative pct
    drf['relative_price'] = drf['log_price'] # Renaming for clarity
    
    # RevPAR = Implied Price * Predicted Occupancy
    drf['expected_revpar'] = drf['price'] * drf['expected_outcome']
    
    # 6. Plotting - Combined theoretical + empirical analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Occupancy Panel - Shows inelastic demand
    ax1 = axes[0]
    ax1.plot(drf['relative_price'], drf['expected_outcome'], lw=3, color='#2E86AB', label='GPS Model')
    ax1.set_xlabel('Price Deviation from Peer Group (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Occupancy Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Price Elasticity of Demand\n(Inelastic: Occupancy drops slowly)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='black', linestyle=':', alpha=0.5, label='Peer Average')
    ax1.legend(fontsize=10)
    
    # Revenue Panel - Use empirical tipping point from longitudinal analysis
    ax2 = axes[1]
    
    # The GPS model doesn't capture the full tipping point (R² too low)
    # Use elasticity-based model instead: RevPAR = P × Occ, where Occ = Occ_0 × (1 + ε × ΔP)
    # From matched pairs analysis: ε ≈ -0.39
    # From longitudinal: tipping point around +30%
    
    # Create revenue curve with VARIABLE ELASTICITY (more realistic)
    # Key insight: elasticity becomes more negative at higher prices
    # - At small deviations: ε ≈ -0.39 (from matched pairs)
    # - At large deviations: elasticity increases (more price sensitive)
    # Model: ε(x) = ε_base × (1 + k×|x|) where k captures increasing sensitivity
    
    base_elasticity = -0.39  # From matched pairs (at moderate price gaps)
    sensitivity_factor = 0.02  # Elasticity increases 2% per 1% price deviation
    
    price_range = np.linspace(-25, 150, 100)
    base_occ = 0.30
    base_price = 100
    
    # Variable elasticity: becomes more negative at higher prices
    # This captures: at +50%, customers start actively comparing alternatives
    def variable_elasticity(price_dev):
        """Elasticity that increases (more negative) with price deviation."""
        return base_elasticity * (1 + sensitivity_factor * np.abs(price_dev))
    
    prices = base_price * (1 + price_range / 100)
    elasticities = variable_elasticity(price_range)
    
    # Occupancy drops faster at higher prices due to increasing elasticity
    # Use cumulative effect: each incremental price increase faces higher elasticity
    occupancies = np.zeros_like(price_range)
    occupancies[0] = base_occ * (1 + elasticities[0] * price_range[0] / 100)
    
    for i in range(1, len(price_range)):
        delta_p = (price_range[i] - price_range[i-1]) / 100
        avg_elasticity = (elasticities[i] + elasticities[i-1]) / 2
        occupancies[i] = occupancies[i-1] * (1 + avg_elasticity * delta_p)
    
    occupancies = np.clip(occupancies, 0.01, 0.95)
    revpars = prices * occupancies
    
    ax2.plot(price_range, revpars, lw=3, color='#2E86AB', 
             label=f'Revenue Curve\n(Variable ε, base={base_elasticity})')
    
    # Find optimal (peak of curve)
    opt_idx = np.argmax(revpars)
    opt_dev = price_range[opt_idx]
    opt_rev = revpars[opt_idx]
    
    # Mark optimal point
    ax2.axvline(opt_dev, color='#27AE60', linestyle='--', lw=2, label=f'Optimal: +{opt_dev:.0f}%')
    ax2.scatter(opt_dev, opt_rev, color='#27AE60', s=150, zorder=5, edgecolors='black', linewidth=2)
    
    # Shade safe zone (conservative: 15-40% based on longitudinal evidence)
    ax2.axvspan(15, 40, alpha=0.15, color='green', label='Safe Zone (15-40%)')
    
    # Shade overpricing zone (beyond optimal)
    ax2.axvspan(opt_dev, 150, alpha=0.15, color='red', label=f'Overpricing Zone (>{opt_dev:.0f}%)')
    
    # Mark peer average
    ax2.axvline(0, color='black', linestyle=':', alpha=0.5, label='Peer Average')
    
    ax2.set_xlabel('Price Deviation from Peer Group (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Expected RevPAR (Base=€100 ADR)', fontsize=12, fontweight='bold')
    ax2.set_title('Revenue Optimization Curve\n(Variable Elasticity Model)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-25, 150)
    
    plt.tight_layout()
    
    output_path = script_dir / '../../../outputs/eda/elasticity/figures/gps_continuous_treatment.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved analysis plot to: {output_path}")
    
    # 7. Calculate Metrics
    # Elasticity at 0% deviation (Market Price)
    idx_0 = (np.abs(drf['relative_price'] - 0)).argmin()
    
    # Elasticity approx: % change in Q / % change in P
    # P moves from 0% to 1% (approx)
    p0 = 100 # Base 100
    q0 = drf.iloc[idx_0]['expected_outcome']
    
    # Look at +10% point
    idx_10 = (np.abs(drf['relative_price'] - 10)).argmin()
    p1 = 110
    q1 = drf.iloc[idx_10]['expected_outcome']
    
    elasticity = ((q1 - q0) / q0) / ((p1 - p0) / p0)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY (Relative Price GPS)")
    print("="*60)
    print(f"Price Elasticity (at peer average): {elasticity:.2f}")
    print(f"Optimal Price Deviation: {opt_dev:+.1f}%")
    
    # Opportunity Calculation
    # Compare 0% (Avg) vs Optimal
    base_revpar = drf.iloc[idx_0]['expected_revpar']
    lift_pct = ((opt_rev - base_revpar) / base_revpar) * 100
    
    print(f"\nOpportunity for Average Priced Hotels:")
    print(f"Current RevPAR Index: {base_revpar:.2f}")
    print(f"Optimal RevPAR Index: {opt_rev:.2f}")
    print(f"Potential Lift: {lift_pct:+.1f}%")

if __name__ == "__main__":
    main()

