# ===== MULTIPLE REGRESSION ANALYSIS FOR ENVIRONMENTAL IMPACT ON USER ATTITUDE =====
# This script performs multiple regression analysis to identify which environmental awareness 
# factors decisively influence user attitude

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def run_analysis():
    # ========== STEP 1: DATA LOADING ==========
    # Load the dataset from CSV file
    try:
        df = pd.read_csv('../../dataset.csv')
    except FileNotFoundError:
        print("Error: dataset.csv not found.")
        return

    # ========== STEP 2: VARIABLE DEFINITION ==========
    # Step 2.1: Define dependent variable components (User Attitude indicators)
    y_cols = ['satisfaction', 'trust', 'feelings']
    
    # Step 2.2: Define independent variables (Environmental awareness factors)
    x_cols = [
        'No deforestation',
        'natural resources protecting',
        'recyclable packaging',
        'reduced use of energy',
        'low carbon emissions',
        'reduced use of pesticides/fertilizers',
        'water sparingly',
        'familiarity'
    ]

    # ========== STEP 3: DATA VALIDATION ==========
    # Verify that all required columns exist in the dataset
    missing_cols = [c for c in y_cols + x_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        return

    # ========== STEP 4: DATA PREPROCESSING ==========
    # Step 4.1: Record original dataset size
    original_len = len(df)
    
    # Step 4.2: Remove rows with missing values in relevant columns
    df = df.dropna(subset=y_cols + x_cols)
    
    # Step 4.3: Report data cleaning results
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with missing values.")

    # ========== STEP 5: TARGET VARIABLE CALCULATION ==========
    # Calculate User_Attitude as the mean of satisfaction, trust, and feelings
    df['User_Attitude'] = df[y_cols].mean(axis=1)

    # ========== STEP 6: PREPARE REGRESSION VARIABLES ==========
    # Step 6.1: Extract independent variables (predictors)
    X = df[x_cols]
    
    # Step 6.2: Extract dependent variable (target)
    y = df['User_Attitude']

    # ========== STEP 7: ADD INTERCEPT TERM ==========
    # Add constant term to the model for intercept estimation
    X = sm.add_constant(X)

    # ========== STEP 8: MODEL FITTING ==========
    # Fit Ordinary Least Squares (OLS) regression model
    model = sm.OLS(y, X).fit()

    # ========== STEP 9: DISPLAY REGRESSION RESULTS ==========
    # Print complete regression summary including coefficients, R-squared, p-values, etc.
    print(model.summary())

    # ========== STEP 10: IDENTIFY DECISIVE FACTOR ==========
    print("\nAnalysis of Decisive Factor:")
    
    # Step 10.1: Filter for statistically significant variables (p < 0.05)
    significant_params = model.params[model.pvalues < 0.05].drop('const', errors='ignore')
    
    if not significant_params.empty:
        # Step 10.2: Find the variable with the largest absolute coefficient among significant ones
        best_var = significant_params.abs().idxmax()
        coef = significant_params[best_var]
        print(f"The most decisive factor (highest absolute coefficient with p < 0.05) is '{best_var}' with a coefficient of {coef:.4f}")
        
        # Step 10.3: Display all significant variables for context
        print("\nSignificant variables (p < 0.05):")
        for var in significant_params.index:
            print(f"- {var}: {model.params[var]:.4f}")
    else:
        print("No variables are statistically significant at p < 0.05")

    # ========== STEP 11: DATA VISUALIZATION ==========
    print("\nGenerating visualizations...")
    
    # Set visualization style
    sns.set_style("whitegrid")
    from scipy import stats
    
    # Get coefficients excluding the constant
    coef_data = model.params.drop('const', errors='ignore')
    pvalues_data = model.pvalues.drop('const', errors='ignore')
    
    # ---------- Visualization 1: Coefficient Bar Plot ----------
    print("\n[1/4] Displaying Coefficient Bar Plot...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute coefficient value
    coef_sorted = coef_data.abs().sort_values(ascending=True)
    
    # Create color map: significant (p<0.05) in green, non-significant in gray
    colors = ['#2ecc71' if pvalues_data[var] < 0.05 else '#95a5a6' for var in coef_sorted.index]
    
    # Create horizontal bar plot
    bars = ax1.barh(range(len(coef_sorted)), [coef_data[var] for var in coef_sorted.index], color=colors)
    ax1.set_yticks(range(len(coef_sorted)))
    ax1.set_yticklabels(coef_sorted.index, fontsize=11)
    ax1.set_xlabel('Coefficient Value (β)', fontsize=13, fontweight='bold')
    ax1.set_title('Regression Coefficients by Environmental Factor\n(Green = Significant p<0.05, Gray = Not Significant)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (var, bar) in enumerate(zip(coef_sorted.index, bars)):
        value = coef_data[var]
        x_pos = value + (0.01 if value > 0 else -0.01)
        ha = 'left' if value > 0 else 'right'
        ax1.text(x_pos, i, f'{value:.4f}', va='center', ha=ha, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('regression_coefficients.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'regression_coefficients.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 2: Actual vs Predicted Values ----------
    print("\n[2/4] Displaying Actual vs Predicted Values...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Get predicted values
    y_pred = model.fittedvalues
    
    # Create scatter plot
    ax2.scatter(y, y_pred, alpha=0.5, s=30, color='#3498db', edgecolors='#2c3e50', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Labels and formatting
    ax2.set_xlabel('Actual User Attitude', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Predicted User Attitude', fontsize=13, fontweight='bold')
    ax2.set_title(f'Actual vs Predicted Values\n(R² = {model.rsquared:.4f})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'regression_actual_vs_predicted.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 3: Residual Distribution ----------
    print("\n[3/4] Displaying Residual Distribution...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    # Calculate residuals
    residuals = model.resid
    
    # Create histogram with KDE
    ax3.hist(residuals, bins=40, color='#9b59b6', alpha=0.7, edgecolor='black', density=True)
    
    # Add KDE curve
    kde_x = np.linspace(residuals.min(), residuals.max(), 100)
    kde = stats.gaussian_kde(residuals)
    ax3.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    normal_curve = stats.norm.pdf(kde_x, mu, sigma)
    ax3.plot(kde_x, normal_curve, 'g--', linewidth=2, label='Normal Distribution')
    
    ax3.set_xlabel('Residuals', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax3.set_title(f'Residual Distribution\n(Mean = {mu:.4f}, Std = {sigma:.4f})', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'regression_residuals.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 4: P-value Significance Chart ----------
    print("\n[4/4] Displaying P-value Significance Chart...")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Get p-values excluding constant
    pval_data = model.pvalues.drop('const', errors='ignore').sort_values()
    
    # Plot p-values
    colors_pval = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in pval_data]
    bars_pval = ax4.barh(range(len(pval_data)), pval_data.values, color=colors_pval)
    
    # Add significance threshold line
    ax4.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (α=0.05)')
    
    ax4.set_yticks(range(len(pval_data)))
    ax4.set_yticklabels(pval_data.index, fontsize=11)
    ax4.set_xlabel('P-value', fontsize=13, fontweight='bold')
    ax4.set_title('Statistical Significance of Each Variable\n(Red = Significant p<0.05, Gray = Not Significant)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlim(0, max(0.1, pval_data.max() * 1.1))
    ax4.legend(fontsize=11)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (var, p) in enumerate(pval_data.items()):
        ax4.text(p + 0.005, i, f'{p:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('regression_pvalues.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'regression_pvalues.png'")
    plt.show()
    plt.close()
    
    print("\n" + "="*70)
    print("Analysis complete! All visualizations saved.")
    print("="*70)

# ========== STEP 12: PROGRAM EXECUTION ==========
# Run the analysis when script is executed directly
if __name__ == "__main__":
    run_analysis()

