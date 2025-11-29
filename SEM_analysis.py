# ===== STRUCTURAL EQUATION MODELING (SEM) ANALYSIS FOR ENVIRONMENTAL IMPACT =====
# This script performs SEM analysis to examine how environmental awareness factors 
# influence user attitude through a latent variable approach

import pandas as pd
import semopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_sem_analysis():
    # ========== STEP 1: DATA LOADING ==========
    # Load the dataset from CSV file
    try:
        df = pd.read_csv('../../dataset.csv')
    except FileNotFoundError:
        print("Error: dataset.csv not found.")
        return

    # ========== STEP 2: COLUMN RENAMING FOR SEM SYNTAX ==========
    # Step 2.1: Create mapping dictionary to rename columns 
    # (semopy requires variable names without spaces or special characters)
    rename_map = {
        'No deforestation': 'No_deforestation',
        'natural resources protecting': 'natural_resources_protecting',
        'recyclable packaging': 'recyclable_packaging',
        'reduced use of energy': 'reduced_use_of_energy',
        'low carbon emissions': 'low_carbon_emissions',
        'reduced use of pesticides/fertilizers': 'reduced_use_of_pesticides',
        'water sparingly': 'water_sparingly',
        'familiarity': 'familiarity',
        'satisfaction': 'satisfaction',
        'trust': 'trust',
        'feelings': 'feelings'
    }
    
    # ========== STEP 3: DATA VALIDATION ==========
    # Check if all required columns exist in the dataset before renaming
    missing = [k for k in rename_map.keys() if k not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return

    # Step 3.1: Apply column renaming
    df = df.rename(columns=rename_map)
    
    # ========== STEP 4: DATA PREPROCESSING ==========
    # Step 4.1: Extract only the columns needed for the model
    model_cols = list(rename_map.values())
    
    # Step 4.2: Record original dataset size
    original_len = len(df)
    
    # Step 4.3: Select relevant columns and remove rows with missing values
    df = df[model_cols].dropna()
    
    # Step 4.4: Report data cleaning results
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with missing values.")

    # ========== STEP 5: SEM MODEL SPECIFICATION ==========
    # Define the SEM model with two components:
    # - Measurement model: Defines latent variable 'Attitude' measured by observed indicators
    # - Structural model: Defines relationships between predictors and latent variable
    desc = """
    # Measurement model
    Attitude =~ satisfaction + trust + feelings
    
    # Structural model
    Attitude ~ No_deforestation + natural_resources_protecting + recyclable_packaging + reduced_use_of_energy + low_carbon_emissions + reduced_use_of_pesticides + water_sparingly + familiarity
    """

    print("Fitting SEM model...")
    
    # ========== STEP 6: MODEL FITTING ==========
    # Step 6.1: Create SEM model object from specification
    # Step 6.2: Fit the model to the data
    try:
        model = semopy.Model(desc)
        results = model.fit(df)
    except Exception as e:
        print(f"Error fitting model: {e}")
        return
    
    # ========== STEP 7: EXTRACT MODEL RESULTS ==========
    # Retrieve parameter estimates, standard errors, and p-values from fitted model
    stats = model.inspect()
    
    # ========== STEP 7.5: MODEL FIT EVALUATION ==========
    print("\n" + "="*70)
    print("MODEL FIT INDICES")
    print("="*70)
    
    try:
        # Use semopy's calc_stats to get fit indices
        fit_stats = semopy.calc_stats(model)
        
        print("\n[Fit Indices] Model Fit Statistics:")
        print("="*70)
        
        # Helper function to extract scalar value from Series
        def get_scalar(series_or_value):
            if hasattr(series_or_value, 'item'):
                return series_or_value.item()
            elif hasattr(series_or_value, 'values'):
                return series_or_value.values[0] if len(series_or_value.values) > 0 else series_or_value
            else:
                return series_or_value
        
        # Display available fit indices
        if 'DoF' in fit_stats:
            dof = get_scalar(fit_stats['DoF'])
            print(f"\nDegrees of Freedom (DoF): {dof}")
        
        if 'MLchi_sq' in fit_stats:
            chi2 = get_scalar(fit_stats['MLchi_sq'])
            print(f"\nChi-Square (χ²): {chi2:.4f}")
            if 'DoF' in fit_stats and get_scalar(fit_stats['DoF']) > 0:
                from scipy.stats import chi2 as chi2_dist
                p_value = 1 - chi2_dist.cdf(chi2, get_scalar(fit_stats['DoF']))
                print(f"P-value: {p_value:.4f}")
                if p_value > 0.05:
                    print("✓ Model fits well (p > 0.05)")
                else:
                    print("⚠ Model fit may be poor (p < 0.05)")
        
        if 'CFI' in fit_stats:
            cfi = get_scalar(fit_stats['CFI'])
            print(f"\nCFI (Comparative Fit Index): {cfi:.4f}")
            print("Rule: CFI > 0.95 = excellent, 0.90-0.95 = acceptable, < 0.90 = poor")
            if cfi > 0.95:
                print("✓ Excellent fit")
            elif cfi > 0.90:
                print("✓ Acceptable fit")
            else:
                print("⚠ Poor fit")
        
        if 'RMSEA' in fit_stats:
            rmsea = get_scalar(fit_stats['RMSEA'])
            print(f"\nRMSEA (Root Mean Square Error of Approximation): {rmsea:.4f}")
            print("Rule: RMSEA < 0.05 = excellent, 0.05-0.08 = acceptable, > 0.10 = poor")
            if rmsea < 0.05:
                print("✓ Excellent fit")
            elif rmsea < 0.08:
                print("✓ Acceptable fit")
            elif rmsea < 0.10:
                print("⚠ Mediocre fit")
            else:
                print("⚠ Poor fit")
        
        if 'SRMR' in fit_stats:
            srmr = get_scalar(fit_stats['SRMR'])
            print(f"\nSRMR (Standardized Root Mean Square Residual): {srmr:.4f}")
            print("Rule: SRMR < 0.05 = excellent, 0.05-0.08 = acceptable, > 0.10 = poor")
            if srmr < 0.05:
                print("✓ Excellent fit")
            elif srmr < 0.08:
                print("✓ Acceptable fit")
            else:
                print("⚠ Poor fit")
        
        if 'GFI' in fit_stats:
            gfi = get_scalar(fit_stats['GFI'])
            print(f"\nGFI (Goodness of Fit Index): {gfi:.4f}")
            print("Rule: GFI > 0.95 = good fit")
            if gfi > 0.95:
                print("✓ Good fit")
            else:
                print("⚠ May need improvement")
        
        if 'AGFI' in fit_stats:
            agfi = get_scalar(fit_stats['AGFI'])
            print(f"\nAGFI (Adjusted Goodness of Fit Index): {agfi:.4f}")
            print("Rule: AGFI > 0.90 = acceptable")
            if agfi > 0.90:
                print("✓ Acceptable fit")
            else:
                print("⚠ May need improvement")
        
        # Create summary table
        print("\n" + "="*70)
        print("FIT SUMMARY")
        print("="*70)
        
        summary_data = []
        if 'CFI' in fit_stats:
            cfi_val = get_scalar(fit_stats['CFI'])
            summary_data.append(['CFI', f"{cfi_val:.4f}", '> 0.95', 
                                '✓ Good' if cfi_val > 0.95 else ('✓ OK' if cfi_val > 0.90 else '⚠ Poor')])
        if 'RMSEA' in fit_stats:
            rmsea_val = get_scalar(fit_stats['RMSEA'])
            summary_data.append(['RMSEA', f"{rmsea_val:.4f}", '< 0.05', 
                                '✓ Good' if rmsea_val < 0.05 else ('✓ OK' if rmsea_val < 0.08 else '⚠ Poor')])
        if 'SRMR' in fit_stats:
            srmr_val = get_scalar(fit_stats['SRMR'])
            summary_data.append(['SRMR', f"{srmr_val:.4f}", '< 0.05', 
                                '✓ Good' if srmr_val < 0.05 else ('✓ OK' if srmr_val < 0.08 else '⚠ Poor')])
        if 'GFI' in fit_stats:
            gfi_val = get_scalar(fit_stats['GFI'])
            summary_data.append(['GFI', f"{gfi_val:.4f}", '> 0.95', 
                                '✓ Good' if gfi_val > 0.95 else '⚠ Check'])
        
        if summary_data:
            fit_summary = pd.DataFrame(summary_data, columns=['Index', 'Value', 'Cutoff', 'Status'])
            print(fit_summary.to_string(index=False))
        else:
            print("No fit indices available. This may be a just-identified model.")
        
    except Exception as e:
        print(f"\n⚠ Could not calculate fit indices: {e}")
        print("Note: Fit indices are typically only available for over-identified models.")
        print("Your model may be saturated (just-identified), which means it perfectly fits the data by design.")
    
    print("\n" + "="*70)
    print("END OF MODEL FIT EVALUATION")
    print("="*70)
    
    # ========== STEP 8: FILTER STRUCTURAL COEFFICIENTS ==========
    # Extract only the regression coefficients (op='~') for the 'Attitude' variable
    # This gives us the direct effects of environmental factors on Attitude
    reg_results = stats[(stats['op'] == '~') & (stats['lval'] == 'Attitude')]
    
    # ========== STEP 9: DISPLAY SEM RESULTS ==========
    print("\n--- SEM Analysis Results ---")
    print(reg_results[['rval', 'Estimate', 'p-value', 'Std. Err']])

    # ========== STEP 10: IDENTIFY DECISIVE FACTOR ==========
    print("\n--- Analysis of Decisive Factor ---")
    
    # Step 10.1: Filter for statistically significant results (p < 0.05)
    significant = reg_results[reg_results['p-value'] < 0.05]
    
    if not significant.empty:
        # Step 10.2: Identify the factor with the largest absolute coefficient
        best_row = significant.loc[significant['Estimate'].abs().idxmax()]
        
        # Step 10.3: Display the most decisive factor and its statistics
        print(f"The most decisive factor (highest absolute coefficient with p < 0.05) is '{best_row['rval']}'")
        print(f"Coefficient: {best_row['Estimate']:.4f}")
        print(f"P-value: {best_row['p-value']:.4e}")
    else:
        print("No variables are statistically significant at p < 0.05.")

    # ========== STEP 11: DATA VISUALIZATION ==========
    print("\nGenerating visualizations...")
    
    # Set visualization style
    sns.set_style("whitegrid")
    
    # ---------- Visualization 1: SEM Path Coefficients ----------
    print("\n[1/5] Displaying SEM Path Coefficients...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Sort by absolute estimate
    reg_sorted = reg_results.copy()
    reg_sorted['abs_estimate'] = reg_sorted['Estimate'].abs()
    reg_sorted = reg_sorted.sort_values('abs_estimate', ascending=True)
    
    # Create color map based on significance
    colors = ['#27ae60' if p < 0.05 else '#95a5a6' for p in reg_sorted['p-value']]
    
    # Create horizontal bar plot
    bars = ax1.barh(range(len(reg_sorted)), reg_sorted['Estimate'].values, color=colors)
    ax1.set_yticks(range(len(reg_sorted)))
    ax1.set_yticklabels(reg_sorted['rval'].values, fontsize=11)
    ax1.set_xlabel('Path Coefficient (β)', fontsize=13, fontweight='bold')
    ax1.set_title('SEM Path Coefficients: Environmental Factors → Attitude\n(Green = Significant p<0.05, Gray = Not Significant)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(reg_sorted.iterrows()):
        value = row['Estimate']
        x_pos = value + (0.01 if value > 0 else -0.01)
        ha = 'left' if value > 0 else 'right'
        ax1.text(x_pos, i, f"{value:.4f}", va='center', ha=ha, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('SEM_path_coefficients.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'SEM_path_coefficients.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 2: P-value Comparison ----------
    print("\n[2/5] Displaying P-value Comparison...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Sort by p-value
    pval_sorted = reg_results.sort_values('p-value')
    colors_pval = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in pval_sorted['p-value']]
    
    bars_pval = ax2.barh(range(len(pval_sorted)), pval_sorted['p-value'].values, color=colors_pval)
    ax2.set_yticks(range(len(pval_sorted)))
    ax2.set_yticklabels(pval_sorted['rval'].values, fontsize=11)
    
    # Add significance threshold
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (α=0.05)')
    
    ax2.set_xlabel('P-value', fontsize=13, fontweight='bold')
    ax2.set_title('Statistical Significance of Path Coefficients\n(Red = Significant, Gray = Not Significant)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(0.1, pval_sorted['p-value'].max() * 1.1))
    ax2.legend(fontsize=11)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(pval_sorted.iterrows()):
        p = row['p-value']
        ax2.text(p + 0.005, i, f'{p:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('SEM_pvalues.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'SEM_pvalues.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 3: Correlation Heatmap ----------
    print("\n[3/5] Displaying Correlation Heatmap...")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    
    # Calculate correlation matrix for environmental variables
    env_vars = ['No_deforestation', 'natural_resources_protecting', 'recyclable_packaging',
                'reduced_use_of_energy', 'low_carbon_emissions', 'reduced_use_of_pesticides',
                'water_sparingly', 'familiarity']
    
    corr_matrix = df[env_vars].corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax3, vmin=-1, vmax=1, annot_kws={'size': 10})
    
    # Format labels
    labels = [var.replace('_', ' ').title() for var in env_vars]
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax3.set_yticklabels(labels, rotation=0, fontsize=10)
    ax3.set_title('Correlation Matrix: Environmental Awareness Variables', 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('SEM_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'SEM_correlation_heatmap.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 4: Coefficient Comparison with Error Bars ----------
    print("\n[4/5] Displaying Coefficients with Confidence Intervals...")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    reg_sorted2 = reg_results.sort_values('Estimate', ascending=True)
    
    # Create error bars plot
    y_pos = range(len(reg_sorted2))
    estimates = reg_sorted2['Estimate'].values
    std_errors = reg_sorted2['Std. Err'].values
    
    # Color based on significance
    colors_err = ['#3498db' if p < 0.05 else '#95a5a6' for p in reg_sorted2['p-value']]
    
    # Plot points with error bars
    for i, (est, err, color) in enumerate(zip(estimates, std_errors, colors_err)):
        ax4.errorbar(est, i, xerr=err*1.96, fmt='o', color=color, 
                    markersize=10, capsize=6, capthick=2, linewidth=2)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(reg_sorted2['rval'].values, fontsize=11)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Coefficient Estimate ± 95% CI', fontsize=13, fontweight='bold')
    ax4.set_title('SEM Path Coefficients with 95% Confidence Intervals\n(Blue = Significant, Gray = Not Significant)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('SEM_confidence_intervals.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'SEM_confidence_intervals.png'")
    plt.show()
    plt.close()
    
    # ---------- Visualization 5: Measurement Model ----------
    print("\n[5/5] Displaying Measurement Model...")
    
    # Get measurement model loadings
    measurement_stats = stats[(stats['op'] == '=~')]
    
    if not measurement_stats.empty:
        fig5, ax5 = plt.subplots(figsize=(10, 7))
        
        # Create bar plot for factor loadings
        loadings = measurement_stats['Estimate'].values
        indicators = measurement_stats['rval'].values
        colors_loading = ['#16a085', '#27ae60', '#2ecc71']
        
        bars_loading = ax5.bar(range(len(loadings)), loadings, color=colors_loading, 
                              edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax5.set_xticks(range(len(loadings)))
        ax5.set_xticklabels(indicators, fontsize=13, fontweight='bold')
        ax5.set_ylabel('Factor Loading', fontsize=13, fontweight='bold')
        ax5.set_title('Measurement Model: Attitude Latent Variable\nFactor Loadings on Observed Indicators', 
                     fontsize=14, fontweight='bold', pad=20)
        ax5.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars_loading, loadings)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('SEM_measurement_model.png', dpi=300, bbox_inches='tight')
        print("✓ Saved as 'SEM_measurement_model.png'")
        plt.show()
        plt.close()
    
    print("\n" + "="*70)
    print("SEM Analysis complete! All visualizations saved.")
    print("="*70)

# ========== STEP 12: PROGRAM EXECUTION ==========
# Run the SEM analysis when script is executed directly
if __name__ == "__main__":
    run_sem_analysis()
