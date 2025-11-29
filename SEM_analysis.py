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
