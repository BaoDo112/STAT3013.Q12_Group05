# ===== MODEL COMPARISON: COMMON METRICS ONLY =====
# This script compares MLR and SEM using ONLY metrics that both models share
# Focuses on fair comparison using common ground

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import semopy
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_mlr_model(df):
    """Run Multiple Linear Regression and return common metrics"""
    print("\n" + "="*70)
    print("RUNNING MULTIPLE LINEAR REGRESSION MODEL")
    print("="*70)
    
    # Define variables
    y_cols = ['satisfaction', 'trust', 'feelings']
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
    
    # Create User_Attitude
    df['User_Attitude'] = df[y_cols].mean(axis=1)
    
    # Prepare data
    X = df[x_cols]
    y = df['User_Attitude']
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    # Get predictions
    y_pred = model.fittedvalues
    
    # Calculate ONLY common metrics
    mlr_metrics = {}
    
    # 1. Prediction Accuracy (COMMON)
    mlr_metrics['RMSE'] = np.sqrt(mean_squared_error(y, y_pred))
    mlr_metrics['MAE'] = mean_absolute_error(y, y_pred)
    mlr_metrics['MAPE'] = calculate_mape(y, y_pred)
    
    # 2. R-squared (COMMON - both have some form of R²)
    mlr_metrics['R²'] = model.rsquared
    
    # 3. Model Complexity (COMMON)
    mlr_metrics['Number of Parameters'] = len(model.params)
    
    # 4. Statistical Significance (COMMON)
    significant_count = sum(model.pvalues.drop('const', errors='ignore') < 0.05)
    mlr_metrics['Significant Predictors'] = significant_count
    
    # Get significant variables for detailed comparison
    sig_vars = model.pvalues.drop('const', errors='ignore')
    sig_vars = sig_vars[sig_vars < 0.05].index.tolist()
    mlr_metrics['Significant Variables'] = sig_vars
    
    # Get coefficients for significant variables
    sig_coefs = {var: model.params[var] for var in sig_vars}
    mlr_metrics['Significant Coefficients'] = sig_coefs
    
    print("✓ MLR Model fitted successfully")
    
    return mlr_metrics, model, y, y_pred

def run_sem_model(df):
    """Run SEM and return common metrics"""
    print("\n" + "="*70)
    print("RUNNING STRUCTURAL EQUATION MODELING")
    print("="*70)
    
    # Rename columns for SEM
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
    
    df_sem = df.rename(columns=rename_map)
    
    # Define SEM model
    desc = """
    # Measurement model
    Attitude =~ satisfaction + trust + feelings
    
    # Structural model
    Attitude ~ No_deforestation + natural_resources_protecting + recyclable_packaging + reduced_use_of_energy + low_carbon_emissions + reduced_use_of_pesticides + water_sparingly + familiarity
    """
    
    # Fit SEM
    model_sem = semopy.Model(desc)
    results = model_sem.fit(df_sem)
    
    # Get statistics
    stats_sem = model_sem.inspect()
    
    # Calculate ONLY common metrics
    sem_metrics = {}
    
    # Get predictions
    reg_results = stats_sem[(stats_sem['op'] == '~') & (stats_sem['lval'] == 'Attitude')]
    
    y_cols = ['satisfaction', 'trust', 'feelings']
    df_sem['User_Attitude'] = df_sem[y_cols].mean(axis=1)
    
    # For prediction, we'll use the structural model coefficients
    predictors = reg_results['rval'].values
    coefficients = reg_results['Estimate'].values
    
    X_sem = df_sem[predictors].values
    y_pred_sem = X_sem @ coefficients
    
    y_true_sem = df_sem['User_Attitude'].values
    
    # Scale predictions to match observed range
    y_pred_sem_scaled = (y_pred_sem - y_pred_sem.mean()) * y_true_sem.std() + y_true_sem.mean()
    
    # 1. Prediction metrics (COMMON)
    sem_metrics['RMSE'] = np.sqrt(mean_squared_error(y_true_sem, y_pred_sem_scaled))
    sem_metrics['MAE'] = mean_absolute_error(y_true_sem, y_pred_sem_scaled)
    sem_metrics['MAPE'] = calculate_mape(y_true_sem, y_pred_sem_scaled)
    
    # 2. R² approximation (COMMON)
    sem_metrics['R²'] = r2_score(y_true_sem, y_pred_sem_scaled)
    
    # 3. Model complexity (COMMON)
    sem_metrics['Number of Parameters'] = len(stats_sem)
    
    # 4. Significant paths (COMMON)
    significant_paths = sum(reg_results['p-value'] < 0.05)
    sem_metrics['Significant Predictors'] = significant_paths
    
    # Get significant variables
    sig_results = reg_results[reg_results['p-value'] < 0.05]
    sig_vars = sig_results['rval'].tolist()
    sem_metrics['Significant Variables'] = sig_vars
    
    # Get coefficients for significant variables
    sig_coefs = {row['rval']: row['Estimate'] for _, row in sig_results.iterrows()}
    sem_metrics['Significant Coefficients'] = sig_coefs
    
    print("✓ SEM Model fitted successfully")
    
    return sem_metrics, model_sem, stats_sem, y_true_sem, y_pred_sem_scaled

def create_common_comparison_table(mlr_metrics, sem_metrics):
    """Create comparison table with ONLY common metrics"""
    print("\n" + "="*80)
    print("MODEL COMPARISON - COMMON METRICS ONLY")
    print("="*80)
    print("\nFocus: Comparing only metrics that BOTH models can calculate")
    print("="*80)
    
    comparison_data = []
    
    # Category 1: PREDICTION ACCURACY
    print("\n[1] PREDICTION ACCURACY")
    print("-" * 80)
    
    comparison_data.append({
        'Category': 'Prediction',
        'Metric': 'RMSE',
        'MLR': f"{mlr_metrics['RMSE']:.4f}",
        'SEM': f"{sem_metrics['RMSE']:.4f}",
        'Difference': f"{abs(mlr_metrics['RMSE'] - sem_metrics['RMSE']):.4f}",
        'Better': 'MLR' if mlr_metrics['RMSE'] < sem_metrics['RMSE'] else 'SEM',
        'Interpretation': 'Lower is better - Root Mean Square Error'
    })
    
    comparison_data.append({
        'Category': 'Prediction',
        'Metric': 'MAE',
        'MLR': f"{mlr_metrics['MAE']:.4f}",
        'SEM': f"{sem_metrics['MAE']:.4f}",
        'Difference': f"{abs(mlr_metrics['MAE'] - sem_metrics['MAE']):.4f}",
        'Better': 'MLR' if mlr_metrics['MAE'] < sem_metrics['MAE'] else 'SEM',
        'Interpretation': 'Lower is better - Mean Absolute Error'
    })
    
    comparison_data.append({
        'Category': 'Prediction',
        'Metric': 'MAPE (%)',
        'MLR': f"{mlr_metrics['MAPE']:.2f}%",
        'SEM': f"{sem_metrics['MAPE']:.2f}%",
        'Difference': f"{abs(mlr_metrics['MAPE'] - sem_metrics['MAPE']):.2f}%",
        'Better': 'MLR' if mlr_metrics['MAPE'] < sem_metrics['MAPE'] else 'SEM',
        'Interpretation': 'Lower is better - Mean Absolute Percentage Error'
    })
    
    # Category 2: EXPLANATORY POWER
    print("\n[2] EXPLANATORY POWER")
    print("-" * 80)
    
    comparison_data.append({
        'Category': 'Explanatory Power',
        'Metric': 'R²',
        'MLR': f"{mlr_metrics['R²']:.4f}",
        'SEM': f"{sem_metrics['R²']:.4f}",
        'Difference': f"{abs(mlr_metrics['R²'] - sem_metrics['R²']):.4f}",
        'Better': 'MLR' if mlr_metrics['R²'] > sem_metrics['R²'] else 'SEM',
        'Interpretation': 'Higher is better - Variance explained'
    })
    
    # Category 3: STATISTICAL SIGNIFICANCE
    print("\n[3] STATISTICAL SIGNIFICANCE")
    print("-" * 80)
    
    comparison_data.append({
        'Category': 'Significance',
        'Metric': 'Number of Significant Predictors',
        'MLR': str(mlr_metrics['Significant Predictors']),
        'SEM': str(sem_metrics['Significant Predictors']),
        'Difference': str(abs(mlr_metrics['Significant Predictors'] - sem_metrics['Significant Predictors'])),
        'Better': 'Equal' if mlr_metrics['Significant Predictors'] == sem_metrics['Significant Predictors'] 
                  else f"{'MLR' if mlr_metrics['Significant Predictors'] > sem_metrics['Significant Predictors'] else 'SEM'} (more)",
        'Interpretation': 'Predictors with p < 0.05'
    })
    
    # Category 4: MODEL COMPLEXITY
    print("\n[4] MODEL COMPLEXITY (Parsimony)")
    print("-" * 80)
    
    comparison_data.append({
        'Category': 'Complexity',
        'Metric': 'Number of Parameters',
        'MLR': str(mlr_metrics['Number of Parameters']),
        'SEM': str(sem_metrics['Number of Parameters']),
        'Difference': str(abs(mlr_metrics['Number of Parameters'] - sem_metrics['Number of Parameters'])),
        'Better': 'MLR' if mlr_metrics['Number of Parameters'] < sem_metrics['Number of Parameters'] else 'SEM',
        'Interpretation': 'Lower is better - Simpler model (Occam\'s Razor)'
    })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display table
    print("\n" + "="*80)
    print("COMPARISON TABLE - COMMON METRICS")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(comparison_df.to_string(index=False))
    
    # Detailed significant variables comparison
    print("\n" + "="*80)
    print("SIGNIFICANT VARIABLES COMPARISON")
    print("="*80)
    
    mlr_sig = set(mlr_metrics['Significant Variables'])
    sem_sig = set(sem_metrics['Significant Variables'])
    
    # Map SEM variable names back to original
    sem_sig_original = set()
    name_map = {
        'No_deforestation': 'No deforestation',
        'natural_resources_protecting': 'natural resources protecting',
        'recyclable_packaging': 'recyclable packaging',
        'reduced_use_of_energy': 'reduced use of energy',
        'low_carbon_emissions': 'low carbon emissions',
        'reduced_use_of_pesticides': 'reduced use of pesticides/fertilizers',
        'water_sparingly': 'water sparingly',
        'familiarity': 'familiarity'
    }
    
    for var in sem_sig:
        if var in name_map:
            sem_sig_original.add(name_map[var])
    
    both = mlr_sig & sem_sig_original
    only_mlr = mlr_sig - sem_sig_original
    only_sem = sem_sig_original - mlr_sig
    
    print(f"\nSignificant in BOTH models ({len(both)}):")
    for var in sorted(both):
        mlr_coef = mlr_metrics['Significant Coefficients'].get(var, 'N/A')
        # Find SEM coefficient
        sem_var_name = [k for k, v in name_map.items() if v == var][0] if var in [v for v in name_map.values()] else var
        sem_coef = sem_metrics['Significant Coefficients'].get(sem_var_name, 'N/A')
        print(f"  ✓ {var}")
        print(f"      MLR coef: {mlr_coef:.4f} | SEM coef: {sem_coef:.4f}")
    
    if only_mlr:
        print(f"\nSignificant ONLY in MLR ({len(only_mlr)}):")
        for var in sorted(only_mlr):
            mlr_coef = mlr_metrics['Significant Coefficients'].get(var, 'N/A')
            print(f"  • {var} (coef: {mlr_coef:.4f})")
    
    if only_sem:
        print(f"\nSignificant ONLY in SEM ({len(only_sem)}):")
        for var in sorted(only_sem):
            sem_var_name = [k for k, v in name_map.items() if v == var][0] if var in [v for v in name_map.values()] else var
            sem_coef = sem_metrics['Significant Coefficients'].get(sem_var_name, 'N/A')
            print(f"  • {var} (coef: {sem_coef:.4f})")
    
    print("\n" + "="*80)
    
    return comparison_df

def create_common_visualizations(comparison_df, mlr_metrics, sem_metrics):
    """Create visualizations for common metrics only"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS - COMMON METRICS")
    print("="*70)
    
    sns.set_style("whitegrid")
    
    # Visualization 1: Side-by-side comparison
    print("\n[1/3] Creating Side-by-Side Metrics Comparison...")
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Model Comparison: Common Metrics Only\nMLR vs SEM', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: R² Comparison
    ax1 = axes[0, 0]
    r2_data = [mlr_metrics['R²'], sem_metrics['R²']]
    colors_r2 = ['#3498db', '#e74c3c']
    bars1 = ax1.bar(['MLR', 'SEM'], r2_data, color=colors_r2, edgecolor='black', linewidth=2, width=0.6)
    ax1.set_ylabel('R² Value', fontsize=12, fontweight='bold')
    ax1.set_title('R² - Explanatory Power\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(r2_data) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, r2_data):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Error Metrics
    ax2 = axes[0, 1]
    error_metrics = ['RMSE', 'MAE']
    mlr_errors = [mlr_metrics['RMSE'], mlr_metrics['MAE']]
    sem_errors = [sem_metrics['RMSE'], sem_metrics['MAE']]
    
    x = np.arange(len(error_metrics))
    width = 0.35
    bars2a = ax2.bar(x - width/2, mlr_errors, width, label='MLR', color='#3498db', edgecolor='black')
    bars2b = ax2.bar(x + width/2, sem_errors, width, label='SEM', color='#e74c3c', edgecolor='black')
    
    ax2.set_ylabel('Error Value', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Errors\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_metrics, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: MAPE Comparison
    ax3 = axes[1, 0]
    mape_data = [mlr_metrics['MAPE'], sem_metrics['MAPE']]
    bars3 = ax3.bar(['MLR', 'SEM'], mape_data, color=['#9b59b6', '#e67e22'], edgecolor='black', linewidth=2, width=0.6)
    ax3.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Mean Absolute Percentage Error\n(Lower is Better)', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, mape_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 4: Complexity and Significance
    ax4 = axes[1, 1]
    metrics_labels = ['Significant\nPredictors', 'Total\nParameters']
    mlr_complex = [mlr_metrics['Significant Predictors'], mlr_metrics['Number of Parameters']]
    sem_complex = [sem_metrics['Significant Predictors'], sem_metrics['Number of Parameters']]
    
    x2 = np.arange(len(metrics_labels))
    width2 = 0.35
    bars4a = ax4.bar(x2 - width2/2, mlr_complex, width2, label='MLR', color='#3498db', edgecolor='black')
    bars4b = ax4.bar(x2 + width2/2, sem_complex, width2, label='SEM', color='#e74c3c', edgecolor='black')
    
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('Model Complexity & Significance', fontsize=13, fontweight='bold')
    ax4.set_xticks(x2)
    ax4.set_xticklabels(metrics_labels, fontsize=11)
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('common_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'common_metrics_comparison.png'")
    plt.show()
    plt.close()
    
    # Visualization 2: Difference Chart
    print("\n[2/3] Creating Difference Chart...")
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate percentage differences (MLR as baseline)
    metrics_names = ['R²', 'RMSE', 'MAE', 'MAPE']
    differences = [
        ((sem_metrics['R²'] - mlr_metrics['R²']) / mlr_metrics['R²']) * 100,
        ((sem_metrics['RMSE'] - mlr_metrics['RMSE']) / mlr_metrics['RMSE']) * 100,
        ((sem_metrics['MAE'] - mlr_metrics['MAE']) / mlr_metrics['MAE']) * 100,
        ((sem_metrics['MAPE'] - mlr_metrics['MAPE']) / mlr_metrics['MAPE']) * 100
    ]
    
    colors_diff = ['#27ae60' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in differences]
    
    # Note: For error metrics (RMSE, MAE, MAPE), negative difference means SEM is better (lower error)
    # For R², positive difference means SEM is better (higher R²)
    
    bars = ax.barh(metrics_names, differences, color=colors_diff, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Percentage Difference (%)\n← SEM Better (for errors) | MLR Better (for errors) →', 
                  fontsize=12, fontweight='bold')
    ax.set_title('SEM vs MLR: Percentage Difference\n(Using MLR as Baseline)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, diff, metric in zip(bars, differences, metrics_names):
        x_pos = diff + (0.05 if diff > 0 else -0.05)
        ha = 'left' if diff > 0 else 'right'
        
        # Interpretation
        if metric == 'R²':
            better = 'SEM' if diff > 0 else 'MLR' if diff < 0 else 'Equal'
        else:  # Errors
            better = 'SEM' if diff < 0 else 'MLR' if diff > 0 else 'Equal'
        
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{diff:+.2f}%\n({better})', ha=ha, va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('common_metrics_differences.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'common_metrics_differences.png'")
    plt.show()
    plt.close()
    
    # Visualization 3: Summary Table
    print("\n[3/3] Creating Summary Table...")
    fig3, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['METRIC', 'MLR', 'SEM', 'DIFFERENCE', 'WINNER'],
        ['', '', '', '', ''],
        ['R² (Explanatory Power)', f"{mlr_metrics['R²']:.4f}", f"{sem_metrics['R²']:.4f}",
         f"{abs(mlr_metrics['R²'] - sem_metrics['R²']):.4f}",
         '✓ MLR' if mlr_metrics['R²'] > sem_metrics['R²'] else '✓ SEM'],
        ['', '', '', '', ''],
        ['RMSE (Prediction Error)', f"{mlr_metrics['RMSE']:.4f}", f"{sem_metrics['RMSE']:.4f}",
         f"{abs(mlr_metrics['RMSE'] - sem_metrics['RMSE']):.4f}",
         '✓ MLR' if mlr_metrics['RMSE'] < sem_metrics['RMSE'] else '✓ SEM'],
        ['MAE (Prediction Error)', f"{mlr_metrics['MAE']:.4f}", f"{sem_metrics['MAE']:.4f}",
         f"{abs(mlr_metrics['MAE'] - sem_metrics['MAE']):.4f}",
         '✓ MLR' if mlr_metrics['MAE'] < sem_metrics['MAE'] else '✓ SEM'],
        ['MAPE (% Error)', f"{mlr_metrics['MAPE']:.2f}%", f"{sem_metrics['MAPE']:.2f}%",
         f"{abs(mlr_metrics['MAPE'] - sem_metrics['MAPE']):.2f}%",
         '✓ MLR' if mlr_metrics['MAPE'] < sem_metrics['MAPE'] else '✓ SEM'],
        ['', '', '', '', ''],
        ['Significant Predictors', str(mlr_metrics['Significant Predictors']), 
         str(sem_metrics['Significant Predictors']),
         str(abs(mlr_metrics['Significant Predictors'] - sem_metrics['Significant Predictors'])),
         '~Equal' if mlr_metrics['Significant Predictors'] == sem_metrics['Significant Predictors'] else 
         ('MLR' if mlr_metrics['Significant Predictors'] > sem_metrics['Significant Predictors'] else 'SEM')],
        ['Total Parameters', str(mlr_metrics['Number of Parameters']), 
         str(sem_metrics['Number of Parameters']),
         str(abs(mlr_metrics['Number of Parameters'] - sem_metrics['Number of Parameters'])),
         '✓ MLR (simpler)' if mlr_metrics['Number of Parameters'] < sem_metrics['Number of Parameters'] else '✓ SEM (simpler)'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # Style the table
    for i, row in enumerate(summary_data):
        for j in range(5):
            cell = table[(i, j)]
            
            # Header row
            if i == 0:
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white', fontsize=12)
            # Empty rows
            elif row[0] == '':
                cell.set_facecolor('#ecf0f1')
            # Data rows
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#ffffff')
                else:
                    cell.set_facecolor('#f8f9fa')
                
                # Highlight winner column
                if j == 4 and '✓' in str(row[j]):
                    cell.set_facecolor('#d5f4e6')
                    cell.set_text_props(weight='bold', color='#27ae60')
    
    plt.title('Summary Comparison Table - Common Metrics Only', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('common_metrics_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved as 'common_metrics_table.png'")
    plt.show()
    plt.close()
    
    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("="*70)

def provide_common_recommendations(mlr_metrics, sem_metrics):
    """Provide recommendations based on common metrics only"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS - BASED ON COMMON METRICS")
    print("="*80)
    
    mlr_score = 0
    sem_score = 0
    
    print("\nScoring based on common metrics:")
    print("-" * 80)
    
    # 1. R²
    if mlr_metrics['R²'] > sem_metrics['R²']:
        mlr_score += 2
        print(f"✓ MLR: Higher R² ({mlr_metrics['R²']:.4f} vs {sem_metrics['R²']:.4f}) +2 points")
    else:
        sem_score += 2
        print(f"✓ SEM: Higher R² ({sem_metrics['R²']:.4f} vs {mlr_metrics['R²']:.4f}) +2 points")
    
    # 2. RMSE
    if mlr_metrics['RMSE'] < sem_metrics['RMSE']:
        mlr_score += 2
        print(f"✓ MLR: Lower RMSE ({mlr_metrics['RMSE']:.4f} vs {sem_metrics['RMSE']:.4f}) +2 points")
    else:
        sem_score += 2
        print(f"✓ SEM: Lower RMSE ({sem_metrics['RMSE']:.4f} vs {mlr_metrics['RMSE']:.4f}) +2 points")
    
    # 3. MAE
    if mlr_metrics['MAE'] < sem_metrics['MAE']:
        mlr_score += 1
        print(f"✓ MLR: Lower MAE ({mlr_metrics['MAE']:.4f} vs {sem_metrics['MAE']:.4f}) +1 point")
    else:
        sem_score += 1
        print(f"✓ SEM: Lower MAE ({sem_metrics['MAE']:.4f} vs {mlr_metrics['MAE']:.4f}) +1 point")
    
    # 4. MAPE
    if mlr_metrics['MAPE'] < sem_metrics['MAPE']:
        mlr_score += 1
        print(f"✓ MLR: Lower MAPE ({mlr_metrics['MAPE']:.2f}% vs {sem_metrics['MAPE']:.2f}%) +1 point")
    else:
        sem_score += 1
        print(f"✓ SEM: Lower MAPE ({sem_metrics['MAPE']:.2f}% vs {mlr_metrics['MAPE']:.2f}%) +1 point")
    
    # 5. Parsimony
    if mlr_metrics['Number of Parameters'] < sem_metrics['Number of Parameters']:
        mlr_score += 1
        print(f"✓ MLR: Simpler model ({mlr_metrics['Number of Parameters']} vs {sem_metrics['Number of Parameters']} params) +1 point")
    else:
        sem_score += 1
        print(f"✓ SEM: Simpler model ({sem_metrics['Number of Parameters']} vs {mlr_metrics['Number of Parameters']} params) +1 point")
    
    # 6. Significant predictors
    if mlr_metrics['Significant Predictors'] == sem_metrics['Significant Predictors']:
        print(f"~ Equal: Same number of significant predictors ({mlr_metrics['Significant Predictors']}) +0 points each")
    elif mlr_metrics['Significant Predictors'] > sem_metrics['Significant Predictors']:
        mlr_score += 1
        print(f"✓ MLR: More significant predictors ({mlr_metrics['Significant Predictors']} vs {sem_metrics['Significant Predictors']}) +1 point")
    else:
        sem_score += 1
        print(f"✓ SEM: More significant predictors ({sem_metrics['Significant Predictors']} vs {mlr_metrics['Significant Predictors']}) +1 point")
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL SCORE & RECOMMENDATION")
    print("="*80)
    print(f"\n{'🔵 MLR Score:':<20} {mlr_score} / 7")
    print(f"{'🔴 SEM Score:':<20} {sem_score} / 7")
    
    print("\n" + "-"*80)
    
    if mlr_score > sem_score:
        margin = mlr_score - sem_score
        print(f"\n🏆 WINNER: Multiple Linear Regression (MLR)")
        print(f"   Margin: {margin} point(s)")
        print("\n📊 Key Strengths:")
        if mlr_metrics['R²'] > sem_metrics['R²']:
            print("   • Higher explanatory power (R²)")
        if mlr_metrics['RMSE'] < sem_metrics['RMSE']:
            print("   • Better prediction accuracy (lower errors)")
        if mlr_metrics['Number of Parameters'] < sem_metrics['Number of Parameters']:
            print("   • Simpler model (fewer parameters)")
            
    elif sem_score > mlr_score:
        margin = sem_score - mlr_score
        print(f"\n🏆 WINNER: Structural Equation Modeling (SEM)")
        print(f"   Margin: {margin} point(s)")
        print("\n📊 Key Strengths:")
        if sem_metrics['R²'] > mlr_metrics['R²']:
            print("   • Higher explanatory power (R²)")
        if sem_metrics['RMSE'] < mlr_metrics['RMSE']:
            print("   • Better prediction accuracy (lower errors)")
        if sem_metrics['Number of Parameters'] < mlr_metrics['Number of Parameters']:
            print("   • Simpler model (fewer parameters)")
            
    else:
        print("\n⚖️ TIE: Both models perform equally well on common metrics")
        print("\n   Consider other factors:")
        print("   • Theoretical framework requirements")
        print("   • Interpretability needs")
        print("   • Stakeholder familiarity")
    
    # Practical differences
    print("\n" + "="*80)
    print("PRACTICAL INTERPRETATION")
    print("="*80)
    
    r2_diff = abs(mlr_metrics['R²'] - sem_metrics['R²']) * 100
    rmse_diff = abs(mlr_metrics['RMSE'] - sem_metrics['RMSE'])
    mae_diff = abs(mlr_metrics['MAE'] - sem_metrics['MAE'])
    
    print(f"\nDifferences in absolute terms:")
    print(f"  • R² difference: {r2_diff:.2f} percentage points")
    print(f"  • RMSE difference: {rmse_diff:.4f}")
    print(f"  • MAE difference: {mae_diff:.4f}")
    
    if r2_diff < 1 and rmse_diff < 0.01:
        print("\n💡 INSIGHT: Differences are very small!")
        print("   Both models perform nearly identically on these common metrics.")
        print("   Choice should be based on:")
        print("   • Your theoretical framework")
        print("   • Need for latent variables (→ SEM)")
        print("   • Preference for simplicity (→ MLR)")
    else:
        print("\n💡 INSIGHT: There are meaningful differences between models.")
        print("   The winning model shows clearly better performance.")
    
    print("\n" + "="*80)

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: COMMON METRICS ONLY")
    print("Fair comparison using metrics both MLR and SEM share")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv('../dataset.csv')
        print(f"\n✓ Data loaded successfully: {len(df)} observations")
    except FileNotFoundError:
        print("\n✗ Error: dataset.csv not found")
        return
    
    # Define required columns
    required_cols = [
        'satisfaction', 'trust', 'feelings',
        'No deforestation', 'natural resources protecting',
        'recyclable packaging', 'reduced use of energy',
        'low carbon emissions', 'reduced use of pesticides/fertilizers',
        'water sparingly', 'familiarity'
    ]
    
    # Check and clean
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"\n✗ Error: Missing columns: {missing_cols}")
        return
    
    original_len = len(df)
    df = df[required_cols].dropna()
    if len(df) < original_len:
        print(f"✓ Removed {original_len - len(df)} rows with missing values")
    print(f"✓ Final dataset: {len(df)} complete observations\n")
    
    # Run both models
    mlr_metrics, mlr_model, y_mlr, y_pred_mlr = run_mlr_model(df.copy())
    sem_metrics, sem_model, sem_stats, y_sem, y_pred_sem = run_sem_model(df.copy())
    
    # Create comparison
    comparison_df = create_common_comparison_table(mlr_metrics, sem_metrics)
    
    # Save table
    comparison_df.to_csv('common_metrics_comparison_table.csv', index=False)
    print(f"\n✓ Comparison table saved as 'common_metrics_comparison_table.csv'")
    
    # Create visualizations
    create_common_visualizations(comparison_df, mlr_metrics, sem_metrics)
    
    # Recommendations
    provide_common_recommendations(mlr_metrics, sem_metrics)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files (COMMON METRICS ONLY):")
    print("  1. common_metrics_comparison_table.csv - Detailed comparison")
    print("  2. common_metrics_comparison.png - Side-by-side charts")
    print("  3. common_metrics_differences.png - Percentage differences")
    print("  4. common_metrics_table.png - Summary table")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
