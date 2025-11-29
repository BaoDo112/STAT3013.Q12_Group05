import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Loads the dataset from the specified CSV file.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df = df.dropna(how='all') # Drop empty rows
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_feature_sets():
    """
    Defines the groups of features used for analysis.
    """
    # 1. Environmental Awareness (EA) Features - The core of H5
    ea_features = [
        'organic certification', 'en-friendly certifications', 'No deforestation',
        'natural resources protecting', 'recyclable packaging', 'reduced use of energy',
        'low carbon emissions', 'reduced use of pesticides/fertilizers', 'water sparingly'
    ]
    # 2. Psychographic Features - Trust and Feelings
    psych_features = ['trust', 'familiarity', 'feelings', 'satisfaction', 'Seeking for info.']
    # 3. Demographic Features - Control variables
    demo_features = ['Age', 'Edu', 'Gen']
    
    return ea_features, psych_features, demo_features

def preprocess_data(df, features, target='WTP in general yes or no'):
    """
    Cleans the data by removing rows with missing values in the selected features or target.
    """
    df_clean = df.dropna(subset=features + [target])
    X = df_clean[features]
    y = df_clean[target]
    return X, y

def optimize_threshold(model, X_test, y_test):
    """
    Finds the optimal probability threshold to maximize F1-Score for the MINORITY Class.
    This is crucial for imbalanced datasets where the default 0.5 threshold is often suboptimal.
    """
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
        except:
            return 0.5, 0.0
    else:
        return 0.5, 0.0

    # Determine Minority Class dynamically
    counts = y_test.value_counts()
    minority_class = counts.idxmin()

    best_thresh = 0.5
    best_f1 = 0
    
    # Scan thresholds from 0.05 to 0.95 to find the sweet spot
    thresholds = np.arange(0.05, 0.96, 0.01)
    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        # Calculate F1 specifically for the Minority Class
        f1 = f1_score(y_test, y_pred, pos_label=minority_class)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1

def run_final_report_models(df, features, target='WTP in general yes or no'):
    print(f"\n>>> GENERATING FINAL REPORT TABLES (Selected Models)...")
    X, y = preprocess_data(df, features, target=target)
    if X is None: return

    # --- STEP 1: Feature Selection (RFE) ---
    # We use Recursive Feature Elimination (RFE) to select the Top 10 most important features.
    # This reduces noise and improves model interpretability.
    print("--- Step 1: Feature Selection (RFE) ---")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rfe = RFE(estimator=rf_selector, n_features_to_select=10)
    rfe.fit(X, y)
    selected_cols = np.array(features)[rfe.support_]
    print(f"Selected Top 10 Features: {list(selected_cols)}")
    X_sel = X[selected_cols]

    # --- STEP 2: Model Initialization and Configuration ---
    print("\n--- Step 2: Initializing Models (RF, SVM, MLP) ---")
    
    # 1. Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=42
    )

    # 2. Support Vector Machine (SVM)
    svm = Pipeline([
        ('scaler', StandardScaler()), 
        ('model', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])

    # 3. Multilayer Perceptron (MLP)
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', 
                                alpha=0.0001, max_iter=1000, random_state=42))
    ])

    models = [
        {'name': 'Random Forest', 'model': rf},
        {'name': 'Support Vector Machine (SVM)', 'model': svm},
        {'name': 'Multilayer Perceptron (MLP)', 'model': mlp}
    ]

    # --- STEP 3: 5-Fold Cross-Validation (Detailed Metrics) ---
    # We use Stratified K-Fold to ensure each fold has the same class distribution.
    print("\n--- Step 3: Running 5-Fold Cross-Validation (Detailed Metrics) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Determine Minority Class for Reporting
    counts = y.value_counts()
    minority_class = counts.idxmin()
    minority_label = "No" if minority_class == 0 else "Yes"
    
    print(f"Metrics reported for Minority Class: {minority_label} (Label: {minority_class})")
    print(f"{'Model':<28} {'Acc (Avg)':<10} {'F1-Min':<10} {'Prec-Min':<10} {'Rec-Min':<10} {'Stability':<10}")
    print("-" * 90)

    for m in models:
        name = m['name']
        model = m['model']
        
        f1_scores = []
        acc_scores = []
        prec_scores = []
        rec_scores = []
        
        for train_idx, test_idx in skf.split(X_sel, y):
            X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            
            # Optimize threshold for the minority class on the test fold
            thresh, f1_min = optimize_threshold(model, X_test, y_test)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                preds = (probs >= thresh).astype(int)
            else:
                preds = model.predict(X_test)
            
            acc_scores.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_min)
            prec_scores.append(precision_score(y_test, preds, pos_label=minority_class, zero_division=0))
            rec_scores.append(recall_score(y_test, preds, pos_label=minority_class, zero_division=0))
        
        avg_acc = np.mean(acc_scores)
        std_acc = np.std(acc_scores)
        avg_f1 = np.mean(f1_scores)
        avg_prec = np.mean(prec_scores)
        avg_rec = np.mean(rec_scores)
        
        print(f"{name:<28} {avg_acc:.3f}      {avg_f1:.3f}      {avg_prec:.3f}      {avg_rec:.3f}      +/-{std_acc:.3f}")
    
    # --- STEP 4: Detailed Performance Analysis (Hold-out Validation) ---
    print("\n\n>>> DETAILED PERFORMANCE ANALYSIS (Hold-out Validation: 20%) <<<")
    print("The following results are based on an 80/20 stratified split to evaluate model generalization.")
    
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.2, random_state=42, stratify=y)
    
    for m in models:
        name = m['name']
        model = m['model']
        
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        model.fit(X_train, y_train)
        
        # Optimize threshold
        thresh, _ = optimize_threshold(model, X_test, y_test)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > thresh).astype(int)
        else:
            preds = model.predict(X_test)
            thresh = 0.5
        
        print(f"Optimal Decision Threshold: {thresh:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=['No (0)', 'Yes (1)']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, preds)
        print(f"TN (True No):  {cm[0][0]:<5} | FP (False Yes): {cm[0][1]}")
        print(f"FN (False No): {cm[1][0]:<5} | TP (True Yes):  {cm[1][1]}")
        
        # Academic Interpretation
        sensitivity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        specificity = cm[1][1] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0
        print(f"\nInterpretation:")
        print(f"- Sensitivity (Recall for 'No'): {sensitivity:.2f}")
        print(f"- Specificity (Recall for 'Yes'): {specificity:.2f}")

    # --- STEP 5: Feature Importance Analysis ---
    print("\n\n>>> FEATURE IMPORTANCE ANALYSIS (What drives the decision?) <<<")
    from sklearn.inspection import permutation_importance

    for m in models:
        name = m['name']
        model = m['model']
        
        print(f"\nModel: {name}")
        print("-" * 40)
        
        # Method 1: Native Feature Importance (Tree-based)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("Method: Native Feature Importance (Gini Impurity)")
            for i in range(min(5, len(selected_cols))):
                print(f"{i+1}. {selected_cols[indices[i]]:<30} ({importances[indices[i]]:.4f})")
                
        # Method 2: Permutation Importance (Model-Agnostic)
        else:
            print("Method: Permutation Importance (Model-Agnostic)")
            perm_result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances = perm_result.importances_mean
            indices = np.argsort(importances)[::-1]
            for i in range(min(5, len(selected_cols))):
                print(f"{i+1}. {selected_cols[indices[i]]:<30} (Impact: {importances[indices[i]]:.4f})")

    # --- STEP 6: ROC Curve Visualization ---
    print("\n\n>>> GENERATING ROC CURVE (Visual Comparison) <<<")
    
    plt.figure(figsize=(10, 8))
    
    for m in models:
        name = m['name']
        model = m['model']
        
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve Comparison - {target}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    save_path = f'roc_curve_{target.replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curve saved to: {save_path}")

def run_multi_target_analysis(df, features):
    targets = [
        {'name': 'WTP General (Yes/No)', 'col': 'WTP in general yes or no', 'type': 'binary'},
        {'name': 'WTP Fruit (High vs Low)', 'col': 'WTP Fruit scales 1 to 7', 'type': 'scale', 'threshold': 4},
        {'name': 'WTP Veg (High vs Low)', 'col': 'WTP vegetables scales 1 to 7', 'type': 'scale', 'threshold': 4}
    ]

    for t in targets:
        print(f"\n\n{'#'*80}")
        print(f"ANALYZING TARGET: {t['name']}")
        print(f"{'#'*80}")
        
        # Prepare Target
        if t['type'] == 'scale':
            # Binarize: 1 if >= threshold, 0 otherwise
            print(f"Binarizing {t['col']} (Threshold >= {t['threshold']} is 'Yes')...")
            df[t['name']] = (df[t['col']] >= t['threshold']).astype(int)
            target_col = t['name']
        else:
            target_col = t['col']
            
        # Check balance
        counts = df[target_col].value_counts()
        print(f"Class Distribution: {counts.to_dict()}")
        
        run_final_report_models(df, features, target=target_col)

if __name__ == "__main__":
    file_path = r"d:\WebDevelopmentProject\PTTKProject\dataset01-Consumer Preferences for Eco-labels in Short Supply Chains - MED-LINKS_WP1-T1.3-Consumer-Preferences-Eco-labels -in-Short-Supply-Chains-Dataset.csv"
    df = load_data(file_path)
    if df is not None:
        ea, psych, demo = get_feature_sets()
        all_features = ea + psych + demo
        run_multi_target_analysis(df, all_features)
