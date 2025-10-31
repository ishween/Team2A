import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/processed/cleaned_data.csv")

# Creating target variable
# A lead will be qualified if it moves past "Prospecting" stage
# So this includes Engaging, Won, or Lost - all show the lead warranted sales attention
df["qualified_lead"] = (
    (df.get("deal_stage_ENGAGING", 0) == 1) |
    (df.get("deal_stage_WON", 0) == 1) |
    (df.get("deal_stage_LOST", 0) == 1)
).astype(int)

TARGET = "qualified_lead"

# Remove any columns that would leak the answer
temporal_cols = ['engage_date', 'close_date', 'engage_year', 'engage_month', 
                 'engage_dayofweek', 'days_to_close', 'closed_within_30d']
outcome_cols = ['deal_stage_PROSPECTING', 'deal_stage_ENGAGING', 
                'deal_stage_WON', 'deal_stage_LOST', 'won_deal', 
                'has_close_date', 'close_value', 'close_value_log']
remove_cols = temporal_cols + outcome_cols

# Keeping transformed features, droping raw versions
raw_features = ['revenue', 'employees', 'sales_price']

# Preparing training and test sets
feature_cols = [c for c in df.columns if c not in remove_cols + raw_features + [TARGET]]
X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Class distribution: {y_train.value_counts(normalize=True).to_dict()}")

# Baseline model comparison
print("\n" + "="*60)
print("BASELINE MODEL COMPARISON")
print("="*60)

def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te):
    """train and evaluate a model"""
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    
    # Calculate metrics
    roc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)
    
    # Default 0.5 threshold predictions
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_te, preds)
    
    print(f"\n{name}")
    print(f"  ROC-AUC: {roc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(classification_report(y_te, preds, digits=3))
    
    return probs, {"roc_auc": roc, "pr_auc": pr_auc}

# Model 1: Logistic Regression (Simple baseline)
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_probs, lr_metrics = evaluate_classifier("Logistic Regression", lr_model, 
                                           X_train, y_train, X_test, y_test)

# Model 2: Random Forest (Tree-based ensemble)
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
rf_probs, rf_metrics = evaluate_classifier("Random Forest", rf_model,
                                           X_train, y_train, X_test, y_test)

# Model 3: XGBoost (Gradient boosting)
scale_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])
xgb_model = XGBClassifier(
    n_estimators=600, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    scale_pos_weight=scale_weight, tree_method="hist",
    eval_metric="logloss", random_state=42
)
xgb_probs, xgb_metrics = evaluate_classifier("XGBoost", xgb_model,
                                             X_train, y_train, X_test, y_test)

# Finding optimal threshold
def find_optimal_threshold(y_true, probabilities):
    """find threshold that maximizes F1"""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    
    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find best threshold
    best_idx = np.nanargmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return optimal_threshold, precision[best_idx], recall[best_idx], f1_scores[best_idx]

# Hyperparameter tuning with Random Forest
print("\n" + "="*60)
print("HYPERPARAMETER TUNING (Random Forest)")
print("="*60)

# Cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Broad random search
param_distributions = {
    "n_estimators": randint(250, 1000),
    "max_depth": randint(3, 35),
    "min_samples_split": randint(2, 25),
    "min_samples_leaf": randint(1, 12),
    "max_features": uniform(0.15, 0.85),
    "bootstrap": [True, False],
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    param_distributions=param_distributions,
    n_iter=50,
    scoring="average_precision",
    cv=cv_strategy,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nPhase 1: Random Search...")
random_search.fit(X_train, y_train)
print(f"Best CV PR-AUC: {random_search.best_score_:.4f}")
print(f"Best parameters: {random_search.best_params_}")

# Fine-tuning with grid search
best = random_search.best_params_
param_grid = {
    "n_estimators": [best["n_estimators"], best["n_estimators"] + 150],
    "max_depth": [max(3, best["max_depth"]-2), best["max_depth"], best["max_depth"]+2],
    "min_samples_split": [max(2, best["min_samples_split"]-3), 
                          best["min_samples_split"], 
                          best["min_samples_split"]+3],
    "min_samples_leaf": [max(1, best["min_samples_leaf"]-1), 
                         best["min_samples_leaf"], 
                         best["min_samples_leaf"]+1],
    "max_features": [max(0.1, best["max_features"]-0.15), 
                     best["max_features"], 
                     min(1.0, best["max_features"]+0.15)],
    "bootstrap": [best["bootstrap"]],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1),
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)

print("\nPhase 2: Grid Search Refinement...")
grid_search.fit(X_train, y_train)
print(f"Refined CV PR-AUC: {grid_search.best_score_:.4f}")
print(f"Final parameters: {grid_search.best_params_}")

# Final model evaluation
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE")
print("="*60)

final_model = grid_search.best_estimator_
final_probs = final_model.predict_proba(X_test)[:, 1]

# Find optimal threshold
opt_thresh, opt_prec, opt_rec, opt_f1 = find_optimal_threshold(y_test, final_probs)
print(f"\nOptimal threshold: {opt_thresh:.4f}")
print(f"  Precision: {opt_prec:.3f}")
print(f"  Recall: {opt_rec:.3f}")
print(f"  F1-Score: {opt_f1:.3f}")

# Evaluate at optimal threshold
final_preds = (final_probs >= opt_thresh).astype(int)
final_cm = confusion_matrix(y_test, final_preds)

print(f"\nFinal Test Performance:")
print(f"  ROC-AUC: {roc_auc_score(y_test, final_probs):.4f}")
print(f"  PR-AUC: {average_precision_score(y_test, final_probs):.4f}")
print(f"\nConfusion Matrix:\n{final_cm}")
print(classification_report(y_test, final_preds, digits=3))

# Export model
import pickle

model_package = {
    "classifier": final_model,
    "optimal_threshold": opt_thresh,
    "hyperparameters": grid_search.best_params_,
    "cv_score": grid_search.best_score_,
    "test_metrics": {
        "roc_auc": roc_auc_score(y_test, final_probs),
        "pr_auc": average_precision_score(y_test, final_probs),
        "f1": opt_f1
    },
    "model_version": "1.0",
    "notes": "Optimized Random Forest for lead qualification with custom threshold"
}

with open("lead_scoring_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("\nModel saved to lead_scoring_model.pkl")
