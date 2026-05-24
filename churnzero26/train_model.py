import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import os

print("Starting model training pipeline...")

# 1. Load Datasets
train_path = r"C:\Users\loq\Downloads\problemstatement\ChurnZero_dataset_v1.csv"
test_path = r"C:\Users\loq\Downloads\problemstatement\ChurnZero_test_v1.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Keep track of Customer IDs
train_ids = df_train['customer_id']
test_ids = df_test['customer_id']

# Separate features and target
X = df_train.drop(columns=['customer_id', 'churn'])
y = df_train['churn']
X_test = df_test.drop(columns=['customer_id'])

# 2. Preprocessing
# Fill missing app ratings
median_rating = X['app_rating_given'].median()
X['app_rating_given'] = X['app_rating_given'].fillna(median_rating)
X_test['app_rating_given'] = X_test['app_rating_given'].fillna(median_rating)

# Convert categorical columns to category type for native support
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {cat_cols}")
for col in cat_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# 3. Model Configuration
RANDOM_STATE = 42

lgb_params = {
    'n_estimators': 800,
    'learning_rate': 0.03,
    'max_depth': 6,
    'num_leaves': 31,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}

xgb_params = {
    'n_estimators': 800,
    'learning_rate': 0.03,
    'max_depth': 6,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'enable_categorical': True,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'n_jobs': -1,
    'verbosity': 0
}

cat_params = {
    'iterations': 800,
    'learning_rate': 0.03,
    'depth': 6,
    'auto_class_weights': 'Balanced',
    'cat_features': cat_cols,
    'random_state': RANDOM_STATE,
    'verbose': 0
}

# 4. Cross-Validation Framework
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_preds_lgb = np.zeros(len(X))
oof_preds_xgb = np.zeros(len(X))
oof_preds_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

trained_lgb_models = []
trained_xgb_models = []
trained_cat_models = []

print("Training Stratified 5-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(X_train, y_train)
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5.0
    trained_lgb_models.append(model_lgb)
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(X_train, y_train)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5.0
    trained_xgb_models.append(model_xgb)
    
    # CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(X_train, y_train)
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / 5.0
    trained_cat_models.append(model_cat)
    
    print(f"  Fold {fold+1} complete.")

# 5. Evaluate Individual Models (PR-AUC / Average Precision)
auc_lgb = average_precision_score(y, oof_preds_lgb)
auc_xgb = average_precision_score(y, oof_preds_xgb)
auc_cat = average_precision_score(y, oof_preds_cat)

print(f"\nModel PR-AUC (Average Precision) Scores:")
print(f"  LightGBM PR-AUC: {auc_lgb:.4f}")
print(f"  XGBoost  PR-AUC: {auc_xgb:.4f}")
print(f"  CatBoost PR-AUC: {auc_cat:.4f}")

# 6. Ensemble Predictions (Average of the 3 models)
oof_ensemble = (oof_preds_lgb + oof_preds_xgb + oof_preds_cat) / 3.0
test_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3.0
ensemble_auc = average_precision_score(y, oof_ensemble)
print(f"\nEnsemble PR-AUC Score: {ensemble_auc:.4f}")

# 7. Optimize Threshold based on Business Cost
# Business Cost = FN * 40,000 + FP * 500
thresholds = np.arange(0.01, 1.00, 0.01)
best_threshold = 0.5
min_cost = float('inf')

for t in thresholds:
    y_pred = (oof_ensemble >= t).astype(int)
    fn = ((y == 1) & (y_pred == 0)).sum()
    fp = ((y == 0) & (y_pred == 1)).sum()
    cost = fn * 40000 + fp * 500
    if cost < min_cost:
        min_cost = cost
        best_threshold = t

# Baseline cost (No interventions at all -> all positives are FNs)
baseline_fn = (y == 1).sum()
baseline_cost = baseline_fn * 40000

print(f"\nBusiness Cost Optimization:")
print(f"  Baseline Cost (No Interventions): INR {baseline_cost:,.2f}")
print(f"  Optimized Cost with Model:       INR {min_cost:,.2f}")
print(f"  Cost Savings:                     INR {baseline_cost - min_cost:,.2f} ({(baseline_cost - min_cost)/baseline_cost:.2%})")
print(f"  Optimal Probability Threshold:    {best_threshold:.2f}")

# Final validation metrics at optimized threshold
final_val_preds = (oof_ensemble >= best_threshold).astype(int)
final_f1 = f1_score(y, final_val_preds)
print(f"  F1-Score at optimal threshold:  {final_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y, final_val_preds, target_names=['Stayed', 'Churned']))

# 8. Export Predictions on Test Set
test_predictions = (test_ensemble >= best_threshold).astype(int)

df_preds = pd.DataFrame({
    'customer_id': test_ids,
    'churn_prediction': test_predictions,
    'churn_probability': test_ensemble
})

preds_filename = "ChurnZero_Smitrax_Predictions.csv"
df_preds.to_csv(preds_filename, index=False)
print(f"\nExported test set predictions to: {preds_filename} (Rows: {len(df_preds)})")

# 9. Compute SHAP Values (using first LightGBM model for explainability)
import shap
print("Computing SHAP values...")
# Convert categorical features to numeric code for SHAP tree explainer compatibility
X_shap = X.copy()
for col in cat_cols:
    X_shap[col] = X_shap[col].cat.codes

model_lgb_shap = lgb.LGBMClassifier(**lgb_params)
model_lgb_shap.fit(X_shap, y)

explainer = shap.TreeExplainer(model_lgb_shap)
shap_values = explainer.shap_values(X_shap)

# Save precomputed SHAP data
shap_data = {
    'shap_values': shap_values,
    'X': X_shap,
    'feature_names': X.columns.tolist()
}
joblib.dump(shap_data, "shap_data.joblib")
print("Saved SHAP values to: shap_data.joblib")

# Save model artifacts and predictions dataset for dashboard
model_artifacts = {
    'features': X.columns.tolist(),
    'cat_cols': cat_cols,
    'optimal_threshold': best_threshold,
    'average_precision': ensemble_auc,
    'business_cost': min_cost,
    'baseline_cost': baseline_cost
}
joblib.dump(model_artifacts, "model_artifacts.joblib")
print("Saved model artifacts to: model_artifacts.joblib")

# Save a file with train predictions for dashboard usage
df_train_predictions = df_train.copy()
df_train_predictions['ChurnProb'] = oof_ensemble
# LTV model: we can use the actual customer_lifetime_value column from the dataset!
# Wait, let's verify if the dataset has customer_lifetime_value column. Yes, the PDF mentions it!
# Let's check how the LTV is represented.
df_train_predictions['LTV'] = df_train['customer_lifetime_value']
df_train_predictions['RiskSegment'] = df_train_predictions['ChurnProb'].apply(
    lambda p: 'Critical' if p >= 0.7 else ('High Risk' if p >= 0.5 else ('Medium Risk' if p >= 0.3 else 'Low Risk'))
)
df_train_predictions.to_csv("customer_predictions.csv", index=False)
print("Saved scored train predictions to: customer_predictions.csv")
print("All tasks completed successfully!")
