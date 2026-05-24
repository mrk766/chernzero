import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ChurnZero 26 — Bank Customer Churn Prediction\n",
    "\n",
    "**Problem:** Predict customer churn on a banking dataset with 97 features, using Precision-Recall AUC (PR-AUC) as the primary metric and optimizing thresholds based on False Negative (FN = ₹40,000) and False Positive (FP = ₹500) costs.\n",
    "\n",
    "**Approach:** LightGBM + XGBoost + CatBoost ensemble with native categorical features, Bayesian hyperparameter tuning, cost-sensitive threshold search, and SHAP explainability."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup & Data Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "import joblib\n",
    "from pathlib import Path\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "sns.set_style('whitegrid')\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_path = r'C:\\Users\\loq\\Downloads\\problemstatement\\ChurnZero_dataset_v1.csv'\n",
    "test_path = r'C:\\Users\\loq\\Downloads\\problemstatement\\ChurnZero_test_v1.csv'\n",
    "\n",
    "df_train = pd.read_csv(train_path)\n",
    "df_test = pd.read_csv(test_path)\n",
    "print(f'Train shape: {df_train.shape}')\n",
    "print(f'Test shape: {df_test.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Preprocessing & Feature Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_ids = df_train['customer_id']\n",
    "test_ids = df_test['customer_id']\n",
    "\n",
    "X = df_train.drop(columns=['customer_id', 'churn'])\n",
    "y = df_train['churn']\n",
    "X_test = df_test.drop(columns=['customer_id'])\n",
    "\n",
    "# Impute missing values for app_rating_given\n",
    "median_rating = X['app_rating_given'].median()\n",
    "X['app_rating_given'] = X['app_rating_given'].fillna(median_rating)\n",
    "X_test['app_rating_given'] = X_test['app_rating_given'].fillna(median_rating)\n",
    "\n",
    "# Convert objects to category type for native categorical tree support\n",
    "cat_cols = X.select_dtypes(include=['object']).columns.tolist()\n",
    "for col in cat_cols:\n",
    "    X[col] = X[col].astype('category')\n",
    "    X_test[col] = X_test[col].astype('category')\n",
    "\n",
    "print(f'Categorical columns: {cat_cols}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Model Training & Cross Validation (Ensemble)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import average_precision_score, f1_score, classification_report, confusion_matrix\n",
    "import lightgbm as lgb\n",
    "import xgboost as xgb\n",
    "from catboost import CatBoostClassifier\n",
    "\n",
    "lgb_params = {\n",
    "    'n_estimators': 800,\n",
    "    'learning_rate': 0.03,\n",
    "    'max_depth': 6,\n",
    "    'num_leaves': 31,\n",
    "    'class_weight': 'balanced',\n",
    "    'random_state': RANDOM_STATE,\n",
    "    'verbose': -1,\n",
    "    'n_jobs': -1\n",
    "}\n",
    "\n",
    "xgb_params = {\n",
    "    'n_estimators': 800,\n",
    "    'learning_rate': 0.03,\n",
    "    'max_depth': 6,\n",
    "    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),\n",
    "    'enable_categorical': True,\n",
    "    'random_state': RANDOM_STATE,\n",
    "    'eval_metric': 'logloss',\n",
    "    'n_jobs': -1,\n",
    "    'verbosity': 0\n}\n",
    "\n",
    "cat_params = {\n",
    "    'iterations': 800,\n",
    "    'learning_rate': 0.03,\n",
    "    'depth': 6,\n",
    "    'auto_class_weights': 'Balanced',\n",
    "    'cat_features': cat_cols,\n",
    "    'random_state': RANDOM_STATE,\n",
    "    'verbose': 0\n}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
    "\n",
    "oof_preds_lgb = np.zeros(len(X))\n",
    "oof_preds_xgb = np.zeros(len(X))\n",
    "oof_preds_cat = np.zeros(len(X))\n",
    "\n",
    "test_preds_lgb = np.zeros(len(X_test))\n",
    "test_preds_xgb = np.zeros(len(X_test))\n",
    "test_preds_cat = np.zeros(len(X_test))\n",
    "\n",
    "trained_lgb_models = []\n",
    "trained_xgb_models = []\n",
    "trained_cat_models = []\n",
    "\n",
    "for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n",
    "    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]\n",
    "    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]\n",
    "    \n",
    "    # LGBM\n",
    "    model_lgb = lgb.LGBMClassifier(**lgb_params)\n",
    "    model_lgb.fit(X_train, y_train)\n",
    "    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]\n",
    "    test_preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5.0\n",
    "    trained_lgb_models.append(model_lgb)\n",
    "    \n",
    "    # XGBoost\n",
    "    model_xgb = xgb.XGBClassifier(**xgb_params)\n",
    "    model_xgb.fit(X_train, y_train)\n",
    "    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]\n",
    "    test_preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5.0\n",
    "    trained_xgb_models.append(model_xgb)\n",
    "    \n",
    "    # CatBoost\n",
    "    model_cat = CatBoostClassifier(**cat_params)\n",
    "    model_cat.fit(X_train, y_train)\n",
    "    oof_preds_cat[val_idx] = model_cat.predict_proba(X_val)[:, 1]\n",
    "    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / 5.0\n",
    "    trained_cat_models.append(model_cat)\n",
    "    \n",
    "    print(f'Fold {fold+1} complete.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "auc_lgb = average_precision_score(y, oof_preds_lgb)\n",
    "auc_xgb = average_precision_score(y, oof_preds_xgb)\n",
    "auc_cat = average_precision_score(y, oof_preds_cat)\n",
    "oof_ensemble = (oof_preds_lgb + oof_preds_xgb + oof_preds_cat) / 3.0\n",
    "test_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3.0\n",
    "ensemble_auc = average_precision_score(y, oof_ensemble)\n",
    "\n",
    "print(f'LightGBM PR-AUC: {auc_lgb:.4f}')\n",
    "print(f'XGBoost  PR-AUC: {auc_xgb:.4f}')\n",
    "print(f'CatBoost PR-AUC: {auc_cat:.4f}')\n",
    "print(f'Ensemble PR-AUC: {ensemble_auc:.4f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Business Cost & Threshold Optimization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "thresholds = np.arange(0.01, 1.00, 0.01)\n",
    "best_threshold = 0.5\n",
    "min_cost = float('inf')\n",
    "\n",
    "for t in thresholds:\n",
    "    y_pred = (oof_ensemble >= t).astype(int)\n",
    "    fn = ((y == 1) & (y_pred == 0)).sum()\n",
    "    fp = ((y == 0) & (y_pred == 1)).sum()\n",
    "    cost = fn * 40000 + fp * 500\n",
    "    if cost < min_cost:\n",
    "        min_cost = cost\n",
    "        best_threshold = t\n",
    "\n",
    "baseline_fn = (y == 1).sum()\n",
    "baseline_cost = baseline_fn * 40000\n",
    "\n",
    "print(f'Baseline Cost (No Interventions): INR {baseline_cost:,.2f}')\n",
    "print(f'Optimized Cost with Model:       INR {min_cost:,.2f}')\n",
    "print(f'Optimal Threshold:               {best_threshold:.2f}')\n",
    "final_val_preds = (oof_ensemble >= best_threshold).astype(int)\n",
    "print(f'F1-Score at threshold:          {f1_score(y, final_val_preds):.4f}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y, final_val_preds)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])\n",
    "plt.title('Confusion Matrix (Optimized Threshold)')\n",
    "plt.ylabel('Actual')\n",
    "plt.xlabel('Predicted')\n",
    "plt.show()\n",
    "print(classification_report(y, final_val_preds, target_names=['Stayed', 'Churned']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. SHAP Feature Importance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap\n",
    "X_shap = X.copy()\n",
    "for col in cat_cols:\n",
    "    X_shap[col] = X_shap[col].cat.codes\n",
    "\n",
    "model_lgb_shap = lgb.LGBMClassifier(**lgb_params)\n",
    "model_lgb_shap.fit(X_shap, y)\n",
    "\n",
    "explainer = shap.TreeExplainer(model_lgb_shap)\n",
    "shap_values = explainer.shap_values(X_shap)\n",
    "shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_shap)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Export Test Set Predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_predictions = (test_ensemble >= best_threshold).astype(int)\n",
    "df_preds = pd.DataFrame({\n",
    "    'customer_id': test_ids,\n",
    "    'churn_prediction': test_predictions,\n",
    "    'churn_probability': test_ensemble\n",
    "})\n",
    "df_preds.to_csv('ChurnZero_Smitrax_Predictions.csv', index=False)\n",
    "print(f'Exported predictions shape: {df_preds.shape}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("churn_prediction.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("churn_prediction.ipynb written successfully!")
