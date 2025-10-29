# 🚦 Road Accident Risk — Residual-Boosted Risk Model

Playground Series S5E10 (Kaggle)

---

## 🎯 What is this?

Predict `accident_risk` for each road segment as a calibrated score in `[0,1]`.

Goal:

* Stable CV, not leaderboard luck
* Interpretable signals (why is this road risky?)
* Zero leakage

---

## 🧠 Modeling Pipeline (3 stages)

### 1. LightGBM (main learner)

* Train LightGBM directly on `accident_risk`
* Bag multiple random seeds → smoother OOF preds
* Output: `oof_lgb`, `pred_lgb`

### 2. XGBoost residual (prior-corrected)

* Build an interpretable safety prior `risk_prior` ∈ `[0,1]`

  * high curvature
  * high speed limit
  * night lighting
  * bad weather
* Train XGBoost on the residual:
  `residual_target = accident_risk - risk_prior`
* At inference:
  `pred = risk_prior + predicted_residual`
* Output: `oof_xgb`, `pred_xgb`

Why? Stage 2 is only learning what the simple prior missed.

### 3. NNLS blend (non-negative)

* Fit Non-Negative Least Squares (NNLS) on `[oof_lgb, oof_xgb]`
* Get blend weights ≥ 0 (no negative canceling)
* Apply same weights to test preds
* Clip final predictions to `[0,1]`
* Output: `final_test` → `submission.csv`

Result:

* Lower OOF RMSE
* More consistent folds
* Predictions always in a valid range

---

## 🔬 Features & CV

**Feature engineering**

* `curv_speed` = curvature × speed_limit
* `acc_per_lane` = num_reported_accidents / num_lanes
* `critical_zone` = high curvature & high speed
* `risk_prior` = human-readable baseline danger score

**Cross-validation**

* Stratified K-Fold on binned target quantiles
* Keeps each fold balanced (safe vs dangerous segments)
* All metrics are out-of-fold (OOF)

---

## 📂 Output

The notebook will:

1. Train Stage 1 → Stage 2 → Stage 3
2. Blend predictions
3. Write `submission.csv` under `/kaggle/working/`

No external data. No test target leakage.
