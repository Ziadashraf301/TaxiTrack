import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Config ----
MODEL_PATH = "models/ridge_model.pkl"   # path to your saved model
OUTPUT_CSV = "ridge_feature_importance.csv"

# ---- Load model ----
model = joblib.load(MODEL_PATH)
print("Loaded object type:", type(model))

# ---- Get coefficients ----
if hasattr(model, "coef_"):  
    # Case 1: model is Ridge directly
    coefs = model.coef_
    try:
        features = model.feature_names_in_  # sklearn >= 1.0
    except AttributeError:
        features = [f"feature_{i}" for i in range(len(coefs))]
else:
    raise ValueError("The loaded model does not have coefficients. Check if it's a Ridge model.")

# ---- Sort by importance ----
indices = np.argsort(np.abs(coefs))[::-1]
sorted_features = [features[i] for i in indices]
sorted_coefs = coefs[indices]

# ---- Save to CSV ----
df_importance = pd.DataFrame({
    "feature": sorted_features,
    "coefficient": sorted_coefs,
    "abs_importance": np.abs(sorted_coefs)
})
df_importance.to_csv(OUTPUT_CSV, index=False)
print(f"Feature importance saved to {OUTPUT_CSV}")

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.barh(sorted_features, sorted_coefs, color="skyblue")
plt.xlabel("Coefficient Value")
plt.title("Ridge Regression Feature Importance")
plt.gca().invert_yaxis()
plt.show()
