# api/predict_helper.py
import os, joblib, pandas as pd
from typing import List

# ✅ import the modules you just added to this repo
from .preprocess import build_features_infer, load_feature_columns, clean_and_parse
from .recommend import recommend_assemblies_from_predictions

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))

# Load once at import time
model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
feature_columns = load_feature_columns(os.path.join(MODEL_DIR, 'feature_columns.json'))

def predict_rows(df_raw: pd.DataFrame):
    X = build_features_infer(df_raw, feature_columns)
    return model.predict(X)

def recommend(df_raw: pd.DataFrame,
              max_cost: float,
              max_u: float,
              target_shgc: float,
              tolerance: float,
              top_n: int) -> pd.DataFrame:
    y_pred = predict_rows(df_raw)
    df_clean = clean_and_parse(df_raw)
    return recommend_assemblies_from_predictions(
        df_original=df_clean,
        y_pred=y_pred,
        max_cost=max_cost,
        max_u=max_u,
        target_shgc=target_shgc,
        tolerance=tolerance,
        top_n=top_n
    )
def derive_schema(feature_columns):
    """
    Build a simple schema for the UI:
    - categorical options (extracted from one-hot columns)
    - numeric fields the API accepts
    """
    numeric_required = [
        "Glazing λ (W/m·K)",
        "Glazing Thickness (mm)",
        "Gas λ (W/m·K)",
        "Spacer λ (W/m·K)",
        "Spacer Width (mm)",
        "Sealant λ (W/m·K)",
        "Frame λ (W/m·K)",
        "Frame Thickness (mm)",
        "Thermal Break λ (W/m·K)",
    ]

    cats = ["Glazing Name", "Gas Fill Name", "Spacer Name",
            "Sealant Name", "Frame Name", "Thermal Break Name"]
    opts = {c: [] for c in cats}

    for col in feature_columns:
        for c in cats:
            prefix = f"{c}_"
            if col.startswith(prefix):
                opts[c].append(col[len(prefix):])

    # Make sure “None” appears for spacer/thermal break
    for c in ["Spacer Name", "Thermal Break Name"]:
        if "None" not in opts[c]:
            opts[c].append("None")

    for c in cats:
        opts[c] = sorted(set(opts[c]))

    return {
        "categorical": opts,
        "numeric_required": numeric_required,
        "notes": "Send one or more rows to /recommend/json or /predict/json. "
                 "Omit spacer/thermal-break numeric fields when Name is 'None'."
    }


