import os, sys, joblib, pandas as pd
from typing import List

# Make sure we can import your Colab modules
sys.path.append('/content')

from windai_preprocess import build_features_infer, load_feature_columns, clean_and_parse
from windai_recommend import recommend_assemblies_from_predictions

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
