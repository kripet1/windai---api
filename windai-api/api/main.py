from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# one clean import from your helper
from .predict_helper import (
    recommend,
    derive_schema,
    feature_columns,
    predict_rows,
)

app = FastAPI(title="WindAI API", version="0.1.0")

# CORS (MVP: allow all; later restrict to your Lovable domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema")
def schema():
    return derive_schema(feature_columns)

# ----- payload models -----
class PredictPayload(BaseModel):
    rows: list  # list[dict]

# ----- prediction-only endpoint -----
@app.post("/predict/json")
def predict_json(payload: PredictPayload):
    df = pd.DataFrame(payload.rows)
    y = predict_rows(df)
    out = []
    for i in range(len(df)):
        out.append({
            "Predicted U-Factor": float(y[i, 0]),
            "Predicted SHGC": float(y[i, 1]),
            "Predicted Cost": float(y[i, 2]),
        })
    return {"predictions": out}

# ----- recommend from JSON rows -----
@app.post("/recommend/json")
def recommend_json(
    payload: PredictPayload,
    max_cost: float = Query(400.0),
    max_u: float = Query(1.2),
    target_shgc: float = Query(0.45),
    tolerance: float = Query(0.05),
    top_n: int = Query(5),
):
    df = pd.DataFrame(payload.rows)
    top = recommend(
        df_raw=df,
        max_cost=max_cost,
        max_u=max_u,
        target_shgc=target_shgc,
        tolerance=tolerance,
        top_n=top_n
    )
    return {"results": top.to_dict(orient="records")}

# ----- recommend from uploaded CSV/XLSX (for your own testing) -----
@app.post("/recommend/csv")
async def recommend_csv(
    file: UploadFile = File(...),
    max_cost: float = Form(400.0),
    max_u: float = Form(1.2),
    target_shgc: float = Form(0.45),
    tolerance: float = Form(0.05),
    top_n: int = Form(5),
):
    if file.filename.lower().endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        # needs 'openpyxl' in requirements.txt
        df = pd.read_excel(file.file)

    top = recommend(
        df_raw=df,
        max_cost=max_cost,
        max_u=max_u,
        target_shgc=target_shgc,
        tolerance=tolerance,
        top_n=top_n
    )
    return {"results": top.to_dict(orient="records")}


