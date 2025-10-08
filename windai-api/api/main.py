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

# === Inverse design: suggest assemblies for target metrics ===
import json, random
from typing import Optional
from pydantic import BaseModel
import pandas as pd
from .predict_helper import predict_rows  # make sure this import is already at the top

# load option space
with open("model/option_space.json", "r") as f:
    _OPT = json.load(f)
_OPTIONS = _OPT["options"]
_DEFAULTS = _OPT["defaults"]

class DesignRequest(BaseModel):
    target_u: float = 1.2
    target_shgc: float = 0.45
    max_cost: float = 500.0
    tolerance: float = 0.10
    top_n: int = 5
    samples: int = 3000
    seed: Optional[int] = 42

def _sample_candidate() -> dict:
    pick = lambda k: random.choice(_OPTIONS[k]) if _OPTIONS.get(k) else None
    row = {
        "Glazing Name": pick("Glazing Name"),
        "Gas Fill Name": pick("Gas Fill Name"),
        "Spacer Name": pick("Spacer Name"),
        "Sealant Name": pick("Sealant Name"),
        "Frame Name": pick("Frame Name"),
        "Thermal Break Name": pick("Thermal Break Name"),
        "Glazing Thickness (mm)": _DEFAULTS["Glazing Thickness (mm)"],
        "Gas λ (W/m·K)": _DEFAULTS["Gas λ (W/m·K)"],
        "Spacer λ (W/m·K)": _DEFAULTS["Spacer λ (W/m·K)"],
        "Spacer Width (mm)": _DEFAULTS["Spacer Width (mm)"],
        "Sealant λ (W/m·K)": _DEFAULTS["Sealant λ (W/m·K)"],
        "Frame λ (W/m·K)": _DEFAULTS["Frame λ (W/m·K)"],
        "Frame Thickness (mm)": _DEFAULTS["Frame Thickness (mm)"],
        "Thermal Break λ (W/m·K)": _DEFAULTS["Thermal Break λ (W/m·K)"],
    }
    return row

@app.post("/design/json")
def design_json(req: DesignRequest):
    if req.seed is not None:
        random.seed(req.seed)

    rows = [_sample_candidate() for _ in range(req.samples)]
    df = pd.DataFrame(rows)
    y = predict_rows(df)

    lo = req.target_shgc * (1 - req.tolerance)
    hi = req.target_shgc * (1 + req.tolerance)

    results = []
    for i, r in enumerate(rows):
        u, shgc, cost = float(y[i,0]), float(y[i,1]), float(y[i,2])
        feasible = (u <= req.target_u) and (cost <= req.max_cost) and (lo <= shgc <= hi)
        shgc_dev = max(0.0, abs(shgc - req.target_shgc) - req.tolerance*req.target_shgc)
        score = (
            (u / max(req.target_u, 1e-6)) +
            (cost / max(req.max_cost, 1e-6)) +
            (shgc_dev * 10.0)
        )
        if not feasible:
            score += 100.0

        results.append({
            **r,
            "Predicted U-Factor": u,
            "Predicted SHGC": shgc,
            "Predicted Cost": cost,
            "Score": score,
            "Feasible": feasible
        })

    results_sorted = sorted(results, key=lambda x: x["Score"])
    top = [x for x in results_sorted if x["Feasible"]][:req.top_n]
    if len(top) < req.top_n:
        extras = [x for x in results_sorted if not x["Feasible"]][: (req.top_n - len(top))]
        top.extend(extras)

    keep = [
        "Glazing Name","Gas Fill Name","Spacer Name","Sealant Name",
        "Frame Name","Thermal Break Name",
        "Glazing Thickness (mm)","Spacer Width (mm)",
        "Predicted U-Factor","Predicted SHGC","Predicted Cost","Feasible","Score"
    ]
    return {"recommendations": [{k:v for k,v in x.items() if k in keep} for x in top]}


