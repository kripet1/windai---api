from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from .predict_helper import recommend

app = FastAPI(title="WindAI API", version="0.1.0")

# For MVP, allow all origins (tighten later to your Lovable domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendParams(BaseModel):
    max_cost: float = 400.0
    max_u: float = 1.2
    target_shgc: float = 0.45
    tolerance: float = 0.05
    top_n: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend/json")
def recommend_json(payload: dict, params: RecommendParams = RecommendParams()):
    # Expected payload: {"rows": [{...}, {...}]}
    rows = payload.get("rows", [])
    df = pd.DataFrame(rows)
    top = recommend(
        df_raw=df,
        max_cost=params.max_cost,
        max_u=params.max_u,
        target_shgc=params.target_shgc,
        tolerance=params.tolerance,
        top_n=params.top_n
    )
    return {"results": top.to_dict(orient="records")}

@app.post("/recommend/csv")
async def recommend_csv(file: UploadFile = File(...),
                        max_cost: float = Form(400.0),
                        max_u: float = Form(1.2),
                        target_shgc: float = Form(0.45),
                        tolerance: float = Form(0.05),
                        top_n: int = Form(5)):
    # Accept CSV or Excel uploads
    if file.filename.lower().endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
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
