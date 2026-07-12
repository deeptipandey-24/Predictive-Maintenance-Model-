
import sys
sys.path.append('/content/drive/MyDrive/predictive_maintenance_project/src')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from predict import load_production_model, predict_machine_failure_risk
from logging_utils import setup_logger

MODELS_DIR = '/content/drive/MyDrive/predictive_maintenance_project/models'
LOG_PATH = '/content/drive/MyDrive/predictive_maintenance_project/logs/prediction_logs.jsonl'

app = FastAPI(title="Predictive Maintenance API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, config = load_production_model(MODELS_DIR)
logger = setup_logger(LOG_PATH)

class MachineInput(BaseModel):
    Type: str = Field(..., description="Machine type category, e.g. 'L', 'M', 'H'")
    Air_temperature_K: float = Field(..., alias="Air temperature [K]")
    Process_temperature_K: float = Field(..., alias="Process temperature [K]")
    Rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]")
    Torque_Nm: float = Field(..., alias="Torque [Nm]")
    Tool_wear_min: float = Field(..., alias="Tool wear [min]")
    machine_id: str | None = None

    class Config:
        populate_by_name = True

@app.get("/")
def root():
    return {"status": "ok", "message": "Predictive Maintenance API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "threshold": config['threshold']}

@app.post("/predict")
def predict(payload: MachineInput):
    raw_input = payload.dict(by_alias=True, exclude={'machine_id'})
    try:
        result = predict_machine_failure_risk(
            raw_input, model, config,
            logger=logger, machine_id=payload.machine_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
