from pathlib import Path
from fastapi.responses import FileResponse
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.predictor import get_model_info, predict_patient
from app.history import (
    delete_patient_history,
    get_patient_history,
    save_prediction,
)

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="FedFairChain API",
    description=(
        "Fairness-aware federated multi-label "
        "comorbidity prediction research prototype."
    ),
    version="1.1.0",
)


DiagnosisStatus = Literal[
    "yes",
    "no",
    "unknown",
]


class PatientRequest(BaseModel):

    AGE: float = Field(
        ...,
        ge=20,
        le=80,
    )

    BMI: Optional[float] = Field(
        None,
        ge=14,
        le=93,
    )

    WAIST_CM: Optional[float] = Field(
        None,
        ge=55,
        le=180,
    )

    SBP_MEAN: Optional[float] = Field(
        None,
        ge=70,
        le=220,
    )

    DBP_MEAN: Optional[float] = Field(
        None,
        ge=35,
        le=145,
    )

    HBA1C: Optional[float] = Field(
        None,
        ge=2.5,
        le=17,
    )

    TOTAL_CHOLESTEROL: Optional[float] = Field(
        None,
        ge=60,
        le=450,
    )

    HDL_CHOLESTEROL: Optional[float] = Field(
        None,
        ge=5,
        le=190,
    )

    CREATININE: Optional[float] = Field(
        None,
        ge=0.2,
        le=15,
    )

    BUN: Optional[float] = Field(
        None,
        ge=1,
        le=80,
    )

    ALBUMIN: Optional[float] = Field(
        None,
        ge=2,
        le=5.5,
    )

    URIC_ACID: Optional[float] = Field(
        None,
        ge=0.5,
        le=16,
    )

    SERUM_GLUCOSE: Optional[float] = Field(
        None,
        ge=35,
        le=650,
    )

    SMOKING_STATUS: Optional[int] = Field(
        None,
        ge=0,
        le=2,
    )

    PHYSICAL_ACTIVITY_ANY: Optional[int] = Field(
        None,
        ge=0,
        le=1,
    )

    known_diabetes: DiagnosisStatus = "unknown"
    known_cvd: DiagnosisStatus = "unknown"
    known_hypertension: DiagnosisStatus = "unknown"

    # Optional local history. Use an anonymous code only; never a name,
    # phone number, Aadhaar number, or other direct identifier.
    patient_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    save_history: bool = False


@app.get("/")
def root():

    return FileResponse(
        BASE_DIR / "index.html"
    )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.get("/model-info")
def model_info():
    """Return safe, non-patient model metadata for transparency."""
    return get_model_info()


@app.get("/history/{patient_id}")
def history(patient_id: str):
    """Return locally saved prediction visits for one anonymous ID."""
    try:
        records = get_patient_history(patient_id)
        return {
            "patient_id": patient_id,
            "count": len(records),
            "records": records,
            "note": (
                "Saved history is local to this application and is not used "
                "for model training or automatic disease-progression prediction."
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/history/{patient_id}")
def clear_history(patient_id: str):
    """Delete all locally saved visits for one anonymous ID."""
    try:
        deleted = delete_patient_history(patient_id)
        return {"patient_id": patient_id, "deleted_records": deleted}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PatientRequest):

    try:

        patient = {
            "AGE": request.AGE,
            "BMI": request.BMI,
            "WAIST_CM": request.WAIST_CM,
            "SBP_MEAN": request.SBP_MEAN,
            "DBP_MEAN": request.DBP_MEAN,
            "HBA1C": request.HBA1C,
            "TOTAL_CHOLESTEROL": request.TOTAL_CHOLESTEROL,
            "HDL_CHOLESTEROL": request.HDL_CHOLESTEROL,
            "CREATININE": request.CREATININE,
            "BUN": request.BUN,
            "ALBUMIN": request.ALBUMIN,
            "URIC_ACID": request.URIC_ACID,
            "SERUM_GLUCOSE": request.SERUM_GLUCOSE,
            "SMOKING_STATUS": request.SMOKING_STATUS,
            "PHYSICAL_ACTIVITY_ANY": (
                request.PHYSICAL_ACTIVITY_ANY
            ),
        }

        known_diagnoses = {
            "DIABETES": request.known_diabetes,
            "CVD": request.known_cvd,
            "HYPERTENSION": (
                request.known_hypertension
            ),
        }

        prediction = predict_patient(
            patient=patient,
            known_diagnoses=known_diagnoses,
            top_n_explanations=5,
        )

        prediction["history_saved"] = False
        prediction["history_record_id"] = None

        if request.save_history:
            if not request.patient_id:
                raise ValueError(
                    "Enter an Anonymous Patient ID before saving prediction history."
                )
            record_id = save_prediction(
                patient_id=request.patient_id,
                patient=patient,
                known_diagnoses=known_diagnoses,
                prediction=prediction,
            )
            prediction["history_saved"] = True
            prediction["history_record_id"] = record_id

        return prediction

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc