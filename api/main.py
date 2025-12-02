from fastapi import FastAPI
from api.routes import predict

# Initialize the main FastAPI application instance
app = FastAPI(
    title="HealthGuard AI Fraud Detection API",
    description="Real-time XGBoost model serving with SHAP explainability for fraud detection.",
    version="1.0.0"
)

# Include the prediction router
app.include_router(predict.router, prefix="/v1")

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "HealthGuard-AI Fraud Detector"}
