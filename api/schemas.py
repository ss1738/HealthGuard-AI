from pydantic import BaseModel, Field
from typing import List

# Defines the features required by the XGBoost model
class TransactionInput(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; 
    V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; 
    V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float; V25: float; 
    V26: float; V27: float; V28: float
    Amount_Scaled: float = Field(..., description="Transaction amount, standardized.")
    
# Defines the output structure (Prediction + SHAP Explanation)
class PredictionOutput(BaseModel):
    is_fraud: bool = Field(..., description="True if transaction is predicted as fraudulent.")
    confidence_score: float = Field(..., description="Probability of being a fraudulent transaction (0 to 1).")
    explanation_top_3_features: List[str] = Field(..., description="Top 3 features driving the model's decision.")
