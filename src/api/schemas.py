"""
Pydantic schemas for the Fraud Detection API.

This module defines request and response models for all API endpoints
using Pydantic for data validation, serialization, and automatic
OpenAPI documentation generation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import constr, confloat, conint

from ..config import config


class TransactionType(str, Enum):
    """Enumeration of valid transaction types in the PaySim dataset."""
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class RiskLevel(str, Enum):
    """Risk level categories for fraud predictions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ========================================
# REQUEST SCHEMAS
# ========================================

class TransactionInput(BaseModel):
    """
    Input schema for a single transaction fraud prediction.
    
    This schema defines the required and optional fields for submitting
    a transaction for fraud analysis.
    """
    
    # Core transaction identifiers
    transaction_id: Optional[str] = Field(
        None,
        description="Unique identifier for the transaction",
        example="TXN_123456789"
    )
    
    sender_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Unique identifier of the transaction sender",
        example="C1234567890"
    )
    
    receiver_id: constr(min_length=1, max_length=50) = Field(
        ...,
        description="Unique identifier of the transaction receiver", 
        example="M987654321"
    )
    
    # Transaction details
    amount: confloat(gt=0, le=config.MAX_AMOUNT_THRESHOLD) = Field(
        ...,
        description="Transaction amount in the base currency",
        example=150000.50
    )
    
    type: TransactionType = Field(
        ...,
        description="Type of transaction",
        example=TransactionType.TRANSFER
    )
    
    # Optional timing information
    step: Optional[conint(ge=0)] = Field(
        None,
        description="Time step of the transaction (hours from start of simulation)",
        example=1440
    )
    
    # Optional balance information
    sender_old_balance: Optional[confloat(ge=0)] = Field(
        None,
        description="Sender's balance before the transaction",
        example=50000.0
    )
    
    sender_new_balance: Optional[confloat(ge=0)] = Field(
        None,
        description="Sender's balance after the transaction",
        example=25000.0
    )
    
    receiver_old_balance: Optional[confloat(ge=0)] = Field(
        None,
        description="Receiver's balance before the transaction",
        example=100000.0
    )
    
    receiver_new_balance: Optional[confloat(ge=0)] = Field(
        None,
        description="Receiver's balance after the transaction",
        example=125000.0
    )
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the transaction",
        example={"channel": "mobile", "location": "NYC"}
    )
    
    @validator('sender_id', 'receiver_id')
    def validate_user_ids(cls, v):
        """Validate user ID format."""
        if not v or len(v.strip()) == 0:
            raise ValueError("User ID cannot be empty")
        return v.strip()
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > config.MAX_AMOUNT_THRESHOLD:
            raise ValueError(f"Amount exceeds maximum threshold of {config.MAX_AMOUNT_THRESHOLD}")
        return v
    
    @root_validator
    def validate_balance_consistency(cls, values):
        """Validate that balance changes are consistent with transaction amount."""
        sender_old = values.get('sender_old_balance')
        sender_new = values.get('sender_new_balance')
        receiver_old = values.get('receiver_old_balance')
        receiver_new = values.get('receiver_new_balance')
        amount = values.get('amount')
        
        if all(x is not None for x in [sender_old, sender_new, amount]):
            expected_sender_new = sender_old - amount
            if abs(sender_new - expected_sender_new) > 0.01:  # Allow for small floating point errors
                raise ValueError("Sender balance change inconsistent with transaction amount")
        
        if all(x is not None for x in [receiver_old, receiver_new, amount]):
            expected_receiver_new = receiver_old + amount
            if abs(receiver_new - expected_receiver_new) > 0.01:
                raise ValueError("Receiver balance change inconsistent with transaction amount")
        
        return values


class BatchTransactionInput(BaseModel):
    """Schema for batch transaction prediction requests."""
    
    transactions: List[TransactionInput] = Field(
        ...,
        description="List of transactions to analyze",
        min_items=1,
        max_items=100  # Limit batch size for performance
    )
    
    batch_id: Optional[str] = Field(
        None,
        description="Unique identifier for this batch",
        example="BATCH_20231201_001"
    )
    
    include_explanations: bool = Field(
        False,
        description="Whether to include AI-generated explanations for predictions"
    )


class ExplanationRequest(BaseModel):
    """Schema for requesting explanations for existing predictions."""
    
    transaction_id: str = Field(
        ...,
        description="ID of the transaction to explain",
        example="TXN_123456789"
    )
    
    prediction_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the prediction to help with explanation"
    )
    
    explanation_depth: Optional[str] = Field(
        "standard",
        description="Depth of explanation ('basic', 'standard', 'detailed')",
        regex="^(basic|standard|detailed)$"
    )


# ========================================
# RESPONSE SCHEMAS
# ========================================

class PredictionOutput(BaseModel):
    """
    Output schema for fraud prediction results.
    
    Contains the fraud probability, binary prediction, and confidence metrics.
    """
    
    transaction_id: Optional[str] = Field(
        None,
        description="Transaction ID that was analyzed"
    )
    
    fraud_probability: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Probability that the transaction is fraudulent (0-1)",
        example=0.7523
    )
    
    is_fraud_predicted: bool = Field(
        ...,
        description="Binary prediction: True if fraud is predicted",
        example=True
    )
    
    confidence: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Confidence in the prediction (0-1, higher is more confident)",
        example=0.8945
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Risk level category",
        example=RiskLevel.HIGH
    )
    
    # Timing information
    prediction_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the prediction was made"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process the prediction in milliseconds",
        example=145.7
    )
    
    # Optional model information
    model_version: Optional[str] = Field(
        None,
        description="Version of the model used for prediction",
        example="v1.2.3"
    )


class ExplanationOutput(BaseModel):
    """Schema for AI-generated fraud explanations."""
    
    transaction_id: str = Field(
        ...,
        description="Transaction ID that was explained"
    )
    
    explanation_text: str = Field(
        ...,
        description="Human-readable explanation of the fraud prediction",
        example="This transaction shows high risk due to unusual amount patterns and sender's history..."
    )
    
    key_factors: List[str] = Field(
        ...,
        description="List of key factors that influenced the prediction",
        example=["High transaction amount", "Unusual time pattern", "Sender risk profile"]
    )
    
    risk_indicators: Dict[str, Union[str, float]] = Field(
        ...,
        description="Specific risk indicators and their values",
        example={
            "amount_percentile": 95.2,
            "sender_fraud_history": "elevated", 
            "network_centrality": 0.78
        }
    )
    
    recommendation: str = Field(
        ...,
        description="Recommended action based on the analysis",
        example="Manual review recommended due to high risk indicators"
    )
    
    explanation_confidence: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Confidence in the explanation quality",
        example=0.89
    )
    
    generated_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the explanation was generated"
    )


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction results."""
    
    batch_id: Optional[str] = Field(None, description="ID of the processed batch")
    
    predictions: List[PredictionOutput] = Field(
        ...,
        description="Individual predictions for each transaction"
    )
    
    batch_summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics for the batch",
        example={
            "total_transactions": 50,
            "fraud_predicted": 3,
            "fraud_rate": 0.06,
            "average_confidence": 0.84,
            "high_risk_count": 5
        }
    )
    
    processing_time_ms: float = Field(
        ...,
        description="Total time to process the batch in milliseconds"
    )


class ModelStatus(BaseModel):
    """Schema for model status and health information."""
    
    model_name: str = Field(..., description="Name of the loaded model")
    model_version: Optional[str] = Field(None, description="Version of the model")
    model_stage: Optional[str] = Field(None, description="Stage of the model (Production, Staging, etc.)")
    
    is_healthy: bool = Field(..., description="Whether the model is healthy and ready")
    last_prediction: Optional[datetime] = Field(None, description="Timestamp of last prediction")
    
    model_metrics: Dict[str, float] = Field(
        ...,
        description="Key performance metrics of the model",
        example={
            "accuracy": 0.94,
            "precision": 0.89,
            "recall": 0.92,
            "f1_score": 0.90,
            "roc_auc": 0.96
        }
    )
    
    system_info: Dict[str, Any] = Field(
        ...,
        description="System information",
        example={
            "device": "cuda",
            "memory_usage_mb": 512.5,
            "total_parameters": 125000
        }
    )


class HealthCheck(BaseModel):
    """Schema for API health check response."""
    
    status: str = Field(..., example="healthy")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field(..., example="1.0.0")
    
    services: Dict[str, str] = Field(
        ...,
        description="Status of dependent services",
        example={
            "model": "healthy",
            "database": "healthy", 
            "mlflow": "healthy"
        }
    )
    
    uptime_seconds: float = Field(..., description="API uptime in seconds")


# ========================================
# ERROR SCHEMAS
# ========================================

class ErrorDetail(BaseModel):
    """Schema for detailed error information."""
    
    error_code: str = Field(..., description="Specific error code")
    error_message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error (if applicable)")
    
    
class ErrorResponse(BaseModel):
    """Schema for API error responses."""
    
    success: bool = Field(False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Unique request identifier for tracking")


class ValidationErrorResponse(BaseModel):
    """Schema for validation error responses."""
    
    success: bool = Field(False)
    message: str = Field(..., example="Validation failed")
    errors: List[ErrorDetail] = Field(..., description="List of validation errors")
    timestamp: datetime = Field(default_factory=datetime.now)


# ========================================
# UTILITY SCHEMAS
# ========================================

class APIResponse(BaseModel):
    """Generic wrapper for successful API responses."""
    
    success: bool = Field(True, description="Indicates successful operation")
    data: Any = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Optional success message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    
    page: conint(ge=1) = Field(1, description="Page number (1-based)")
    page_size: conint(ge=1, le=100) = Field(20, description="Number of items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", regex="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Schema for paginated response data."""
    
    items: List[Any] = Field(..., description="List of items for current page")
    total_items: conint(ge=0) = Field(..., description="Total number of items")
    total_pages: conint(ge=0) = Field(..., description="Total number of pages")
    current_page: conint(ge=1) = Field(..., description="Current page number")
    page_size: conint(ge=1) = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


# ========================================
# CONFIGURATION SCHEMAS
# ========================================

class PredictionConfig(BaseModel):
    """Configuration options for predictions."""
    
    threshold: confloat(ge=0.0, le=1.0) = Field(
        0.5,
        description="Decision threshold for binary classification"
    )
    
    include_confidence: bool = Field(
        True,
        description="Whether to include confidence metrics"
    )
    
    include_explanation_features: bool = Field(
        False,
        description="Whether to include features for explainability"
    )
    
    use_subgraph: bool = Field(
        True,
        description="Whether to use subgraph-based prediction when available"
    )
    
    subgraph_hops: conint(ge=1, le=5) = Field(
        2,
        description="Number of hops for subgraph extraction"
    )


