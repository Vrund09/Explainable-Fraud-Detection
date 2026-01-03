"""
FastAPI application for the Explainable Fraud Detection system.

This module provides REST API endpoints for fraud prediction and explanation
using the trained Graph Neural Network model. It includes comprehensive
error handling, logging, and integration with the AI explanation agent.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
import torch

from .schemas import (
    TransactionInput, BatchTransactionInput, ExplanationRequest,
    PredictionOutput, ExplanationOutput, BatchPredictionOutput,
    ModelStatus, HealthCheck, ErrorResponse, ValidationErrorResponse,
    APIResponse, PredictionConfig
)
from ..gnn_model.predict import FraudPredictor, load_production_model, validate_transaction_input
from ..explainability.agent import AIInvestigator
from ..config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global variables for model and services
fraud_predictor: Optional[FraudPredictor] = None
ai_investigator: Optional[AIInvestigator] = None
app_start_time = time.time()


# ========================================
# STARTUP AND SHUTDOWN HANDLERS
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    
    Handles startup and shutdown operations including model loading
    and resource cleanup.
    """
    # Startup
    logger.info("Starting Fraud Detection API...")
    
    try:
        # Load fraud prediction model
        global fraud_predictor
        logger.info("Loading fraud prediction model...")
        fraud_predictor = load_production_model()
        logger.info("✓ Fraud prediction model loaded successfully")
        
        # Initialize AI investigator (will be implemented in explainability module)
        global ai_investigator
        logger.info("Initializing AI investigator...")
        try:
            from ..explainability.agent import AIInvestigator
            ai_investigator = AIInvestigator()
            logger.info("✓ AI investigator initialized successfully")
        except Exception as e:
            logger.warning(f"AI investigator initialization failed: {str(e)}")
            logger.warning("Explanation endpoints will not be available")
        
        logger.info("🚀 Fraud Detection API startup complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")
    
    # Cleanup resources if needed
    if fraud_predictor:
        fraud_predictor.clear_history()
    
    logger.info("👋 Shutdown complete")


# ========================================
# FASTAPI APPLICATION SETUP
# ========================================

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure appropriately for production
)


# ========================================
# DEPENDENCY FUNCTIONS
# ========================================

def get_fraud_predictor() -> FraudPredictor:
    """Dependency to get the loaded fraud predictor."""
    if fraud_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud prediction model is not loaded"
        )
    return fraud_predictor


def get_ai_investigator() -> AIInvestigator:
    """Dependency to get the AI investigator."""
    if ai_investigator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI investigator is not available"
        )
    return ai_investigator


def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())


# ========================================
# EXCEPTION HANDLERS
# ========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        errors.append({
            "error_code": "VALIDATION_ERROR",
            "error_message": error["msg"],
            "field": " -> ".join([str(x) for x in error["loc"]])
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            message="Input validation failed",
            errors=errors,
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error={
                "error_code": f"HTTP_{exc.status_code}",
                "error_message": exc.detail
            },
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error={
                "error_code": "INTERNAL_ERROR",
                "error_message": "An unexpected error occurred"
            },
            request_id=getattr(request.state, 'request_id', None)
        ).dict()
    )


# ========================================
# MIDDLEWARE FOR REQUEST TRACKING
# ========================================

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add request ID to all requests for tracking."""
    request_id = generate_request_id()
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response


# ========================================
# HEALTH CHECK ENDPOINTS
# ========================================

@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns the current status of the API and its dependencies.
    """
    services = {
        "model": "healthy" if fraud_predictor else "unhealthy",
        "ai_investigator": "healthy" if ai_investigator else "unavailable",
        "mlflow": "unknown"  # Could add actual MLflow connectivity check
    }
    
    return HealthCheck(
        status="healthy" if fraud_predictor else "degraded",
        version=config.API_VERSION,
        services=services,
        uptime_seconds=time.time() - app_start_time
    )


@app.get("/model/status", response_model=ModelStatus, tags=["Model"])
async def get_model_status(predictor: FraudPredictor = Depends(get_fraud_predictor)):
    """
    Get detailed information about the loaded model.
    
    Returns model version, performance metrics, and system information.
    """
    model_info = predictor.get_model_info()
    prediction_stats = predictor.get_prediction_statistics()
    
    return ModelStatus(
        model_name=model_info.get("model_name", "unknown"),
        model_version=model_info.get("model_version"),
        model_stage=model_info.get("model_stage"),
        is_healthy=True,
        last_prediction=None,  # Could track this from prediction history
        model_metrics={
            "total_predictions": prediction_stats.get("total_predictions", 0),
            "fraud_rate": prediction_stats.get("fraud_rate", 0.0),
            "avg_confidence": prediction_stats.get("avg_fraud_probability", 0.0),
        },
        system_info={
            "device": model_info.get("device", "unknown"),
            "total_parameters": model_info.get("model_summary", {}).get("total_parameters", 0)
        }
    )


# ========================================
# PREDICTION ENDPOINTS
# ========================================

@app.post("/predict", response_model=APIResponse, tags=["Prediction"])
async def predict_fraud(
    transaction: TransactionInput,
    config: Optional[PredictionConfig] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    predictor: FraudPredictor = Depends(get_fraud_predictor),
    request: Request = None
) -> APIResponse:
    """
    Predict fraud probability for a single transaction.
    
    This endpoint analyzes a transaction and returns the fraud probability,
    binary prediction, and confidence metrics using the trained GNN model.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        # Validate transaction input
        transaction_dict = transaction.dict()
        if not validate_transaction_input(transaction_dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid transaction data"
            )
        
        # Make prediction
        logger.info(f"Processing fraud prediction for transaction {transaction.transaction_id}")
        
        prediction_result = predictor.predict_fraud(
            transaction_dict,
            return_confidence=config.include_confidence if config else True,
            return_explanation=config.include_explanation_features if config else False
        )
        
        # Create response
        prediction_output = PredictionOutput(
            transaction_id=prediction_result.get('transaction_id'),
            fraud_probability=prediction_result['fraud_probability'],
            is_fraud_predicted=prediction_result['is_fraud_predicted'],
            confidence=prediction_result.get('confidence', 0.0),
            risk_level=prediction_result.get('risk_level', 'LOW'),
            processing_time_ms=(time.time() - start_time) * 1000,
            model_version=predictor.get_model_info().get('model_version')
        )
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction_async, 
            transaction_dict, 
            prediction_result, 
            request_id
        )
        
        logger.info(f"Prediction completed: {prediction_result['fraud_probability']:.4f} for transaction {transaction.transaction_id}")
        
        return APIResponse(
            success=True,
            data=prediction_output,
            message="Fraud prediction completed successfully",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for transaction {transaction.transaction_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=APIResponse, tags=["Prediction"])
async def predict_fraud_batch(
    batch_request: BatchTransactionInput,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    predictor: FraudPredictor = Depends(get_fraud_predictor),
    request: Request = None
) -> APIResponse:
    """
    Predict fraud probabilities for a batch of transactions.
    
    This endpoint processes multiple transactions in a single request,
    providing efficient batch processing for high-volume scenarios.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        logger.info(f"Processing batch prediction for {len(batch_request.transactions)} transactions")
        
        # Convert to list of dictionaries
        transactions_dict = [t.dict() for t in batch_request.transactions]
        
        # Validate all transactions
        for i, transaction in enumerate(transactions_dict):
            if not validate_transaction_input(transaction):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid transaction data at index {i}"
                )
        
        # Make batch prediction
        batch_results = predictor.predict_batch(transactions_dict)
        
        # Convert results to response format
        predictions = []
        successful_predictions = 0
        total_fraud_predictions = 0
        total_confidence = 0.0
        
        for result in batch_results:
            if 'error' not in result:
                prediction = PredictionOutput(
                    transaction_id=result.get('transaction_id'),
                    fraud_probability=result['fraud_probability'],
                    is_fraud_predicted=result['is_fraud_predicted'],
                    confidence=result.get('confidence', 0.0),
                    risk_level=result.get('risk_level', 'LOW')
                )
                predictions.append(prediction)
                
                successful_predictions += 1
                if result['is_fraud_predicted']:
                    total_fraud_predictions += 1
                total_confidence += result.get('confidence', 0.0)
            else:
                # Handle failed predictions
                logger.warning(f"Failed prediction: {result['error']}")
        
        # Calculate batch summary
        processing_time = (time.time() - start_time) * 1000
        batch_summary = {
            "total_transactions": len(batch_request.transactions),
            "successful_predictions": successful_predictions,
            "fraud_predicted": total_fraud_predictions,
            "fraud_rate": total_fraud_predictions / max(successful_predictions, 1),
            "average_confidence": total_confidence / max(successful_predictions, 1),
            "processing_time_ms": processing_time
        }
        
        batch_output = BatchPredictionOutput(
            batch_id=batch_request.batch_id,
            predictions=predictions,
            batch_summary=batch_summary,
            processing_time_ms=processing_time
        )
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction_async,
            batch_request.batch_id,
            batch_summary,
            request_id
        )
        
        logger.info(f"Batch prediction completed: {successful_predictions}/{len(batch_request.transactions)} successful")
        
        return APIResponse(
            success=True,
            data=batch_output,
            message=f"Batch prediction completed: {successful_predictions}/{len(batch_request.transactions)} successful",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ========================================
# EXPLANATION ENDPOINTS
# ========================================

@app.post("/explain", response_model=APIResponse, tags=["Explanation"])
async def explain_transaction(
    explanation_request: ExplanationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    investigator: AIInvestigator = Depends(get_ai_investigator),
    request: Request = None
) -> APIResponse:
    """
    Generate AI-powered explanation for a fraud prediction.
    
    This endpoint uses the AI investigator to analyze transaction context
    and provide human-readable explanations for fraud decisions.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        transaction_id = explanation_request.transaction_id
        logger.info(f"Generating explanation for transaction {transaction_id}")
        
        # Generate explanation using AI investigator
        explanation_result = await investigator.explain_transaction(
            transaction_id,
            depth=explanation_request.explanation_depth,
            context=explanation_request.prediction_context
        )
        
        # Create response
        explanation_output = ExplanationOutput(
            transaction_id=transaction_id,
            explanation_text=explanation_result.get('explanation_text', ''),
            key_factors=explanation_result.get('key_factors', []),
            risk_indicators=explanation_result.get('risk_indicators', {}),
            recommendation=explanation_result.get('recommendation', ''),
            explanation_confidence=explanation_result.get('confidence', 0.0)
        )
        
        # Log explanation generation (background task)
        background_tasks.add_task(
            log_explanation_async,
            transaction_id,
            explanation_request.explanation_depth,
            request_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Explanation generated for transaction {transaction_id} in {processing_time:.2f}ms")
        
        return APIResponse(
            success=True,
            data=explanation_output,
            message="Explanation generated successfully",
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Explanation generation failed for transaction {explanation_request.transaction_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(e)}"
        )


# ========================================
# UTILITY ENDPOINTS
# ========================================

@app.get("/predictions/history", response_model=APIResponse, tags=["Utility"])
async def get_prediction_history(
    limit: int = 100,
    predictor: FraudPredictor = Depends(get_fraud_predictor)
) -> APIResponse:
    """Get recent prediction history and statistics."""
    try:
        stats = predictor.get_prediction_statistics()
        
        # Get recent predictions (limited)
        recent_predictions = predictor.prediction_history[-limit:] if predictor.prediction_history else []
        
        return APIResponse(
            success=True,
            data={
                "statistics": stats,
                "recent_predictions": recent_predictions,
                "total_returned": len(recent_predictions)
            },
            message=f"Retrieved {len(recent_predictions)} recent predictions"
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction history"
        )


@app.delete("/predictions/history", response_model=APIResponse, tags=["Utility"])
async def clear_prediction_history(
    predictor: FraudPredictor = Depends(get_fraud_predictor)
) -> APIResponse:
    """Clear the prediction history."""
    try:
        count_before = len(predictor.prediction_history)
        predictor.clear_history()
        
        return APIResponse(
            success=True,
            data={"cleared_predictions": count_before},
            message=f"Cleared {count_before} predictions from history"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear prediction history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear prediction history"
        )


# ========================================
# BACKGROUND TASKS
# ========================================

async def log_prediction_async(
    transaction_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    request_id: str
) -> None:
    """Log prediction details asynchronously."""
    try:
        logger.info(f"[{request_id}] Logged prediction for transaction {transaction_data.get('transaction_id')}")
        # Here you could add database logging, metrics collection, etc.
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")


async def log_batch_prediction_async(
    batch_id: Optional[str],
    batch_summary: Dict[str, Any],
    request_id: str
) -> None:
    """Log batch prediction details asynchronously."""
    try:
        logger.info(f"[{request_id}] Logged batch prediction {batch_id}: {batch_summary}")
        # Here you could add database logging, metrics collection, etc.
    except Exception as e:
        logger.error(f"Failed to log batch prediction: {str(e)}")


async def log_explanation_async(
    transaction_id: str,
    explanation_depth: str,
    request_id: str
) -> None:
    """Log explanation generation details asynchronously."""
    try:
        logger.info(f"[{request_id}] Logged explanation generation for transaction {transaction_id} (depth: {explanation_depth})")
        # Here you could add database logging, metrics collection, etc.
    except Exception as e:
        logger.error(f"Failed to log explanation: {str(e)}")


# ========================================
# APPLICATION ENTRY POINT
# ========================================

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.API_DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )


