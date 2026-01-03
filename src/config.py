"""
Configuration module for the Explainable Fraud Detection System.

This module manages all constants, paths, and environment-based configurations
used throughout the application. It follows the principle of centralized
configuration management for better maintainability and reproducibility.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class containing all system constants and settings.
    
    This class organizes configuration into logical sections:
    - Data paths and directories
    - Model parameters and paths
    - Database connections
    - API configurations
    - MLOps settings
    """
    
    # ========================================
    # PROJECT PATHS
    # ========================================
    
    # Root project directory
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # PaySim dataset paths
    PAYSIM_RAW_PATH: Path = RAW_DATA_DIR / "paysim.csv"
    GRAPH_NODES_PATH: Path = PROCESSED_DATA_DIR / "graph_nodes.csv"
    GRAPH_EDGES_PATH: Path = PROCESSED_DATA_DIR / "graph_edges.csv"
    
    # Model directories
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
    
    # Notebook directory
    NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
    
    # ========================================
    # NEO4J DATABASE CONFIGURATION
    # ========================================
    
    # Neo4j connection settings (from environment variables)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Connection pool settings
    NEO4J_MAX_CONNECTION_LIFETIME: int = 3600  # 1 hour
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = 50
    
    # ========================================
    # LLM & AI AGENT CONFIGURATION
    # ========================================
    
    # Gemini API settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-1.5-pro-latest"
    
    # OpenAI API settings (alternative LLM)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME: str = "gpt-4"
    
    # Agent configuration
    MAX_EXPLANATION_LENGTH: int = 1000
    CONTEXT_WINDOW_SIZE: int = 10  # Number of recent transactions to consider
    
    # ========================================
    # GRAPH NEURAL NETWORK CONFIGURATION
    # ========================================
    
    # Model architecture parameters
    GNN_INPUT_DIM: int = 10  # Node feature dimension
    GNN_HIDDEN_DIM: int = 128
    GNN_OUTPUT_DIM: int = 64
    GNN_NUM_LAYERS: int = 2
    GNN_DROPOUT_RATE: float = 0.2
    
    # Classification head parameters
    CLASSIFIER_HIDDEN_DIM: int = 32
    CLASSIFIER_OUTPUT_DIM: int = 1  # Binary classification
    
    # ========================================
    # TRAINING CONFIGURATION
    # ========================================
    
    # Training hyperparameters
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 512
    NUM_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Data splitting
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    
    # Model checkpointing
    SAVE_EVERY_N_EPOCHS: int = 10
    KEEP_N_CHECKPOINTS: int = 5
    
    # ========================================
    # MLFLOW CONFIGURATION
    # ========================================
    
    # MLflow tracking
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME: str = "fraud-detection-gnn"
    MLFLOW_ARTIFACT_ROOT: Optional[str] = os.getenv("MLFLOW_ARTIFACT_ROOT")
    
    # Model registry
    MLFLOW_MODEL_NAME: str = "fraud-detection-model"
    MLFLOW_MODEL_STAGE: str = "Production"
    
    # ========================================
    # API CONFIGURATION
    # ========================================
    
    # FastAPI settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    API_DEBUG: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # API metadata
    API_TITLE: str = "Explainable Fraud Detection API"
    API_DESCRIPTION: str = "Graph Neural Network-based fraud detection with AI-powered explanations"
    API_VERSION: str = "1.0.0"
    
    # Rate limiting
    API_RATE_LIMIT: int = 100  # requests per minute
    
    # ========================================
    # DATA PROCESSING CONFIGURATION
    # ========================================
    
    # Transaction types in PaySim dataset
    TRANSACTION_TYPES: list[str] = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    
    # Feature engineering
    MIN_TRANSACTION_COUNT: int = 5  # Minimum transactions for a user to be included
    MAX_AMOUNT_THRESHOLD: float = 1e6  # Maximum transaction amount to consider
    
    # Graph construction
    MAX_GRAPH_SIZE: int = 1000000  # Maximum number of nodes in the graph
    EDGE_WEIGHT_THRESHOLD: float = 0.1  # Minimum edge weight to include
    
    # ========================================
    # LOGGING CONFIGURATION
    # ========================================
    
    # Logging levels
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log files
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    API_LOG_FILE: str = "api.log"
    TRAINING_LOG_FILE: str = "training.log"
    
    # ========================================
    # SECURITY CONFIGURATION
    # ========================================
    
    # API security
    API_KEY_HEADER: str = "X-API-Key"
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "dev-secret-key-change-in-production")
    
    # Database security
    ENCRYPT_DATABASE_CREDENTIALS: bool = True
    
    # ========================================
    # DEVELOPMENT & DEBUGGING
    # ========================================
    
    # Development flags
    IS_DEVELOPMENT: bool = os.getenv("ENVIRONMENT", "development") == "development"
    ENABLE_PROFILING: bool = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    
    # Debugging
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    VERBOSE_LOGGING: bool = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
    
    @classmethod
    def create_directories(cls) -> None:
        """
        Create all necessary directories if they don't exist.
        
        This method ensures that all required directories are present
        before the application starts processing data or training models.
        """
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate the configuration settings.
        
        Returns:
            bool: True if configuration is valid, False otherwise.
            
        Raises:
            ValueError: If critical configuration values are missing or invalid.
        """
        # Check if critical environment variables are set
        if not cls.NEO4J_PASSWORD or cls.NEO4J_PASSWORD == "password":
            if not cls.IS_DEVELOPMENT:
                raise ValueError("NEO4J_PASSWORD must be set in production environment")
        
        if not cls.GEMINI_API_KEY and not cls.IS_DEVELOPMENT:
            raise ValueError("GEMINI_API_KEY must be set for AI explanations to work")
        
        # Validate data ratios
        if abs((cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO) - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Validate model parameters
        if cls.GNN_HIDDEN_DIM <= 0 or cls.GNN_OUTPUT_DIM <= 0:
            raise ValueError("GNN dimensions must be positive integers")
        
        return True
    
    @classmethod
    def get_database_url(cls) -> str:
        """
        Get the complete Neo4j database connection URL.
        
        Returns:
            str: Complete database connection URL.
        """
        return f"{cls.NEO4J_URI}"
    
    @classmethod
    def get_mlflow_config(cls) -> dict[str, str]:
        """
        Get MLflow configuration as a dictionary.
        
        Returns:
            dict: MLflow configuration settings.
        """
        config = {
            "tracking_uri": cls.MLFLOW_TRACKING_URI,
            "experiment_name": cls.MLFLOW_EXPERIMENT_NAME,
            "model_name": cls.MLFLOW_MODEL_NAME,
        }
        
        if cls.MLFLOW_ARTIFACT_ROOT:
            config["artifact_root"] = cls.MLFLOW_ARTIFACT_ROOT
            
        return config


# Global configuration instance
config = Config()

# Validate configuration on import
if __name__ != "__main__":
    config.validate_config()
    config.create_directories()

