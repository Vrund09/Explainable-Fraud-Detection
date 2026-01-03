"""
Graph Constructor for PaySim Fraud Detection Dataset.

This module handles the transformation of raw PaySim transaction data into
a graph structure suitable for Graph Neural Network training. It creates
nodes (users) and edges (transactions) and provides functionality to
ingest this data into a Neo4j graph database.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError
import time

from ..config import config


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class GraphConstructor:
    """
    Constructs graph representation from PaySim transaction data.
    
    This class processes the raw PaySim CSV data and transforms it into
    a graph structure with users as nodes and transactions as edges.
    It also provides functionality to ingest this data into Neo4j.
    
    Attributes:
        raw_data_path (Path): Path to the raw PaySim CSV file
        processed_data_dir (Path): Directory to save processed data
        neo4j_driver (Driver): Neo4j database driver (optional)
    """
    
    def __init__(
        self,
        raw_data_path: Optional[Path] = None,
        processed_data_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the GraphConstructor.
        
        Args:
            raw_data_path: Path to the raw PaySim CSV file.
                         Defaults to config.PAYSIM_RAW_PATH
            processed_data_dir: Directory to save processed data.
                              Defaults to config.PROCESSED_DATA_DIR
        """
        self.raw_data_path = raw_data_path or config.PAYSIM_RAW_PATH
        self.processed_data_dir = processed_data_dir or config.PROCESSED_DATA_DIR
        self.neo4j_driver: Optional[Driver] = None
        
        # Ensure processed data directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.raw_data: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.edges_df: Optional[pd.DataFrame] = None
        
        logger.info(f"GraphConstructor initialized with data path: {self.raw_data_path}")
    
    def load_raw_data(self, use_kagglehub: bool = True) -> pd.DataFrame:
        """
        Load the raw PaySim dataset from Kaggle using kagglehub or local CSV.
        
        Args:
            use_kagglehub: Whether to use kagglehub to download dataset (default: True)
        
        Returns:
            pd.DataFrame: Raw PaySim transaction data
            
        Raises:
            FileNotFoundError: If the PaySim CSV file doesn't exist and kagglehub fails
            pd.errors.EmptyDataError: If the CSV file is empty
            Exception: For other data loading errors
        """
        try:
            # First try to use kagglehub to download the dataset
            if use_kagglehub:
                try:
                    logger.info("Attempting to download PaySim dataset using kagglehub...")
                    import kagglehub
                    import glob
                    
                    # Download the dataset files using new API
                    path = kagglehub.dataset_download("mtalaltariq/paysim-data")
                    logger.info(f"Downloaded to: {path}")
                    
                    # Find and load the CSV file
                    csv_files = glob.glob(f"{path}/*.csv")
                    if csv_files:
                        self.raw_data = pd.read_csv(csv_files[0])
                    else:
                        raise Exception("No CSV files found in downloaded dataset")
                    
                    logger.info("✅ Successfully downloaded PaySim dataset using kagglehub")
                    logger.info(f"Dataset shape: {self.raw_data.shape}")
                    logger.info(f"Columns: {list(self.raw_data.columns)}")
                    
                    # Save the downloaded data locally for future use
                    self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
                    self.raw_data.to_csv(self.raw_data_path, index=False)
                    logger.info(f"💾 Saved dataset locally to {self.raw_data_path}")
                    
                    return self.raw_data
                    
                except Exception as e:
                    logger.warning(f"Failed to download using kagglehub: {str(e)}")
                    logger.info("Falling back to local file...")
            
            # Fallback to loading from local file
            if not self.raw_data_path.exists():
                error_message = f"""
PaySim dataset not found at {self.raw_data_path}.

Options to get the dataset:
1. Automatic download (recommended):
   - Run with use_kagglehub=True (default)
   - Requires kagglehub: pip install kagglehub

2. Manual download:
   - Visit: https://www.kaggle.com/datasets/mtalaltariq/paysim-data
   - Download the CSV file
   - Place it at: {self.raw_data_path}

3. Alternative dataset:
   - Original PaySim: https://www.kaggle.com/ntnu-testimon/paysim1
"""
                raise FileNotFoundError(error_message)
            
            logger.info(f"Loading raw data from local file: {self.raw_data_path}")
            self.raw_data = pd.read_csv(self.raw_data_path)
            
            if self.raw_data.empty:
                raise pd.errors.EmptyDataError("The PaySim CSV file is empty")
            
            logger.info(f"Loaded {len(self.raw_data)} transactions from PaySim dataset")
            logger.info(f"Dataset shape: {self.raw_data.shape}")
            logger.info(f"Columns: {list(self.raw_data.columns)}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def validate_data_schema(self) -> bool:
        """
        Validate that the loaded data has the expected PaySim schema.
        
        Returns:
            bool: True if schema is valid
            
        Raises:
            ValueError: If required columns are missing
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        required_columns = [
            'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
            'isFraud', 'isFlaggedFraud'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info("Data schema validation passed")
        return True
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the raw PaySim data for graph construction.
        
        This method performs data cleaning, feature engineering, and filtering
        to prepare the data for graph construction.
        
        Returns:
            pd.DataFrame: Preprocessed transaction data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        logger.info("Starting data preprocessing...")
        df = self.raw_data.copy()
        
        # Remove transactions with invalid amounts
        initial_count = len(df)
        df = df[df['amount'] > 0]
        df = df[df['amount'] <= config.MAX_AMOUNT_THRESHOLD]
        logger.info(f"Filtered {initial_count - len(df)} transactions with invalid amounts")
        
        # Create additional features
        df['amount_log'] = np.log1p(df['amount'])
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Add normalized time features
        df['hour_of_day'] = df['step'] % 24
        df['day_of_month'] = (df['step'] // 24) % 30
        
        # Create transaction type encoding
        df['type_encoded'] = pd.Categorical(df['type']).codes
        
        logger.info(f"Preprocessing completed. {len(df)} transactions remaining.")
        return df
    
    def create_nodes_dataframe(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a nodes DataFrame from preprocessed transaction data.
        
        Each unique user (nameOrig and nameDest) becomes a node with aggregated features.
        
        Args:
            preprocessed_data: Preprocessed transaction DataFrame
            
        Returns:
            pd.DataFrame: Nodes DataFrame with user features
        """
        logger.info("Creating nodes DataFrame...")
        
        # Get all unique users (both originators and destinations)
        orig_users = preprocessed_data[['nameOrig', 'type']].copy()
        orig_users.columns = ['user_id', 'transaction_type']
        orig_users['role'] = 'originator'
        
        dest_users = preprocessed_data[['nameDest', 'type']].copy()
        dest_users.columns = ['user_id', 'transaction_type']
        dest_users['role'] = 'destination'
        
        # Combine all users
        all_users = pd.concat([orig_users, dest_users], ignore_index=True)
        
        # Calculate user-level features
        user_stats = []
        
        for user_id in all_users['user_id'].unique():
            # Get transactions where this user was the originator
            orig_transactions = preprocessed_data[preprocessed_data['nameOrig'] == user_id]
            # Get transactions where this user was the destination
            dest_transactions = preprocessed_data[preprocessed_data['nameDest'] == user_id]
            
            # Calculate features
            total_transactions = len(orig_transactions) + len(dest_transactions)
            total_amount_sent = orig_transactions['amount'].sum()
            total_amount_received = dest_transactions['amount'].sum()
            avg_amount_sent = orig_transactions['amount'].mean() if len(orig_transactions) > 0 else 0
            avg_amount_received = dest_transactions['amount'].mean() if len(dest_transactions) > 0 else 0
            
            # Fraud-related features
            fraud_transactions_as_orig = orig_transactions['isFraud'].sum()
            fraud_transactions_as_dest = dest_transactions['isFraud'].sum()
            total_fraud_transactions = fraud_transactions_as_orig + fraud_transactions_as_dest
            fraud_rate = total_fraud_transactions / total_transactions if total_transactions > 0 else 0
            
            # Transaction type diversity
            unique_types_as_orig = len(orig_transactions['type'].unique()) if len(orig_transactions) > 0 else 0
            unique_types_as_dest = len(dest_transactions['type'].unique()) if len(dest_transactions) > 0 else 0
            
            user_stats.append({
                'user_id': user_id,
                'total_transactions': total_transactions,
                'transactions_as_originator': len(orig_transactions),
                'transactions_as_destination': len(dest_transactions),
                'total_amount_sent': total_amount_sent,
                'total_amount_received': total_amount_received,
                'avg_amount_sent': avg_amount_sent,
                'avg_amount_received': avg_amount_received,
                'net_amount': total_amount_received - total_amount_sent,
                'fraud_transactions': total_fraud_transactions,
                'fraud_rate': fraud_rate,
                'unique_transaction_types_orig': unique_types_as_orig,
                'unique_transaction_types_dest': unique_types_as_dest,
                'is_active_sender': len(orig_transactions) >= config.MIN_TRANSACTION_COUNT,
                'is_active_receiver': len(dest_transactions) >= config.MIN_TRANSACTION_COUNT
            })
        
        nodes_df = pd.DataFrame(user_stats)
        
        # Filter users with minimum transaction count
        initial_users = len(nodes_df)
        nodes_df = nodes_df[nodes_df['total_transactions'] >= config.MIN_TRANSACTION_COUNT]
        logger.info(f"Filtered {initial_users - len(nodes_df)} users with insufficient transactions")
        
        # Add normalized features
        if len(nodes_df) > 0:
            nodes_df['amount_sent_normalized'] = (
                (nodes_df['total_amount_sent'] - nodes_df['total_amount_sent'].mean()) /
                (nodes_df['total_amount_sent'].std() + 1e-8)
            )
            nodes_df['amount_received_normalized'] = (
                (nodes_df['total_amount_received'] - nodes_df['total_amount_received'].mean()) /
                (nodes_df['total_amount_received'].std() + 1e-8)
            )
        
        self.nodes_df = nodes_df
        logger.info(f"Created nodes DataFrame with {len(nodes_df)} users")
        
        return nodes_df
    
    def create_edges_dataframe(self, preprocessed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create an edges DataFrame from preprocessed transaction data.
        
        Each transaction becomes an edge with associated features.
        
        Args:
            preprocessed_data: Preprocessed transaction DataFrame
            
        Returns:
            pd.DataFrame: Edges DataFrame with transaction features
        """
        logger.info("Creating edges DataFrame...")
        
        # Filter to only include users that exist in the nodes DataFrame
        if self.nodes_df is None:
            raise ValueError("Nodes DataFrame not created. Call create_nodes_dataframe() first.")
        
        valid_users = set(self.nodes_df['user_id'])
        edges_df = preprocessed_data[
            (preprocessed_data['nameOrig'].isin(valid_users)) &
            (preprocessed_data['nameDest'].isin(valid_users))
        ].copy()
        
        # Rename columns for clarity
        edges_df = edges_df.rename(columns={
            'nameOrig': 'source_user',
            'nameDest': 'target_user',
            'isFraud': 'is_fraud',
            'isFlaggedFraud': 'is_flagged_fraud'
        })
        
        # Add edge-specific features
        edges_df['transaction_id'] = range(len(edges_df))
        edges_df['weight'] = 1.0  # Default weight, can be modified based on amount or frequency
        
        # Create edge weight based on transaction amount (normalized)
        if len(edges_df) > 0:
            max_amount = edges_df['amount'].max()
            edges_df['weight'] = edges_df['amount'] / max_amount
            
            # Ensure minimum edge weight
            edges_df['weight'] = edges_df['weight'].clip(lower=config.EDGE_WEIGHT_THRESHOLD)
        
        # Select relevant columns for the final edges DataFrame
        edge_columns = [
            'transaction_id', 'source_user', 'target_user', 'step', 'type', 
            'amount', 'amount_log', 'balance_change_orig', 'balance_change_dest',
            'hour_of_day', 'day_of_month', 'type_encoded', 'weight',
            'is_fraud', 'is_flagged_fraud'
        ]
        
        edges_df = edges_df[edge_columns]
        self.edges_df = edges_df
        
        logger.info(f"Created edges DataFrame with {len(edges_df)} transactions")
        logger.info(f"Fraud rate in edges: {edges_df['is_fraud'].mean():.4f}")
        
        return edges_df
    
    def process_paysim_data(self, use_kagglehub: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete pipeline to process PaySim data into nodes and edges DataFrames.
        
        This method orchestrates the entire data processing pipeline:
        1. Load raw data (automatically from Kaggle using kagglehub)
        2. Validate schema
        3. Preprocess data
        4. Create nodes DataFrame
        5. Create edges DataFrame
        
        Args:
            use_kagglehub: Whether to automatically download dataset using kagglehub (default: True)
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (nodes_df, edges_df)
        """
        logger.info("Starting complete PaySim data processing pipeline...")
        
        try:
            # Step 1: Load and validate data (with automatic download)
            self.load_raw_data(use_kagglehub=use_kagglehub)
            self.validate_data_schema()
            
            # Step 2: Preprocess data
            preprocessed_data = self.preprocess_data()
            
            # Step 3: Create graph components
            nodes_df = self.create_nodes_dataframe(preprocessed_data)
            edges_df = self.create_edges_dataframe(preprocessed_data)
            
            logger.info("PaySim data processing completed successfully")
            logger.info(f"Final graph: {len(nodes_df)} nodes, {len(edges_df)} edges")
            
            return nodes_df, edges_df
            
        except Exception as e:
            logger.error(f"Error in PaySim data processing: {str(e)}")
            raise
    
    def save_processed_data(
        self,
        nodes_df: Optional[pd.DataFrame] = None,
        edges_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Save the processed nodes and edges DataFrames to CSV files.
        
        Args:
            nodes_df: Nodes DataFrame to save. Uses self.nodes_df if None.
            edges_df: Edges DataFrame to save. Uses self.edges_df if None.
        """
        nodes_df = nodes_df or self.nodes_df
        edges_df = edges_df or self.edges_df
        
        if nodes_df is None or edges_df is None:
            raise ValueError("No processed data available to save")
        
        # Save nodes
        nodes_path = config.GRAPH_NODES_PATH
        nodes_df.to_csv(nodes_path, index=False)
        logger.info(f"Saved {len(nodes_df)} nodes to {nodes_path}")
        
        # Save edges
        edges_path = config.GRAPH_EDGES_PATH
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"Saved {len(edges_df)} edges to {edges_path}")
    
    def connect_to_neo4j(self) -> None:
        """
        Establish connection to Neo4j database.
        
        Raises:
            ServiceUnavailable: If Neo4j service is not available
            AuthError: If authentication fails
            Exception: For other connection errors
        """
        try:
            logger.info(f"Connecting to Neo4j at {config.NEO4J_URI}")
            
            self.neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
                max_connection_lifetime=config.NEO4J_MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=config.NEO4J_MAX_CONNECTION_POOL_SIZE
            )
            
            # Test the connection
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                if test_value != 1:
                    raise Exception("Connection test failed")
            
            logger.info("Successfully connected to Neo4j")
            
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {str(e)}")
            raise
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def create_neo4j_constraints(self) -> None:
        """
        Create necessary constraints and indexes in Neo4j for optimal performance.
        """
        if not self.neo4j_driver:
            raise ValueError("Not connected to Neo4j. Call connect_to_neo4j() first.")
        
        constraints_and_indexes = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE INDEX user_fraud_rate IF NOT EXISTS FOR (u:User) ON (u.fraud_rate)",
            "CREATE INDEX transaction_amount IF NOT EXISTS FOR ()-[t:TRANSACTION]-() ON (t.amount)",
            "CREATE INDEX transaction_fraud IF NOT EXISTS FOR ()-[t:TRANSACTION]-() ON (t.is_fraud)",
            "CREATE INDEX transaction_type IF NOT EXISTS FOR ()-[t:TRANSACTION]-() ON (t.type)"
        ]
        
        with self.neo4j_driver.session() as session:
            for constraint_query in constraints_and_indexes:
                try:
                    session.run(constraint_query)
                    logger.info(f"Created constraint/index: {constraint_query.split()[1]}")
                except Exception as e:
                    logger.warning(f"Could not create constraint/index: {str(e)}")
    
    def ingest_nodes_to_neo4j(self, nodes_df: Optional[pd.DataFrame] = None) -> None:
        """
        Ingest nodes (users) into Neo4j database.
        
        Args:
            nodes_df: Nodes DataFrame to ingest. Uses self.nodes_df if None.
        """
        nodes_df = nodes_df or self.nodes_df
        if nodes_df is None:
            raise ValueError("No nodes data available to ingest")
        
        if not self.neo4j_driver:
            raise ValueError("Not connected to Neo4j. Call connect_to_neo4j() first.")
        
        logger.info(f"Ingesting {len(nodes_df)} nodes into Neo4j...")
        
        # Convert DataFrame to list of dictionaries for efficient ingestion
        nodes_data = nodes_df.to_dict('records')
        
        # Batch insert nodes
        batch_size = 1000
        with self.neo4j_driver.session() as session:
            for i in range(0, len(nodes_data), batch_size):
                batch = nodes_data[i:i + batch_size]
                
                session.run("""
                    UNWIND $nodes AS node
                    MERGE (u:User {user_id: node.user_id})
                    SET u.total_transactions = node.total_transactions,
                        u.transactions_as_originator = node.transactions_as_originator,
                        u.transactions_as_destination = node.transactions_as_destination,
                        u.total_amount_sent = node.total_amount_sent,
                        u.total_amount_received = node.total_amount_received,
                        u.avg_amount_sent = node.avg_amount_sent,
                        u.avg_amount_received = node.avg_amount_received,
                        u.net_amount = node.net_amount,
                        u.fraud_transactions = node.fraud_transactions,
                        u.fraud_rate = node.fraud_rate,
                        u.is_active_sender = node.is_active_sender,
                        u.is_active_receiver = node.is_active_receiver
                """, nodes=batch)
                
                logger.info(f"Ingested batch {i // batch_size + 1}/{(len(nodes_data) + batch_size - 1) // batch_size}")
        
        logger.info("Nodes ingestion completed successfully")
    
    def ingest_edges_to_neo4j(self, edges_df: Optional[pd.DataFrame] = None) -> None:
        """
        Ingest edges (transactions) into Neo4j database.
        
        Args:
            edges_df: Edges DataFrame to ingest. Uses self.edges_df if None.
        """
        edges_df = edges_df or self.edges_df
        if edges_df is None:
            raise ValueError("No edges data available to ingest")
        
        if not self.neo4j_driver:
            raise ValueError("Not connected to Neo4j. Call connect_to_neo4j() first.")
        
        logger.info(f"Ingesting {len(edges_df)} edges into Neo4j...")
        
        # Convert DataFrame to list of dictionaries
        edges_data = edges_df.to_dict('records')
        
        # Batch insert edges
        batch_size = 500  # Smaller batch size for relationship creation
        with self.neo4j_driver.session() as session:
            for i in range(0, len(edges_data), batch_size):
                batch = edges_data[i:i + batch_size]
                
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (source:User {user_id: edge.source_user})
                    MATCH (target:User {user_id: edge.target_user})
                    CREATE (source)-[t:TRANSACTION {
                        transaction_id: edge.transaction_id,
                        step: edge.step,
                        type: edge.type,
                        amount: edge.amount,
                        amount_log: edge.amount_log,
                        balance_change_orig: edge.balance_change_orig,
                        balance_change_dest: edge.balance_change_dest,
                        hour_of_day: edge.hour_of_day,
                        day_of_month: edge.day_of_month,
                        weight: edge.weight,
                        is_fraud: edge.is_fraud,
                        is_flagged_fraud: edge.is_flagged_fraud
                    }]->(target)
                """, edges=batch)
                
                logger.info(f"Ingested batch {i // batch_size + 1}/{(len(edges_data) + batch_size - 1) // batch_size}")
        
        logger.info("Edges ingestion completed successfully")
    
    def ingest_to_neo4j(
        self,
        nodes_df: Optional[pd.DataFrame] = None,
        edges_df: Optional[pd.DataFrame] = None,
        clear_existing: bool = False
    ) -> None:
        """
        Complete pipeline to ingest processed data into Neo4j.
        
        Args:
            nodes_df: Nodes DataFrame to ingest. Uses self.nodes_df if None.
            edges_df: Edges DataFrame to ingest. Uses self.edges_df if None.
            clear_existing: Whether to clear existing data before ingestion.
        """
        try:
            # Connect to Neo4j
            self.connect_to_neo4j()
            
            # Clear existing data if requested
            if clear_existing:
                logger.info("Clearing existing data from Neo4j...")
                with self.neo4j_driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                logger.info("Existing data cleared")
            
            # Create constraints and indexes
            self.create_neo4j_constraints()
            
            # Ingest data
            self.ingest_nodes_to_neo4j(nodes_df)
            self.ingest_edges_to_neo4j(edges_df)
            
            logger.info("Neo4j ingestion completed successfully")
            
        except Exception as e:
            logger.error(f"Error during Neo4j ingestion: {str(e)}")
            raise
        finally:
            if self.neo4j_driver:
                self.neo4j_driver.close()
                logger.info("Neo4j connection closed")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the constructed graph.
        
        Returns:
            Dict[str, Any]: Graph statistics
        """
        stats = {}
        
        if self.nodes_df is not None:
            stats['num_nodes'] = len(self.nodes_df)
            stats['avg_transactions_per_user'] = self.nodes_df['total_transactions'].mean()
            stats['median_transactions_per_user'] = self.nodes_df['total_transactions'].median()
            stats['users_with_fraud'] = (self.nodes_df['fraud_transactions'] > 0).sum()
            stats['avg_fraud_rate'] = self.nodes_df['fraud_rate'].mean()
        
        if self.edges_df is not None:
            stats['num_edges'] = len(self.edges_df)
            stats['fraud_edges'] = self.edges_df['is_fraud'].sum()
            stats['fraud_rate_edges'] = self.edges_df['is_fraud'].mean()
            stats['avg_transaction_amount'] = self.edges_df['amount'].mean()
            stats['median_transaction_amount'] = self.edges_df['amount'].median()
            stats['transaction_types'] = self.edges_df['type'].value_counts().to_dict()
        
        return stats

