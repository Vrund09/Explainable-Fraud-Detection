#!/usr/bin/env python3
"""
PaySim Dataset Download Script using KaggleHub.

This script demonstrates how to automatically download the PaySim dataset
using the integrated kagglehub functionality in our graph constructor.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.graph_constructor import GraphConstructor
from config import config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to download and process the PaySim dataset."""
    
    print("🔍 Explainable Fraud Detection - Dataset Download")
    print("=" * 50)
    
    try:
        # Initialize the graph constructor
        logger.info("Initializing GraphConstructor...")
        constructor = GraphConstructor()
        
        # Option 1: Just download the dataset
        print("\n📥 Downloading PaySim dataset using kagglehub...")
        constructor.load_raw_data(use_kagglehub=True)
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"   📍 Saved to: {constructor.raw_data_path}")
        print(f"   📊 Shape: {constructor.raw_data.shape}")
        print(f"   📋 Columns: {list(constructor.raw_data.columns)}")
        
        # Display basic statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Total transactions: {len(constructor.raw_data):,}")
        print(f"   • Transaction types: {constructor.raw_data['type'].unique().tolist()}")
        print(f"   • Fraud transactions: {constructor.raw_data['isFraud'].sum():,}")
        print(f"   • Fraud rate: {constructor.raw_data['isFraud'].mean():.4f} ({constructor.raw_data['isFraud'].mean() * 100:.2f}%)")
        print(f"   • Amount range: ${constructor.raw_data['amount'].min():.2f} - ${constructor.raw_data['amount'].max():,.2f}")
        
        # Ask user if they want to continue with full processing
        response = input("\n🚀 Would you like to continue with full data processing and graph construction? (y/N): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\n🏗️  Starting full data processing pipeline...")
            
            # Process the data into graph format
            nodes_df, edges_df = constructor.process_paysim_data(use_kagglehub=False)  # Data already loaded
            
            # Save processed data
            constructor.save_processed_data(nodes_df, edges_df)
            
            print(f"\n✅ Data processing completed!")
            print(f"   📊 Graph created with:")
            print(f"     • Nodes (users): {len(nodes_df):,}")
            print(f"     • Edges (transactions): {len(edges_df):,}")
            print(f"   💾 Saved to:")
            print(f"     • Nodes: {config.GRAPH_NODES_PATH}")
            print(f"     • Edges: {config.GRAPH_EDGES_PATH}")
            
            # Optional: Ingest to Neo4j
            neo4j_response = input("\n🗄️  Would you like to ingest data to Neo4j database? (y/N): ").lower().strip()
            
            if neo4j_response in ['y', 'yes']:
                print("\n🔄 Ingesting data to Neo4j...")
                try:
                    constructor.ingest_to_neo4j(nodes_df, edges_df, clear_existing=True)
                    print("✅ Data successfully ingested to Neo4j!")
                except Exception as e:
                    print(f"❌ Failed to ingest to Neo4j: {str(e)}")
                    print("   Make sure Neo4j is running and credentials are correct.")
        
        print("\n🎉 Process completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Train the GraphSAGE model: python -m src.gnn_model.training")
        print("   2. Start the API server: python -m uvicorn src.api.main:app --reload")
        print("   3. Open API documentation: http://localhost:8000/docs")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("\n📦 Please install required packages:")
        print("   pip install kagglehub kaggle")
    
    except Exception as e:
        logger.error(f"Error during dataset download: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   • Check your internet connection")
        print("   • Ensure kagglehub is installed: pip install kagglehub")
        print("   • Try manual download from: https://www.kaggle.com/datasets/mtalaltariq/paysim-data")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

