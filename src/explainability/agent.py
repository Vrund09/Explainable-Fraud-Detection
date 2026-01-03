"""
AI Investigator Agent for Explainable Fraud Detection.

This module implements an AI-powered investigator that uses LangChain and
Gemini LLM to provide human-readable explanations for fraud predictions.
It integrates with Neo4j to retrieve transaction context and generate
comprehensive fraud analysis reports.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import AsyncCallbackHandler
from langchain.schema import LLMResult

import google.generativeai as genai
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable

from ..config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)


@dataclass
class TransactionContext:
    """Data class for transaction context information."""
    transaction_id: str
    user_id: str
    transaction_history: List[Dict[str, Any]]
    network_neighbors: List[Dict[str, Any]]
    risk_indicators: Dict[str, Any]
    graph_metrics: Dict[str, Any]


class Neo4jTransactionTool(BaseTool):
    """
    Custom LangChain tool for retrieving transaction context from Neo4j.
    
    This tool allows the AI agent to query the Neo4j graph database
    to get contextual information about users and their transaction patterns.
    """
    
    name = "get_transaction_context"
    description = """
    Retrieves comprehensive transaction context from the graph database.
    
    Input: user_id (string) - The ID of the user to investigate
    
    Returns: JSON object containing:
    - user_profile: Basic user information and statistics
    - recent_transactions: List of recent transactions for the user
    - network_neighbors: Information about connected users
    - risk_indicators: Calculated risk metrics
    - graph_metrics: Network centrality and connectivity metrics
    """
    
    def __init__(self, neo4j_driver: Driver):
        super().__init__()
        self.driver = neo4j_driver
    
    def _run(self, user_id: str) -> str:
        """Execute the tool to retrieve transaction context."""
        try:
            logger.info(f"Retrieving transaction context for user: {user_id}")
            
            with self.driver.session() as session:
                # Get user profile and statistics
                user_profile = self._get_user_profile(session, user_id)
                
                # Get recent transactions
                recent_transactions = self._get_recent_transactions(session, user_id)
                
                # Get network neighbors
                network_neighbors = self._get_network_neighbors(session, user_id)
                
                # Calculate risk indicators
                risk_indicators = self._calculate_risk_indicators(session, user_id)
                
                # Get graph metrics
                graph_metrics = self._get_graph_metrics(session, user_id)
                
                # Combine all context
                context = {
                    "user_id": user_id,
                    "user_profile": user_profile,
                    "recent_transactions": recent_transactions,
                    "network_neighbors": network_neighbors,
                    "risk_indicators": risk_indicators,
                    "graph_metrics": graph_metrics,
                    "retrieved_at": datetime.now().isoformat()
                }
                
                logger.info(f"Retrieved context for user {user_id}: {len(recent_transactions)} transactions, {len(network_neighbors)} neighbors")
                
                return json.dumps(context, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to retrieve transaction context for {user_id}: {str(e)}")
            return f"Error retrieving context: {str(e)}"
    
    async def _arun(self, user_id: str) -> str:
        """Async version of the tool execution."""
        return self._run(user_id)
    
    def _get_user_profile(self, session, user_id: str) -> Dict[str, Any]:
        """Get basic user profile information."""
        query = """
        MATCH (u:User {user_id: $user_id})
        RETURN u.user_id as user_id,
               u.total_transactions as total_transactions,
               u.total_amount_sent as total_amount_sent,
               u.total_amount_received as total_amount_received,
               u.fraud_rate as fraud_rate,
               u.fraud_transactions as fraud_transactions
        """
        
        result = session.run(query, user_id=user_id)
        record = result.single()
        
        if record:
            return dict(record)
        else:
            return {"user_id": user_id, "status": "user_not_found"}
    
    def _get_recent_transactions(self, session, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent transactions involving the user."""
        query = """
        MATCH (u:User {user_id: $user_id})-[t:TRANSACTION]->(other:User)
        RETURN t.transaction_id as transaction_id,
               t.amount as amount,
               t.type as type,
               t.step as step,
               t.is_fraud as is_fraud,
               other.user_id as counterparty,
               'sent' as direction
        ORDER BY t.step DESC
        LIMIT $limit
        
        UNION ALL
        
        MATCH (other:User)-[t:TRANSACTION]->(u:User {user_id: $user_id})
        RETURN t.transaction_id as transaction_id,
               t.amount as amount,
               t.type as type,
               t.step as step,
               t.is_fraud as is_fraud,
               other.user_id as counterparty,
               'received' as direction
        ORDER BY t.step DESC
        LIMIT $limit
        """
        
        result = session.run(query, user_id=user_id, limit=limit)
        transactions = [dict(record) for record in result]
        
        # Sort by step (time) descending
        transactions.sort(key=lambda x: x.get('step', 0), reverse=True)
        
        return transactions[:limit]
    
    def _get_network_neighbors(self, session, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about users connected to this user."""
        query = """
        MATCH (u:User {user_id: $user_id})-[t:TRANSACTION]-(neighbor:User)
        WITH neighbor, 
             count(t) as transaction_count,
             sum(t.amount) as total_amount,
             collect(DISTINCT t.type) as transaction_types,
             sum(CASE WHEN t.is_fraud THEN 1 ELSE 0 END) as fraud_count
        RETURN neighbor.user_id as neighbor_id,
               neighbor.fraud_rate as neighbor_fraud_rate,
               transaction_count,
               total_amount,
               transaction_types,
               fraud_count,
               fraud_count * 1.0 / transaction_count as fraud_rate_with_user
        ORDER BY transaction_count DESC
        LIMIT $limit
        """
        
        result = session.run(query, user_id=user_id, limit=limit)
        return [dict(record) for record in result]
    
    def _calculate_risk_indicators(self, session, user_id: str) -> Dict[str, Any]:
        """Calculate various risk indicators for the user."""
        query = """
        MATCH (u:User {user_id: $user_id})
        OPTIONAL MATCH (u)-[sent:TRANSACTION]->()
        OPTIONAL MATCH ()-[received:TRANSACTION]->(u)
        
        WITH u,
             collect(DISTINCT sent.type) as sent_types,
             collect(DISTINCT received.type) as received_types,
             avg(sent.amount) as avg_sent_amount,
             max(sent.amount) as max_sent_amount,
             count(sent) as sent_count,
             count(received) as received_count,
             sum(CASE WHEN sent.is_fraud THEN 1 ELSE 0 END) as fraud_sent,
             sum(CASE WHEN received.is_fraud THEN 1 ELSE 0 END) as fraud_received
        
        RETURN u.fraud_rate as overall_fraud_rate,
               sent_types,
               received_types,
               avg_sent_amount,
               max_sent_amount,
               sent_count,
               received_count,
               fraud_sent,
               fraud_received,
               fraud_sent * 1.0 / CASE WHEN sent_count > 0 THEN sent_count ELSE 1 END as sent_fraud_rate,
               fraud_received * 1.0 / CASE WHEN received_count > 0 THEN received_count ELSE 1 END as received_fraud_rate
        """
        
        result = session.run(query, user_id=user_id)
        record = result.single()
        
        if record:
            return dict(record)
        else:
            return {}
    
    def _get_graph_metrics(self, session, user_id: str) -> Dict[str, Any]:
        """Calculate graph-based metrics for the user."""
        # Get degree centrality (number of connections)
        degree_query = """
        MATCH (u:User {user_id: $user_id})-[t:TRANSACTION]-()
        WITH u, count(DISTINCT t) as degree
        RETURN degree
        """
        
        degree_result = session.run(degree_query, user_id=user_id)
        degree_record = degree_result.single()
        degree = degree_record["degree"] if degree_record else 0
        
        # Get clustering coefficient (how connected are the neighbors)
        clustering_query = """
        MATCH (u:User {user_id: $user_id})-[:TRANSACTION]-(neighbor1:User)
        MATCH (u)-[:TRANSACTION]-(neighbor2:User)
        WHERE neighbor1 <> neighbor2
        WITH u, count(DISTINCT neighbor1) as neighbors,
             count(DISTINCT neighbor2) as neighbors2
        OPTIONAL MATCH (neighbor1)-[:TRANSACTION]-(neighbor2)
        WITH u, neighbors, count(*) as connected_neighbors
        RETURN connected_neighbors * 2.0 / (neighbors * (neighbors - 1)) as clustering_coefficient
        """
        
        try:
            clustering_result = session.run(clustering_query, user_id=user_id)
            clustering_record = clustering_result.single()
            clustering_coeff = clustering_record["clustering_coefficient"] if clustering_record else 0.0
        except:
            clustering_coeff = 0.0
        
        return {
            "degree_centrality": degree,
            "clustering_coefficient": clustering_coeff or 0.0
        }


class LoggingCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for logging agent interactions."""
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log when LLM starts processing."""
        logger.debug(f"LLM started processing {len(prompts)} prompts")
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log when LLM finishes processing."""
        logger.debug(f"LLM finished processing, generated {len(response.generations)} responses")
    
    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Log agent actions."""
        logger.info(f"Agent action: {action.tool} with input: {action.tool_input}")
    
    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Log when agent finishes."""
        logger.info(f"Agent finished with output length: {len(str(finish.return_values))}")


class AIInvestigator:
    """
    AI-powered fraud investigation agent.
    
    This class combines Neo4j graph queries with Gemini LLM to provide
    detailed, context-aware explanations for fraud predictions.
    """
    
    def __init__(
        self,
        neo4j_uri: str = config.NEO4J_URI,
        neo4j_username: str = config.NEO4J_USERNAME,
        neo4j_password: str = config.NEO4J_PASSWORD,
        gemini_api_key: Optional[str] = config.GEMINI_API_KEY,
        model_name: str = config.GEMINI_MODEL_NAME
    ):
        """
        Initialize the AI Investigator.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            gemini_api_key: Google Gemini API key
            model_name: Gemini model name to use
        """
        self.neo4j_driver: Optional[Driver] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.agent: Optional[AgentExecutor] = None
        self.memory: Optional[ConversationBufferWindowMemory] = None
        
        # Initialize components
        self._initialize_neo4j(neo4j_uri, neo4j_username, neo4j_password)
        self._initialize_llm(gemini_api_key, model_name)
        self._initialize_agent()
        
        logger.info("AI Investigator initialized successfully")
    
    def _initialize_neo4j(self, uri: str, username: str, password: str) -> None:
        """Initialize Neo4j database connection."""
        try:
            self.neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1").single()
            
            logger.info("✓ Neo4j connection established")
            
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j connection failed: {str(e)}")
            self.neo4j_driver = None
            # Continue without Neo4j - explanations will be limited
        except Exception as e:
            logger.error(f"❌ Neo4j initialization error: {str(e)}")
            self.neo4j_driver = None
    
    def _initialize_llm(self, api_key: Optional[str], model_name: str) -> None:
        """Initialize the Gemini LLM."""
        try:
            if not api_key:
                logger.warning("No Gemini API key provided - explanations will be limited")
                return
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize LangChain wrapper
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3,  # Lower temperature for more consistent explanations
                max_output_tokens=1000
            )
            
            logger.info("✓ Gemini LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {str(e)}")
            self.llm = None
    
    def _initialize_agent(self) -> None:
        """Initialize the LangChain agent with tools."""
        if not self.llm or not self.neo4j_driver:
            logger.warning("Cannot initialize agent - LLM or Neo4j not available")
            return
        
        try:
            # Create tools
            tools = [Neo4jTransactionTool(self.neo4j_driver)]
            
            # Create memory for conversation context
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,  # Keep last 5 interactions
                return_messages=True
            )
            
            # Initialize agent
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            logger.info("✓ AI Agent initialized with Neo4j tool")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {str(e)}")
            self.agent = None
    
    def _create_investigation_prompt(
        self, 
        transaction_id: str, 
        depth: str = "standard",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a detailed prompt for fraud investigation.
        
        Args:
            transaction_id: ID of the transaction to investigate
            depth: Level of detail ('basic', 'standard', 'detailed')
            context: Additional context about the prediction
            
        Returns:
            str: Formatted investigation prompt
        """
        base_prompt = f"""
You are a Senior Fraud Analyst investigating transaction {transaction_id}. 
Your task is to provide a comprehensive analysis explaining why this transaction 
was flagged as potentially fraudulent.

INVESTIGATION GUIDELINES:
1. Use the get_transaction_context tool to gather information about the users involved
2. Analyze patterns in transaction history, amounts, timing, and network connections
3. Identify specific risk factors and anomalies
4. Provide actionable recommendations

ANALYSIS DEPTH: {depth.upper()}
"""
        
        if depth == "basic":
            prompt_detail = """
Focus on:
- Key risk indicators
- Simple explanation in 2-3 sentences
- Clear recommendation (approve/reject/review)
"""
        elif depth == "detailed":
            prompt_detail = """
Provide comprehensive analysis including:
- Detailed user behavior patterns
- Network analysis and connections
- Historical context and trends
- Statistical comparisons
- Multiple recommendation scenarios
- Confidence assessment
"""
        else:  # standard
            prompt_detail = """
Include:
- Main risk factors and patterns
- User transaction behavior
- Network context
- Clear explanation (4-5 sentences)
- Specific recommendation with reasoning
"""
        
        context_info = ""
        if context:
            context_info = f"\nADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        full_prompt = base_prompt + prompt_detail + context_info + """
Start your investigation by using the get_transaction_context tool to gather 
information about the users involved in this transaction.
"""
        
        return full_prompt
    
    async def explain_transaction(
        self,
        transaction_id: str,
        depth: str = "standard",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation for a fraud prediction.
        
        Args:
            transaction_id: ID of the transaction to explain
            depth: Level of detail ('basic', 'standard', 'detailed')
            context: Additional context about the prediction
            
        Returns:
            Dict[str, Any]: Explanation results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting investigation for transaction {transaction_id} (depth: {depth})")
            
            if not self.agent:
                # Fallback explanation without AI agent
                return self._create_fallback_explanation(transaction_id, context)
            
            # Create investigation prompt
            prompt = self._create_investigation_prompt(transaction_id, depth, context)
            
            # Run the agent
            logger.info("Running AI investigation agent...")
            result = await asyncio.to_thread(self.agent.run, prompt)
            
            # Parse the agent's response
            explanation_result = self._parse_agent_response(result, transaction_id)
            
            # Add metadata
            explanation_result.update({
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "investigation_depth": depth,
                "agent_used": True,
                "data_sources": ["neo4j", "gemini_llm"]
            })
            
            logger.info(f"Investigation completed for transaction {transaction_id}")
            
            return explanation_result
            
        except Exception as e:
            logger.error(f"Investigation failed for transaction {transaction_id}: {str(e)}")
            
            # Return fallback explanation
            return self._create_fallback_explanation(
                transaction_id, 
                context, 
                error=str(e)
            )
    
    def _parse_agent_response(self, agent_output: str, transaction_id: str) -> Dict[str, Any]:
        """
        Parse the agent's response into structured format.
        
        Args:
            agent_output: Raw output from the agent
            transaction_id: Transaction ID being investigated
            
        Returns:
            Dict[str, Any]: Structured explanation
        """
        # This is a simplified parser - in practice, you might want more sophisticated parsing
        lines = agent_output.split('\n')
        
        explanation_text = agent_output
        key_factors = []
        risk_indicators = {}
        recommendation = "Manual review recommended"
        confidence = 0.7
        
        # Try to extract structured information from the response
        for line in lines:
            if "risk factor" in line.lower() or "indicator" in line.lower():
                key_factors.append(line.strip())
            elif "recommend" in line.lower():
                recommendation = line.strip()
        
        # Extract any numeric values as risk indicators
        import re
        numbers = re.findall(r'\d+\.?\d*', agent_output)
        if numbers:
            risk_indicators["analysis_score"] = float(numbers[0]) if numbers else 0.0
        
        return {
            "explanation_text": explanation_text,
            "key_factors": key_factors[:5],  # Limit to top 5 factors
            "risk_indicators": risk_indicators,
            "recommendation": recommendation,
            "confidence": confidence
        }
    
    def _create_fallback_explanation(
        self,
        transaction_id: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a basic explanation when the AI agent is unavailable.
        
        Args:
            transaction_id: Transaction ID
            context: Available context
            error: Error message if applicable
            
        Returns:
            Dict[str, Any]: Basic explanation
        """
        explanation_text = f"""
Fraud Analysis for Transaction {transaction_id}:

This transaction has been flagged as potentially fraudulent based on our 
Graph Neural Network model analysis. The model detected patterns that 
are commonly associated with fraudulent transactions.
"""
        
        if error:
            explanation_text += f"\n\nNote: Advanced AI explanation unavailable due to: {error}"
        
        key_factors = [
            "Pattern matching with known fraud indicators",
            "Network-based anomaly detection",
            "Machine learning model prediction"
        ]
        
        if context:
            amount = context.get('amount', 0)
            if amount > 100000:
                key_factors.append("High transaction amount")
            
            trans_type = context.get('type', '')
            if trans_type in ['CASH_OUT', 'TRANSFER']:
                key_factors.append(f"High-risk transaction type: {trans_type}")
        
        return {
            "explanation_text": explanation_text,
            "key_factors": key_factors,
            "risk_indicators": {
                "model_confidence": 0.8,
                "fallback_explanation": True
            },
            "recommendation": "Manual review recommended due to fraud prediction",
            "confidence": 0.6,
            "agent_used": False,
            "data_sources": ["gnn_model"]
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the AI Investigator.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        status = {
            "neo4j_connected": self.neo4j_driver is not None,
            "llm_initialized": self.llm is not None,
            "agent_available": self.agent is not None,
            "overall_health": "healthy"
        }
        
        # Determine overall health
        if not status["neo4j_connected"] and not status["llm_initialized"]:
            status["overall_health"] = "unhealthy"
        elif not status["agent_available"]:
            status["overall_health"] = "degraded"
        
        return status
    
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")
        
        logger.info("AI Investigator resources cleaned up")


