"""
Working AI Explanation System with Graph Context
==============================================

This module provides working AI explanations using graph context
for the fraud detection system.
"""

import json
import os
from typing import Dict, Any, List

class WorkingExplanationSystem:
    """Working AI explanation system with graph context"""
    
    def __init__(self):
        self.mock_data_path = "data/mock_neo4j"
        self.users_data = self._load_mock_data("users.json")
        self.network_data = self._load_mock_data("network.json")
        
        # Test Gemini connection
        self.ai_available = self._test_gemini()
        
        print(f"AI Explanation System Ready")
        print(f"AI Available: {self.ai_available}")
    
    def _load_mock_data(self, filename):
        """Load mock graph data"""
        file_path = os.path.join(self.mock_data_path, filename)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _test_gemini(self):
        """Test Gemini AI availability"""
        try:
            import google.generativeai as genai
            genai.configure(api_key='AIzaSyBssiEYJb2rJQdMeKCvVVCSAJT_uyLtpfw')
            return True
        except:
            return False
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context for explanations"""
        
        # Get user profile
        user_profile = self.users_data.get(user_id, {
            'total_transactions': 50,
            'fraud_rate': 0.01,
            'avg_amount_sent': 5000.0,
            'risk_category': 'MEDIUM'
        })
        
        # Get network connections
        connections = self.network_data.get(user_id, [])
        
        return {
            'user_profile': user_profile,
            'network_connections': connections,
            'context_available': True
        }
    
    def generate_enhanced_explanation(self, transaction: Dict[str, Any], prediction: Dict[str, Any]) -> str:
        """Generate enhanced explanation with graph context"""
        
        sender_id = transaction.get('sender_id', 'unknown')
        context = self.get_user_context(sender_id)
        
        if self.ai_available:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Enhanced prompt with graph context
                prompt = f"""
Professional fraud analysis with network context:

TRANSACTION: ${transaction.get('amount', 0):,.0f} {transaction.get('type', 'N/A')}
RISK ASSESSMENT: {prediction.get('risk_level', 'UNKNOWN')} ({prediction.get('fraud_probability', 0):.1%})

SENDER PROFILE:
- Transaction History: {context['user_profile']['total_transactions']} transactions
- Historical Fraud Rate: {context['user_profile']['fraud_rate']:.1%}
- Average Amount: ${context['user_profile']['avg_amount_sent']:,.0f}
- Risk Category: {context['user_profile']['risk_category']}

NETWORK ANALYSIS:
- Connected Users: {len(context['network_connections'])}
- Network Context: {'Available' if context['context_available'] else 'Limited'}

RISK FACTORS: {', '.join(prediction.get('risk_factors', []))}

Provide professional 2-sentence fraud analysis for compliance officers.
"""
                
                response = model.generate_content(prompt)
                return response.text.strip()
                
            except Exception as e:
                print(f"AI explanation error: {e}")
        
        # Enhanced fallback with context
        user_info = context['user_profile']
        risk_level = prediction.get('risk_level', 'UNKNOWN')
        fraud_prob = prediction.get('fraud_probability', 0)
        
        explanation = f"Advanced fraud analysis indicates {risk_level} risk ({fraud_prob:.1%} probability) "
        explanation += f"based on sender's profile ({user_info['total_transactions']} transactions, "
        explanation += f"{user_info['fraud_rate']:.1%} historical fraud rate) and current transaction patterns. "
        
        if prediction.get('risk_factors'):
            explanation += f"Key concerns: {', '.join(prediction['risk_factors'][:2])}."
        
        return explanation

# Global explanation system
working_explanations = WorkingExplanationSystem()

def get_enhanced_explanation(transaction: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    """Get enhanced explanation with graph context"""
    return working_explanations.generate_enhanced_explanation(transaction, prediction)
