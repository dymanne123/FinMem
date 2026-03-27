"""
Finance Dialogue Dataset - Test with FinMem Agent (Long-Short Term Memory + Skill Routing)
==========================================================================================

This script integrates FinMem agent into the test framework, supporting:
- Custom model selection (OpenAI, Qwen, DeepSeek, Llama, etc.)
- Custom API key and base URL
- Long-short term memory for personalized responses
- Skill routing based on query type (investment, tax, retirement, insurance, debt)

Usage:
    python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --base-url https://api.vveai.com/v1
    python test_finmem.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
    python test_finmem.py --model deepseek-chat --api-key YOUR_KEY
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import sys

# Import FinMem components
from finmem import (
    FinMemAgent,
    UserPersona,
    FinancialEvent,
    QueryType
)

# Import evaluation metrics from test_class
try:
    from test_class import (
        calculate_bleu_score,
        calculate_f1_score,
        calculate_exact_match,
        LLMEvaluator,
        load_dataset,
        extract_test_queries,
        calculate_metrics,
        analyze_by_question_type,
        print_classification_report
    )
except ImportError:
    # Fallback: simple metrics if test_class not available
    def calculate_bleu_score(ref, hyp):
        ref_tokens = set(ref.lower().split())
        hyp_tokens = set(hyp.lower().split())
        if not ref_tokens:
            return 0.0
        return len(ref_tokens & hyp_tokens) / len(ref_tokens)
    
    def calculate_f1_score(ref, hyp):
        ref_tokens = set(ref.lower().split())
        hyp_tokens = set(hyp.lower().split())
        if not ref_tokens or not hyp_tokens:
            return 0.0
        tp = len(ref_tokens & hyp_tokens)
        precision = tp / len(hyp_tokens) if hyp_tokens else 0
        recall = tp / len(ref_tokens) if ref_tokens else 0
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_exact_match(ref, hyp):
        return 1.0 if ref.lower().strip() == hyp.lower().strip() else 0.0
    
    def load_dataset(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_test_queries(data, max_per_user=1000, include_dialogue_context=False):
        queries = []
        for user_data in data:
            persona = user_data.get("persona", {})
            timeline = user_data.get("timeline", [])
            sessions = user_data.get("sessions", [])
            qa_pairs = user_data.get("qa_pairs", [])
            
            session_dict = {s.get("session_id"): s for s in sessions}
            
            #for qa in qa_pairs[:max_per_user]:
            for qa in qa_pairs:
                context = {
                    "persona": persona,
                    "timeline": timeline,
                    "session_id": qa.get("context_session_id", "")
                }
                
                if include_dialogue_context:
                    session_id = qa.get("context_session_id", "")
                    if session_id in session_dict:
                        session = session_dict[session_id]
                        dialogue_history = []
                        for turn in session.get("turns", []):
                            dialogue_history.append({
                                "speaker": turn.get("speaker", ""),
                                "message": turn.get("message", "")
                            })
                        context["dialogue_history"] = dialogue_history
                        context["session_topic"] = session.get("topic", "")
                        context["session_topic_description"] = session.get("topic_description", "")
                
                # Get question_type directly from dataset
                question_type = get_question_type_from_data(qa)
                
                queries.append({
                    "user_id": persona.get("user_id", "unknown"),
                    "query": qa.get("query", ""),
                    "expected_response": qa.get("response", ""),
                    "question_type": question_type,  # Directly read from dataset
                    "context": context
                })
        
        return queries
    
# ==================== Question Type Classification ====================

# Valid question types from dataset V2
VALID_QUESTION_TYPES = ["summary", "planning", "investment_advice", "preference", "calculation"]


def get_question_type_from_data(qa: Dict[str, Any]) -> str:
    """Get question_type directly from the dataset qa_pair.
    
    Returns the question_type if valid, otherwise 'general' as fallback.
    """
    qtype = qa.get("question_type", "general")
    # Validate against known types
    if qtype in VALID_QUESTION_TYPES:
        return qtype
    return "general"


# Question type keywords for classification (fallback only)
QUESTION_TYPE_KEYWORDS = {
    "calculation": ["how much", "amount", "calculate", "percentage", "percent", "$", "dollar", "cost", "price", "number", "多少", "计算", "金额", "人民币"],
    "recommendation": ["should", "recommend", "best", "advice", "suggestion", "which", "ought to", "must", "应该", "建议", "推荐", "哪个", "最好"],
    "explanation": ["what is", "explain", "difference", "define", "meaning", "tell me about", "describe", "how does", "什么是", "解释", "区别", "定义", "说明"],
    "temporal": ["when", "timeline", "how long", "age", "year", "month", "frequency", "date", "period", "什么时候", "多久", "时间", "年龄", "年份"],
    "possibility": ["can i", "able", "possible", "may", "could i", "allow", "will i", "可以", "能否", "可能", "允许"],
    "general": []  # Default type
}


def classify_question(query: str) -> str:
    """Classify a query into a question type based on keywords (fallback only)."""
    query_lower = query.lower()
    
    for qtype, keywords in QUESTION_TYPE_KEYWORDS.items():
        if qtype == "general":
            continue
        for keyword in keywords:
            if keyword in query_lower:
                return qtype
    
    return "general"


def calculate_metrics(results):
    if not results:
        return {
            "count": 0,
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "avg_bleu": 0.0,
            "avg_f1": 0.0,
            "avg_exact_match": 0.0,
            "avg_llm_overall": 0.0,
            "avg_llm_relevance": 0.0,
            "avg_llm_accuracy": 0.0,
            "avg_llm_personalization": 0.0,
            "avg_llm_helpfulness": 0.0
        }
    
    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    
    avg_latency = sum(r.get("latency_seconds", 0) for r in results) / total
    avg_bleu = sum(r.get("bleu_score", 0) for r in results) / total
    avg_f1 = sum(r.get("f1_score", 0) for r in results) / total
    avg_exact_match = sum(r.get("exact_match", 0) for r in results) / total
    
    # LLM evaluation metrics
    llm_results = [r for r in results if r.get("llm_overall_score", 0) > 0]
    if llm_results:
        llm_total = len(llm_results)
        avg_llm_overall = sum(r.get("llm_overall_score", 0) for r in llm_results) / llm_total
        avg_llm_relevance = sum(r.get("llm_relevance", 0) for r in llm_results) / llm_total
        avg_llm_accuracy = sum(r.get("llm_accuracy", 0) for r in llm_results) / llm_total
        avg_llm_personalization = sum(r.get("llm_personalization", 0) for r in llm_results) / llm_total
        avg_llm_helpfulness = sum(r.get("llm_helpfulness", 0) for r in llm_results) / llm_total
    else:
        avg_llm_overall = 0.0
        avg_llm_relevance = 0.0
        avg_llm_accuracy = 0.0
        avg_llm_personalization = 0.0
        avg_llm_helpfulness = 0.0
    
    return {
        "count": total,
        "success_rate": success_count / total if total > 0 else 0.0,
        "avg_latency": avg_latency,
        "avg_bleu": avg_bleu,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_exact_match,
        "avg_llm_overall": avg_llm_overall,
        "avg_llm_relevance": avg_llm_relevance,
        "avg_llm_accuracy": avg_llm_accuracy,
        "avg_llm_personalization": avg_llm_personalization,
        "avg_llm_helpfulness": avg_llm_helpfulness
    }


def analyze_by_question_type(results):
    """Analyze results grouped by question type.
    
    Uses question_type directly from the dataset results.
    """
    from collections import defaultdict
    
    # Group by question type directly from data
    type_groups = defaultdict(list)
    for r in results:
        qtype = r.get("question_type", "general")
        type_groups[qtype].append(r)
    
    # Calculate metrics per type
    type_metrics = {}
    for qtype, type_results in type_groups.items():
        type_metrics[qtype] = calculate_metrics(type_results)
    
    return type_metrics


def print_classification_report(type_metrics, overall_metrics):
    """Print the per-question-type evaluation report."""
    print("\n" + "=" * 80)
    print("FINANCE DIALOGUE DATASET - EVALUATION REPORT BY QUESTION TYPE")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define column widths
    col_type = 18
    col_count = 8
    col_success = 10
    col_latency = 10
    col_bleu = 10
    col_f1 = 10
    col_exact = 10
    col_llm = 10
    
    # Header
    header = f"{'Question Type':<{col_type}}{'Count':>{col_count}}{'Success%':>{col_success}}{'Latency':>{col_latency}}{'BLEU':>{col_bleu}}{'F1':>{col_f1}}{'Exact':>{col_exact}}{'LLM':>{col_llm}}"
    print(header)
    print("-" * 80)
    
    # Question type order (V2 dataset types)
    type_order = ["summary", "planning", "investment_advice", "preference", "calculation"]
    
    # Print each question type
    for qtype in type_order:
        if qtype not in type_metrics:
            continue
        
        m = type_metrics[qtype]
        row = f"{qtype.capitalize().replace('_', ' '):<{col_type}}{m['count']:>{col_count}}{m['success_rate']*100:>8.1f}%{m['avg_latency']:>8.2f}s{m['avg_bleu']:>9.4f}{m['avg_f1']:>9.4f}{m['avg_exact_match']:>9.4f}{m['avg_llm_overall']:>9.1f}"
        print(row)
    
    print("-" * 80)
    
    # Overall row
    row = f"{'OVERALL':<{col_type}}{overall_metrics['count']:>{col_count}}{overall_metrics['success_rate']*100:>8.1f}%{overall_metrics['avg_latency']:>8.2f}s{overall_metrics['avg_bleu']:>9.4f}{overall_metrics['avg_f1']:>9.4f}{overall_metrics['avg_exact_match']:>9.4f}{overall_metrics['avg_llm_overall']:>9.1f}"
    print(row)
    print()
    
    # Detailed LLM metrics section
    print("\n" + "=" * 80)
    print("DETAILED LLM EVALUATION METRICS BY QUESTION TYPE")
    print("=" * 80)
    
    header2 = f"{'Question Type':<{col_type}}{'Relevance':>12}{'Accuracy':>12}{'Personaliz.':>12}{'Helpful':>12}{'Overall':>12}"
    print(header2)
    print("-" * 72)
    
    for qtype in type_order:
        if qtype not in type_metrics:
            continue
        
        m = type_metrics[qtype]
        row = f"{qtype.capitalize().replace('_', ' '):<{col_type}}{m['avg_llm_relevance']:>11.1f}{m['avg_llm_accuracy']:>11.1f}{m['avg_llm_personalization']:>11.1f}{m['avg_llm_helpfulness']:>11.1f}{m['avg_llm_overall']:>11.1f}"
        print(row)
    
    print("-" * 72)
    
    # Overall detailed
    row = f"{'OVERALL':<{col_type}}{overall_metrics['avg_llm_relevance']:>11.1f}{overall_metrics['avg_llm_accuracy']:>11.1f}{overall_metrics['avg_llm_personalization']:>11.1f}{overall_metrics['avg_llm_helpfulness']:>11.1f}{overall_metrics['avg_llm_overall']:>11.1f}"
    print(row)
    print()
    
    class LLMEvaluator:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate_response(self, *args, **kwargs):
            return {"success": False}


# ==================== User Persona Conversion ====================

def persona_to_user_persona(persona_dict: Dict) -> UserPersona:
    """Convert dataset persona to FinMem UserPersona"""
    
    # Map risk tolerance
    risk_map = {
        "low": "conservative",
        "medium": "moderate", 
        "high": "aggressive",
        "conservative": "conservative",
        "moderate": "moderate",
        "aggressive": "aggressive"
    }
    
    risk = persona_dict.get("risk_tolerance", "moderate")
    risk_tolerance = risk_map.get(risk.lower(), "moderate")
    
    # Parse financial goals
    financial_goals = persona_dict.get("financial_goals", [])
    if isinstance(financial_goals, str):
        financial_goals = [financial_goals]
    
    # Create UserPersona
    return UserPersona(
        user_id=persona_dict.get("user_id", "unknown"),
        name=persona_dict.get("name", "User"),
        age=persona_dict.get("age", 30),
        occupation=persona_dict.get("occupation", "Unknown"),
        income_level=persona_dict.get("income_level", "Unknown"),
        family_status=persona_dict.get("family_status", "Unknown"),
        risk_tolerance=risk_tolerance,
        financial_goals=financial_goals,
        # Default time perception values
        perceived_short_term_days=90,
        perceived_medium_term_days=365,
        perceived_long_term_days=1095,
        investment_cycle_days=365
    )


def timeline_to_events(timeline: List[Dict]) -> List[FinancialEvent]:
    """Convert timeline to FinancialEvent list
    
    Handles different timestamp formats:
    - %Y-%m-%d (e.g., "2001-06-15")
    - %Y%m%d%H%M (e.g., "202501010000")
    - %Y%m%d (e.g., "20250101")
    """
    
    def parse_timestamp(ts: str) -> str:
        """Parse timestamp to standard format %Y%m%d%H%M"""
        if not ts:
            return "202501010000"
        
        # Try different formats
        formats = ["%Y-%m-%d", "%Y%m%d%H%M", "%Y%m%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]
        
        for fmt in formats:
            try:
                from datetime import datetime
                dt = datetime.strptime(ts, fmt)
                return dt.strftime("%Y%m%d%H%M")
            except ValueError:
                continue
        
        # If all formats fail, return default
        print(f"Warning: Could not parse timestamp '{ts}', using default")
        return "202501010000"
    
    events = []
    for i, item in enumerate(timeline):
        # Determine event type
        event_type = "expense"
        impact = "neutral"
        
        if "income" in item.get("description", "").lower():
            event_type = "income"
            impact = "positive"
        elif "invest" in item.get("description", "").lower():
            event_type = "investment"
            impact = "positive"
        
        # Parse timestamp to standard format
        timestamp = parse_timestamp(item.get("timestamp", ""))
        
        event = FinancialEvent(
            event_id=f"evt_{i}",
            timestamp=timestamp,
            event_type=event_type,
            amount=float(item.get("amount", 0)),
            category=item.get("category", "General"),
            description=item.get("description", ""),
            impact=impact
        )
        events.append(event)
    
    return events


# ==================== FinMem LLM Wrapper ====================

class FinMemLLM:
    """
    Wrapper class to use FinMemAgent in the test framework
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        persona: Dict,
        timeline: List[Dict],
        enable_memory: bool = True,
        enable_skill_routing: bool = True
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.enable_memory = enable_memory
        self.enable_skill_routing = enable_skill_routing
        
        # Convert persona
        self.user_persona = persona_to_user_persona(persona)
        
        # Create FinMem agent
        llm_config = {
            "provider": "openai",
            "model": model,
            "api_key": api_key,
            "base_url": base_url
        }
        
        self.agent = FinMemAgent(
            user_persona=self.user_persona,
            llm_config=llm_config,
            enable_skill_routing=enable_skill_routing
        )
        
        # Add timeline events to memory
        if timeline:
            events = timeline_to_events(timeline)
            for event in events:
                self.agent.add_financial_event(event)
        
        # Add preference memories based on persona
        if persona.get("risk_tolerance"):
            self.agent.add_memory(
                content=f"用户风险偏好: {persona.get('risk_tolerance')}",
                memory_type="preference",
                importance=1.5,
                keywords=["风险偏好", persona.get("risk_tolerance", "")]
            )
        
        if persona.get("financial_goals"):
            self.agent.add_memory(
                content=f"用户财务目标: {', '.join(persona.get('financial_goals', []))}",
                memory_type="goal",
                importance=1.5,
                keywords=persona.get("financial_goals", [])
            )
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate response using FinMem agent"""
        
        if not self.enable_memory:
            # Fallback: direct LLM call without memory
            return self._direct_generate(prompt, context)
        
        try:
            # Use FinMem agent to process query
            result = self.agent.process_query(prompt)
            return result.get("response", "")
        except Exception as e:
            print(f"Error in FinMem generate: {e}")
            return f"Error: {str(e)}"
    
    def _direct_generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Direct LLM call without memory"""
        
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Build system prompt
        persona = context.get("persona", {})
        system_prompt = f"""You are a personalized financial advisor assistant.
Your client's name is {persona.get('name', 'User')}.
Age: {persona.get('age', 'N/A')}, Occupation: {persona.get('occupation', 'N/A')},
Income: {persona.get('income_level', 'N/A')}, Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}"""
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


# ==================== Model Configurations ====================

MODEL_CONFIGS = {
    # OpenAI models
    "gpt-5-mini": {
        "provider": "openai",
        "base_url": "https://api.vveai.com/v1"
    },
    "gpt-5": {
        "provider": "openai", 
        "base_url": "https://api.vveai.com/v1"
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "base_url": "https://api.vveai.com/v1"
    },
    "gpt-4o": {
        "provider": "openai",
        "base_url": "https://api.vveai.com/v1"
    },
    
    # Qwen models
    "qwen-turbo": {
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "qwen-plus": {
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    "qwen-max": {
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    
    # DeepSeek models
    "deepseek-chat": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1"
    },
    "deepseek-coder": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1"
    },
    
    # Anthropic models (if supported)
    "claude-3-opus": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1"
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1"
    },
    
    # Local/Ollama models
    "llama2": {
        "provider": "local",
        "base_url": "http://localhost:11434"
    },
    "llama3": {
        "provider": "local",
        "base_url": "http://localhost:11434"
    },
    "qwen2.5": {
        "provider": "local",
        "base_url": "http://localhost:11434"
    },
}


def get_model_config(model_name: str) -> Dict:
    """Get default config for a model, can be overridden by CLI args"""
    return MODEL_CONFIGS.get(model_name, {"provider": "openai", "base_url": "https://api.vveai.com/v1"})


# ==================== LLM Evaluator (from test_class.py) ====================

class LLMEvaluator:
    """LLM-based evaluator using GPT-4o-mini for quality assessment"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.vveai.com/v1"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gpt-4o-mini"
    
    def evaluate_response(
        self, 
        query: str, 
        expected_response: str, 
        actual_response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate response quality using LLM"""
        
        persona = context.get("persona", {})
        persona_info = f"User: {persona.get('name', 'User')}, Age: {persona.get('age', 'N/A')}, "
        persona_info += f"Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}"
        
        prompt = f"""You are an expert evaluator for a financial advisory assistant.
Evaluate the quality of the AI assistant's response compared to the expected response.

Context (persona/profile):
{persona_info}

User Query: {query}

Expected Response: {expected_response}

Actual Response: {actual_response}

EVALUATION PRINCIPLES (moderately strict):
- Use the Expected Response as the primary reference. The best answers will closely match its meaning and key points.
- Allow small wording differences and minor reordering IF the meaning stays the same.
- Do NOT give full credit for generic “reasonable” advice that fails to cover the key points in the Expected Response.
- Penalize for missing any major elements from the Expected Response (key steps, constraints, rationale, or any required numbers).
- Penalize for adding substantial new recommendations, assumptions, or numbers that are not supported by the Expected Response (minor helpful clarifications are allowed if they do not change the plan).
- If the Expected Response includes specific numbers/ranges/options, the Actual Response should match them. Small rounding differences are acceptable unless they change the decision.
- If this is multiple-choice and the Expected Response implies a specific option, the Actual Response must identify the same option to score well on accuracy.

Score the actual response on a scale of 0-100 for each criterion:
1. Relevance (0-100): Addresses the user’s query and covers the same core topics as the expected response.
2. Accuracy (0-100): Financial statements and conclusions align with the expected response; numeric details align when present.
3. Personalization (0-100): Incorporates the user’s situation to a similar degree and in a similar way as the expected response.
4. Helpfulness (0-100): Comparable actionability/clarity to the expected response without drifting into a different plan.

Compute:
- overall_score = round((relevance + accuracy + personalization + helpfulness) / 4)

Provide your evaluation in the following JSON format:
{{
  "relevance": <score>,
  "accuracy": <score>,
  "personalization": <score>,
  "helpfulness": <score>,
  "overall_score": <average>,
  "reason": "<brief explanation focusing on the biggest match/mismatch vs Expected Response>"
}}

Only respond with valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON from response
            try:
                import re
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    eval_result = json.loads(json_match.group())
                    return {
                        "relevance": eval_result.get("relevance", 0),
                        "accuracy": eval_result.get("accuracy", 0),
                        "personalization": eval_result.get("personalization", 0),
                        "helpfulness": eval_result.get("helpfulness", 0),
                        "overall_score": eval_result.get("overall_score", 0),
                        "reason": eval_result.get("reason", ""),
                        "success": True
                    }
            except json.JSONDecodeError:
                pass
            
            # Fallback: parse scores from text
            return self._parse_evaluation(content)
            
        except Exception as e:
            return {
                "relevance": 0,
                "accuracy": 0,
                "personalization": 0,
                "helpfulness": 0,
                "overall_score": 0,
                "reason": f"Error: {str(e)}",
                "success": False
            }
    
    def _parse_evaluation(self, text: str) -> Dict[str, Any]:
        """Parse evaluation scores from text"""
        import re
        scores = {}
        
        for key in ["relevance", "accuracy", "personalization", "helpfulness", "overall_score"]:
            match = re.search(rf'{key}["\s:]+(\d+)', text, re.IGNORECASE)
            if match:
                scores[key] = int(match.group(1))
            else:
                scores[key] = 0
        
        return {
            "relevance": scores.get("relevance", 0),
            "accuracy": scores.get("accuracy", 0),
            "personalization": scores.get("personalization", 0),
            "helpfulness": scores.get("helpfulness", 0),
            "overall_score": scores.get("overall_score", 0),
            "reason": "Parsed from text",
            "success": True
        }


# ==================== Test Functions ====================

def run_tests(
    llm: FinMemLLM,
    queries: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
    eval_llm: LLMEvaluator = None,
    enable_llm_eval: bool = False,
    enable_classification: bool = True
) -> List[Dict]:
    """Run tests on all queries
    
    Args:
        llm: The FinMemLLM to test
        queries: List of test queries
        model_name: Name of the model being tested
        output_dir: Directory to save results
        eval_llm: Optional LLM evaluator for quality assessment
        enable_llm_eval: Whether to enable LLM-based evaluation
        enable_classification: Whether to classify questions by type
    """
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with FinMem Agent")
    print(f"LLM Evaluation: {'Enabled' if enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"{'='*60}")
    
    for i, query_data in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query_data['query'][:50]}...")
        
        start_time = time.time()
        
        # Generate response using FinMem
        actual_response = llm.generate(
            query_data["query"],
            query_data["context"]
        )
        
        latency = time.time() - start_time
        
        # Calculate metrics
        expected_response = query_data["expected_response"]
        bleu = calculate_bleu_score(expected_response, actual_response) if actual_response else 0.0
        f1 = calculate_f1_score(expected_response, actual_response) if actual_response else 0.0
        exact_match = calculate_exact_match(expected_response, actual_response) if actual_response else 0.0
        
        # LLM evaluation (optional, can be slow)
        llm_relevance = 0.0
        llm_accuracy = 0.0
        llm_personalization = 0.0
        llm_helpfulness = 0.0
        llm_overall_score = 0.0
        llm_reason = ""
        
        if enable_llm_eval and eval_llm and not actual_response.startswith("Error:"):
            print(f"    Running LLM evaluation...")
            llm_eval_result = eval_llm.evaluate_response(
                query_data["query"],
                expected_response,
                actual_response,
                query_data["context"]
            )
            llm_relevance = llm_eval_result.get("relevance", 0)
            llm_accuracy = llm_eval_result.get("accuracy", 0)
            llm_personalization = llm_eval_result.get("personalization", 0)
            llm_helpfulness = llm_eval_result.get("helpfulness", 0)
            llm_overall_score = llm_eval_result.get("overall_score", 0)
            llm_reason = llm_eval_result.get("reason", "")
        
        result = {
            "model_name": model_name,
            "user_id": query_data["user_id"],
            "query": query_data["query"],
            "expected_response": expected_response,
            "actual_response": actual_response,
            "question_type": query_data.get("question_type", "general"),  # Include question type
            "latency_seconds": latency,
            "timestamp": datetime.now().isoformat(),
            "success": True if not actual_response.startswith("Error:") else False,
            "bleu_score": bleu,
            "f1_score": f1,
            "exact_match": exact_match,
            # LLM evaluation results
            "llm_relevance": llm_relevance,
            "llm_accuracy": llm_accuracy,
            "llm_personalization": llm_personalization,
            "llm_helpfulness": llm_helpfulness,
            "llm_overall_score": llm_overall_score,
            "llm_evaluation_reason": llm_reason
        }
        
        results.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(queries)} completed")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{model_name.replace(' ', '_').replace('/', '_')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    metrics = calculate_metrics(results)
    print(f"\nSummary for {model_name}:")
    print(f"  - Total queries: {metrics['count']}")
    print(f"  - Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"  - Avg latency: {metrics['avg_latency']:.2f}s")
    print(f"  - Avg BLEU: {metrics['avg_bleu']:.4f}")
    print(f"  - Avg F1: {metrics['avg_f1']:.4f}")
    
    # Print LLM evaluation summary if enabled
    if enable_llm_eval and eval_llm:
        print(f"  - LLM Avg Overall: {metrics.get('avg_llm_overall', 0):.2f}")
        print(f"  - LLM Avg Relevance: {metrics.get('avg_llm_relevance', 0):.2f}")
        print(f"  - LLM Avg Accuracy: {metrics.get('avg_llm_accuracy', 0):.2f}")
        print(f"  - LLM Avg Personalization: {metrics.get('avg_llm_personalization', 0):.2f}")
        print(f"  - LLM Avg Helpfulness: {metrics.get('avg_llm_helpfulness', 0):.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test FinMem Agent on finance dialogue dataset with LLM evaluation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with OpenAI (basic)
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --dataset ./output/finance_dialogue.json

  # Test with Qwen
  python test_finmem.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

  # Test with DeepSeek
  python test_finmem.py --model deepseek-chat --api-key YOUR_KEY

  # Test with memory disabled (direct LLM)
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --no-memory

  # Test specific number of queries
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --max-queries 10

  # ===== LLM Evaluation Examples =====
  
  # Enable LLM evaluation (uses GPT-4o-mini to evaluate responses)
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval
  
  # LLM evaluation with custom eval API key
  python test_finmem.py --model qwen-plus --api-key YOUR_KEY --enable-llm-eval --eval-api-key YOUR_EVAL_KEY
  
  # LLM evaluation with custom eval base URL
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval --eval-base-url https://api.vveai.com/v1
  
  # Full example with all options
  python test_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval --eval-api-key YOUR_EVAL_KEY --max-queries 20
        """
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the model (default: from environment variable)")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for API")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str,
                       default="./output/finance_dialogue.json",
                       help="Path to finance dialogue dataset")
    parser.add_argument("--output-dir", type=str,
                       default="./test_results_finmem",
                       help="Output directory for results")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to test")
    parser.add_argument("--max-per-user", type=int, default=1000,
                       help="Max queries per user")
    
    # FinMem options
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable memory, use direct LLM call")
    parser.add_argument("--disable-skill-routing", action="store_true",
                       help="Disable skill routing, use general prompt")
    parser.add_argument("--include-dialogue-context", action="store_true",
                       help="Include dialogue history from sessions")
    
    # LLM Evaluation options
    parser.add_argument("--enable-llm-eval", action="store_true",
                       help="Enable LLM-based evaluation using GPT-4o-mini (slower but more accurate)")
    parser.add_argument("--eval-api-key", type=str, default=None,
                       help="API key for LLM evaluation (defaults to OPENAI_API_KEY)")
    parser.add_argument("--eval-base-url", type=str, default=None,
                       help="Base URL for LLM evaluation (default: https://api.vveai.com/v1)")
    
    args = parser.parse_args()
    
    # ==================== Get API Keys ====================
    
    # Model API key
    api_key = args.api_key
    if not api_key:
        if "qwen" in args.model.lower() or "dashscope" in (args.base_url or ""):
            api_key = os.environ.get("DASHSCOPE_API_KEY")
        elif "deepseek" in args.model.lower():
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No API key provided. Use --api-key or set environment variable.")
        print("  - For OpenAI models: set OPENAI_API_KEY")
        print("  - For Qwen models: set DASHSCOPE_API_KEY")
        print("  - For DeepSeek models: set DEEPSEEK_API_KEY")
        sys.exit(1)
    
    # Get base URL
    base_url = args.base_url
    if not base_url:
        config = get_model_config(args.model)
        base_url = config.get("base_url", "https://api.vveai.com/v1")
    
    # ==================== Initialize LLM Evaluator ====================
    
    eval_llm = None
    if args.enable_llm_eval:
        eval_api_key = args.eval_api_key or os.environ.get("OPENAI_API_KEY")
        eval_base_url = args.eval_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        
        if eval_api_key:
            print(f"\nInitializing LLM evaluator with GPT-4o-mini...")
            print(f"  Eval API Base URL: {eval_base_url}")
            eval_llm = LLMEvaluator(api_key=eval_api_key, base_url=eval_base_url)
        else:
            print("\nWarning: No API key available for LLM evaluation. Disabling...")
            args.enable_llm_eval = False
    
    # ==================== Print Configuration ====================
    
    print(f"\n{'='*60}")
    print("FinMem Agent Test Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Base URL: {base_url}")
    print(f"API Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else api_key}")
    print(f"Memory: {'Enabled' if not args.no_memory else 'Disabled'}")
    print(f"LLM Evaluation: {'Enabled' if args.enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"Dataset: {args.dataset}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    try:
        data = load_dataset(args.dataset)
        print(f"Loaded {len(data)} user profiles")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please generate the dataset first or provide correct path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Extract queries
    queries = extract_test_queries(
        data,
        max_per_user=args.max_per_user,
        include_dialogue_context=args.include_dialogue_context
    )
    
    if args.max_queries:
        queries = queries[:args.max_queries]
    
    print(f"Testing {len(queries)} queries")
    
    # Group queries by user for per-user FinMem agent
    user_queries = {}
    for q in queries:
        user_id = q["user_id"]
        if user_id not in user_queries:
            user_queries[user_id] = []
        user_queries[user_id].append(q)
    
    print(f"Testing with {len(user_queries)} unique users")
    
    # ==================== Run Tests ====================
    
    all_results = []
    
    for user_id, user_qs in user_queries.items():
        # Get user data from original dataset
        user_data = None
        for d in data:
            if d.get("persona", {}).get("user_id") == user_id:
                user_data = d
                break
        
        if not user_data:
            continue
        
        persona = user_data.get("persona", {})
        timeline = user_data.get("timeline", [])
        
        # Create FinMem LLM for this user
        finmem_llm = FinMemLLM(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            persona=persona,
            timeline=timeline,
            enable_memory=not args.no_memory
        )
        
        # Run tests for this user (with LLM evaluation if enabled)
        user_results = run_tests(
            finmem_llm,
            user_qs,
            f"{args.model}_{user_id}",
            args.output_dir,
            eval_llm=eval_llm,
            enable_llm_eval=args.enable_llm_eval,
            enable_classification=True
        )
        
        all_results.extend(user_results)
    
    # ==================== Print Overall Summary ====================
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    overall_metrics = calculate_metrics(all_results)
    print(f"Total queries: {overall_metrics['count']}")
    print(f"Success rate: {overall_metrics['success_rate']*100:.1f}%")
    print(f"Avg latency: {overall_metrics['avg_latency']:.2f}s")
    print(f"Avg BLEU: {overall_metrics['avg_bleu']:.4f}")
    print(f"Avg F1: {overall_metrics['avg_f1']:.4f}")
    print(f"Avg Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    
    # Print LLM evaluation summary if enabled
    if args.enable_llm_eval and eval_llm:
        print(f"\nLLM Evaluation Metrics (Overall):")
        print(f"  - Avg Overall Score: {overall_metrics['avg_llm_overall']:.2f}")
        print(f"  - Avg Relevance: {overall_metrics['avg_llm_relevance']:.2f}")
        print(f"  - Avg Accuracy: {overall_metrics['avg_llm_accuracy']:.2f}")
        print(f"  - Avg Personalization: {overall_metrics['avg_llm_personalization']:.2f}")
        print(f"  - Avg Helpfulness: {overall_metrics['avg_llm_helpfulness']:.2f}")
    
    # ==================== Analyze by Question Type ====================
    
    print(f"\n{'='*60}")
    print("ANALYSIS BY QUESTION TYPE")
    print(f"{'='*60}")
    
    # Classify all results by question type
    type_metrics = analyze_by_question_type(all_results)
    
    # Print classification report
    print_classification_report(type_metrics, overall_metrics)
    
    # Save overall summary
    summary_file = os.path.join(args.output_dir, "summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": base_url,
        "memory_enabled": not args.no_memory,
        "llm_evaluation_enabled": args.enable_llm_eval,
        "dataset": args.dataset,
        "num_queries": len(all_results),
        "metrics": overall_metrics,
        "metrics_by_question_type": type_metrics
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
