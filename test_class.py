"""
Finance Dialogue Dataset - LLM Testing Script with Per-Question-Type Analysis
Tests different LLMs (Llama, Qwen, DeepSeek, GPT) on finance dialogue dataset

Evaluation Metrics:
- BLEU: Text similarity score
- F1: Word-level precision/recall
- LLM Evaluation: Use GPT-4o-mini to evaluate response quality
- Per-Question-Type Analysis: Classify queries and report metrics by type

Question Types: calculation, recommendation, explanation, temporal, possibility, general
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import re
from collections import defaultdict

# Try to import required libraries
try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not installed. GPT tests will not work.")

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Some API tests may not work.")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("Warning: nltk not installed. BLEU score will not be available.")


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


def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference and hypothesis"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        # Tokenize
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0
        
        # Use smoothing for short sentences
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        # Fallback: simple word overlap
        ref_set = set(reference.lower().split())
        hyp_set = set(hypothesis.lower().split())
        if len(ref_set) == 0:
            return 0.0
        overlap = len(ref_set & hyp_set)
        return overlap / len(ref_set)


def calculate_f1_score(reference: str, hypothesis: str) -> float:
    """Calculate F1 score between reference and hypothesis"""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0
    
    # Calculate precision, recall, F1
    tp = len(ref_tokens & hyp_tokens)
    precision = tp / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall = tp / len(ref_tokens) if len(ref_tokens) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_exact_match(reference: str, hypothesis: str) -> float:
    """Calculate exact match ratio"""
    ref_clean = reference.lower().strip()
    hyp_clean = hypothesis.lower().strip()
    return 1.0 if ref_clean == hyp_clean else 0.0


class LLMEvaluator:
    """LLM-based evaluator using GPT-4o-mini"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.vveai.com/v1"):
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

{persona_info}

User Query: {query}

Expected Response: {expected_response}

Actual Response: {actual_response}

Evaluate the actual response on a scale of 0-100 for each criterion:
1. Relevance (0-100): How relevant is the response to the user's query?
2. Accuracy (0-100): How accurate is the financial information provided?
3. Personalization (0-100): How well does it consider the user's profile?
4. Helpfulness (0-100): How helpful is the overall response?

Provide your evaluation in the following JSON format:
{{
    "relevance": <score>,
    "accuracy": <score>,
    "personalization": <score>,
    "helpfulness": <score>,
    "overall_score": <average>,
    "reason": "<brief explanation>"
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
                # Try to extract JSON
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
            
            # Fallback: try to parse scores from text
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


@dataclass
class TestResult:
    """Result of a single test"""
    model_name: str
    user_id: str
    session_id: str
    query: str
    expected_response: str
    actual_response: str
    latency_seconds: float
    timestamp: str
    success: bool = True
    error_message: Optional[str] = None
    # New evaluation metrics
    bleu_score: float = 0.0
    f1_score: float = 0.0
    exact_match: float = 0.0
    # Question type classification
    question_type: str = "general"
    # LLM evaluation
    llm_relevance: float = 0.0
    llm_accuracy: float = 0.0
    llm_personalization: float = 0.0
    llm_helpfulness: float = 0.0
    llm_overall_score: float = 0.0
    llm_evaluation_reason: str = ""


@dataclass
class ModelConfig:
    """Configuration for an LLM"""
    name: str
    provider: str  # openai, anthropic, local, etc.
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 500


class BaseLLM:
    """Base class for LLM implementations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a response"""
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    """OpenAI GPT implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(context)},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        persona = context.get("persona", {})
        timeline = context.get("timeline", [])
        
        prompt = "You are a personalized financial advisor assistant. "
        prompt += f"Your client's name is {persona.get('name', 'User')}. "
        """
        prompt += f"Age: {persona.get('age', 'N/A')}, "
        prompt += f"Occupation: {persona.get('occupation', 'N/A')}, "
        prompt += f"Income: {persona.get('income_level', 'N/A')}, "
        prompt += f"Family Status: {persona.get('family_status', 'N/A')}, "
        prompt += f"Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}, "
        prompt += f"Financial Goals: {', '.join(persona.get('financial_goals', []))}. "
        
        if timeline:
            prompt += "\nRecent financial events:\n"
            for event in timeline[-5:]:  # Last 5 events
                prompt += f"- {event.get('timestamp')}: {event.get('description')} ({event.get('impact')})\n"
        
        # Include dialogue history if available
        dialogue_history = context.get("dialogue_history", [])
        if dialogue_history:
            prompt += "\nConversation history:\n"
            for turn in dialogue_history:
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                prompt += f"{speaker}: {message}\n"
        
        # Include session topic info if available
        session_topic = context.get("session_topic", "")
        session_topic_desc = context.get("session_topic_description", "")
        if session_topic:
            prompt += f"\nCurrent topic: {session_topic}"
            if session_topic_desc:
                prompt += f" - {session_topic_desc}"
            prompt += "\n"
        """
        prompt += "\nProvide helpful, personalized financial advice based on the client's profile and history."
        return prompt


class QwenLLM(BaseLLM):
    """Qwen LLM implementation (compatible with OpenAI API)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.qwen.com/v1"
        )
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(context)},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        persona = context.get("persona", {})
        timeline = context.get("timeline", [])
        
        prompt = "You are a personalized financial advisor assistant. "
        """
        prompt += f"Your client's name is {persona.get('name', 'User')}. "
        prompt += f"Age: {persona.get('age', 'N/A')}, "
        prompt += f"Occupation: {persona.get('occupation', 'N/A')}, "
        prompt += f"Income: {persona.get('income_level', 'N/A')}, "
        prompt += f"Family Status: {persona.get('family_status', 'N/A')}, "
        prompt += f"Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}, "
        prompt += f"Financial Goals: {', '.join(persona.get('financial_goals', []))}. "
        
        if timeline:
            prompt += "\nRecent financial events:\n"
            for event in timeline[-5:]:
                prompt += f"- {event.get('timestamp')}: {event.get('description')} ({event.get('impact')})\n"
        
        # Include dialogue history if available
        dialogue_history = context.get("dialogue_history", [])
        if dialogue_history:
            prompt += "\nConversation history:\n"
            for turn in dialogue_history:
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                prompt += f"{speaker}: {message}\n"
        
        # Include session topic info if available
        session_topic = context.get("session_topic", "")
        session_topic_desc = context.get("session_topic_description", "")
        if session_topic:
            prompt += f"\nCurrent topic: {session_topic}"
            if session_topic_desc:
                prompt += f" - {session_topic_desc}"
            prompt += "\n"
        """
        prompt += "\nProvide helpful, personalized financial advice based on the client's profile and history."
        return prompt


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.deepseek.com/v1"
        )
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(context)},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        persona = context.get("persona", {})
        timeline = context.get("timeline", [])
        
        prompt = "You are a personalized financial advisor assistant. "
        """"""
        prompt += f"Your client's name is {persona.get('name', 'User')}. "
        prompt += f"Age: {persona.get('age', 'N/A')}, "
        prompt += f"Occupation: {persona.get('occupation', 'N/A')}, "
        prompt += f"Income: {persona.get('income_level', 'N/A')}, "
        prompt += f"Family Status: {persona.get('family_status', 'N/A')}, "
        prompt += f"Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}, "
        prompt += f"Financial Goals: {', '.join(persona.get('financial_goals', []))}. "
        
        if timeline:
            prompt += "\nRecent financial events:\n"
            for event in timeline[-5:]:
                prompt += f"- {event.get('timestamp')}: {event.get('description')} ({event.get('impact')})\n"
        
        # Include dialogue history if available
        dialogue_history = context.get("dialogue_history", [])
        if dialogue_history:
            prompt += "\nConversation history:\n"
            for turn in dialogue_history:
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                prompt += f"{speaker}: {message}\n"
        
        # Include session topic info if available
        session_topic = context.get("session_topic", "")
        session_topic_desc = context.get("session_topic_description", "")
        if session_topic:
            prompt += f"\nCurrent topic: {session_topic}"
            if session_topic_desc:
                prompt += f" - {session_topic_desc}"
            prompt += "\n"
        """"""
        prompt += "\nProvide helpful, personalized financial advice based on the client's profile and history."
        return prompt


class LlamaLLM(BaseLLM):
    """Local Llama implementation using transformers (GPU)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._tokenizer = None
        self._model = None
        self._device = None
        self.model_dir = "../TG-LLM/src/Llama-3.1-8B"
    
    def _load_model(self):
        """Lazy load the model on first use"""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading local Llama model from {self.model_dir} on {device}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self._model.eval()
            self._device = device
            print(f"Model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        try:
            self._load_model()
            
            # Build full prompt with system message and user input
            system_prompt = self._build_system_prompt(context)
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            
            # Tokenize and generate
            import torch
            inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    do_sample=True,
                    top_k=10,
                    top_p=0.9,
                    temperature=self.config.temperature,
                    repetition_penalty=1.15,
                    num_return_sequences=1,
                    eos_token_id=self._tokenizer.eos_token_id,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            return response[len(full_prompt):].strip()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        persona = context.get("persona", {})
        timeline = context.get("timeline", [])
        
        prompt = "You are a personalized financial advisor assistant. "
        prompt += f"Your client's name is {persona.get('name', 'User')}. "
        """
        prompt += f"Age: {persona.get('age', 'N/A')}, "
        prompt += f"Occupation: {persona.get('occupation', 'N/A')}, "
        prompt += f"Income: {persona.get('income_level', 'N/A')}, "
        prompt += f"Family Status: {persona.get('family_status', 'N/A')}, "
        prompt += f"Risk Tolerance: {persona.get('risk_tolerance', 'N/A')}, "
        prompt += f"Financial Goals: {', '.join(persona.get('financial_goals', []))}. "
        
        if timeline:
            prompt += "\nRecent financial events:\n"
            for event in timeline[-5:]:
                prompt += f"- {event.get('timestamp')}: {event.get('description')} ({event.get('impact')})\n"
        
        # Include dialogue history if available
        dialogue_history = context.get("dialogue_history", [])
        if dialogue_history:
            prompt += "\nConversation history:\n"
            for turn in dialogue_history:
                speaker = turn.get("speaker", "Unknown")
                message = turn.get("message", "")
                prompt += f"{speaker}: {message}\n"
        
        # Include session topic info if available
        session_topic = context.get("session_topic", "")
        session_topic_desc = context.get("session_topic_description", "")
        if session_topic:
            prompt += f"\nCurrent topic: {session_topic}"
            if session_topic_desc:
                prompt += f" - {session_topic_desc}"
            prompt += "\n"
        """
        prompt += "\nProvide helpful, personalized financial advice based on the client's profile and history."
        return prompt


def create_llm(config: ModelConfig) -> BaseLLM:
    """Factory function to create LLM instances"""
    provider = config.provider.lower()
    
    if provider == "openai" or config.model_id.startswith("gpt"):
        return OpenAILLM(config)
    elif provider == "qwen" or "qwen" in config.model_id.lower():
        return QwenLLM(config)
    elif provider == "deepseek" or "deepseek" in config.model_id.lower():
        return DeepSeekLLM(config)
    elif provider == "llama" or "llama" in config.model_id.lower() or provider == "local" or provider == "gpu":
        return LlamaLLM(config)
    else:
        # Default to OpenAI-compatible
        return OpenAILLM(config)


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load the finance dialogue dataset"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_test_queries(data: List[Dict[str, Any]], max_per_user: int = 3, 
                        include_dialogue_context: bool = False) -> List[Dict[str, Any]]:
    """Extract test queries from the dataset
    
    Args:
        data: The dataset containing user profiles with sessions and qa_pairs
        max_per_user: Maximum number of QA pairs to extract per user
        include_dialogue_context: If True, include full dialogue history from sessions
    """
    queries = []
    
    for user_data in data:
        persona = user_data.get("persona", {})
        timeline = user_data.get("timeline", [])
        sessions = user_data.get("sessions", [])
        qa_pairs = user_data.get("qa_pairs", [])
        
        # Build session lookup for dialogue context
        session_dict = {s.get("session_id"): s for s in sessions}
        
        # Take first few QA pairs as test queries
        for qa in qa_pairs[:max_per_user]:
            context = {
                "persona": persona,
                "timeline": timeline,
                "session_id": qa.get("context_session_id", "")
            }
            
            # Optionally add full dialogue context from the session
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
                    
                    # Also include session topic info
                    context["session_topic"] = session.get("topic", "")
                    context["session_topic_description"] = session.get("topic_description", "")
            
            # Get question_type directly from dataset
            question_type = get_question_type_from_data(qa)
            
            queries.append({
                "user_id": persona.get("user_id", "unknown"),
                "query": qa.get("query", ""),
                "expected_response": qa.get("response", ""),
                "question_type": question_type,  #直接从数据集读取
                "context": context
            })
    
    return queries


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate average metrics from results."""
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
    
    success_count = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
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


def analyze_by_question_type(results: List[Dict]) -> Dict[str, Dict]:
    """Analyze results grouped by question type.
    
    Uses question_type directly from the dataset results.
    """
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


def print_classification_report(type_metrics: Dict[str, Dict], overall_metrics: Dict[str, float]):
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


def run_tests(
    llm: BaseLLM,
    queries: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
    eval_llm: LLMEvaluator = None,
    enable_llm_eval: bool = False,
    enable_classification: bool = True
) -> List[TestResult]:
    """Run tests on all queries with the given LLM
    
    Args:
        llm: The LLM to test
        queries: List of test queries
        model_name: Name of the model being tested
        output_dir: Directory to save results
        eval_llm: Optional LLM evaluator for quality assessment
        enable_llm_eval: Whether to enable LLM-based evaluation
        enable_classification: Whether to classify questions by type
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"LLM Evaluation: {'Enabled' if enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"Question Classification: {'Enabled' if enable_classification else 'Disabled'}")
    print(f"{'='*60}")
    
    for i, query_data in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query_data['query'][:50]}...")
        
        start_time = time.time()
        actual_response = llm.generate(
            query_data["query"],
            query_data["context"]
        )
        latency = time.time() - start_time
        
        # Calculate BLEU and F1 scores
        expected_response = query_data["expected_response"]
        bleu = calculate_bleu_score(expected_response, actual_response) if actual_response else 0.0
        f1 = calculate_f1_score(expected_response, actual_response) if actual_response else 0.0
        exact_match = calculate_exact_match(expected_response, actual_response) if actual_response else 0.0
        
        # Get question type directly from dataset
        question_type = query_data.get("question_type", "general")
        
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
        
        result = TestResult(
            model_name=model_name,
            user_id=query_data["user_id"],
            session_id=query_data["context"].get("session_id", ""),
            query=query_data["query"],
            expected_response=expected_response,
            actual_response=actual_response,
            latency_seconds=latency,
            timestamp=datetime.now().isoformat(),
            success=True if not actual_response.startswith("Error:") else False,
            error_message=None if not actual_response.startswith("Error:") else actual_response,
            bleu_score=bleu,
            f1_score=f1,
            exact_match=exact_match,
            question_type=question_type,
            llm_relevance=llm_relevance,
            llm_accuracy=llm_accuracy,
            llm_personalization=llm_personalization,
            llm_helpfulness=llm_helpfulness,
            llm_overall_score=llm_overall_score,
            llm_evaluation_reason=llm_reason
        )
        
        results.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(queries)} completed")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{model_name.replace(' ', '_').replace('/', '_')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary with all metrics
    success_count = sum(1 for r in results if r.success)
    avg_latency = sum(r.latency_seconds for r in results) / len(results) if results else 0
    avg_bleu = sum(r.bleu_score for r in results) / len(results) if results else 0
    avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0
    
    print(f"\nSummary for {model_name}:")
    print(f"  - Total queries: {len(results)}")
    print(f"  - Successful: {success_count}")
    print(f"  - Failed: {len(results) - success_count}")
    print(f"  - Avg latency: {avg_latency:.2f}s")
    print(f"  - Avg BLEU: {avg_bleu:.4f}")
    print(f"  - Avg F1: {avg_f1:.4f}")
    
    if enable_llm_eval and eval_llm:
        avg_llm_overall = sum(r.llm_overall_score for r in results) / len(results) if results else 0
        avg_llm_relevance = sum(r.llm_relevance for r in results) / len(results) if results else 0
        avg_llm_accuracy = sum(r.llm_accuracy for r in results) / len(results) if results else 0
        avg_llm_personalization = sum(r.llm_personalization for r in results) / len(results) if results else 0
        avg_llm_helpfulness = sum(r.llm_helpfulness for r in results) / len(results) if results else 0
        
        print(f"  - LLM Avg Overall: {avg_llm_overall:.2f}")
        print(f"  - LLM Avg Relevance: {avg_llm_relevance:.2f}")
        print(f"  - LLM Avg Accuracy: {avg_llm_accuracy:.2f}")
        print(f"  - LLM Avg Personalization: {avg_llm_personalization:.2f}")
        print(f"  - LLM Avg Helpfulness: {avg_llm_helpfulness:.2f}")
    
    # Print per-question-type analysis
    if enable_classification:
        type_metrics = analyze_by_question_type([asdict(r) for r in results])
        overall_metrics = calculate_metrics([asdict(r) for r in results])
        print_classification_report(type_metrics, overall_metrics)
    
    return results


def print_sample_comparison(results: List[TestResult], num_samples: int = 3):
    """Print sample comparisons of expected vs actual responses"""
    print(f"\n{'='*60}")
    print("Sample Comparisons (Expected vs Actual)")
    print(f"{'='*60}")
    
    for i, result in enumerate(results[:num_samples]):
        print(f"\n[Query {i+1}]")
        print(f"User: {result.user_id}")
        print(f"Query: {result.query[:100]}...")
        print(f"Type: {result.question_type}")
        print(f"\nExpected: {result.expected_response[:200]}...")
        print(f"\nActual: {result.actual_response[:200]}...")
        print(f"Latency: {result.latency_seconds:.2f}s | Success: {result.success}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Test different LLMs on finance dialogue dataset with per-question-type analysis")
    parser.add_argument("--dataset", type=str, 
                       default="./output/finance_dialogue.json",
                       help="Path to the finance dialogue dataset")
    parser.add_argument("--output-dir", type=str,
                       default="./test_results",
                       help="Directory to save test results")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to test per model (default: all)")
    parser.add_argument("--all", action="store_true",
                       help="Test all queries in the dataset")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["gpt-4o-mini", "qwen-turbo", "deepseek-chat", "llama3"],
                       help="Models to test")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for models")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for API")
    parser.add_argument("--enable-llm-eval", action="store_true",
                       help="Enable LLM-based evaluation using GPT-4o-mini (slower but more accurate)")
    parser.add_argument("--eval-api-key", type=str, default=None,
                       help="API key for LLM evaluation (defaults to OPENAI_API_KEY)")
    parser.add_argument("--eval-base-url", type=str, default=None,
                       help="Base URL for LLM evaluation")
    parser.add_argument("--disable-classification", action="store_true",
                       help="Disable per-question-type classification and reporting")
    parser.add_argument("--include-dialogue-context", action="store_true",
                       help="Include full dialogue history from sessions in the context")
    
    args = parser.parse_args()
    
    # Initialize LLM evaluator if enabled
    eval_llm = None
    if args.enable_llm_eval:
        eval_api_key = args.eval_api_key or os.environ.get("OPENAI_API_KEY")
        eval_base_url = args.eval_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        
        if eval_api_key:
            print(f"\nInitializing LLM evaluator with GPT-4o-mini...")
            print(f"  API Base URL: {eval_base_url}")
            eval_llm = LLMEvaluator(api_key=eval_api_key, base_url=eval_base_url)
        else:
            print("\nWarning: No API key available for LLM evaluation. Disabling...")
            args.enable_llm_eval = False
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"Loaded {len(data)} user profiles")
    
    # Calculate total QA pairs in dataset
    total_qa = sum(len(u.get("qa_pairs", [])) for u in data)
    print(f"Total QA pairs in dataset: {total_qa}")
    
    # Extract test queries
    if args.all or args.max_queries is None:
        # Extract ALL queries from the dataset
        queries = extract_test_queries(data, max_per_user=999999, 
                                       include_dialogue_context=args.include_dialogue_context)  # Large number to get all
        if args.max_queries:
            queries = queries[:args.max_queries]
        print(f"Testing ALL queries: {len(queries)}")
    else:
        # Extract limited queries
        max_per_user = max(1, args.max_queries // len(data))
        queries = extract_test_queries(data, max_per_user=max_per_user,
                                       include_dialogue_context=args.include_dialogue_context)
        queries = queries[:args.max_queries]
        print(f"Testing {len(queries)} queries (max_per_user={max_per_user})")
    
    if args.include_dialogue_context:
        print("Dialogue context: ENABLED")
    else:
        print("Dialogue context: DISABLED (use --include-dialogue-context to enable)")
    
    # Define model configurations
    model_configs = {
        "gpt-5-mini": ModelConfig(
            name="GPT-5 Mini",
            provider="openai",
            model_id="gpt-5-mini",
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        ),
        "gpt-5": ModelConfig(
            name="GPT-5",
            provider="openai",
            model_id="gpt-5",
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        ),
        "gpt-4o-mini": ModelConfig(
            name="GPT-4o Mini",
            provider="openai",
            model_id="gpt-4o-mini",
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        ),
        "gpt-4o": ModelConfig(
            name="GPT-4o",
            provider="openai",
            model_id="gpt-4o",
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.vveai.com/v1")
        ),
        "qwen-turbo": ModelConfig(
            name="Qwen Turbo",
            provider="qwen",
            model_id="qwen-turbo",
            api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "qwen-plus": ModelConfig(
            name="Qwen Plus",
            provider="qwen",
            model_id="qwen-plus",
            api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "deepseek-chat": ModelConfig(
            name="DeepSeek Chat",
            provider="deepseek",
            model_id="deepseek-chat",
            api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        ),
        "llama3": ModelConfig(
            name="Llama 3",
            provider="local",
            model_id="llama3",
            base_url="http://localhost:11434"
        ),
        "llama3.1": ModelConfig(
            name="Llama 3.1",
            provider="local",
            model_id="llama3.1",
            base_url="http://localhost:11434"
        ),
        "llama-3.1-8b": ModelConfig(
            name="Llama 3.1 8B (GPU)",
            provider="gpu",
            model_id="llama3.1-8b",
            base_url="../TG-LLM/src/Llama-3.1-8B"
        )
    }
    
    # Run tests for each model
    all_results = {}
    enable_classification = not args.disable_classification
    
    for model_key in args.models:
        if model_key not in model_configs:
            print(f"\nSkipping unknown model: {model_key}")
            continue
        
        config = model_configs[model_key]
        
        # Check if API key is available
        if not config.api_key and config.provider != "gpu" and config.provider != "local":
            print(f"\nSkipping {config.name} - No API key available")
            continue
        
        try:
            llm = create_llm(config)
            # Pass LLM evaluator and classification flag to run_tests
            results = run_tests(
                llm, 
                queries, 
                config.name, 
                args.output_dir,
                eval_llm=eval_llm,
                enable_llm_eval=args.enable_llm_eval,
                enable_classification=enable_classification
            )
            all_results[config.name] = results
            
            # Print sample comparisons
            print_sample_comparison(results, num_samples=2)
            
        except Exception as e:
            print(f"\nError testing {model_key}: {str(e)}")
            continue
    
    # Print final summary with all metrics
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        success_count = sum(1 for r in results if r.success)
        avg_latency = sum(r.latency_seconds for r in results) / len(results) if results else 0
        avg_bleu = sum(r.bleu_score for r in results) / len(results) if results else 0
        avg_f1 = sum(r.f1_score for r in results) / len(results) if results else 0
        
        print(f"\n{model_name}:")
        print(f"  Success: {success_count:3d}/{len(results):3d} | Latency: {avg_latency:6.2f}s")
        print(f"  BLEU: {avg_bleu:.4f} | F1: {avg_f1:.4f}")
        
        if args.enable_llm_eval and eval_llm:
            avg_llm_overall = sum(r.llm_overall_score for r in results) / len(results) if results else 0
            avg_llm_relevance = sum(r.llm_relevance for r in results) / len(results) if results else 0
            avg_llm_accuracy = sum(r.llm_accuracy for r in results) / len(results) if results else 0
            avg_llm_personalization = sum(r.llm_personalization for r in results) / len(results) if results else 0
            avg_llm_helpfulness = sum(r.llm_helpfulness for r in results) / len(results) if results else 0
            
            print(f"  LLM Overall: {avg_llm_overall:.2f} | Relevance: {avg_llm_relevance:.2f} | Accuracy: {avg_llm_accuracy:.2f}")
            print(f"  LLM Personalization: {avg_llm_personalization:.2f} | Helpfulness: {avg_llm_helpfulness:.2f}")
        
        # Print per-question-type summary for this model
        if enable_classification:
            type_metrics = analyze_by_question_type([asdict(r) for r in results])
            print(f"\n  Per-Question-Type Performance:")
            for qtype in ["summary", "planning", "investment_advice", "preference", "calculation"]:
                if qtype in type_metrics:
                    m = type_metrics[qtype]
                    print(f"    {qtype.replace('_', ' ').capitalize():20s}: BLEU={m['avg_bleu']:.4f}, F1={m['avg_f1']:.4f}, LLM={m['avg_llm_overall']:.1f}, Count={m['count']}")
    
    # Save aggregated results with all metrics
    summary_file = os.path.join(args.output_dir, "summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "num_queries": len(queries),
        "llm_evaluation_enabled": args.enable_llm_eval,
        "classification_enabled": enable_classification,
        "results": {}
    }
    
    for model_name, results in all_results.items():
        summary["results"][model_name] = {
            "total": len(results),
            "success": sum(1 for r in results if r.success),
            "avg_latency": sum(r.latency_seconds for r in results) / len(results) if results else 0,
            "avg_bleu": sum(r.bleu_score for r in results) / len(results) if results else 0,
            "avg_f1": sum(r.f1_score for r in results) / len(results) if results else 0,
            "avg_exact_match": sum(r.exact_match for r in results) / len(results) if results else 0
        }
        
        # Add per-question-type metrics
        if enable_classification:
            type_metrics = analyze_by_question_type([asdict(r) for r in results])
            summary["results"][model_name]["by_question_type"] = type_metrics
        
        if args.enable_llm_eval and eval_llm:
            summary["results"][model_name]["llm_eval"] = {
                "avg_overall": sum(r.llm_overall_score for r in results) / len(results) if results else 0,
                "avg_relevance": sum(r.llm_relevance for r in results) / len(results) if results else 0,
                "avg_accuracy": sum(r.llm_accuracy for r in results) / len(results) if results else 0,
                "avg_personalization": sum(r.llm_personalization for r in results) / len(results) if results else 0,
                "avg_helpfulness": sum(r.llm_helpfulness for r in results) / len(results) if results else 0
            }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
