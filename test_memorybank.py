"""
Finance Dialogue Dataset - Test with MemoryBank
=================================================

This script tests MemoryBank on the finance dialogue dataset.
MemoryBank provides long-term memory capabilities using:
- FAISS for efficient memory retrieval
- Sentence embeddings for semantic similarity
- Ebbinghaus forgetting curve for memory management

It supports:
- Custom model selection (OpenAI, Qwen, DeepSeek, etc.)
- Custom API key and base URL
- Memory built from dialogue conversations
- User profile and personality tracking
- Forgetting mechanism for memory management
- LLM-based evaluation for response quality assessment

Usage:
    python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY --base-url https://api.vveai.com/v1
    python test_memorybank.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
    
    # With LLM evaluation:
    python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval --eval-api-key YOUR_EVAL_KEY
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

# Import MemoryBank components
from memorybank import (
    MemoryBank,
    SiliconFriend,
    MemoryStorage,
    MemoryRetriever,
    LLMInterface,
    OpenAILLM
)

# Import for LLM evaluation
try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not installed. GPT tests will not work.")

# Import evaluation metrics
try:
    from test_finmem import (
        calculate_bleu_score,
        calculate_f1_score,
        calculate_exact_match,
        load_dataset,
        get_model_config
    )
except ImportError:
    # Fallback: simple metrics if test_finmem not available
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
    
    def get_model_config(model_name):
        return {"provider": "openai", "base_url": "https://api.vveai.com/v1"}


# ==================== LLM Evaluator Class ====================

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


# ==================== Query Extraction for MemoryBank ====================

def extract_test_queries_memorybank(
    data, 
    max_per_user: int = 1000, 
    include_dialogue_context: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract test queries from dataset for MemoryBank testing.
    """
    queries = []
    for user_data in data:
        persona = user_data.get("persona", {})
        timeline = user_data.get("timeline", [])
        sessions = user_data.get("sessions", [])
        qa_pairs = user_data.get("qa_pairs", [])
        
        session_dict = {s.get("session_id"): s for s in sessions}
        
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
            
            question_type = get_question_type_from_data(qa)
            
            queries.append({
                "user_id": persona.get("user_id", "unknown"),
                "query": qa.get("query", ""),
                "expected_response": qa.get("response", ""),
                "question_type": question_type,
                "context": context,
                "sessions": sessions,
                "persona": persona
            })
    
    return queries


def get_question_type_from_data(qa: Dict[str, Any]) -> str:
    """Get question_type directly from the dataset qa_pair."""
    VALID_QUESTION_TYPES = ["summary", "planning", "investment_advice", "preference", "calculation"]
    qtype = qa.get("question_type", "general")
    if qtype in VALID_QUESTION_TYPES:
        return qtype
    return "general"


# ==================== Custom LLM for base_url support ====================

class CustomOpenAILLM(LLMInterface):
    """Custom LLM that supports custom base URL"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None):
        try:
            from openai import OpenAI
            if base_url:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def summarize_events(self, conversations: str) -> str:
        prompt = f"Summarize the events and key information in the following content:\n\n{conversations}"
        return self.generate(prompt)
    
    def analyze_personality(self, conversations: str) -> str:
        prompt = f"Based on the following dialogue, please summarize the user's personality traits and emotions:\n\n{conversations}"
        return self.generate(prompt)


class MockLLM(LLMInterface):
    """Mock LLM for testing without API"""
    
    def __init__(self):
        pass
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return f"[Mock Response] Processed: {prompt[:50]}..."
    
    def summarize_events(self, conversations: str) -> str:
        return "User discussed various topics."
    
    def analyze_personality(self, conversations: str) -> str:
        return "The user appears curious and friendly."


# ==================== MemoryBank LLM Wrapper ====================

class MemoryBankLLM:
    """
    Wrapper class to use MemoryBank in the test framework.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        sessions: List[Dict],
        persona: Dict,
        enable_forgetting: bool = True,
        forgetting_threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_mock_llm: bool = False
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.sessions = sessions
        #self.persona = persona
        self.persona = None
        self.enable_forgetting = enable_forgetting
        
        self.user_id = persona.get("user_id", "default_user")
        
        # Create LLM interface
        if use_mock_llm:
            self.llm = MockLLM()
        else:
            try:
                self.llm = CustomOpenAILLM(
                    api_key=api_key,
                    model=model,
                    base_url=base_url
                )
            except Exception as e:
                print(f"Warning: Failed to create LLM: {e}. Using mock LLM.")
                self.llm = MockLLM()
        
        
        # Create MemoryBank instance
        self.memory_bank = MemoryBank(
            user_id=self.user_id,
            llm=self.llm,
            embedding_model=embedding_model,
            enable_forgetting=enable_forgetting,
            forgetting_threshold=forgetting_threshold
        )
        
        # Build memory from dialogue sessions
        if sessions:
            self._build_memory_from_sessions(sessions)
        
        # Set initial user profile
        self._set_user_profile(persona)
    
    def _build_memory_from_sessions(self, sessions: List[Dict]):
        """Build memory from dialogue sessions"""
        for session in sessions:
            topic = session.get("topic", "")
            topic_desc = session.get("topic_description", "")
            turns = session.get("turns", [])
            
            for turn in turns:
                speaker = turn.get("speaker", "")
                message = turn.get("message", "")
                
                if speaker == "user":
                    self.memory_bank.add_dialogue("user", message)
                elif speaker == "assistant":
                    self.memory_bank.add_dialogue("assistant", message)
    
    def _set_user_profile(self, persona: Dict = None):
        """Set user profile information - start empty, build from memory
        
        Note: Persona is not directly set from dataset. Instead, user profile
        will be dynamically built from conversations through the LLM's
        analyze_personality method when add_dialogue is called.
        """
        # Start with empty profile - will be built from memory over time
        # The MemoryBank's build_augmented_prompt will retrieve relevant
        # memories which can include user information discovered from chats
        pass
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate response using MemoryBank"""
        try:
            augmented_prompt = self.memory_bank.build_augmented_prompt(prompt)
            
            system_prompt = """You are a helpful financial advisor assistant with long-term memory.
You have access to past conversations and user profile information to provide personalized advice.
Use the context provided to give tailored financial guidance."""
            
            response = self.llm.generate(augmented_prompt, system_prompt)
            return response
        except Exception as e:
            print(f"Error in MemoryBank generate: {e}")
            return f"Error: {str(e)}"
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        storage = self.memory_bank.storage
        return {
            "total_memories": len(storage.memories),
            "daily_events": len(storage.user_profile.daily_events),
            "daily_personalities": len(storage.user_profile.daily_personalities),
            "global_events": len(storage.user_profile.global_events) if storage.user_profile.global_events else 0,
            "global_personality": len(storage.user_profile.global_personality) if storage.user_profile.global_personality else 0
        }


# ==================== Test Functions ====================

def run_tests(
    llm: MemoryBankLLM,
    queries: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
    memory_mode: str = "memorybank",
    eval_llm: LLMEvaluator = None,
    enable_llm_eval: bool = False
) -> List[Dict]:
    """Run tests on all queries using MemoryBank"""
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with MemoryBank")
    print(f"Memory Mode: {memory_mode}")
    print(f"LLM Evaluation: {'Enabled' if enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"{'='*60}")
    
    mem_stats = llm.get_memory_stats()
    print(f"\nMemory Stats:")
    print(f"  - Total memories stored: {mem_stats['total_memories']}")
    print(f"  - Daily events: {mem_stats['daily_events']}")
    print(f"  - Daily personalities: {mem_stats['daily_personalities']}")
    print(f"  - Global events: {mem_stats['global_events']}")
    print(f"  - Global personality: {mem_stats['global_personality']}")
    
    for i, query_data in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query_data['query'][:50]}...")
        
        start_time = time.time()
        
        actual_response = llm.generate(
            query_data["query"],
            query_data["context"]
        )
        
        latency = time.time() - start_time
        
        expected_response = query_data["expected_response"]
        bleu = calculate_bleu_score(expected_response, actual_response) if actual_response else 0.0
        f1 = calculate_f1_score(expected_response, actual_response) if actual_response else 0.0
        exact_match = calculate_exact_match(expected_response, actual_response) if actual_response else 0.0
        
        # LLM evaluation (optional)
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
            "question_type": query_data.get("question_type", "general"),
            "latency_seconds": latency,
            "timestamp": datetime.now().isoformat(),
            "success": True if not actual_response.startswith("Error:") else False,
            "bleu_score": bleu,
            "f1_score": f1,
            "exact_match": exact_match,
            "memory_mode": memory_mode,
            # LLM evaluation metrics
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
    output_file = os.path.join(output_dir, f"results_{model_name.replace(' ', '_').replace('/', '_')}_{memory_mode.replace(' ', '_')}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    metrics = calculate_metrics(results)
    print(f"\nSummary for {model_name} ({memory_mode}):")
    print(f"  - Total queries: {metrics['count']}")
    print(f"  - Success rate: {metrics['success_rate']*100:.1f}%")
    print(f"  - Avg latency: {metrics['avg_latency']:.2f}s")
    print(f"  - Avg BLEU: {metrics['avg_bleu']:.4f}")
    print(f"  - Avg F1: {metrics['avg_f1']:.4f}")
    print(f"  - Avg Exact Match: {metrics['avg_exact_match']:.4f}")
    
    if enable_llm_eval and eval_llm:
        print(f"  - Avg LLM Overall: {metrics['avg_llm_overall']:.2f}")
        print(f"  - Avg LLM Relevance: {metrics['avg_llm_relevance']:.2f}")
        print(f"  - Avg LLM Accuracy: {metrics['avg_llm_accuracy']:.2f}")
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate average metrics from results."""
    if not results:
        return {
            "count": 0, "success_rate": 0, "avg_latency": 0,
            "avg_bleu": 0, "avg_f1": 0, "avg_exact_match": 0,
            "avg_llm_overall": 0, "avg_llm_relevance": 0, "avg_llm_accuracy": 0,
            "avg_llm_personalization": 0, "avg_llm_helpfulness": 0
        }
    
    total = len(results)
    success = sum(1 for r in results if r.get("success", False))
    
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
        avg_llm_overall = avg_llm_relevance = avg_llm_accuracy = 0
        avg_llm_personalization = avg_llm_helpfulness = 0
    
    return {
        "count": total,
        "success_rate": success / total if total > 0 else 0,
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
    """Analyze results grouped by question type."""
    type_groups = defaultdict(list)
    for r in results:
        qtype = r.get("question_type", "general")
        type_groups[qtype].append(r)
    
    type_metrics = {}
    for qtype, type_results in type_groups.items():
        type_metrics[qtype] = calculate_metrics(type_results)
    
    return type_metrics


def print_classification_report(type_metrics: Dict[str, Dict], overall_metrics: Dict[str, float]):
    """Print the per-question-type evaluation report."""
    print("\n" + "=" * 80)
    print("FINANCE DIALOGUE DATASET - EVALUATION REPORT BY QUESTION TYPE")
    print("=" * 80)
    
    col_type = 18
    col_count = 8
    col_success = 10
    col_latency = 10
    col_bleu = 10
    col_f1 = 10
    col_exact = 10
    col_llm = 10
    
    header = f"{'Question Type':<{col_type}}{'Count':>{col_count}}{'Success%':>{col_success}}{'Latency':>{col_latency}}{'BLEU':>{col_bleu}}{'F1':>{col_f1}}{'Exact':>{col_exact}}{'LLM':>{col_llm}}"
    print(header)
    print("-" * 80)
    
    type_order = ["summary", "planning", "investment_advice", "preference", "calculation"]
    
    for qtype in type_order:
        if qtype not in type_metrics:
            continue
        
        m = type_metrics[qtype]
        row = f"{qtype.capitalize().replace('_', ' '):<{col_type}}{m['count']:>{col_count}}{m['success_rate']*100:>8.1f}%{m['avg_latency']:>8.2f}s{m['avg_bleu']:>9.4f}{m['avg_f1']:>9.4f}{m['avg_exact_match']:>9.4f}{m['avg_llm_overall']:>9.1f}"
        print(row)
    
    print("-" * 80)
    
    row = f"{'OVERALL':<{col_type}}{overall_metrics['count']:>{col_count}}{overall_metrics['success_rate']*100:>8.1f}%{overall_metrics['avg_latency']:>8.2f}s{overall_metrics['avg_bleu']:>9.4f}{overall_metrics['avg_f1']:>9.4f}{overall_metrics['avg_exact_match']:>9.4f}{overall_metrics['avg_llm_overall']:>9.1f}"
    print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test MemoryBank on finance dialogue dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with OpenAI
  python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY

  # Test with Qwen
  python test_memorybank.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

  # Test with specific number of queries
  python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY --max-queries 10

  # Test with mock LLM (no API key required)
  python test_memorybank.py --use-mock-llm

  # Disable forgetting mechanism
  python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY --disable-forgetting

  # With LLM evaluation
  python test_memorybank.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval --eval-api-key YOUR_EVAL_KEY
        """
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model name (default: gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for the model")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for API")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str,
                       default="./output/finance_dialogue.json",
                       help="Path to finance dialogue dataset")
    parser.add_argument("--output-dir", type=str,
                       default="./test_results_memorybank",
                       help="Output directory for results")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to test")
    parser.add_argument("--max-per-user", type=int, default=1000,
                       help="Max queries per user")
    
    # MemoryBank options
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model for memory (default: all-MiniLM-L6-v2)")
    parser.add_argument("--disable-forgetting", action="store_true",
                       help="Disable forgetting mechanism")
    parser.add_argument("--forgetting-threshold", type=float, default=0.3,
                       help="Forgetting threshold (default: 0.3)")
    parser.add_argument("--use-mock-llm", action="store_true",
                       help="Use mock LLM (no API key required)")
    
    # LLM evaluation options
    parser.add_argument("--enable-llm-eval", action="store_true",
                       help="Enable LLM-based evaluation using GPT-4o-mini")
    parser.add_argument("--eval-api-key", type=str, default=None,
                       help="API key for LLM evaluation (defaults to OPENAI_API_KEY)")
    parser.add_argument("--eval-base-url", type=str, default=None,
                       help="Base URL for LLM evaluation")
    
    args = parser.parse_args()
    
    # ==================== Initialize LLM Evaluator ====================
    
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
    
    # ==================== Get API Keys ====================
    
    api_key = args.api_key
    if not args.use_mock_llm:
        if not api_key:
            if "qwen" in args.model.lower() or "dashscope" in (args.base_url or ""):
                api_key = os.environ.get("DASHSCOPE_API_KEY")
            elif "deepseek" in args.model.lower():
                api_key = os.environ.get("DEEPSEEK_API_KEY")
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key and not args.use_mock_llm:
        print("Warning: No API key provided. Use --api-key or set environment variable.")
        print("Falling back to mock LLM...")
        args.use_mock_llm = True
    
    # Get base URL
    base_url = args.base_url
    if not base_url and not args.use_mock_llm:
        config = get_model_config(args.model)
        base_url = config.get("base_url", "https://api.vveai.com/v1")
    
    # ==================== Print Configuration ====================
    
    print(f"\n{'='*60}")
    print("MemoryBank Test Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    if not args.use_mock_llm:
        print(f"Base URL: {base_url}")
        print(f"API Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else api_key}")
    else:
        print("LLM: Mock LLM (no API required)")
    print(f"Forgetting: {'Enabled' if not args.disable_forgetting else 'Disabled'}")
    print(f"Forgetting Threshold: {args.forgetting_threshold}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Dataset: {args.dataset}")
    print(f"LLM Evaluation: {'Enabled' if args.enable_llm_eval else 'Disabled'}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    try:
        data = load_dataset(args.dataset)
        print(f"Loaded {len(data)} user profiles")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Extract queries
    queries = extract_test_queries_memorybank(
        data,
        max_per_user=args.max_per_user,
        include_dialogue_context=True
    )
    
    if args.max_queries:
        queries = queries[:args.max_queries]
    
    print(f"Testing {len(queries)} queries")
    
    # Group queries by user
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
        user_data = None
        for d in data:
            if d.get("persona", {}).get("user_id") == user_id:
                user_data = d
                break
        
        if not user_data:
            continue
        
        sessions = user_data.get("sessions", [])
        persona = user_data.get("persona", {})
        
        memorybank_llm = MemoryBankLLM(
            model=args.model,
            api_key=api_key if api_key else "",
            base_url=base_url if base_url else "",
            sessions=sessions,
            persona=persona,
            enable_forgetting=not args.disable_forgetting,
            forgetting_threshold=args.forgetting_threshold,
            embedding_model=args.embedding_model,
            use_mock_llm=args.use_mock_llm
        )
        
        user_results = run_tests(
            memorybank_llm,
            user_qs,
            f"{args.model}_{user_id}",
            args.output_dir,
            memory_mode="memorybank",
            eval_llm=eval_llm,
            enable_llm_eval=args.enable_llm_eval
        )
        
        all_results.extend(user_results)
    
    # ==================== Print Overall Summary ====================
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY (MemoryBank)")
    print(f"{'='*60}")
    
    overall_metrics = calculate_metrics(all_results)
    print(f"Total queries: {overall_metrics['count']}")
    print(f"Success rate: {overall_metrics['success_rate']*100:.1f}%")
    print(f"Avg latency: {overall_metrics['avg_latency']:.2f}s")
    print(f"Avg BLEU: {overall_metrics['avg_bleu']:.4f}")
    print(f"Avg F1: {overall_metrics['avg_f1']:.4f}")
    print(f"Avg Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    
    if args.enable_llm_eval:
        print(f"Avg LLM Overall: {overall_metrics['avg_llm_overall']:.2f}")
        print(f"Avg LLM Relevance: {overall_metrics['avg_llm_relevance']:.2f}")
        print(f"Avg LLM Accuracy: {overall_metrics['avg_llm_accuracy']:.2f}")
        print(f"Avg LLM Personalization: {overall_metrics['avg_llm_personalization']:.2f}")
        print(f"Avg LLM Helpfulness: {overall_metrics['avg_llm_helpfulness']:.2f}")
    
    # ==================== Analyze by Question Type ====================
    
    print(f"\n{'='*60}")
    print("ANALYSIS BY QUESTION TYPE")
    print(f"{'='*60}")
    
    type_metrics = analyze_by_question_type(all_results)
    print_classification_report(type_metrics, overall_metrics)
    
    # Save overall summary
    summary_file = os.path.join(args.output_dir, f"summary_memorybank.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": base_url if not args.use_mock_llm else "mock",
        "memory_mode": "memorybank",
        "forgetting_enabled": not args.disable_forgetting,
        "forgetting_threshold": args.forgetting_threshold,
        "embedding_model": args.embedding_model,
        "dataset": args.dataset,
        "num_queries": len(all_results),
        "llm_evaluation_enabled": args.enable_llm_eval,
        "metrics": overall_metrics,
        "metrics_by_question_type": type_metrics
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()