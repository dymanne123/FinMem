"""
Finance Dialogue Dataset - Test with DialogueOnlyFinMem Agent
=============================================================

This script tests DialogueOnlyFinMemAgent which builds memory purely from dialogue
(conversations/sessions) without requiring user persona or financial timeline.

It supports:
- Custom model selection (OpenAI, Qwen, DeepSeek, Llama, etc.)
- Custom API key and base URL
- Memory built ONLY from dialogue conversations
- Skill routing based on query type (investment, tax, retirement, insurance, debt)
- No-memory mode for baseline comparison (pure LLM without any memory context)

Usage:
    python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --base-url https://api.vveai.com/v1
    python test_dialogue_only_finmem.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
    python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --no-memory
"""

import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import sys

# Import DialogueOnlyFinMem components
from finmem import (
    DialogueOnlyFinMemAgent,
    DialogueOnlyMemorySystem,
    QueryType
)

# Import evaluation metrics from test_finmem
try:
    from test_finmem import (
        calculate_bleu_score,
        calculate_f1_score,
        calculate_exact_match,
        LLMEvaluator,
        load_dataset,
        extract_test_queries,
        calculate_metrics,
        analyze_by_question_type,
        print_classification_report,
        get_model_config,
        MODEL_CONFIGS
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


# ==================== Query Extraction for Dialogue-Only ====================

def extract_test_queries_dialogue_only(
    data, 
    max_per_user: int = 1000, 
    include_dialogue_context: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract test queries from dataset for dialogue-only testing.
    
    This function:
    1. Uses only dialogue sessions for memory building
    2. Does NOT use user persona or timeline
    3. Includes dialogue history in context for query answering
    """
    queries = []
    for user_data in data:
        persona = user_data.get("persona", {})
        timeline = user_data.get("timeline", [])  # Not used in dialogue-only mode
        sessions = user_data.get("sessions", [])
        qa_pairs = user_data.get("qa_pairs", [])
        
        session_dict = {s.get("session_id"): s for s in sessions}
        
        for qa in qa_pairs:
            # Get dialogue context for the query
            context = {
                "persona": persona,  # Keep for reference but not used for memory
                "timeline": timeline,  # Not used in dialogue-only mode
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
            
            # Get question_type from dataset
            question_type = get_question_type_from_data(qa)
            
            queries.append({
                "user_id": persona.get("user_id", "unknown"),
                "query": qa.get("query", ""),
                "expected_response": qa.get("response", ""),
                "question_type": question_type,
                "context": context,
                "sessions": sessions  # Pass all sessions for memory building
            })
    
    return queries


def get_question_type_from_data(qa: Dict[str, Any]) -> str:
    """Get question_type directly from the dataset qa_pair."""
    VALID_QUESTION_TYPES = ["summary", "planning", "investment_advice", "preference", "calculation"]
    qtype = qa.get("question_type", "general")
    if qtype in VALID_QUESTION_TYPES:
        return qtype
    return "general"


# ==================== Dialogue-Only FinMem LLM Wrapper ====================

class NoMemoryLLM:
    """
    Wrapper class for pure LLM without any memory context.
    
    This is used for baseline comparison - no FinMem agent, no memory.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        sessions: List[Dict] = None,
        enable_skill_routing: bool = False,
        skill_version: str = "v1"
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.sessions = sessions or []
        
        # Import LLM client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai package")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Build simple dialogue context for prompt
        self.dialogue_context = self._build_dialogue_context()
    
    def _build_dialogue_context(self) -> str:
        """Build simple dialogue context from sessions"""
        context_parts = []
        for session in self.sessions:
            topic = session.get("topic", "Unknown topic")
            turns = session.get("turns", [])
            context_parts.append(f"Topic: {topic}")
            for turn in turns:
                speaker = turn.get("speaker", "")
                message = turn.get("message", "")
                context_parts.append(f"{speaker}: {message}")
        return "\n".join(context_parts)
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate response using pure LLM without memory"""
        
        try:
            # Build system prompt
            system_prompt = """You are a helpful financial advisor assistant. 
Answer the user's question based on the conversation context provided."""
            
            # Build user message with context
            user_message = f"""Current question: {prompt}"""
            """
            if self.dialogue_context:
                user_message = fPrevious conversation:
{self.dialogue_context}

Current question: {prompt}
            """

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in NoMemory LLM generate: {e}")
            return f"Error: {str(e)}"
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics - returns zeros for no-memory mode"""
        return {
            "conversation_turns": 0,
            "extracted_preferences": 0,
            "extracted_goals": 0,
            "extracted_financial_info": 0,
            "dialogue_memories": 0
        }


class DialogueOnlyFinMemLLM:
    """
    Wrapper class to use DialogueOnlyFinMemAgent in the test framework.
    
    This agent ONLY uses dialogue sessions to build memory - no user persona or timeline.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        sessions: List[Dict],
        enable_skill_routing: bool = True,
        skill_version: str = "v1",
        use_no_memory: bool = False
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.enable_skill_routing = enable_skill_routing
        self.skill_version = skill_version
        self.use_no_memory = use_no_memory
        
        if use_no_memory:
            # Use pure LLM without memory
            self.agent = None
            self.no_memory_llm = NoMemoryLLM(
                model=model,
                api_key=api_key,
                base_url=base_url,
                sessions=sessions,
                enable_skill_routing=enable_skill_routing,
                skill_version=skill_version
            )
        else:
            # Use DialogueOnlyFinMemAgent with memory
            self.no_memory_llm = None
            llm_config = {
                "provider": "openai",
                "model": model,
                "api_key": api_key,
                "base_url": base_url
            }
            
            self.agent = DialogueOnlyFinMemAgent(
                llm_config=llm_config,
                enable_skill_routing=enable_skill_routing,
                skill_version=skill_version
            )
            
            # Build memory ONLY from dialogue sessions
            if sessions:
                self.agent.build_from_sessions(sessions)
        
        # Store session info for reference
        self.sessions = sessions
    
    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate response using DialogueOnlyFinMem agent or pure LLM"""
        
        if self.use_no_memory:
            return self.no_memory_llm.generate(prompt, context)
        
        try:
            # Use DialogueOnlyFinMem agent to process query
            result = self.agent.process_query(prompt)
            return result.get("response", "")
        except Exception as e:
            print(f"Error in DialogueOnlyFinMem generate: {e}")
            return f"Error: {str(e)}"
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics"""
        if self.use_no_memory:
            return self.no_memory_llm.get_memory_stats()
        
        memory = self.agent.memory
        return {
            "conversation_turns": len(memory.conversation_turns),
            "extracted_preferences": len(memory.extracted_preferences),
            "extracted_goals": len(memory.extracted_goals),
            "extracted_financial_info": len(memory.extracted_financial_info),
            "dialogue_memories": len(memory.dialogue_memories)
        }


# ==================== Test Functions ====================

def run_tests(
    llm: DialogueOnlyFinMemLLM,
    queries: List[Dict[str, Any]],
    model_name: str,
    output_dir: str,
    eval_llm=None,
    enable_llm_eval: bool = False,
    memory_mode: str = "dialogue_only"
) -> List[Dict]:
    """Run tests on all queries using DialogueOnlyFinMem"""
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with DialogueOnlyFinMem Agent")
    print(f"Memory: {memory_mode}")
    print(f"LLM Evaluation: {'Enabled' if enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"{'='*60}")
    
    # Print memory stats
    mem_stats = llm.get_memory_stats()
    print(f"\nMemory Stats:")
    print(f"  - Conversation turns: {mem_stats['conversation_turns']}")
    print(f"  - Extracted preferences: {mem_stats['extracted_preferences']}")
    print(f"  - Extracted goals: {mem_stats['extracted_goals']}")
    print(f"  - Extracted financial info: {mem_stats['extracted_financial_info']}")
    print(f"  - Dialogue memories: {mem_stats['dialogue_memories']}")
    
    for i, query_data in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] Query: {query_data['query'][:50]}...")
        
        start_time = time.time()
        
        # Generate response using DialogueOnlyFinMem
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
            # LLM evaluation results
            "llm_relevance": llm_relevance,
            "llm_accuracy": llm_accuracy,
            "llm_personalization": llm_personalization,
            "llm_helpfulness": llm_helpfulness,
            "llm_overall_score": llm_overall_score,
            "llm_evaluation_reason": llm_reason,
            # Memory mode info
            "memory_mode": memory_mode
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
    
    if enable_llm_eval and eval_llm:
        print(f"  - LLM Avg Overall: {metrics.get('avg_llm_overall', 0):.2f}")
        print(f"  - LLM Avg Relevance: {metrics.get('avg_llm_relevance', 0):.2f}")
        print(f"  - LLM Avg Accuracy: {metrics.get('avg_llm_accuracy', 0):.2f}")
        print(f"  - LLM Avg Personalization: {metrics.get('avg_llm_personalization', 0):.2f}")
        print(f"  - LLM Avg Helpfulness: {metrics.get('avg_llm_helpfulness', 0):.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test DialogueOnlyFinMem Agent on finance dialogue dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with OpenAI (with dialogue memory)
  python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY

  # Test with Qwen
  python test_dialogue_only_finmem.py --model qwen-plus --api-key YOUR_KEY --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

  # Test with specific number of queries
  python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --max-queries 10

  # Enable LLM evaluation
  python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --enable-llm-eval

  # Test without memory (baseline - pure LLM)
  python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key YOUR_KEY --no-memory
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
                       default="./test_results_dialogue_only",
                       help="Output directory for results")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to test")
    parser.add_argument("--max-per-user", type=int, default=1000,
                       help="Max queries per user")
    
    # DialogueOnlyFinMem options
    parser.add_argument("--skill-version", type=str, default="v2", choices=["v1", "v2"],
                       help="Skill routing version (default: v1)")
    parser.add_argument("--disable-skill-routing", action="store_true",
                       help="Disable skill routing")
    parser.add_argument("--no-memory", action="store_true",
                       help="Disable memory - use LLM without any memory context (baseline mode)")
    
    # LLM Evaluation options
    parser.add_argument("--enable-llm-eval", action="store_true",
                       help="Enable LLM-based evaluation")
    parser.add_argument("--eval-api-key", type=str, default=None,
                       help="API key for LLM evaluation")
    parser.add_argument("--eval-base-url", type=str, default=None,
                       help="Base URL for LLM evaluation")
    
    args = parser.parse_args()
    
    # ==================== Get API Keys ====================
    
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
            eval_llm = LLMEvaluator(api_key=eval_api_key, base_url=eval_base_url)
        else:
            print("\nWarning: No API key for LLM evaluation. Disabling...")
            args.enable_llm_eval = False
    
    # ==================== Determine Memory Mode ====================
    
    if args.no_memory:
        memory_mode_str = "no_memory"
        memory_mode_display = "No Memory (baseline - pure LLM)"
    else:
        memory_mode_str = "dialogue_only"
        memory_mode_display = "Dialogue-Only (no persona/timeline)"
    
    # ==================== Print Configuration ====================
    
    print(f"\n{'='*60}")
    print("DialogueOnlyFinMem Agent Test Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Base URL: {base_url}")
    print(f"API Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else api_key}")
    print(f"Memory Mode: {memory_mode_display}")
    print(f"Skill Routing: {'Enabled' if not args.disable_skill_routing else 'Disabled'}")
    print(f"Skill Version: {args.skill_version}")
    print(f"LLM Evaluation: {'Enabled' if args.enable_llm_eval and eval_llm else 'Disabled'}")
    print(f"Dataset: {args.dataset}")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.dataset}")
    try:
        data = load_dataset(args.dataset)
        print(f"Loaded {len(data)} user profiles")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.dataset}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Extract queries (dialogue-only mode)
    queries = extract_test_queries_dialogue_only(
        data,
        max_per_user=args.max_per_user,
        include_dialogue_context=True
    )
    
    if args.max_queries:
        queries = queries[:args.max_queries]
    
    print(f"Testing {len(queries)} queries")
    
    # Group queries by user for per-user agent
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
        
        # Get sessions (dialogue-only - no persona or timeline used for memory)
        sessions = user_data.get("sessions", [])
        
        # Create DialogueOnlyFinMem LLM for this user
        dialogue_llm = DialogueOnlyFinMemLLM(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            sessions=sessions,
            enable_skill_routing=not args.disable_skill_routing,
            skill_version=args.skill_version,
            use_no_memory=args.no_memory
        )
        
        # Run tests for this user
        user_results = run_tests(
            dialogue_llm,
            user_qs,
            f"{args.model}_{user_id}",
            args.output_dir,
            eval_llm=eval_llm,
            enable_llm_eval=args.enable_llm_eval,
            memory_mode=memory_mode_str
        )
        
        all_results.extend(user_results)
    
    # ==================== Print Overall Summary ====================
    
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY ({memory_mode_display})")
    print(f"{'='*60}")
    
    overall_metrics = calculate_metrics(all_results)
    print(f"Total queries: {overall_metrics['count']}")
    print(f"Success rate: {overall_metrics['success_rate']*100:.1f}%")
    print(f"Avg latency: {overall_metrics['avg_latency']:.2f}s")
    print(f"Avg BLEU: {overall_metrics['avg_bleu']:.4f}")
    print(f"Avg F1: {overall_metrics['avg_f1']:.4f}")
    print(f"Avg Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    
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
    
    type_metrics = analyze_by_question_type(all_results)
    print_classification_report(type_metrics, overall_metrics)
    
    # Save overall summary
    summary_file = os.path.join(args.output_dir, f"summary_{memory_mode_str}.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "base_url": base_url,
        "memory_mode": memory_mode_str,
        "skill_routing_enabled": not args.disable_skill_routing,
        "skill_version": args.skill_version,
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
