"""
FinMem: Financial Memory Agent with Long-Short Term Preference Modeling and Skill Routing

This module implements a financial advisory agent that:
1. Uses long-short term memory for user preference modeling
   - Short-term: Recent consumption/investment behaviors (< 1 year)
   - Long-term: User's overall financial preferences, risk tolerance, investment goals
   - Personalized time horizon based on user's investment cycle
2. Routes queries to specialized financial skills
   - Investment advice skill
   - Tax planning skill
   - Retirement planning skill
   - Insurance advice skill
   - Debt management skill
   - General financial advisory skill
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict

# Try to import required libraries
try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai not installed.")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: sentence-transformers not installed.")

# LLM-based memory extraction prompt
MEMORY_EXTRACTION_PROMPT = """You are a financial memory analyzer. Analyze the following dialogue and extract structured memory information.

DIALOGUE:
{conversation}

Analyze the dialogue and extract the following types of information. Return your analysis in JSON format:

1. **User Preferences**: Extract expressed preferences, likes, dislikes, and stated requirements
2. **Financial Goals**: Extract financial goals, plans, and objectives mentioned
3. **Financial Information**: Extract specific financial details (amounts, accounts, investments, income, expenses)
4. **Key Behaviors**: Extract spending habits, investment patterns, and financial behaviors
5. **Important Constraints**: Extract budget limits, time constraints, or specific requirements

Return JSON format:
{{
    "preferences": [
        {{"content": "description of preference", "importance": "high/medium/low", "keywords": ["keyword1", "keyword2"]}}
    ],
    "goals": [
        {{"content": "description of goal", "time_horizon": "short/medium/long", "importance": "high/medium/low"}}
    ],
    "financial_info": [
        {{"content": "description of financial info", "type": "account/investment/income/expense", "amount": "specific amount if mentioned"}}
    ],
    "behaviors": [
        {{"content": "description of behavior", "frequency": "recurring/occasional/one-time"}}
    ],
    "constraints": [
        {{"content": "description of constraint", "type": "budget/time/risk"}}
    ]
}}

If no information of a certain type is found, return an empty list for that category.

DIALOGUE TO ANALYZE:
{message}

YOUR ANALYSIS (JSON ONLY):"""

# Memory classification prompt for classifying existing memories
MEMORY_CLASSIFICATION_PROMPT = """Classify the following memory content into one of these categories:
- short_term: Recent transactions, expenses, or temporary information
- long_term: Stable information, general knowledge, user profile facts
- preference: User likes, dislikes, and stated preferences
- behavior: Spending habits, investment patterns, recurring actions
- goal: Financial goals, future plans, objectives

Content to classify: "{content}"

Speaker: {speaker}

Return only the category name (one of: short_term, long_term, preference, behavior, goal):"""


class LLMClient:
    """LLM client wrapper for memory extraction"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.vveai.com/v1", 
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM client: {e}")
    
    def extract_memories(self, message: str, conversation_context: str = "") -> Dict[str, Any]:
        """Extract structured memories from a message using LLM"""
        if not self.client:
            return None
        
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(
                conversation=conversation_context,
                message=message
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise financial data analyst. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            return json.loads(result)
        except Exception as e:
            print(f"Error extracting memories with LLM: {e}")
            return None
    
    def classify_memory(self, content: str, speaker: str = "user") -> str:
        """Classify a memory into one of the predefined categories"""
        if not self.client:
            return "short_term"  # Default fallback
        
        try:
            prompt = MEMORY_CLASSIFICATION_PROMPT.format(
                content=content,
                speaker=speaker
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise classifier. Return only the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Validate the result
            valid_categories = ["short_term", "long_term", "preference", "behavior", "goal"]
            for cat in valid_categories:
                if cat in result:
                    return cat
            
            return "short_term"  # Default fallback
        except Exception as e:
            print(f"Error classifying memory: {e}")
            return "short_term"


# ==================== Data Structures ====================

class TimeHorizon(Enum):
    """User's perceived time horizon for investment"""
    SHORT_TERM = "short_term"      # < 1 year
    MEDIUM_TERM = "medium_term"    # 1-3 years
    LONG_TERM = "long_term"        # 3-10 years
    VERY_LONG_TERM = "very_long_term"  # > 10 years


class QueryType(Enum):
    """Types of financial queries - Version 1 (Original)"""
    INVESTMENT = "investment"
    TAX = "tax"
    RETIREMENT = "retirement"
    INSURANCE = "insurance"
    DEBT = "debt"
    CONSUMPTION = "consumption"
    GENERAL = "general"


class QueryTypeV2(Enum):
    """Types of financial queries - Version 2 (New: Domain-based)"""
    USER_PREFERENCE = "user_preference"      # 用户偏好 - 风险偏好、财务目标
    FINANCIAL_PLANNING = "financial_planning"  # 财务规划 - 综合规划
    INVESTMENT_ADVICE = "investment_advice"    # 投资建议 - 投资相关
    FINANCIAL_SUMMARY = "financial_summary"   # 财务总结 - 状况分析
    FINANCIAL_CALCULATION = "financial_calculation"  # 财务计算 - 计算理财


class SkillVersion(Enum):
    """Skill routing version"""
    V1 = "v1"  # Original: investment, tax, retirement, insurance, debt, general
    V2 = "v2"  # New: user_preference, financial_planning, investment_advice, financial_summary, financial_calculation


@dataclass
class UserPersona:
    """User profile with financial preferences"""
    user_id: str
    name: str
    age: int
    occupation: str
    income_level: str
    family_status: str
    risk_tolerance: str  # conservative, moderate, aggressive
    financial_goals: List[str]
    # Time horizon preference - different users understand "long-term" differently
    perceived_short_term_days: int = 90    # User's perception of short-term
    perceived_medium_term_days: int = 365   # User's perception of medium-term
    perceived_long_term_days: int = 1095    # User's perception of long-term (3 years)
    # Investment cycle preference
    investment_cycle_days: int = 365        # User's typical investment cycle
    # Memory weights
    short_term_weight: float = 0.3          # Weight for short-term memory
    long_term_weight: float = 0.7            # Weight for long-term memory


@dataclass
class DialogueTurnInfo:
    """A dialogue turn for building memory from conversations"""
    turn_id: str
    speaker: str  # user, assistant
    message: str
    turn_number: int
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class FinancialEvent:
    """A financial event (income, expense, investment, etc.)"""
    event_id: str
    timestamp: str
    event_type: str  # income, expense, investment, withdrawal, etc.
    amount: float
    category: str
    description: str
    impact: str  # positive, negative, neutral
    is_recurring: bool = False
    periodicity: Optional[str] = None  # monthly, yearly, etc.


@dataclass
class MemoryBlock:
    """A memory block containing financial information"""
    memory_id: str
    content: str
    memory_type: str  # short_term, long_term, preference, behavior, goal
    timestamp: str
    importance_score: float = 1.0
    retrieval_count: int = 0
    keywords: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    # Time-related fields
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    # User-specific time horizon
    user_perceived_horizon: str = "medium_term"


@dataclass
class DialogueTurn:
    """A dialogue turn in the conversation"""
    turn_id: str
    speaker: str  # user, assistant
    message: str
    timestamp: str
    query_type: Optional[str] = None
    skill_used: Optional[str] = None
    memory_retrieved: List[str] = field(default_factory=list)


# ==================== Dialogue-Only Memory System (LLM-enhanced) ====================

class DialogueOnlyMemorySystem:
    """
    Memory system built purely from dialogue conversations using LLM-based extraction.
    Does NOT use user persona or timeline - only extracts information from dialogues.
    
    This is designed for the scenario where user profile and timeline are not accessible,
    but we can still build useful memories from the conversation history.
    
    Uses LLM for intelligent memory extraction instead of keyword matching.
    """
    
    def __init__(
        self,
        embedding_model: str = 'all-MiniLM-L6-v2',
        max_memories: int = 100,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        self.embedding_model = embedding_model
        self.max_memories = max_memories
        
        # LLM configuration for memory extraction
        self.llm_config = llm_config or {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.vveai.com/v1",
            "model": "gpt-4o-mini"
        }
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            api_key=self.llm_config.get("api_key"),
            base_url=self.llm_config.get("base_url", "https://api.vveai.com/v1"),
            model=self.llm_config.get("model", "gpt-4o-mini")
        )
        
        # Memory storage - organized by conversation context
        self.dialogue_memories: List[MemoryBlock] = []  # All dialogue-based memories
        self.short_term_memories: List[MemoryBlock] = []  # Recent conversation context
        self.long_term_memories: List[MemoryBlock] = []  # Important recurring info
        
        # Extracted user information from dialogues (LLM-extracted)
        self.extracted_preferences: List[MemoryBlock] = []  # User preferences mentioned
        self.extracted_financial_info: List[MemoryBlock] = []  # Financial details mentioned
        self.extracted_goals: List[MemoryBlock] = []  # Goals mentioned by user
        self.extracted_behaviors: List[MemoryBlock] = []  # Behaviors and habits
        self.extracted_constraints: List[MemoryBlock] = []  # Budget/time constraints
        
        # Embedding model for semantic search
        try:
            self.encoder = SentenceTransformer(embedding_model)
            self.use_embeddings = True
        except:
            self.use_embeddings = False
            print("Warning: Embedding model not available, using keyword search")
        
        # Conversation tracking
        self.conversation_turns: List[DialogueTurnInfo] = []
        self.current_session_id: Optional[str] = None
        
        # Batch processing settings
        self._batch_size = 5  # Number of turns to accumulate before LLM extraction
        self._pending_turns: List[DialogueTurnInfo] = []
    
    def add_dialogue_turn(
        self,
        turn: DialogueTurnInfo
    ):
        """Add a dialogue turn and extract memories from it using LLM"""
        self.conversation_turns.append(turn)
        self._pending_turns.append(turn)
        
        # Extract information from this turn using LLM
        self._extract_memory_from_turn_llm(turn)
    
    def _extract_memory_from_turn_llm(self, turn: DialogueTurnInfo):
        """Extract memory-worthy information from a dialogue turn using LLM"""
        
        # Prepare conversation context for LLM
        context_turns = self.conversation_turns[-10:] if len(self.conversation_turns) > 10 else self.conversation_turns
        conversation_context = "\n".join([
            f"{'User' if t.speaker == 'user' else 'Assistant'}: {t.message}"
            for t in context_turns[:-1]  # Exclude current turn
        ])
        
        # Try LLM extraction first
        if self.llm_client.client:
            try:
                result = self.llm_client.extract_memories(
                    message=turn.message,
                    conversation_context=conversation_context
                )
                
                if result:
                    self._process_llm_extracted_memories(result, turn)
                    return
            except Exception as e:
                print(f"LLM extraction failed, falling back to keyword extraction: {e}")
        
        # Fallback to keyword-based extraction if LLM fails
        self._extract_memory_from_turn_fallback(turn)
    
    def _process_llm_extracted_memories(self, result: Dict[str, Any], turn: DialogueTurnInfo):
        """Process memory extraction results from LLM"""
        timestamp = turn.timestamp or datetime.now().strftime("%Y%m%d%H%M")
        
        # Process preferences
        for pref in result.get("preferences", []):
            content = pref.get("content", "")
            if content:
                importance = {"high": 1.5, "medium": 1.2, "low": 1.0}.get(
                    pref.get("importance", "medium"), 1.2
                )
                memory = MemoryBlock(
                    memory_id=f"pref_{len(self.extracted_preferences)}_{turn.turn_number}",
                    content=content,
                    memory_type="preference",
                    timestamp=timestamp,
                    importance_score=importance,
                    keywords=pref.get("keywords", self._extract_keywords(content))
                )
                self.extracted_preferences.append(memory)
        
        # Process goals
        for goal in result.get("goals", []):
            content = goal.get("content", "")
            if content:
                importance = {"high": 1.5, "medium": 1.2, "low": 1.0}.get(
                    goal.get("importance", "medium"), 1.2
                )
                memory = MemoryBlock(
                    memory_id=f"goal_{len(self.extracted_goals)}_{turn.turn_number}",
                    content=content,
                    memory_type="goal",
                    timestamp=timestamp,
                    importance_score=importance,
                    keywords=self._extract_keywords(content)
                )
                self.extracted_goals.append(memory)
        
        # Process financial info
        for fin_info in result.get("financial_info", []):
            content = fin_info.get("content", "")
            if content:
                memory = MemoryBlock(
                    memory_id=f"fin_{len(self.extracted_financial_info)}_{turn.turn_number}",
                    content=content,
                    memory_type="behavior",
                    timestamp=timestamp,
                    importance_score=1.3,
                    keywords=self._extract_keywords(content)
                )
                self.extracted_financial_info.append(memory)
        
        # Process behaviors
        for behavior in result.get("behaviors", []):
            content = behavior.get("content", "")
            if content:
                memory = MemoryBlock(
                    memory_id=f"beh_{len(self.extracted_behaviors)}_{turn.turn_number}",
                    content=content,
                    memory_type="behavior",
                    timestamp=timestamp,
                    importance_score=1.1,
                    keywords=self._extract_keywords(content)
                )
                self.extracted_behaviors.append(memory)
        
        # Process constraints
        for constraint in result.get("constraints", []):
            content = constraint.get("content", "")
            if content:
                memory = MemoryBlock(
                    memory_id=f"con_{len(self.extracted_constraints)}_{turn.turn_number}",
                    content=content,
                    memory_type="short_term",
                    timestamp=timestamp,
                    importance_score=1.2,
                    keywords=self._extract_keywords(content)
                )
                self.extracted_constraints.append(memory)
        
        # For assistant responses, also store as dialogue memory
        if turn.speaker == "assistant":
            self._extract_assistant_info(turn)
    
    def _extract_memory_from_turn_fallback(self, turn: DialogueTurnInfo):
        """Fallback to keyword-based extraction when LLM is unavailable"""
        if turn.speaker == "user":
            self._extract_user_preferences(turn)
            self._extract_user_goals(turn)
            self._extract_financial_info(turn)
        else:
            self._extract_assistant_info(turn)
    
    def _extract_user_preferences(self, turn: DialogueTurnInfo):
        """Extract user preference expressions from user messages (keyword fallback)"""
        message = turn.message.lower()
        
        # Keywords indicating preferences
        preference_keywords = [
            "i prefer", "i like", "i don't like", "i want", "i need",
            "i am looking for", "i'm interested in", "my risk",
            "i can afford", "i would like", "my budget",
            "i've been thinking", "i'm considering"
        ]
        
        # Check if this message expresses a preference
        for kw in preference_keywords:
            if kw in message:
                # Create a preference memory
                memory = MemoryBlock(
                    memory_id=f"pref_{len(self.extracted_preferences)}_{turn.turn_number}",
                    content=turn.message,
                    memory_type="preference",
                    timestamp=turn.timestamp or datetime.now().strftime("%Y%m%d%H%M"),
                    importance_score=1.2,
                    keywords=self._extract_keywords(turn.message)
                )
                self.extracted_preferences.append(memory)
                break
    
    def _extract_user_goals(self, turn: DialogueTurnInfo):
        """Extract financial goals from user messages (keyword fallback)"""
        message = turn.message.lower()
        
        # Keywords indicating goals
        goal_keywords = [
            "i want to", "i plan to", "i'm saving for", "i hope to",
            "my goal is", "retirement", "buy a house", "children's education",
            "emergency fund", "financial freedom", "invest for"
        ]
        
        for kw in goal_keywords:
            if kw in message:
                memory = MemoryBlock(
                    memory_id=f"goal_{len(self.extracted_goals)}_{turn.turn_number}",
                    content=turn.message,
                    memory_type="goal",
                    timestamp=turn.timestamp or datetime.now().strftime("%Y%m%d%H%M"),
                    importance_score=1.3,
                    keywords=self._extract_keywords(turn.message)
                )
                self.extracted_goals.append(memory)
                break
    
    def _extract_financial_info(self, turn: DialogueTurnInfo):
        """Extract financial information (amounts, accounts, etc.) from messages (keyword fallback)"""
        message = turn.message
        
        # Look for financial amounts
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s*(?:dollars?|usd)?'
        amounts = re.findall(amount_pattern, message, re.IGNORECASE)
        
        # Look for account types
        account_keywords = ["401k", "ira", "savings", "checking", "brokerage", "fund", "stock", "bond"]
        has_account = any(kw in message.lower() for kw in account_keywords)
        
        if amounts or has_account:
            memory = MemoryBlock(
                memory_id=f"fin_{len(self.extracted_financial_info)}_{turn.turn_number}",
                content=turn.message,
                memory_type="behavior",
                timestamp=turn.timestamp or datetime.now().strftime("%Y%m%d%H%M"),
                importance_score=1.1,
                keywords=self._extract_keywords(turn.message)
            )
            self.extracted_financial_info.append(memory)
    
    def _extract_assistant_info(self, turn: DialogueTurnInfo):
        """Extract key advice/information from assistant responses"""
        # Store assistant responses as long-term memories (they contain valuable info)
        memory = MemoryBlock(
            memory_id=f"assist_{len(self.dialogue_memories)}_{turn.turn_number}",
            content=turn.message,
            memory_type="long_term",
            timestamp=turn.timestamp or datetime.now().strftime("%Y%m%d%H%M"),
            importance_score=1.0,
            keywords=self._extract_keywords(turn.message)
        )
        self.dialogue_memories.append(memory)
        
        # Also add to short-term (recent context)
        self.short_term_memories.append(memory)
        
        # Limit short-term memories
        if len(self.short_term_memories) > 20:
            self.short_term_memories = self.short_term_memories[-20:]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction - split by common delimiters
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                     'by', 'from', 'as', 'into', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                     'so', 'than', 'too', 'very', 's', 't', 'just', 'i', 'me',
                     'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords[:10]  # Limit to 10 keywords
    
    def build_memories_from_sessions(
        self,
        sessions: List[Dict[str, Any]]
    ):
        """Build memories from dialogue sessions
        
        Args:
            sessions: List of session dictionaries, each containing:
                - session_id: str
                - topic: str (optional)
                - turns: List of dialogue turns
        """
        for session in sessions:
            session_id = session.get("session_id", "unknown")
            turns = session.get("turns", [])
            
            for turn_data in turns:
                turn = DialogueTurnInfo(
                    turn_id=f"{session_id}_{turn_data.get('turn_number', 0)}",
                    speaker=turn_data.get("speaker", "user"),
                    message=turn_data.get("message", ""),
                    turn_number=turn_data.get("turn_number", 0),
                    session_id=session_id,
                    timestamp=turn_data.get("timestamp")
                )
                self.add_dialogue_turn(turn)
    
    def get_relevant_memories(
        self,
        query: str,
        max_memories: int = 10
    ) -> Dict[str, List[MemoryBlock]]:
        """Get relevant memories for a query
        
        Returns a dictionary with different memory categories
        """
        all_memories = (
            self.extracted_preferences +
            self.extracted_goals +
            self.extracted_financial_info +
            self.dialogue_memories
        )
        
        if not all_memories:
            return {
                "preferences": [],
                "goals": [],
                "financial_info": [],
                "dialogue": [],
                "recent": []
            }
        
        # Score and sort all memories
        scored = self._score_memories(query, all_memories)
        
        # Get top memories
        top_memories = sorted(scored, key=lambda x: x[1], reverse=True)[:max_memories]
        
        # Organize by type
        result = {
            "preferences": [(m, s) for m, s in top_memories if m.memory_type == "preference"],
            "goals": [(m, s) for m, s in top_memories if m.memory_type == "goal"],
            "financial_info": [(m, s) for m, s in top_memories if m.memory_type == "behavior"],
            "dialogue": [(m, s) for m, s in top_memories if m.memory_type == "long_term"],
            "recent": [(m, s) for m, s in scored[:5]]  # Most recent
        }
        
        return result
    
    def _score_memories(
        self,
        query: str,
        memories: List[MemoryBlock]
    ) -> List[tuple]:
        """Score memories based on relevance to query"""
        
        if not memories:
            return []
        
        if self.use_embeddings:
            try:
                # Semantic similarity scoring
                query_embedding = self.encoder.encode([query])
                memory_contents = [m.content for m in memories]
                memory_embeddings = self.encoder.encode(memory_contents)
                
                similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
                
                scored = []
                for i, mem in enumerate(memories):
                    # Combine similarity with importance
                    combined_score = (
                        similarities[i] * 0.6 +
                        mem.importance_score * 0.4
                    )
                    scored.append((mem, combined_score))
            except:
                # Fallback to keyword matching
                scored = self._keyword_score_memories(query, memories)
        else:
            scored = self._keyword_score_memories(query, memories)
        
        return scored
    
    def _keyword_score_memories(
        self,
        query: str,
        memories: List[MemoryBlock]
    ) -> List[tuple]:
        """Score memories using keyword matching"""
        query_keywords = set(self._extract_keywords(query))
        
        scored = []
        for mem in memories:
            keyword_match = len(query_keywords & set(mem.keywords)) / max(len(mem.keywords), 1)
            scored.append((mem, keyword_match * mem.importance_score))
        
        return scored
    
    def get_dialogue_context_summary(self, max_turns: int = 10) -> str:
        """Get a summary of dialogue context for prompting"""
        
        # Recent turns
        recent = self.conversation_turns[-max_turns:] if self.conversation_turns else []
        
        context_parts = ["Recent Conversation:\n"]
        for turn in recent:
            speaker_label = "User" if turn.speaker == "user" else "Assistant"
            context_parts.append(f"{speaker_label}: {turn.message[:200]}...")
        
        # Extracted information summary
        if self.extracted_preferences:
            context_parts.append(f"\nUser Preferences Identified: {len(self.extracted_preferences)}")
        if self.extracted_goals:
            context_parts.append(f"Financial Goals Mentioned: {len(self.extracted_goals)}")
        if self.extracted_financial_info:
            context_parts.append(f"Financial Information Shared: {len(self.extracted_financial_info)}")
        
        return "\n".join(context_parts)
    
    def get_context_for_skill(
        self,
        skill_name: str,
        query: str
    ) -> str:
        """Get formatted context for a specific skill"""
        
        memories = self.get_relevant_memories(query, max_memories=5)
        
        info_parts = []
        
        # Add preferences
        if memories.get("preferences"):
            info_parts.append("\nUSER PREFERENCES:")
            for mem, score in memories["preferences"][:2]:
                info_parts.append(f"- {mem.content[:150]}...")
        
        # Add goals
        if memories.get("goals"):
            info_parts.append("\nFINANCIAL GOALS:")
            for mem, score in memories["goals"][:2]:
                info_parts.append(f"- {mem.content[:150]}...")
        
        # Add financial info
        if memories.get("financial_info"):
            info_parts.append("\nFINANCIAL INFORMATION:")
            for mem, score in memories["financial_info"][:2]:
                info_parts.append(f"- {mem.content[:150]}...")
        
        # Add recent dialogue
        if memories.get("recent"):
            info_parts.append("\nRECENT CONVERSATION:")
            for mem, score in memories["recent"][:3]:
                info_parts.append(f"- {mem.content[:100]}...")
        
        if not info_parts:
            return "No relevant dialogue context available."
        
        return "\n".join(info_parts)


# ==================== Memory System ====================

class FinancialMemorySystem:
    """
    Memory system with long-short term preference modeling
    
    Key features:
    - Personalized time horizons based on user's investment cycle
    - Short-term memory: Recent behaviors (< user's perceived short-term)
    - Long-term memory: Stable preferences and goals
    - Adaptive weighting based on query type
    """
    
    def __init__(
        self,
        user_persona: UserPersona,
        embedding_model: str = 'all-MiniLM-L6-v2',
        short_term_threshold_days: Optional[int] = None,
        long_term_threshold_days: Optional[int] = None
    ):
        self.user_persona = user_persona
        self.embedding_model = embedding_model
        
        # Set time thresholds based on user's perception (or use defaults)
        self.short_term_threshold_days = (
            short_term_threshold_days or user_persona.perceived_short_term_days
        )
        self.long_term_threshold_days = (
            long_term_threshold_days or user_persona.perceived_long_term_days
        )
        
        # Memory storage
        self.short_term_memories: List[MemoryBlock] = []
        self.long_term_memories: List[MemoryBlock] = []
        self.preference_memories: List[MemoryBlock] = []
        self.behavior_memories: List[MemoryBlock] = []
        self.goal_memories: List[MemoryBlock] = []
        
        # Embedding model for semantic search
        try:
            self.encoder = SentenceTransformer(embedding_model)
            self.use_embeddings = True
        except:
            self.use_embeddings = False
            print("Warning: Embedding model not available, using keyword search")
        
        # Event history for context
        self.financial_events: List[FinancialEvent] = []
        
        # Session context
        self.current_session_id: Optional[str] = None
        self.session_history: List[DialogueTurn] = []
    
    def add_memory(
        self,
        content: str,
        memory_type: str,
        timestamp: Optional[str] = None,
        importance: float = 1.0,
        keywords: Optional[List[str]] = None,
        related_events: Optional[List[str]] = None
    ) -> str:
        """Add a new memory block"""
        timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M")
        
        # Determine time horizon based on user's perception
        if memory_type == "short_term" or memory_type == "behavior":
            horizon = "short_term"
        elif memory_type == "long_term" or memory_type == "preference":
            horizon = "long_term"
        elif memory_type == "goal":
            horizon = "very_long_term"
        else:
            horizon = "medium_term"
        
        memory = MemoryBlock(
            memory_id=f"mem_{len(self.short_term_memories) + len(self.long_term_memories)}_{timestamp}",
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            importance_score=importance,
            keywords=keywords or [],
            related_events=related_events or [],
            user_perceived_horizon=horizon
        )
        
        # Classify into appropriate memory store
        if memory_type == "short_term":
            self.short_term_memories.append(memory)
        elif memory_type == "long_term":
            self.long_term_memories.append(memory)
        elif memory_type == "preference":
            self.preference_memories.append(memory)
        elif memory_type == "behavior":
            self.behavior_memories.append(memory)
        elif memory_type == "goal":
            self.goal_memories.append(memory)
        else:
            # Default to long-term
            self.long_term_memories.append(memory)
        
        return memory.memory_id
    
    def add_financial_event(self, event: FinancialEvent):
        """Add a financial event and create associated memory"""
        self.financial_events.append(event)
        
        # Determine memory type based on event and user's time perception
        event_date = datetime.strptime(event.timestamp, "%Y%m%d%H%M")
        days_ago = (datetime.now() - event_date).days
        
        if days_ago <= self.short_term_threshold_days:
            memory_type = "short_term"
        else:
            memory_type = "long_term"
        
        # Add memory for this event
        self.add_memory(
            content=f"{event.event_type}: {event.description}, Amount: {event.amount}, Category: {event.category}",
            memory_type=memory_type,
            timestamp=event.timestamp,
            importance=1.5 if event.impact == "positive" else 1.0,
            keywords=[event.event_type, event.category, event.impact],
            related_events=[event.event_id]
        )
    
    def get_memories_for_query(
        self,
        query: str,
        query_type: QueryType,
        max_memories: int = 10
    ) -> Dict[str, List[MemoryBlock]]:
        """
        Retrieve relevant memories for a query
        
        Uses adaptive weighting based on query type:
        - Investment queries: favor long-term preferences + recent behaviors
        - Tax queries: favor recent transactions
        - Retirement queries: favor long-term goals
        - etc.
        """
        memories = {
            "short_term": [],
            "long_term": [],
            "preference": [],
            "behavior": [],
            "goal": []
        }
        
        # Determine memory weights based on query type
        weights = self._get_memory_weights_for_query_type(query_type)
        
        # Get memories from each category
        for mem_type, mem_list in [
            ("short_term", self.short_term_memories),
            ("long_term", self.long_term_memories),
            ("preference", self.preference_memories),
            ("behavior", self.behavior_memories),
            ("goal", self.goal_memories)
        ]:
            if mem_list:
                scored_memories = self._score_memories(query, mem_list)
                # Take top memories based on weight
                num_to_take = int(max_memories * weights.get(mem_type, 0.2))
                memories[mem_type] = sorted(
                    scored_memories,
                    key=lambda x: x[1],
                    reverse=True
                )[:num_to_take]
        
        return memories
    
    def _get_memory_weights_for_query_type(
        self,
        query_type: QueryType
    ) -> Dict[str, float]:
        """Get memory weights based on query type"""
        
        weight_configs = {
            QueryType.INVESTMENT: {
                "short_term": 0.25,
                "long_term": 0.25,
                "preference": 0.25,
                "behavior": 0.15,
                "goal": 0.10
            },
            QueryType.TAX: {
                "short_term": 0.50,
                "long_term": 0.20,
                "preference": 0.10,
                "behavior": 0.15,
                "goal": 0.05
            },
            QueryType.RETIREMENT: {
                "short_term": 0.10,
                "long_term": 0.20,
                "preference": 0.20,
                "behavior": 0.10,
                "goal": 0.40
            },
            QueryType.INSURANCE: {
                "short_term": 0.15,
                "long_term": 0.30,
                "preference": 0.30,
                "behavior": 0.15,
                "goal": 0.10
            },
            QueryType.DEBT: {
                "short_term": 0.40,
                "long_term": 0.20,
                "preference": 0.15,
                "behavior": 0.20,
                "goal": 0.05
            },
            QueryType.CONSUMPTION: {
                "short_term": 0.50,
                "long_term": 0.15,
                "preference": 0.15,
                "behavior": 0.15,
                "goal": 0.05
            },
            QueryType.GENERAL: {
                "short_term": 0.20,
                "long_term": 0.25,
                "preference": 0.25,
                "behavior": 0.15,
                "goal": 0.15
            }
        }
        
        return weight_configs.get(query_type, weight_configs[QueryType.GENERAL])
    
    def _score_memories(
        self,
        query: str,
        memories: List[MemoryBlock]
    ) -> List[tuple]:
        """Score memories based on relevance to query"""
        
        if not memories:
            return []
        
        if self.use_embeddings:
            # Semantic similarity scoring
            query_embedding = self.encoder.encode([query])
            memory_contents = [m.content for m in memories]
            memory_embeddings = self.encoder.encode(memory_contents)
            
            similarities = cosine_similarity(query_embedding, memory_embeddings)[0]
            
            # Combine with importance and recency
            scored = []
            for i, mem in enumerate(memories):
                # Recency score (more recent = higher)
                mem_date = datetime.strptime(mem.timestamp, "%Y%m%d%H%M")
                days_ago = (datetime.now() - mem_date).days
                recency_score = 1.0 / (1.0 + days_ago / 30)  # Decay over months
                
                # Combined score
                combined_score = (
                    similarities[i] * 0.5 +
                    mem.importance_score * 0.3 +
                    recency_score * 0.2
                )
                scored.append((mem, combined_score))
        else:
            # Keyword-based scoring
            query_keywords = set(query.lower().split())
            scored = []
            for mem in memories:
                keyword_match = len(query_keywords & set(mem.keywords)) / max(len(mem.keywords), 1)
                scored.append((mem, keyword_match * mem.importance_score))
        
        return scored
    
    def get_context_summary(self) -> str:
        """Get a summary of user's financial context for prompting"""
        
        # User profile summary
        persona = self.user_persona
        profile_summary = f"""
User Profile:
- Name: {persona.name}, Age: {persona.age}
- Occupation: {persona.occupation}
- Income Level: {persona.income_level}
- Family Status: {persona.family_status}
- Risk Tolerance: {persona.risk_tolerance}
- Financial Goals: {', '.join(persona.financial_goals)}
- Investment Cycle: {persona.investment_cycle_days} days
- User's Short-term Perception: {persona.perceived_short_term_days} days
- User's Long-term Perception: {persona.perceived_long_term_days} days
"""
        
        # Recent financial events
        recent_events = sorted(
            self.financial_events,
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        events_summary = "\nRecent Financial Events:\n"
        for event in recent_events:
            events_summary += f"- {event.timestamp}: {event.event_type} - {event.description} (Amount: {event.amount}, Impact: {event.impact})\n"
        
        # Memory summary
        mem_summary = f"""
Memory Summary:
- Short-term memories: {len(self.short_term_memories)}
- Long-term memories: {len(self.long_term_memories)}
- Preferences: {len(self.preference_memories)}
- Behaviors: {len(self.behavior_memories)}
- Goals: {len(self.goal_memories)}
"""
        
        return profile_summary + events_summary + mem_summary


# ==================== Financial Skills ====================

class FinancialSkill:
    """Base class for financial skills"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Execute the skill"""
        raise NotImplementedError
    
    def _extract_relevant_info(self, memories: Dict[str, List]) -> str:
        """Extract relevant information from memories for the skill"""
        if not memories:
            return "No relevant memory context available."
        
        info_parts = []
        
        for mem_type, mem_list in memories.items():
            if mem_list:
                info_parts.append(f"\n{mem_type.upper()} MEMORIES:")
                for mem, score in mem_list[:3]:  # Top 3 from each type
                    info_parts.append(f"- {mem.content} (relevance: {score:.2f})")
        
        if not info_parts:
            return "No relevant memory context available."
        
        return "\n".join(info_parts)


class InvestmentSkill(FinancialSkill):
    """Investment advice skill"""
    
    def __init__(self):
        super().__init__(
            name="investment_advisor",
            description="Provides investment advice including stock, fund, bond, and asset allocation recommendations"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide investment advice based on user profile and memory"""
        
        persona = context.get("persona")
        memories = context.get("memories", {})
        
        # Build context from memories
        risk_tolerance = persona.risk_tolerance if persona else "moderate"
        
        # Get relevant memory context
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a professional investment advisor. Provide personalized investment advice.

User Profile:
- Risk Tolerance: {risk_tolerance}
- Age: {persona.age if persona else 'N/A'}
- Investment Cycle: {persona.investment_cycle_days if persona else 365} days

User's Financial Context:
{relevant_info}

User Query: {query}

Provide investment advice considering:
1. Risk tolerance alignment
2. Diversification recommendations
3. Short-term vs long-term investment strategy
4. Specific asset allocation suggestions

Respond in a helpful, professional manner."""
        
        return prompt


class TaxPlanningSkill(FinancialSkill):
    """Tax planning skill"""
    
    def __init__(self):
        super().__init__(
            name="tax_planner",
            description="Provides tax planning advice including deductions, credits, and tax-efficient strategies"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide tax planning advice"""
        
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a tax planning specialist. Provide tax-efficient financial advice.

User's Financial Context:
{relevant_info}

User Query: {query}

Provide tax planning advice considering:
1. Applicable tax deductions and credits
2. Tax-efficient investment strategies
3. Income timing considerations
4. Retirement account tax benefits

Respond in a helpful, professional manner."""
        
        return prompt


class RetirementPlanningSkill(FinancialSkill):
    """Retirement planning skill"""
    
    def __init__(self):
        super().__init__(
            name="retirement_planner",
            description="Provides retirement planning advice including retirement accounts and savings strategies"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide retirement planning advice"""
        
        persona = context.get("persona")
        memories = context.get("memories", {})
        
        age = persona.age if persona else 65
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a retirement planning specialist. Provide retirement savings and strategy advice.

User Profile:
- Current Age: {age}
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}

User's Financial Context:
{relevant_info}

User Query: {query}

Provide retirement planning advice considering:
1. Retirement account options (401k, IRA, etc.)
2. Savings rate recommendations
3. Target retirement age
4. Social Security considerations

Respond in a helpful, professional manner."""
        
        return prompt


class InsuranceSkill(FinancialSkill):
    """Insurance advice skill"""
    
    def __init__(self):
        super().__init__(
            name="insurance_advisor",
            description="Provides insurance advice including life, health, property, and disability insurance"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide insurance advice"""
        
        persona = context.get("persona")
        memories = context.get("memories", {})
        
        family_status = persona.family_status if persona else "single"
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are an insurance advisor. Provide personalized insurance recommendations.

User Profile:
- Family Status: {family_status}
- Age: {persona.age if persona else 'N/A'}

User's Financial Context:
{relevant_info}

User Query: {query}

Provide insurance advice considering:
1. Life insurance needs
2. Health insurance options
3. Disability insurance
4. Property insurance

Respond in a helpful, professional manner."""
        
        return prompt


class DebtManagementSkill(FinancialSkill):
    """Debt management skill"""
    
    def __init__(self):
        super().__init__(
            name="debt_manager",
            description="Provides debt management advice including repayment strategies and debt consolidation"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide debt management advice"""
        
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a debt management specialist. Provide debt repayment and management advice.

User's Financial Context:
{relevant_info}

User Query: {query}

Provide debt management advice considering:
1. Debt prioritization strategies
2. Repayment options (avalanche, snowball)
3. Debt consolidation considerations
4. Credit score improvement

Respond in a helpful, professional manner."""
        
        return prompt


class GeneralFinancialSkill(FinancialSkill):
    """General financial advisory skill"""
    
    def __init__(self):
        super().__init__(
            name="general_advisor",
            description="Provides general financial advice on various topics"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Provide general financial advice"""
        
        persona = context.get("persona")
        memories = context.get("memories", {})
        
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a financial advisor. Provide helpful financial guidance.

User Profile:
- Name: {persona.name if persona else 'User'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}

User's Financial Context:
{relevant_info}

User Query: {query}

Provide helpful, personalized financial advice."""

        return prompt


# ==================== V2 Skill Router (English Prompts) ====================

class UserPreferenceSkillV2(FinancialSkill):
    """User Preference Expert - V2"""
    
    def __init__(self):
        super().__init__(
            name="user_preference_expert",
            description="User Preference Expert - Analyzes user risk tolerance and financial goals"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        persona = context.get("persona")
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a professional financial preference analyst. Your expertise is understanding user risk tolerance, financial goals, and investment needs.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Occupation: {persona.occupation if persona else 'N/A'}
- Income Level: {persona.income_level if persona else 'N/A'}
- Family Status: {persona.family_status if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'} (conservative/moderate/aggressive)
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}
- Investment Cycle: {persona.investment_cycle_days if persona else 365} days

User History:
{relevant_info}

User Query: {query}

Provide professional analysis and advice based on user's personal information and history. Show deep understanding of user's personalized needs.

Respond in a helpful, professional manner."""
        return prompt


class FinancialPlanningSkillV2(FinancialSkill):
    """Financial Planning Expert - V2"""
    
    def __init__(self):
        super().__init__(
            name="financial_planning_expert",
            description="Financial Planning Expert - Provides comprehensive financial planning and asset allocation"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        persona = context.get("persona")
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a senior financial planning expert. Your expertise is creating comprehensive financial planning solutions for users.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Occupation: {persona.occupation if persona else 'N/A'}
- Income Level: {persona.income_level if persona else 'N/A'}
- Family Status: {persona.family_status if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}
- Short-term Perception: {persona.perceived_short_term_days if persona else 90} days
- Long-term Perception: {persona.perceived_long_term_days if persona else 1095} days

User Financial Status:
{relevant_info}

User Query: {query}

Provide comprehensive financial planning advice including:
1. Short-term and long-term financial goal planning
2. Income-expense balance and savings recommendations
3. Asset allocation strategy
4. Emergency fund planning

Respond in a professional and accessible manner."""
        return prompt


class InvestmentAdviceSkillV2(FinancialSkill):
    """Investment Advice Expert - V2"""
    
    def __init__(self):
        super().__init__(
            name="investment_advice_expert",
            description="Investment Advice Expert - Provides stock, fund, bond investment advice"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        persona = context.get("persona")
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a professional investment advisor. Your expertise is providing analysis and advice on various investment products.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}
- Investment Cycle: {persona.investment_cycle_days if persona else 365} days

User Investment History and Preferences:
{relevant_info}

User Query: {query}

Provide professional investment advice including:
1. Investment product analysis (stocks, funds, bonds, etc.)
2. Asset allocation recommendations
3. Risk control strategies
4. Investment timing advice

Respond with personalized advice based on user's risk tolerance."""
        return prompt


class FinancialSummarySkillV2(FinancialSkill):
    """Financial Summary Expert - V2"""
    
    def __init__(self):
        super().__init__(
            name="financial_summary_expert",
            description="Financial Summary Expert - Analyzes user financial status and provides summary reports"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        persona = context.get("persona")
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a professional financial analyst. Your expertise is analyzing user financial status and providing clear summary reports.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Income Level: {persona.income_level if persona else 'N/A'}
- Family Status: {persona.family_status if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}

User Financial Data:
{relevant_info}

User Query: {query}

Provide professional financial status analysis including:
1. Income and expense analysis
2. Assets and liabilities status
3. Savings and investment status
4. Financial health assessment

Respond in a clear and easy-to-understand report format."""
        return prompt


class FinancialCalculationSkillV2(FinancialSkill):
    """Financial Calculation Expert - V2"""
    
    def __init__(self):
        super().__init__(
            name="financial_calculation_expert",
            description="Financial Calculation Expert - Compound interest, retirement planning, tax calculation"
        )
    
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        persona = context.get("persona")
        memories = context.get("memories", {})
        relevant_info = self._extract_relevant_info(memories)
        
        prompt = f"""You are a professional financial calculation expert. Your expertise is performing various financial calculations including compound interest, retirement planning, and tax calculations.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Income Level: {persona.income_level if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}

User Relevant Financial Data:
{relevant_info}

User Query: {query}

Perform professional financial calculations including:
1. Compound interest calculations (investment returns, regular investments)
2. Retirement savings calculations
3. Tax-related calculations
4. Loan/ installment calculations

Respond with clear calculation process and final results with recommendations."""
        return prompt


class FinancialSkillRouterV2:
    """V2 Router - Domain-based skill routing with 5 experts"""
    
    QUERY_TYPE_KEYWORDS_V2 = {
        "user_preference": [
            "prefer", "risk", "goal", "suitable", "should", "whether", "can i",
            "recommend", "suggest", "适合", "risk preference", "investment goal", "financial goal",
            "conservative", "aggressive", "steady", "my risk"
        ],
        "financial_planning": [
            "plan", "allocation", "arrange", "分配", "方案",
            "planning", "allocation", "financial plan", "asset allocation", "fund allocation",
            "comprehensive", "综合"
        ],
        "investment_advice": [
            "invest", "stock", "fund", "bond", "理财", "return", "earn",
            "investment", "stock", "fund", "bond", "investment advice", "buy",
            "fund", "financial product", "return rate", "asset"
        ],
        "financial_summary": [
            "summary", "analysis", "status", "report", "查看", "了解",
            "summary", "analysis", "report", "how is", "summary",
            "financial status", "analysis report", "health", "check"
        ],
        "financial_calculation": [
            "calculate", "how much", "多少钱", "多少收益", "赚多少",
            "calculate", "compute", "how much", "compound interest", "retire",
            "loan", "interest", "monthly payment", "annual", "rate"
        ]
    }
    
    def __init__(self):
        self.skills_v2 = {
            "user_preference": UserPreferenceSkillV2(),
            "financial_planning": FinancialPlanningSkillV2(),
            "investment_advice": InvestmentAdviceSkillV2(),
            "financial_summary": FinancialSummarySkillV2(),
            "financial_calculation": FinancialCalculationSkillV2()
        }
    
    def classify_query_v2(self, query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for qtype, keywords in self.QUERY_TYPE_KEYWORDS_V2.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            scores[qtype] = score
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "financial_planning"
    
    def route_v2(self, query: str, context: Dict[str, Any]) -> tuple:
        query_type = self.classify_query_v2(query)
        skill = self.skills_v2[query_type]
        prompt = skill.execute(query, context)
        return query_type, skill.name, prompt


# ==================== Skill Router ====================

class FinancialSkillRouter:
    """
    Router that classifies queries and routes to appropriate financial skills
    """
    
    # Keywords for query classification
    QUERY_TYPE_KEYWORDS = {
        QueryType.INVESTMENT: [
            "invest", "stock", "fund", "bond", "portfolio", "asset", "allocation",
            "dividend", "return", " ETF ", "mutual fund", "rebalance", "买入", "投资",
            "基金", "股票", "债券", "资产配置", "收益率"
        ],
        QueryType.TAX: [
            "tax", "deduction", "credit", "irs", "filing", "refund", "withholding",
            "1099", "w-2", "taxable", "tax-efficient", "税务", "报税", "抵扣",
            "税收", "个人所得税"
        ],
        QueryType.RETIREMENT: [
            "retire", "401k", "ira", "pension", "social security", "retirement account",
            "401(k)", "退休", "养老金", "社保", "退休金", "IRA"
        ],
        QueryType.INSURANCE: [
            "insurance", "life insurance", "health insurance", "policy", "premium",
            "coverage", "claim", "保险", "人寿保险", "医疗保险", "保单", "保费"
        ],
        QueryType.DEBT: [
            "debt", "loan", "mortgage", "credit card", "interest rate", "repayment",
            "consolidation", "credit score", "欠款", "贷款", "债务", "信用卡",
            "还款", "信用分"
        ],
        QueryType.CONSUMPTION: [
            "spending", "expense", "budget", "save", "消费", "支出", "预算",
            "节约", "花费", "开支"
        ]
    }
    
    def __init__(self):
        # Initialize skills
        self.skills = {
            QueryType.INVESTMENT: InvestmentSkill(),
            QueryType.TAX: TaxPlanningSkill(),
            QueryType.RETIREMENT: RetirementPlanningSkill(),
            QueryType.INSURANCE: InsuranceSkill(),
            QueryType.DEBT: DebtManagementSkill(),
            QueryType.CONSUMPTION: GeneralFinancialSkill(),  # Use general for consumption
            QueryType.GENERAL: GeneralFinancialSkill()
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query type based on keywords"""
        
        query_lower = query.lower()
        
        # Score each query type
        scores = {}
        for qtype, keywords in self.QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            scores[qtype] = score
        
        # Get the type with highest score
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        # Default to general
        return QueryType.GENERAL
    
    def route(self, query: str, context: Dict[str, Any]) -> tuple:
        """
        Route query to appropriate skill
        
        Returns:
            (skill_name, prompt) - The skill to use and the constructed prompt
        """
        # Classify query
        query_type = self.classify_query(query)
        
        # Get the skill
        skill = self.skills[query_type]
        
        # Execute skill to get prompt
        prompt = skill.execute(query, context)
        
        return query_type.value, skill.name, prompt


# ==================== Main Agent ====================

class FinMemAgent:
    """
    Financial Memory Agent with long-short term preference modeling and skill routing
    Supports V1 (6 skills) and V2 (5 skills) skill routing
    """
    
    def __init__(
        self,
        user_persona: UserPersona,
        llm_config: Optional[Dict[str, Any]] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        enable_skill_routing: bool = True,
        skill_version: Literal["v1", "v2"] = "v1"
    ):
        # Initialize memory system
        self.memory = FinancialMemorySystem(
            user_persona=user_persona,
            embedding_model=embedding_model
        )
        
        # Initialize skill router (can be disabled)
        self.enable_skill_routing = enable_skill_routing
        self.skill_version = skill_version
        
        if enable_skill_routing:
            if skill_version == "v2":
                self.router = FinancialSkillRouterV2()
            else:
                self.router = FinancialSkillRouter()
        
        # LLM configuration
        self.llm_config = llm_config or {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.vveai.com/v1"
        }
        
        # Initialize LLM client
        if self.llm_config.get("api_key"):
            self.llm_client = OpenAI(
                api_key=self.llm_config["api_key"],
                base_url=self.llm_config.get("base_url", "https://api.vveai.com/v1")
            )
        else:
            self.llm_client = None
        
        # Conversation history
        self.conversation_history: List[DialogueTurn] = []
    
    def switch_skill_version(self, version: Literal["v1", "v2"]):
        """Switch skill routing version at runtime"""
        self.skill_version = version
        if self.enable_skill_routing:
            if version == "v2":
                self.router = FinancialSkillRouterV2()
            else:
                self.router = FinancialSkillRouter()
        print(f"Switched to skill version: {version.upper()}")
    
    def _build_general_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build a general prompt without skill routing"""
        
        persona = context.get("persona")
        memories = context.get("memories", {})
        
        # Extract relevant info from memories
        info_parts = []
        for mem_type, mem_list in memories.items():
            if mem_list:
                info_parts.append(f"\n{mem_type.upper()} MEMORIES:")
                for mem, score in mem_list[:3]:
                    info_parts.append(f"- {mem.content}")
        
        relevant_info = "\n".join(info_parts) if info_parts else "No relevant memory context available."
        
        # Build general prompt
        prompt = f"""You are a personalized financial advisor assistant. Provide helpful financial guidance based on the user's profile and context.

User Profile:
- Name: {persona.name if persona else 'User'}
- Age: {persona.age if persona else 'N/A'}
- Occupation: {persona.occupation if persona else 'N/A'}
- Income Level: {persona.income_level if persona else 'N/A'}
- Family Status: {persona.family_status if persona else 'N/A'}
- Risk Tolerance: {persona.risk_tolerance if persona else 'moderate'}
- Financial Goals: {', '.join(persona.financial_goals) if persona and persona.financial_goals else 'N/A'}

User's Financial Context:
{relevant_info}

User Query: {query}

Provide personalized, helpful financial advice considering the user's profile, goals, and financial situation."""
        
        return prompt
    
    def _get_query_type_for_memory(self, query: str):
        """Get query type for memory retrieval (handles both V1 and V2)"""
        if self.skill_version == "v2":
            # V2 uses string-based classification
            return self.router.classify_query_v2(query)
        else:
            # V1 uses enum-based classification
            return self.router.classify_query(query)
    
    def _route_query(self, query: str, context: Dict[str, Any]):
        """Route query to skill (handles both V1 and V2)"""
        if self.skill_version == "v2":
            return self.router.route_v2(query, context)
        else:
            return self.router.route(query, context)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query"""
        
        # Determine query type for memory retrieval
        if self.enable_skill_routing:
            query_type_for_mem = self._get_query_type_for_memory(query)
            # Convert to V1 QueryType for memory retrieval (or use default)
            if self.skill_version == "v2":
                # Map V2 query types to V1 for memory retrieval
                query_type_mapping = {
                    "user_preference": QueryType.GENERAL,
                    "financial_planning": QueryType.INVESTMENT,
                    "investment_advice": QueryType.INVESTMENT,
                    "financial_summary": QueryType.GENERAL,
                    "financial_calculation": QueryType.GENERAL
                }
                query_type_enum = query_type_mapping.get(query_type_for_mem, QueryType.GENERAL)
            else:
                query_type_enum = query_type_for_mem
        else:
            query_type_enum = QueryType.GENERAL
        
        # Get memories from different categories
        memories = self.memory.get_memories_for_query(
            query=query,
            query_type=query_type_enum,
            max_memories=10
        )
        
        # Build context
        context = {
            "persona": self.memory.user_persona,
            "memories": memories,
            "conversation_history": self.conversation_history[-5:]  # Last 5 turns
        }
        
        # Route to skill or use general prompt
        if self.enable_skill_routing:
            query_type, skill_name, skill_prompt = self._route_query(query, context)
        else:
            query_type = "general"
            skill_name = "general_advisor"
            skill_prompt = self._build_general_prompt(query, context)
        
        # Generate response using LLM
        response = self._generate_response(skill_prompt)
        
        # Save conversation turn
        turn = DialogueTurn(
            turn_id=f"turn_{len(self.conversation_history)}",
            speaker="user",
            message=query,
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            query_type=query_type,
            skill_used=skill_name,
            memory_retrieved=[m.memory_id for mem_list in memories.values() for m, _ in mem_list]
        )
        self.conversation_history.append(turn)
        
        # Add assistant response
        assistant_turn = DialogueTurn(
            turn_id=f"turn_{len(self.conversation_history)}",
            speaker="assistant",
            message=response,
            timestamp=datetime.now().strftime("%Y%m%d%H%M"),
            query_type=query_type,
            skill_used=skill_name
        )
        self.conversation_history.append(assistant_turn)
        
        return {
            "query": query,
            "response": response,
            "query_type": query_type,
            "skill_used": skill_name,
            "memories_retrieved": turn.memory_retrieved
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        
        if not self.llm_client:
            # Mock response if no LLM available
            return "This is a placeholder response. Please configure the LLM API."
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def add_financial_event(self, event: FinancialEvent):
        """Add a financial event and update memory"""
        self.memory.add_financial_event(event)
    
    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float = 1.0,
        keywords: Optional[List[str]] = None
    ):
        """Add a memory block"""
        self.memory.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            keywords=keywords
        )
    
    def get_memory_summary(self) -> str:
        """Get summary of user's financial memory"""
        return self.memory.get_context_summary()


# ==================== Dialogue-Only Agent (LLM-enhanced) ====================

class DialogueOnlyFinMemAgent:
    """
    Financial Memory Agent that builds memory ONLY from dialogues using LLM-based extraction.
    Does NOT use user persona or timeline - only extracts information from conversations.
    
    This is designed for scenarios where user profile and timeline are not accessible.
    Uses LLM for intelligent memory extraction instead of keyword matching.
    
    Supports separate LLM models for:
    - Memory extraction (extract memories from dialogues)
    - Response generation (answer user queries)
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        memory_llm_config: Optional[Dict[str, Any]] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        enable_skill_routing: bool = True,
        skill_version: Literal["v1", "v2"] = "v1"
    ):
        # LLM configuration for response generation
        self.llm_config = llm_config or {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": "https://api.vveai.com/v1"
        }
        
        # LLM configuration for memory extraction (can be different from main LLM)
        self.memory_llm_config = memory_llm_config or self.llm_config
        
        # Initialize dialogue-only memory system with separate LLM config for memory extraction
        self.memory = DialogueOnlyMemorySystem(
            embedding_model=embedding_model,
            llm_config=self.memory_llm_config
        )
        
        # Initialize skill router (can be disabled)
        self.enable_skill_routing = enable_skill_routing
        self.skill_version = skill_version
        
        if enable_skill_routing:
            if skill_version == "v2":
                self.router = FinancialSkillRouterV2()
            else:
                self.router = FinancialSkillRouter()
        
        # Initialize LLM client for response generation
        if self.llm_config.get("api_key"):
            self.llm_client = OpenAI(
                api_key=self.llm_config["api_key"],
                base_url=self.llm_config.get("base_url", "https://api.vveai.com/v1")
            )
        else:
            self.llm_client = None
        
        # Conversation history
        self.conversation_history: List[DialogueTurnInfo] = []
    
    def add_dialogue_turn(
        self,
        speaker: str,
        message: str,
        turn_number: Optional[int] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ):
        """Add a dialogue turn to the conversation history"""
        turn = DialogueTurnInfo(
            turn_id=f"turn_{len(self.conversation_history)}",
            speaker=speaker,
            message=message,
            turn_number=turn_number or len(self.conversation_history),
            session_id=session_id,
            timestamp=timestamp
        )
        self.conversation_history.append(turn)
        self.memory.add_dialogue_turn(turn)
    
    def build_from_sessions(self, sessions: List[Dict[str, Any]]):
        """Build memory from dialogue sessions"""
        self.memory.build_memories_from_sessions(sessions)
        
        # Also populate conversation history
        for session in sessions:
            for turn_data in session.get("turns", []):
                turn = DialogueTurnInfo(
                    turn_id=f"{session.get('session_id', 'unknown')}_{turn_data.get('turn_number', 0)}",
                    speaker=turn_data.get("speaker", "user"),
                    message=turn_data.get("message", ""),
                    turn_number=turn_data.get("turn_number", 0),
                    session_id=session.get("session_id"),
                    timestamp=turn_data.get("timestamp")
                )
                self.conversation_history.append(turn)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using dialogue-only context"""
        
        # Get relevant memories from dialogue
        dialogue_memories = self.memory.get_relevant_memories(query, max_memories=10)
        
        # Build context (without persona - only dialogue context)
        context = {
            "persona": None,  # No persona available
            "memories": dialogue_memories,
            "dialogue_context": self.memory.get_context_for_skill("general", query)
        }
        
        # Route to skill or use general prompt
        if self.enable_skill_routing:
            query_type, skill_name, skill_prompt = self._route_query_with_dialogue_context(query, context)
        else:
            query_type = "general"
            skill_name = "general_advisor"
            skill_prompt = self._build_dialogue_only_prompt(query, context)
        
        # Generate response using LLM
        response = self._generate_response(skill_prompt)
        
        return {
            "query": query,
            "response": response,
            "query_type": query_type,
            "skill_used": skill_name,
            "dialogue_context": dialogue_memories
        }
    
    def _route_query_with_dialogue_context(self, query: str, context: Dict[str, Any]):
        """Route query to skill with dialogue-only context"""
        
        if self.skill_version == "v2":
            query_type = self.router.classify_query_v2(query)
            skill = self.router.skills_v2[query_type]
            
            # Build dialogue-only context for V2 skills
            dialogue_context = self.memory.get_context_for_skill(skill.name, query)
            
            # Create prompt with dialogue context only
            prompt = self._build_dialogue_prompt_v2(query, skill.name, dialogue_context)
            return query_type, skill.name, prompt
        else:
            query_type = self.router.classify_query(query)
            skill = self.router.skills[query_type]
            
            # Build dialogue-only context
            dialogue_context = self.memory.get_context_for_skill(skill.name, query)
            
            # Create prompt with dialogue context only
            prompt = self._build_dialogue_prompt_v1(query, skill.name, dialogue_context)
            return query_type.value, skill.name, prompt
    
    def _build_dialogue_prompt_v1(self, query: str, skill_name: str, dialogue_context: str) -> str:
        """Build prompt for V1 skills using only dialogue context"""
        
        skill_descriptions = {
            "investment_advisor": "You are a professional investment advisor.",
            "tax_planner": "You are a tax planning specialist.",
            "retirement_planner": "You are a retirement planning specialist.",
            "insurance_advisor": "You are an insurance advisor.",
            "debt_manager": "You are a debt management specialist.",
            "general_advisor": "You are a financial advisor."
        }
        
        skill_desc = skill_descriptions.get(skill_name, "You are a financial advisor.")
        
        prompt = f"""{skill_desc}

Based on the user's conversation history and questions, provide personalized financial advice.

Conversation Context:
{dialogue_context}

User Query: {query}

Provide helpful financial advice based on the conversation context above. If there is insufficient information, acknowledge what you know from the conversation and provide general guidance.

Respond in a helpful, professional manner."""
        
        return prompt
    
    def _build_dialogue_prompt_v2(self, query: str, skill_name: str, dialogue_context: str) -> str:
        """Build prompt for V2 skills using only dialogue context"""
        
        prompt = f"""You are a professional financial advisor. Based on the user's conversation history and questions, provide personalized financial advice.

Conversation Context:
{dialogue_context}

User Query: {query}

Provide helpful financial advice based on the conversation context above. If there is insufficient information, acknowledge what you know from the conversation and provide general guidance.

Respond in a helpful, professional manner."""
        
        return prompt
    
    def _build_dialogue_only_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build a general prompt without skill routing"""
        
        dialogue_context = context.get("dialogue_context", "No conversation history available.")
        
        prompt = f"""You are a personalized financial advisor assistant. Provide helpful financial guidance based on the user's conversation history.

Conversation Context:
{dialogue_context}

User Query: {query}

Provide personalized, helpful financial advice considering what you've learned from the conversation."""

        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        
        if not self.llm_client:
            # Mock response if no LLM available
            return "This is a placeholder response. Please configure the LLM API."
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a helpful financial advisor assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_dialogue_summary(self) -> str:
        """Get summary of dialogue-based memory"""
        return self.memory.get_dialogue_context_summary()


# ==================== Example Usage ====================

def create_sample_persona() -> UserPersona:
    """Create a sample user persona"""
    return UserPersona(
        user_id="user_001",
        name="张三",
        age=35,
        occupation="软件工程师",
        income_level="中产",
        family_status="已婚",
        risk_tolerance="moderate",
        financial_goals=["退休储蓄", "子女教育", "购房"],
        perceived_short_term_days=90,
        perceived_medium_term_days=365,
        perceived_long_term_days=1825,  # 5 years
        investment_cycle_days=730,  # 2 years
        short_term_weight=0.3,
        long_term_weight=0.7
    )


def main():
    """Example usage of FinMem agent"""
    
    # Create user persona
    persona = create_sample_persona()
    
    # Initialize agent
    agent = FinMemAgent(
        user_persona=persona,
        llm_config={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY")
        }
    )
    
    # Add some sample financial events
    sample_events = [
        FinancialEvent(
            event_id="evt_001",
            timestamp="202501150000",
            event_type="investment",
            amount=50000,
            category="股票",
            description="买入科技股基金",
            impact="positive",
            is_recurring=False
        ),
        FinancialEvent(
            event_id="evt_002",
            timestamp="202502010000",
            event_type="income",
            amount=20000,
            category="工资",
            description="月薪收入",
            impact="positive",
            is_recurring=True,
            periodicity="monthly"
        ),
        FinancialEvent(
            event_id="evt_003",
            timestamp="202502100000",
            event_type="expense",
            amount=3000,
            category="房租",
            description="月度房租",
            impact="negative",
            is_recurring=True,
            periodicity="monthly"
        )
    ]
    
    for event in sample_events:
        agent.add_financial_event(event)
    
    # Add some preference memories
    agent.add_memory(
        content="用户偏好稳健型投资，不喜欢高风险产品",
        memory_type="preference",
        importance=1.5,
        keywords=["风险偏好", "稳健", "保守"]
    )
    
    agent.add_memory(
        content="用户每月固定储蓄比例为收入的30%",
        memory_type="behavior",
        importance=1.2,
        keywords=["储蓄", "收入比例", "理财习惯"]
    )
    
    # Print memory summary
    print("=" * 60)
    print("User Financial Memory Summary")
    print("=" * 60)
    print(agent.get_memory_summary())
    print()
    
    # Test queries
    test_queries = [
        "我应该如何配置我的投资组合？",
        "有哪些税收优惠的退休账户适合我？",
        "我应该购买人寿保险吗？",
        "如何管理我的信用卡债务？"
    ]
    
    print("=" * 60)
    print("Testing Queries")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Route query
        query_type_enum = agent.router.classify_query(query)
        print(f"Query Type: {query_type_enum.value}")
        
        # Get skill info
        memories = agent.memory.get_memories_for_query(query, query_type_enum)
        context = {
            "persona": persona,
            "memories": memories,
            "conversation_history": []
        }
        _, skill_name, _ = agent.router.route(query, context)
        print(f"Skill Used: {skill_name}")
        print("-" * 40)


if __name__ == "__main__":
    main()
