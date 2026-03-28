"""
MemoryBank: 为大语言模型提供长期记忆能力
基于论文 "MemoryBank: Enhancing Large Language Models with Long-Term Memory" 复现
"""

import os
import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# 需要安装的库
# pip install openai sentence-transformers faiss-cpu langchain

import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryItem:
    """单条记忆项"""
    id: str
    content: str
    timestamp: datetime
    memory_strength: int = 1  # 记忆强度 S
    last_access_time: datetime = None
    embedding: np.ndarray = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.last_access_time is None:
            self.last_access_time = self.timestamp


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    daily_personalities: Dict[str, str] = field(default_factory=dict)  # 日期 -> 性格分析
    global_personality: str = ""
    daily_events: Dict[str, str] = field(default_factory=dict)  # 日期 -> 事件摘要
    global_events: str = ""


class EbbinghausForgettingCurve:
    """
    艾宾浩斯遗忘曲线实现
    R = e^(-t/S)
    R: 记忆保留率
    t: 自学习以来经过的时间
    S: 记忆强度
    """
    
    @staticmethod
    def calculate_retention(time_elapsed_days: float, memory_strength: int) -> float:
        """计算记忆保留率"""
        if memory_strength <= 0:
            return 0.0
        return math.exp(-time_elapsed_days / memory_strength)
    
    @staticmethod
    def should_forget(retention: float, threshold: float = 0.3) -> bool:
        """判断是否应该遗忘（保留率低于阈值）"""
        return retention < threshold


class MemoryStorage:
    """记忆存储仓库"""
    
    def __init__(self, user_id: str, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.user_id = user_id
        self.memories: Dict[str, MemoryItem] = {}
        self.user_profile = UserProfile(user_id=user_id)
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 初始化FAISS索引
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # 内积相似度
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
    def add_memory(self, content: str, timestamp: datetime = None, metadata: Dict = None) -> str:
        """添加新记忆"""
        if timestamp is None:
            timestamp = datetime.now()
        if metadata is None:
            metadata = {}
            
        memory_id = f"mem_{len(self.memories)}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        
        # 生成嵌入向量
        embedding = self.embedding_model.encode(content, normalize_embeddings=True)
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            embedding=embedding,
            metadata=metadata
        )
        
        self.memories[memory_id] = memory_item
        
        # 添加到FAISS索引
        idx = len(self.id_to_index)
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.id_to_index[memory_id] = idx
        self.index_to_id[idx] = memory_id
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取指定记忆"""
        return self.memories.get(memory_id)
    
    def get_all_memories(self) -> List[MemoryItem]:
        """获取所有记忆"""
        return list(self.memories.values())
    
    def update_daily_events(self, date: str, events_summary: str):
        """更新每日事件摘要"""
        self.user_profile.daily_events[date] = events_summary
        
    def update_global_events(self, global_summary: str):
        """更新全局事件摘要"""
        self.user_profile.global_events = global_summary
        
    def update_daily_personality(self, date: str, personality: str):
        """更新每日性格分析"""
        self.user_profile.daily_personalities[date] = personality
        
    def update_global_personality(self, global_personality: str):
        """更新全局性格分析"""
        self.user_profile.global_personality = global_personality


class MemoryRetriever:
    """记忆检索器"""
    
    def __init__(self, memory_storage: MemoryStorage):
        self.storage = memory_storage
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """检索与查询最相关的记忆"""
        if len(self.storage.memories) == 0:
            return []
            
        # 编码查询
        query_embedding = self.storage.embedding_model.encode(
            query, normalize_embeddings=True
        ).reshape(1, -1).astype('float32')
        
        # FAISS检索
        k = min(top_k, len(self.storage.memories))
        scores, indices = self.storage.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            memory_id = self.storage.index_to_id.get(idx)
            if memory_id:
                memory = self.storage.get_memory(memory_id)
                if memory:
                    results.append((memory, float(score)))
                    
        return results
    
    def retrieve_by_time_range(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[MemoryItem]:
        """按时间范围检索记忆"""
        return [
            mem for mem in self.storage.get_all_memories()
            if start_time <= mem.timestamp <= end_time
        ]


class MemoryUpdater:
    """记忆更新器 - 基于艾宾浩斯遗忘曲线"""
    
    def __init__(
        self, 
        memory_storage: MemoryStorage, 
        forgetting_threshold: float = 0.3
    ):
        self.storage = memory_storage
        self.forgetting_curve = EbbinghausForgettingCurve()
        self.forgetting_threshold = forgetting_threshold
        
    def reinforce_memory(self, memory_id: str):
        """
        强化记忆：当记忆被回忆时调用
        增加记忆强度S，重置时间t
        """
        memory = self.storage.get_memory(memory_id)
        if memory:
            memory.memory_strength += 1
            memory.last_access_time = datetime.now()
            
    def update_all_memories(self, current_time: datetime = None):
        """
        更新所有记忆的状态
        根据遗忘曲线计算保留率，标记需要遗忘的记忆
        """
        if current_time is None:
            current_time = datetime.now()
            
        memories_to_forget = []
        
        for memory_id, memory in self.storage.memories.items():
            # 计算时间差（天）
            time_elapsed = (current_time - memory.last_access_time).total_seconds() / 86400
            
            # 计算保留率
            retention = self.forgetting_curve.calculate_retention(
                time_elapsed, memory.memory_strength
            )
            
            # 判断是否应该遗忘
            if self.forgetting_curve.should_forget(retention, self.forgetting_threshold):
                memories_to_forget.append(memory_id)
                
        return memories_to_forget
    
    def forget_memories(self, memory_ids: List[str]):
        """执行遗忘操作"""
        for memory_id in memory_ids:
            if memory_id in self.storage.memories:
                del self.storage.memories[memory_id]
        # 注意：实际应用中需要重建FAISS索引


class LLMInterface:
    """LLM接口抽象类"""
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError
        
    def summarize_events(self, conversations: str) -> str:
        raise NotImplementedError
        
    def analyze_personality(self, conversations: str) -> str:
        raise NotImplementedError


class OpenAILLM(LLMInterface):
    """OpenAI API接口实现"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def summarize_events(self, conversations: str) -> str:
        prompt = f"Summarize the events and key information in the following content:\n\n{conversations}"
        return self.generate(prompt)
    
    def analyze_personality(self, conversations: str) -> str:
        prompt = f"Based on the following dialogue, please summarize the user's personality traits and emotions:\n\n{conversations}"
        return self.generate(prompt)


class MemoryBank:
    """
    MemoryBank主类
    整合存储、检索、更新功能
    """
    
    def __init__(
        self,
        user_id: str,
        llm: LLMInterface,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_forgetting: bool = True,
        forgetting_threshold: float = 0.3
    ):
        self.user_id = user_id
        self.llm = llm
        self.enable_forgetting = enable_forgetting
        
        # 初始化组件
        self.storage = MemoryStorage(user_id, embedding_model)
        self.retriever = MemoryRetriever(self.storage)
        self.updater = MemoryUpdater(self.storage, forgetting_threshold)
        
        # 对话历史缓存
        self.current_session_dialogues: List[Dict] = []
        
    def add_dialogue(self, role: str, content: str, timestamp: datetime = None):
        """添加对话到当前会话"""
        if timestamp is None:
            timestamp = datetime.now()
            
        dialogue = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.current_session_dialogues.append(dialogue)
        
        # 将用户消息添加到长期记忆
        if role == "user":
            self.storage.add_memory(
                content=content,
                timestamp=timestamp,
                metadata={"role": role, "session": "current"}
            )
            
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[str]:
        """检索相关记忆"""
        results = self.retriever.retrieve(query, top_k)
        
        # 强化被检索到的记忆
        for memory, score in results:
            self.updater.reinforce_memory(memory.id)
            
        return [mem.content for mem, _ in results]
    
    def end_session(self):
        """
        结束当前会话
        - 生成事件摘要
        - 分析用户性格
        - 更新用户画像
        """
        if not self.current_session_dialogues:
            return
            
        # 格式化对话内容
        dialogue_text = "\n".join([
            f"{d['role']}: {d['content']}" 
            for d in self.current_session_dialogues
        ])
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 生成每日事件摘要
        events_summary = self.llm.summarize_events(dialogue_text)
        self.storage.update_daily_events(today, events_summary)
        
        # 分析用户性格
        personality = self.llm.analyze_personality(dialogue_text)
        self.storage.update_daily_personality(today, personality)
        
        # 更新全局摘要
        self._update_global_summaries()
        
        # 清空当前会话
        self.current_session_dialogues = []
        
        # 执行记忆遗忘（如果启用）
        if self.enable_forgetting:
            memories_to_forget = self.updater.update_all_memories()
            self.updater.forget_memories(memories_to_forget)
            
    def _update_global_summaries(self):
        """更新全局摘要"""
        # 合并所有每日事件
        all_events = "\n".join(self.storage.user_profile.daily_events.values())
        if all_events:
            global_events = self.llm.summarize_events(all_events)
            self.storage.update_global_events(global_events)
            
        # 合并所有性格分析
        all_personalities = "\n".join(
            self.storage.user_profile.daily_personalities.values()
        )
        if all_personalities:
            prompt = f"""The following are the user's exhibited personality traits 
and emotions throughout multiple days. Please provide a highly concise and 
general summary of the user's personality:

{all_personalities}"""
            global_personality = self.llm.generate(prompt)
            self.storage.update_global_personality(global_personality)
            
    def get_user_profile(self) -> Dict:
        """获取用户画像"""
        return {
            "user_id": self.user_id,
            "global_personality": self.storage.user_profile.global_personality,
            "global_events": self.storage.user_profile.global_events,
            "daily_events": self.storage.user_profile.daily_events,
            "daily_personalities": self.storage.user_profile.daily_personalities
        }
    
    def build_augmented_prompt(self, user_message: str) -> str:
        """
        构建增强提示词
        包含：相关记忆、用户画像、事件摘要
        """
        # 检索相关记忆
        relevant_memories = self.retrieve_relevant_memories(user_message)
        
        # 构建提示词
        prompt_parts = []
        
        # 添加用户画像
        if self.storage.user_profile.global_personality:
            prompt_parts.append(f"User Personality: {self.storage.user_profile.global_personality}")
            
        # 添加事件摘要
        if self.storage.user_profile.global_events:
            prompt_parts.append(f"Past Events Summary: {self.storage.user_profile.global_events}")
            
        # 添加相关记忆
        if relevant_memories:
            memories_text = "\n".join([f"- {mem}" for mem in relevant_memories])
            prompt_parts.append(f"Relevant Memories:\n{memories_text}")
            
        # 添加当前消息
        prompt_parts.append(f"Current User Message: {user_message}")
        
        return "\n\n".join(prompt_parts)


class SiliconFriend:
    """
    SiliconFriend: 基于MemoryBank的AI伴侣聊天机器人
    """
    
    SYSTEM_PROMPT = """You are SiliconFriend, a caring and empathetic AI companion. 
You have the ability to remember past conversations with the user and understand their personality.
Based on the context provided, give thoughtful, personalized responses that show you remember 
and care about the user. Be warm, supportive, and helpful."""
    
    def __init__(
        self,
        user_id: str,
        llm: LLMInterface,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_forgetting: bool = True
    ):
        self.memory_bank = MemoryBank(
            user_id=user_id,
            llm=llm,
            embedding_model=embedding_model,
            enable_forgetting=enable_forgetting
        )
        self.llm = llm
        
    def chat(self, user_message: str) -> str:
        """处理用户消息并生成回复"""
        # 添加用户消息到记忆
        self.memory_bank.add_dialogue("user", user_message)
        
        # 构建增强提示词
        augmented_prompt = self.memory_bank.build_augmented_prompt(user_message)
        
        # 生成回复
        response = self.llm.generate(augmented_prompt, self.SYSTEM_PROMPT)
        
        # 添加AI回复到会话
        self.memory_bank.add_dialogue("assistant", response)
        
        return response
    
    def end_conversation(self):
        """结束对话会话"""
        self.memory_bank.end_session()
        
    def get_user_profile(self) -> Dict:
        """获取用户画像"""
        return self.memory_bank.get_user_profile()


# ==================== 使用示例 ====================

def demo_without_api():
    """无需API的演示（使用模拟LLM）"""
    
    class MockLLM(LLMInterface):
        """模拟LLM用于测试"""
        def generate(self, prompt: str, system_prompt: str = None) -> str:
            return f"[Mock Response] I understand your message."
        
        def summarize_events(self, conversations: str) -> str:
            return "User discussed various topics including daily life and interests."
        
        def analyze_personality(self, conversations: str) -> str:
            return "The user appears to be curious, friendly, and open-minded."
    
    print("=" * 60)
    print("MemoryBank 演示 (使用模拟LLM)")
    print("=" * 60)
    
    # 创建SiliconFriend实例
    mock_llm = MockLLM()
    bot = SiliconFriend(
        user_id="demo_user",
        llm=mock_llm,
        enable_forgetting=False  # 演示时禁用遗忘
    )
    
    # 模拟多轮对话
    conversations = [
        "Hi, I'm learning Python programming.",
        "Can you recommend a good book for beginners?",
        "I'm also interested in machine learning.",
        "What projects should I start with?"
    ]
    
    print("\n--- 第一天对话 ---")
    for msg in conversations:
        print(f"User: {msg}")
        response = bot.chat(msg)
        print(f"Bot: {response}\n")
    
    # 结束会话
    bot.end_conversation()
    
    # 显示用户画像
    print("\n--- 用户画像 ---")
    profile = bot.get_user_profile()
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    
    # 新一天的对话 - 测试记忆检索
    print("\n--- 第二天对话（测试记忆检索）---")
    query = "What did I say about programming?"
    print(f"User: {query}")
    
    # 检索相关记忆
    memories = bot.memory_bank.retrieve_relevant_memories(query)
    print(f"\n检索到的相关记忆:")
    for i, mem in enumerate(memories, 1):
        print(f"  {i}. {mem}")
    
    response = bot.chat(query)
    print(f"\nBot: {response}")


def demo_with_openai():
    """使用OpenAI API的完整演示"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("请设置OPENAI_API_KEY环境变量")
        return
    
    print("=" * 60)
    print("MemoryBank 演示 (使用OpenAI API)")
    print("=" * 60)
    
    # 创建LLM实例
    llm = OpenAILLM(api_key=api_key, model="gpt-3.5-turbo")
    
    # 创建SiliconFriend实例
    bot = SiliconFriend(
        user_id="user_001",
        llm=llm,
        enable_forgetting=True,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # 交互式对话
    print("\n开始与SiliconFriend对话（输入'quit'退出，'end'结束当天会话）\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            bot.end_conversation()
            print("对话已结束，再见！")
            break
        elif user_input.lower() == 'end':
            bot.end_conversation()
            print("[会话已保存，开始新的一天]\n")
            continue
        elif user_input.lower() == 'profile':
            print("\n--- 当前用户画像 ---")
            print(json.dumps(bot.get_user_profile(), indent=2, ensure_ascii=False))
            print()
            continue
        elif not user_input:
            continue
            
        response = bot.chat(user_input)
        print(f"SiliconFriend: {response}\n")


def test_forgetting_mechanism():
    """测试遗忘机制"""
    print("=" * 60)
    print("艾宾浩斯遗忘曲线测试")
    print("=" * 60)
    
    curve = EbbinghausForgettingCurve()
    
    # 测试不同时间和记忆强度下的保留率
    print("\n记忆保留率 R = e^(-t/S)")
    print("-" * 40)
    print(f"{'时间(天)':<10} {'强度S=1':<12} {'强度S=2':<12} {'强度S=3':<12}")
    print("-" * 40)
    
    for days in [0, 1, 2, 3, 5, 7, 14, 30]:
        r1 = curve.calculate_retention(days, 1)
        r2 = curve.calculate_retention(days, 2)
        r3 = curve.calculate_retention(days, 3)
        print(f"{days:<10} {r1:<12.4f} {r2:<12.4f} {r3:<12.4f}")
    
    print("\n结论：记忆强度越高，遗忘速度越慢")


if __name__ == "__main__":
    # 运行演示
    print("\n选择演示模式:")
    print("1. 无需API的基础演示")
    print("2. 使用OpenAI API的完整演示")
    print("3. 测试遗忘机制")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        demo_without_api()
    elif choice == "2":
        demo_with_openai()
    elif choice == "3":
        test_forgetting_mechanism()
    else:
        print("运行默认演示...")
        demo_without_api()
        test_forgetting_mechanism()