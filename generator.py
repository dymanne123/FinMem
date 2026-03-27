"""
Finance Dialogue Dataset Generator

A comprehensive framework for generating multi-turn financial assistant dialogue datasets.
Uses GPT-4o-mini for high-quality dialogue generation.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import openai

# ============== Configuration ==============
@dataclass
class Config:
    """Configuration for dataset generation."""
    api_key: str = ""
    base_url: str = ""  # Custom API endpoint (e.g., for private LLM services)
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    num_personas: int = 100
    sessions_per_persona: int = 5
    turns_per_session: int = 15
    output_dir: str = "./output"
    seed: int = 42
    # Image generation options
    generate_images: bool = False  # Whether to generate real images (requires matplotlib)
    image_output_dir: str = "./output/images"  # Directory to save generated images
    image_probability: float = 0.3  # Probability of generating an image for a session (0.0-1.0)

# ============== Persona Templates ==============
# Fixed personas from persona.xlsx
FIXED_PERSONAS = [
    {
        "name": "Alex Rivera",
        "gender": "Male",
        "age": 34,
        "occupation": "Restaurant Owner",
        "income_level": "$90,000",
        "family_status": "Married, planning child within 2 years",
        "financial_knowledge_level": "Advanced",
        "financial_goals": ["Build childcare fund within 2 years", "Medium-term wealth growth"],
        "risk_tolerance": "Aggressive",
        "region": "North America",
        "life_cycle_stage": "Early Career",
        "annual_income": 90000,
        "net_worth": 180000,
        "investable_assets": 120000,
        "investment_experience_years": 13,
        "financial_context": "High liquidity need; variable business income"
    },
    {
        "name": "Carlos Silva",
        "gender": "Male",
        "age": 23,
        "occupation": "Junior Software Developer",
        "income_level": "$28,000",
        "family_status": "Single",
        "financial_knowledge_level": "Basic",
        "financial_goals": ["Long-term capital appreciation", "First home purchase in 8-10 years"],
        "risk_tolerance": "Growth",
        "region": "Latin America",
        "life_cycle_stage": "Early Career",
        "annual_income": 28000,
        "net_worth": 25000,
        "investable_assets": 10000,
        "investment_experience_years": 1,
        "financial_context": "Limited emergency savings; stable salary"
    },
    {
        "name": "Hans Mueller",
        "gender": "Male",
        "age": 42,
        "occupation": "Senior Corporate Executive",
        "income_level": "$150,000",
        "family_status": "Married with 2 children (ages 8 & 12)",
        "financial_knowledge_level": "Intermediate",
        "financial_goals": ["Education funding", "Retirement planning"],
        "risk_tolerance": "Conservative",
        "region": "Europe",
        "life_cycle_stage": "Family Accumulation",
        "annual_income": 150000,
        "net_worth": 1200000,
        "investable_assets": 800000,
        "investment_experience_years": 22,
        "financial_context": "Mortgage outstanding; tax-efficient investing important"
    },
    {
        "name": "Ahmed Al-Fayed",
        "gender": "Male",
        "age": 55,
        "occupation": "Business Owner",
        "income_level": "$220,000",
        "family_status": "Married, children financially independent",
        "financial_knowledge_level": "Professional",
        "financial_goals": ["Preparing for retirement in 8-10 years"],
        "risk_tolerance": "Balanced",
        "region": "Middle East",
        "life_cycle_stage": "Peak Wealth",
        "annual_income": 220000,
        "net_worth": 3500000,
        "investable_assets": 2800000,
        "investment_experience_years": 25,
        "financial_context": "Significant real estate exposure; diversification needed"
    },
    {
        "name": "Wei Chen",
        "gender": "Male",
        "age": 66,
        "occupation": "Retired Civil Servant",
        "income_level": "$22,000",
        "family_status": "Married, grandchildren",
        "financial_knowledge_level": "Basic",
        "financial_goals": ["Wealth preservation", "Inheritance/estate planning"],
        "risk_tolerance": "Very Conservative",
        "region": "Asia-Pacific",
        "life_cycle_stage": "Retirement",
        "annual_income": 22000,
        "net_worth": 900000,
        "investable_assets": 600000,
        "investment_experience_years": 32,
        "financial_context": "Fixed pension income; low risk tolerance"
    }
]

# Legacy templates (kept for reference)
PERSONA_TEMPLATES = {
    "names": {
        "male": ["James", "Michael", "Robert", "John", "David", "William", "Richard", "Joseph", "Thomas", "Christopher",
                 "Daniel", "Matthew", "Anthony", "Mark", "Steven", "Andrew", "Joshua", "Kevin", "Brian", "Ryan",
                 "Alex", "Chris", "Jordan", "Taylor", "Casey", "Morgan", "Cameron", "Quinn", "Avery", "Reese"],
        "female": ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
                   "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle",
                   "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia", "Harper", "Evelyn"]
    },
    "occupations": [
        "Software Engineer", "Data Scientist", "Product Manager", "Financial Analyst", "Accountant",
        "Marketing Manager", "Sales Representative", "Doctor", "Nurse", "Teacher", "Professor",
        "Lawyer", "Consultant", "Project Manager", "UX Designer", "Business Analyst", "HR Manager",
        "Real Estate Agent", "Insurance Agent", "Investment Banker", "Trader", "Portfolio Manager",
        "Architect", "Civil Engineer", "Mechanical Engineer", "Freelancer", "Entrepreneur", "Writer",
        "Journalist", "Chef", "Retail Manager", "Customer Service Representative", "Administrative Assistant"
    ],
    "income_levels": [
        "$30,000 - $50,000",
        "$50,000 - $70,000",
        "$70,000 - $100,000",
        "$100,000 - $150,000",
        "$150,000 - $200,000",
        "$200,000+"
    ],
    "family_statuses": [
        "Single, no children",
        "Single, 1 child",
        "Single, 2 children",
        "Married, no children",
        "Married, 1 child",
        "Married, 2 children",
        "Married, 3+ children",
        "Divorced, no children",
        "Divorced, 1 child",
        "Widowed, no children",
        "Domestic Partnership, no children",
        "Domestic Partnership, 1 child"
    ],
    "financial_knowledge_levels": ["Beginner", "Intermediate", "Advanced"],
    "financial_goals": [
        "Retirement planning",
        "Emergency fund creation",
        "Debt repayment",
        "Homeownership",
        "Children's education fund",
        "Financial independence",
        "Travel fund",
        "Investment portfolio growth",
        "Tax optimization",
        "Estate planning",
        "Healthcare savings",
        "Business startup capital",
        "Vehicle purchase",
        "Wedding fund"
    ]
}

# ============== Timeline Event Templates ==============
TIMELINE_EVENT_TYPES = {
    "income_change": {
        "subtypes": [
            {"description": "Received annual raise", "min_amount": 3000, "max_amount": 15000},
            {"description": "Got a year-end bonus", "min_amount": 5000, "max_amount": 25000},
            {"description": "Started a new job with higher salary", "min_amount": 10000, "max_amount": 30000},
            {"description": "Received promotion", "min_amount": 8000, "max_amount": 20000},
            {"description": "Started freelance work", "min_amount": 2000, "max_amount": 10000}
        ],
        "impact": "positive"
    },
    "expense_increase": {
        "subtypes": [
            {"description": "Had a medical emergency", "min_amount": 1000, "max_amount": 15000},
            {"description": "Car repairs needed", "min_amount": 500, "max_amount": 5000},
            {"description": "Home repairs required", "min_amount": 1000, "max_amount": 10000},
            {"description": "Child's school fees increased", "min_amount": 2000, "max_amount": 8000},
            {"description": "Rent increase", "min_amount": 200, "max_amount": 1000}
        ],
        "impact": "negative"
    },
    "expense_decrease": {
        "subtypes": [
            {"description": "Paid off credit card debt", "min_amount": 1000, "max_amount": 15000},
            {"description": "Finished paying car loan", "min_amount": 300, "max_amount": 2000},
            {"description": "Student loan paid off", "min_amount": 5000, "max_amount": 50000},
            {"description": "Mortgage paid off", "min_amount": 100000, "max_amount": 500000},
            {"description": "Reduced subscription services", "min_amount": 50, "max_amount": 300}
        ],
        "impact": "positive"
    },
    "investment_event": {
        "subtypes": [
            {"description": "Stock market investment gained value", "min_amount": 1000, "max_amount": 50000},
            {"description": "Received dividends from investments", "min_amount": 200, "max_amount": 5000},
            {"description": "Sold investment property", "min_amount": 50000, "max_amount": 200000},
            {"description": "Cryptocurrency investment", "min_amount": 500, "max_amount": 20000},
            {"description": "Bought index funds", "min_amount": 1000, "max_amount": 10000}
        ],
        "impact": "positive"
    },
    "savings_change": {
        "subtypes": [
            {"description": "Opened a high-yield savings account", "min_amount": 5000, "max_amount": 20000},
            {"description": "Increased 401(k) contribution", "min_amount": 200, "max_amount": 1000},
            {"description": "Started automatic savings plan", "min_amount": 100, "max_amount": 500},
            {"description": "Moved savings to money market", "min_amount": 10000, "max_amount": 50000}
        ],
        "impact": "positive"
    },
    "behavior_change": {
        "subtypes": [
            {"description": "Started using budgeting app", "amount": 0},
            {"description": "Began tracking daily expenses", "amount": 0},
            {"description": "Cut down on dining out", "amount": 0},
            {"description": "Reduced impulse purchases", "amount": 0},
            {"description": "Started meal planning to save money", "amount": 0}
        ],
        "impact": "neutral"
    },
    "financial_learning": {
        "subtypes": [
            {"description": "Completed personal finance course", "amount": 0},
            {"description": "Read 3 books on investing", "amount": 0},
            {"description": "Attended financial planning workshop", "amount": 0},
            {"description": "Started following financial news daily", "amount": 0},
            {"description": "Hired a financial advisor", "amount": 0}
        ],
        "impact": "neutral"
    },
    "life_event": {
        "subtypes": [
            {"description": "Got married", "amount": -15000},
            {"description": "Had a baby", "amount": -5000},
            {"description": "Divorced", "amount": -10000},
            {"description": "Retired from job", "amount": -5000},
            {"description": "Moved to new city", "amount": -5000}
        ],
        "impact": "mixed"
    }
}

# ============== Dialogue Topics ==============
DIALOGUE_TOPICS = [
    {
        "topic": "retirement_planning",
        "keywords": ["401k", "IRA", "pension", "social security", "retirement age", "nest egg"],
        "question_types": [
            "How much should I be saving for retirement?",
            "What are the best retirement investment options for my age?",
            "How do I calculate my retirement savings goal?",
            "Should I switch from traditional to Roth IRA?",
            "How much Social Security will I receive?"
        ]
    },
    {
        "topic": "budgeting_expense_tracking",
        "keywords": ["budget", "expenses", "spending", "tracking", "categories", "limits"],
        "question_types": [
            "How do I create a monthly budget?",
            "What's the 50/30/20 rule and should I use it?",
            "How can I reduce my monthly expenses?",
            "What apps do you recommend for expense tracking?",
            "How much should I spend on housing?"
        ]
    },
    {
        "topic": "debt_management",
        "keywords": ["debt", "loan", "credit card", "interest", "pay off", "consolidation"],
        "question_types": [
            "What's the best strategy to pay off credit card debt?",
            "Should I consolidate my loans?",
            "How do I negotiate lower interest rates?",
            "What's the difference between debt avalanche and snowball methods?",
            "When is debt consolidation a good idea?"
        ]
    },
    {
        "topic": "investment_strategy",
        "keywords": ["stocks", "bonds", "mutual funds", "ETF", "portfolio", "diversification"],
        "question_types": [
            "How do I build an investment portfolio?",
            "What's the difference between stocks and bonds?",
            "Should I invest in individual stocks or index funds?",
            "How much should I have in bonds vs stocks?",
            "What is dollar-cost averaging?"
        ]
    },
    {
        "topic": "emergency_fund",
        "keywords": ["emergency fund", "savings", "rainy day", "unexpected expenses", "liquidity"],
        "question_types": [
            "How much should I have in my emergency fund?",
            "Where should I keep my emergency fund?",
            "How long did it take you to build your emergency fund?",
            "Can I invest part of my emergency fund?",
            "What counts as an emergency expense?"
        ]
    },
    {
        "topic": "home_buying",
        "keywords": ["mortgage", "down payment", "home equity", "property tax", "buying vs renting"],
        "question_types": [
            "How much down payment do I need for a house?",
            "Should I buy or continue renting?",
            "What's the best type of mortgage for me?",
            "How do I improve my credit score for a mortgage?",
            "What closing costs should I expect?"
        ]
    },
    {
        "topic": "tax_planning",
        "keywords": ["taxes", "deductions", "credits", "tax brackets", "filing status"],
        "question_types": [
            "What tax deductions am I missing?",
            "Should I itemize or take the standard deduction?",
            "How can I reduce my tax liability?",
            "What is tax-loss harvesting?",
            "When should I make charitable contributions?"
        ]
    },
    {
        "topic": "insurance_planning",
        "keywords": ["insurance", "health", "life", "auto", "coverage", "premium"],
        "question_types": [
            "How much life insurance do I need?",
            "What health insurance plan is best for my family?",
            "Do I need umbrella insurance?",
            "How can I lower my insurance premiums?",
            "What does comprehensive auto insurance cover?"
        ]
    },
    {
        "topic": "education_planning",
        "keywords": ["529 plan", "college savings", "student loans", "FAFSA", "scholarships"],
        "question_types": [
            "How much should I save for my child's college?",
            "What is a 529 plan and how does it work?",
            "Should I prioritize saving for college or retirement?",
            "How do I apply for financial aid?",
            "What are Roth IRAs for education?"
        ]
    },
    {
        "topic": "credit_score",
        "keywords": ["credit score", "credit report", " FICO", "credit utilization", "credit history"],
        "question_types": [
            "How can I improve my credit score quickly?",
            "What is a good credit score?",
            "How often should I check my credit report?",
            "Does checking my credit score hurt it?",
            "How long do negative items stay on credit report?"
        ]
    },
    # New topics for more diverse financial dialogues
    {
        "topic": "financial_product_recommendations",
        "keywords": ["savings account", "checking account", "credit card", "brokerage", "app", "platform"],
        "question_types": [
            "What's the best high-yield savings account for my needs?",
            "Which credit card offers the best rewards for my spending?",
            "What brokerage platform do you recommend for beginners?",
            "Should I use a robo-advisor or traditional advisor?",
            "What are the best apps for managing my finances?"
        ]
    },
    {
        "topic": "market_analysis",
        "keywords": ["stock market", "S&P 500", "bear market", "bull market", "indices", "trends"],
        "question_types": [
            "What do you think about the current market conditions?",
            "How does the S&P 500 performance affect my portfolio?",
            "Are we in a bull market or bear market right now?",
            "What sectors are performing well this year?",
            "Should I be worried about market volatility?"
        ]
    },
    {
        "topic": "cryptocurrency",
        "keywords": ["Bitcoin", "Ethereum", "crypto", "blockchain", "digital assets", "altcoins"],
        "question_types": [
            "Should I invest in cryptocurrency?",
            "What's the difference between Bitcoin and Ethereum?",
            "How much of my portfolio should be in crypto?",
            "What are the tax implications of crypto investments?",
            "Is it safe to keep crypto in exchanges?"
        ]
    },
    {
        "topic": "real_estate_investment",
        "keywords": ["rental property", "REIT", "real estate", "property management", "rental income"],
        "question_types": [
            "Should I invest in rental properties?",
            "What are the pros and cons of REITs?",
            "How do I calculate the return on investment for a property?",
            "What's better: real estate or stock market?",
            "How do I become a landlord?"
        ]
    },
    {
        "topic": "small_business_finance",
        "keywords": ["business loan", "startup", "SBA", "cash flow", "business credit"],
        "question_types": [
            "How do I get funding for my small business?",
            "What are the best business credit cards?",
            "How do I separate personal and business finances?",
            "Should I take out a business loan or investor funding?",
            "How do I build business credit?"
        ]
    },
    {
        "topic": "financial_charts_analysis",
        "keywords": ["chart", "graph", "performance", "returns", "allocation", "pie chart"],
        "question_types": [
            "Can you explain what this portfolio allocation chart shows?",
            "How should I read this performance graph?",
            "What does this stock price chart indicate?",
            "Can you analyze this asset allocation pie chart?",
            "What does this historical return chart tell us?"
        ],
        "supports_image": True  # New: indicates this topic supports image descriptions
    },
    {
        "topic": "financial_report_interpretation",
        "keywords": ["balance sheet", "income statement", "P&L", "earnings", "financial report"],
        "question_types": [
            "What does this balance sheet tell me about the company?",
            "How do I read an income statement?",
            "What should I look for in a company's earnings report?",
            "Can you explain this financial ratio?",
            "What does this cash flow statement show?"
        ],
        "supports_image": True  # New: indicates this topic supports image descriptions
    }
]


# ============== Image Description Templates ==============
# Templates for generating image descriptions in financial contexts
IMAGE_DESCRIPTION_TEMPLATES = {
    "portfolio_allocation": [
        "A pie chart showing portfolio allocation: 60% stocks, 30% bonds, 10% cash",
        "A bar chart comparing asset class returns over the past year",
        "A donut chart showing diversification across different sectors",
        "A stacked bar chart showing portfolio composition changes over time"
    ],
    "performance_charts": [
        "A line graph showing stock price movement over 12 months",
        "A candlestick chart showing daily trading activity",
        "A line chart comparing portfolio performance against S&P 500",
        "A area chart showing cumulative returns over 5 years"
    ],
    "financial_statements": [
        "A balance sheet with assets, liabilities, and equity sections",
        "An income statement showing revenue, expenses, and net income",
        "A cash flow statement with operating, investing, and financing activities",
        "A financial ratio comparison table across multiple years"
    ],
    "budget_tracking": [
        "A bar chart showing monthly expenses by category",
        "A pie chart of spending breakdown: housing 35%, food 20%, transportation 15%",
        "A line graph tracking savings rate over 12 months",
        "A waterfall chart showing income minus expenses over 6 months"
    ],
    "market_trends": [
        "A line chart showing S&P 500 index performance over 10 years",
        "A heat map showing sector performance for the quarter",
        "A comparison chart of different market indices",
        "A volatility chart showing market fluctuations"
    ],
    "product_poster": [
        "A promotional poster for a new investment app with modern design",
        "A product banner for a financial planning service",
        "An advertisement for a premium credit card with benefits",
        "A poster for a financial education course"
    ]
}


# ============== Real Image Generator ==============
class RealImageGenerator:
    """Generates real chart images using matplotlib for multimodal dialogues.
    
    This class creates actual PNG images of financial charts, graphs, and reports
    that can be used in multimodal dialogues. Images are saved to a specified directory.
    """
    
    def __init__(self, output_dir: str = "./output/images", config: Config = None):
        self.output_dir = output_dir
        self.config = config or Config()
        random.seed(self.config.seed)
        
        # Try to import matplotlib
        self.matplotlib_available = False
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            self.plt = plt
            self.np = np
            self.matplotlib_available = True
        except ImportError:
            print("Warning: matplotlib not available. Real image generation disabled.")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_portfolio_pie_chart(self, session_id: str, 
                                     allocations: dict = None) -> Dict:
        """Generate a pie chart showing portfolio allocation.
        
        Args:
            session_id: Session ID for naming the image file
            allocations: Dict of {asset_class: percentage}
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        if allocations is None:
            allocations = {
                'Stocks': random.randint(40, 70),
                'Bonds': random.randint(15, 35),
                'Cash': random.randint(5, 15),
                'Real Estate': random.randint(5, 15)
            }
            # Normalize to 100%
            total = sum(allocations.values())
            allocations = {k: round(v/total*100) for k, v in allocations.items()}
        
        self._ensure_output_dir()
        
        fig, ax = self.plt.subplots(figsize=(8, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        wedges, texts, autotexts = ax.pie(
            allocations.values(), 
            labels=allocations.keys(),
            autopct='%1.1f%%',
            colors=colors[:len(allocations)],
            startangle=90
        )
        
        ax.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')
        
        image_path = f"{self.output_dir}/portfolio_pie_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        # Generate description
        desc_parts = [f"{k}: {v}%" for k, v in allocations.items()]
        description = f"A pie chart showing portfolio allocation: {', '.join(desc_parts)}"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "portfolio_allocation",
            "image_format": "png"
        }
    
    def generate_performance_line_chart(self, session_id: str,
                                        months: int = 12) -> Dict:
        """Generate a line chart showing investment performance over time.
        
        Args:
            session_id: Session ID for naming the image file
            months: Number of months to show
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        self._ensure_output_dir()
        
        # Generate random performance data
        dates = self.np.arange(months)
        # Simulate stock performance with some volatility
        base_value = 10000
        portfolio_values = [base_value]
        benchmark_values = [base_value]
        
        for _ in range(months - 1):
            # Random monthly return: -5% to +8% for portfolio
            port_return = random.uniform(-0.05, 0.08)
            bench_return = random.uniform(-0.04, 0.06)
            
            portfolio_values.append(portfolio_values[-1] * (1 + port_return))
            benchmark_values.append(benchmark_values[-1] * (1 + bench_return))
        
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        ax.plot(dates, portfolio_values, label='Portfolio', linewidth=2, color='#2E86AB')
        ax.plot(dates, benchmark_values, label='S&P 500 Benchmark', linewidth=2, 
                color='#A23B72', linestyle='--')
        
        ax.set_xlabel('Months', fontsize=12)
        ax.set_ylabel('Value ($)', fontsize=12)
        ax.set_title('Portfolio Performance vs S&P 500', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(self.plt.FuncFormatter(
            lambda x, p: f'${x:,.0f}'))
        
        image_path = f"{self.output_dir}/performance_line_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        total_return = (portfolio_values[-1] - base_value) / base_value * 100
        benchmark_return = (benchmark_values[-1] - base_value) / base_value * 100
        
        description = (f"A line chart showing portfolio performance over {months} months. "
                      f"Portfolio return: {total_return:.1f}%, Benchmark return: {benchmark_return:.1f}%")
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "performance_chart",
            "image_format": "png"
        }
    
    def generate_expense_bar_chart(self, session_id: str,
                                   categories: dict = None) -> Dict:
        """Generate a bar chart showing monthly expenses by category.
        
        Args:
            session_id: Session ID for naming the image file
            categories: Dict of {category: amount}
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        if categories is None:
            categories = {
                'Housing': random.randint(1500, 3000),
                'Food': random.randint(400, 800),
                'Transportation': random.randint(200, 500),
                'Utilities': random.randint(100, 300),
                'Entertainment': random.randint(100, 400),
                'Healthcare': random.randint(100, 300),
                'Other': random.randint(200, 500)
            }
        
        self._ensure_output_dir()
        
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        cat_names = list(categories.keys())
        cat_values = list(categories.values())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6B4E71', '#8B9DC3']
        
        bars = ax.bar(cat_names, cat_values, color=colors[:len(categories)])
        
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Monthly Amount ($)', fontsize=12)
        ax.set_title('Monthly Expenses by Category', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.yaxis.set_major_formatter(self.plt.FuncFormatter(
            lambda x, p: f'${x:,.0f}'))
        
        image_path = f"{self.output_dir}/expense_bar_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        total = sum(cat_values)
        description = f"A bar chart showing monthly expenses. Total: ${total:,}"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "budget_tracking",
            "image_format": "png"
        }
    
    def generate_savings_area_chart(self, session_id: str,
                                    months: int = 12) -> Dict:
        """Generate an area chart showing savings progress over time.
        
        Args:
            session_id: Session ID for naming the image file
            months: Number of months to show
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        self._ensure_output_dir()
        
        dates = self.np.arange(months)
        
        # Generate savings data
        savings = [random.randint(500, 2000) for _ in range(months)]
        cumulative = []
        total = 0
        for s in savings:
            total += s
            cumulative.append(total)
        
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        ax.fill_between(dates, cumulative, alpha=0.4, color='#2E86AB')
        ax.plot(dates, cumulative, linewidth=2, color='#2E86AB')
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Cumulative Savings ($)', fontsize=12)
        ax.set_title('Savings Progress Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.yaxis.set_major_formatter(self.plt.FuncFormatter(
            lambda x, p: f'${x:,.0f}'))
        
        image_path = f"{self.output_dir}/savings_area_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        description = f"An area chart showing cumulative savings over {months} months. Total saved: ${cumulative[-1]:,}"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "savings_tracking",
            "image_format": "png"
        }
    
    def generate_stock_candlestick(self, session_id: str,
                                   days: int = 30) -> Dict:
        """Generate a candlestick chart showing stock price movement.
        
        Args:
            session_id: Session ID for naming the image file
            days: Number of days to show
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        self._ensure_output_dir()
        
        # Generate OHLC data
        import numpy as np
        
        base_price = random.uniform(50, 200)
        dates = np.arange(days)
        opens = []
        highs = []
        lows = []
        closes = []
        
        for _ in range(days):
            change = random.uniform(-0.03, 0.04)
            close = base_price * (1 + change)
            open_price = close * random.uniform(0.98, 1.02)
            high = max(open_price, close) * random.uniform(1.0, 1.03)
            low = min(open_price, close) * random.uniform(0.97, 1.0)
            
            opens.append(open_price)
            closes.append(close)
            highs.append(high)
            lows.append(low)
            
            base_price = close
        
        # Create candlestick chart manually
        fig, ax = self.plt.subplots(figsize=(12, 6))
        
        for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
            color = '#2E86AB' if c >= o else '#C73E1D'
            ax.plot([i, i], [l, h], color=color, linewidth=1)
            ax.plot([i-0.3, i+0.3], [o, o], color=color, linewidth=2)
            ax.plot([i-0.3, i+0.3], [c, c], color=color, linewidth=2)
        
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Stock Price ($)', fontsize=12)
        ax.set_title('Stock Price Movement (Candlestick Chart)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.yaxis.set_major_formatter(self.plt.FuncFormatter(
            lambda x, p: f'${x:,.2f}'))
        
        image_path = f"{self.output_dir}/candlestick_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        price_change = (closes[-1] - opens[0]) / opens[0] * 100
        description = f"A candlestick chart showing {days} days of stock price movement. Price change: {price_change:.1f}%"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "stock_chart",
            "image_format": "png"
        }
    
    def generate_market_heatmap(self, session_id: str) -> Dict:
        """Generate a heatmap showing sector performance.
        
        Args:
            session_id: Session ID for naming the image file
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        self._ensure_output_dir()
        
        sectors = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial', 'Utilities', 'Real Estate']
        returns = [random.uniform(-8, 15) for _ in sectors]
        
        fig, ax = self.plt.subplots(figsize=(10, 4))
        
        # Create heatmap
        im = ax.imshow([returns], cmap='RdYlGn', aspect='auto', vmin=-10, vmax=15)
        
        ax.set_xticks(range(len(sectors)))
        ax.set_xticklabels(sectors, rotation=45, ha='right')
        ax.set_yticks([])
        
        # Add colorbar
        cbar = self.plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)', rotation=270, labelpad=15)
        
        ax.set_title('Sector Performance Heatmap (YTD)', fontsize=14, fontweight='bold')
        
        image_path = f"{self.output_dir}/heatmap_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        best_sector = sectors[returns.index(max(returns))]
        worst_sector = sectors[returns.index(min(returns))]
        description = f"A heatmap showing sector performance. Best: {best_sector} ({max(returns):.1f}%), Worst: {worst_sector} ({min(returns):.1f}%)"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "market_trends",
            "image_format": "png"
        }
    
    def generate_product_poster(self, session_id: str,
                                product_type: str = "investment_app") -> Dict:
        """Generate a conceptual product poster/banner image.
        
        Note: This creates a stylized chart representation since we can't generate
        actual promotional posters. In practice, you would use real poster images.
        
        Args:
            session_id: Session ID for naming the image file
            product_type: Type of product (investment_app, credit_card, financial_course)
        
        Returns:
            Dict with image_path, image_description, and image_type
        """
        if not self.matplotlib_available:
            return None
        
        self._ensure_output_dir()
        
        # Create a stylized "poster" with key metrics/benefits
        fig, ax = self.plt.subplots(figsize=(8, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Background
        ax.add_patch(self.plt.Rectangle((0.5, 0.5), 9, 9, fill=True, 
                                         facecolor='#1a1a2e', edgecolor='none'))
        
        # Title
        product_names = {
            "investment_app": "FinancePro App",
            "credit_card": "Platinum Rewards Card", 
            "financial_course": "Investment Masterclass"
        }
        product_name = product_names.get(product_type, "Financial Product")
        
        ax.text(5, 8.5, product_name, ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')
        
        # Key features (as visual elements)
        features = [
            "✓ Low Fees",
            "✓ High Returns",
            "✓ Easy to Use",
            "✓ 24/7 Support"
        ]
        
        for i, feature in enumerate(features):
            ax.text(5, 6.5 - i * 0.8, feature, ha='center', va='center',
                    fontsize=14, color='#00d4ff')
        
        # CTA
        ax.text(5, 2, "Get Started Today!", ha='center', va='center',
                fontsize=16, fontweight='bold', color='#ff6b6b',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        image_path = f"{self.output_dir}/poster_{session_id}.png"
        self.plt.savefig(image_path, dpi=100, bbox_inches='tight')
        self.plt.close()
        
        description = f"A promotional poster/banner for {product_name} showing key features and call-to-action"
        
        return {
            "image_path": image_path,
            "image_description": description,
            "image_type": "product_poster",
            "image_format": "png"
        }
    
    def generate_for_topic(self, topic: str, session_id: str) -> Optional[Dict]:
        """Generate an image for a given topic.
        
        Args:
            topic: The dialogue topic
            session_id: Session ID for naming the image file
        
        Returns:
            Dict with image details or None
        """
        if not self.matplotlib_available:
            return None
        
        image_generators = {
            "financial_charts_analysis": self.generate_performance_line_chart,
            "financial_report_interpretation": self.generate_expense_bar_chart,
            "retirement_planning": self.generate_portfolio_pie_chart,
            "budgeting_expense_tracking": self.generate_expense_bar_chart,
            "investment_strategy": self.generate_performance_line_chart,
            "emergency_fund": self.generate_savings_area_chart,
            "market_analysis": self.generate_market_heatmap,
            "financial_product_recommendations": self.generate_product_poster
        }
        
        generator = image_generators.get(topic)
        if generator:
            try:
                return generator(session_id)
            except Exception as e:
                print(f"Error generating image for topic {topic}: {e}")
                return None
        
        return None
    
    def generate_random(self, session_id: str) -> Optional[Dict]:
        """Generate a random financial chart image.
        
        Args:
            session_id: Session ID for naming the image file
        
        Returns:
            Dict with image details
        """
        if not self.matplotlib_available:
            return None
        
        generators = [
            self.generate_portfolio_pie_chart,
            self.generate_performance_line_chart,
            self.generate_expense_bar_chart,
            self.generate_savings_area_chart,
            self.generate_stock_candlestick,
            self.generate_market_heatmap,
            lambda sid: self.generate_product_poster(sid, random.choice(["investment_app", "credit_card", "financial_course"]))
        ]
        
        generator = random.choice(generators)
        try:
            return generator(session_id)
        except Exception as e:
            print(f"Error generating random image: {e}")
            return None


# ============== LLM Client ==============
class LLMClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, config: Config):
        self.config = config
        # Initialize OpenAI client with optional custom base_url
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self.client = openai.OpenAI(**client_kwargs)
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content


# ============== Data Classes ==============
@dataclass
class UserPersona:
    """Represents a user persona."""
    user_id: str
    name: str
    gender: str
    age: int
    occupation: str
    income_level: str
    family_status: str
    financial_knowledge_level: str
    financial_goals: List[str]
    risk_tolerance: str = "moderate"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TimelineEvent:
    """Represents a timeline event."""
    event_id: str
    timestamp: str
    event_type: str
    description: str
    amount: Optional[float]
    impact: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DialogueTurn:
    """Represents a single turn in a dialogue."""
    turn_number: int
    speaker: str  # "user" or "assistant"
    message: str
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DialogueSession:
    """Represents a multi-turn dialogue session."""
    session_id: str
    topic: str
    topic_description: str
    turns: List[DialogueTurn]
    start_timestamp: str
    end_timestamp: str
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["turns"] = [turn.to_dict() for turn in self.turns]
        return result


@dataclass
class QAPair:
    """Represents a query-answer pair with dialogue history context."""
    query_id: str
    query: str
    response: str
    context_session_id: str
    timestamp: str
    question_type: str
    # New field: dialogue history context
    dialogue_history: List[Dict] = field(default_factory=list)
    referenced_timeline_events: List[Dict] = field(default_factory=list)
    # New field: image description context (for multimodal conversations)
    image_description: Optional[str] = None
    image_type: Optional[str] = None  # e.g., "portfolio_allocation", "performance_chart"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ImageDescriptionGenerator:
    """Generates image descriptions for multimodal financial dialogues.
    
    This class provides functionality to generate text descriptions of financial
    charts, graphs, and reports that can be used in multimodal dialogues.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        random.seed(self.config.seed)
    
    def generate_for_topic(self, topic: str) -> Optional[Dict]:
        """Generate an image description for a given topic.
        
        Args:
            topic: The dialogue topic
            
        Returns:
            Dict with 'image_description' and 'image_type' keys, or None if topic doesn't support images
        """
        # Check if topic supports images
        topic_data = None
        for t in DIALOGUE_TOPICS:
            if t["topic"] == topic:
                topic_data = t
                break
        
        if not topic_data or not topic_data.get("supports_image", False):
            return None
        
        # Select appropriate image type based on topic
        image_type_map = {
            "financial_charts_analysis": "performance_charts",
            "financial_report_interpretation": "financial_statements"
        }
        
        image_type = image_type_map.get(topic, "portfolio_allocation")
        
        # Generate description from template
        description = random.choice(IMAGE_DESCRIPTION_TEMPLATES.get(image_type, 
                                                    IMAGE_DESCRIPTION_TEMPLATES["portfolio_allocation"]))
        
        return {
            "image_description": description,
            "image_type": image_type
        }
    
    def generate_with_llm(self, topic: str, persona: UserPersona = None) -> Optional[Dict]:
        """Generate a more detailed image description using LLM.
        
        Args:
            topic: The dialogue topic
            persona: Optional user persona for personalized descriptions
            
        Returns:
            Dict with 'image_description' and 'image_type' keys, or None
        """
        if not self.llm_client:
            return self.generate_for_topic(topic)
        
        # Check if topic supports images
        topic_data = None
        for t in DIALOGUE_TOPICS:
            if t["topic"] == topic:
                topic_data = t
                break
        
        if not topic_data or not topic_data.get("supports_image", False):
            return None
        
        persona_info = ""
        if persona:
            persona_info = f"\n\nUser Profile: {persona.name}, {persona.age}, {persona.occupation}, Risk tolerance: {persona.risk_tolerance}"
        
        system_prompt = """You are a financial data visualization expert. Generate realistic 
descriptions of financial charts, graphs, and reports that a user might share in a conversation."""
        
        user_prompt = f"""Generate a detailed image description for a financial dialogue about '{topic}'.{persona_info}

The description should be text that could replace an actual image - describing what the chart/graph would show.
Make it realistic with specific numbers and data points.

Format as JSON:
{{"image_description": "...", "image_type": "..."}}

image_type should be one of: portfolio_allocation, performance_charts, financial_statements, budget_tracking, market_trends"""

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            print(f"LLM image description generation failed: {e}, using template fallback")
            return self.generate_for_topic(topic)
    
    def generate_batch(self, topics: List[str], persona: UserPersona = None) -> Dict[str, Optional[Dict]]:
        """Generate image descriptions for multiple topics.
        
        Args:
            topics: List of dialogue topics
            persona: Optional user persona
            
        Returns:
            Dict mapping topic to image description dict (or None)
        """
        results = {}
        for topic in topics:
            if self.llm_client:
                results[topic] = self.generate_with_llm(topic, persona)
            else:
                results[topic] = self.generate_for_topic(topic)
        return results


# ============== Generator Classes ==============
class PersonaGenerator:
    """Generates user personas using random templates."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        random.seed(self.config.seed)
    
    def generate_one(self, user_id: str = None) -> UserPersona:
        """Generate a single user persona."""
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        gender = random.choice(["Male", "Female"])
        names = PERSONA_TEMPLATES["names"]["male" if gender == "Male" else "female"]
        name = random.choice(names)
        age = random.randint(22, 65)
        occupation = random.choice(PERSONA_TEMPLATES["occupations"])
        income_level = random.choice(PERSONA_TEMPLATES["income_levels"])
        family_status = random.choice(PERSONA_TEMPLATES["family_statuses"])
        financial_knowledge = random.choice(PERSONA_TEMPLATES["financial_knowledge_levels"])
        
        # Select 2-4 financial goals
        num_goals = random.randint(2, 4)
        financial_goals = random.sample(PERSONA_TEMPLATES["financial_goals"], num_goals)
        
        # Determine risk tolerance based on age and knowledge
        if age < 35:
            risk_tolerance = random.choice(["moderate", "aggressive"])
        elif age < 50:
            risk_tolerance = random.choice(["conservative", "moderate"])
        else:
            risk_tolerance = random.choice(["conservative", "very conservative"])
        
        # Adjust based on knowledge level
        if financial_knowledge == "Advanced":
            if risk_tolerance in ["conservative", "very conservative"]:
                risk_tolerance = random.choice(["conservative", "moderate"])
        
        return UserPersona(
            user_id=user_id,
            name=name,
            gender=gender,
            age=age,
            occupation=occupation,
            income_level=income_level,
            family_status=family_status,
            financial_knowledge_level=financial_knowledge,
            financial_goals=financial_goals,
            risk_tolerance=risk_tolerance
        )
    
    def generate_batch(self, num_personas: int = 100) -> List[UserPersona]:
        """Generate a batch of user personas."""
        personas = []
        for i in tqdm(range(num_personas), desc="Generating personas"):
            persona = self.generate_one(user_id=f"user_{i+1:03d}")
            personas.append(persona)
        return personas


class FixedPersonaGenerator:
    """Generates user personas using FIXED_PERSONAS from persona.xlsx.
    
    This class uses the 5 fixed personas instead of randomly generating them.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        random.seed(self.config.seed)
    
    def generate_one(self, user_id: str = None, index: int = None) -> UserPersona:
        """Generate a single user persona from FIXED_PERSONAS.
        
        Args:
            user_id: Optional user ID (defaults to user_001, user_002, etc.)
            index: Optional index to select specific persona (0-4)
        
        Returns:
            UserPersona object with fixed persona data
        """
        # Use index if provided, otherwise random selection
        if index is not None and 0 <= index < len(FIXED_PERSONAS):
            persona_data = FIXED_PERSONAS[index]
        else:
            persona_data = random.choice(FIXED_PERSONAS)
        
        # Generate user_id if not provided
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        return UserPersona(
            user_id=user_id,
            name=persona_data["name"],
            gender=persona_data["gender"],
            age=persona_data["age"],
            occupation=persona_data["occupation"],
            income_level=persona_data["income_level"],
            family_status=persona_data["family_status"],
            financial_knowledge_level=persona_data["financial_knowledge_level"],
            financial_goals=persona_data["financial_goals"],
            risk_tolerance=persona_data["risk_tolerance"]
        )
    
    def generate_batch(self, num_personas: int = 5) -> List[UserPersona]:
        """Generate a batch of user personas from FIXED_PERSONAS.
        
        Since we only have 5 fixed personas, this will cycle through them.
        
        Args:
            num_personas: Number of personas to generate (max 5 unique)
        
        Returns:
            List of UserPersona objects
        """
        personas = []
        for i in tqdm(range(num_personas), desc="Generating fixed personas"):
            # Cycle through the 5 fixed personas
            persona = self.generate_one(user_id=f"user_{i+1:03d}", index=i % len(FIXED_PERSONAS))
            personas.append(persona)
        return personas


class JsonPersonaGenerator:
    """Generates user personas from a JSON file.
    
    This class reads personas from generated_personas.json instead of
    regenerating them, ensuring consistency across runs.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None, 
                 json_path: str = "./generated_personas.json"):
        self.llm_client = llm_client
        self.config = config or Config()
        self.json_path = json_path
        self._personas = None
    
    def _load_personas(self) -> List[Dict]:
        """Load personas from JSON file."""
        if self._personas is None:
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self._personas = json.load(f)
                print(f"Loaded {len(self._personas)} personas from {self.json_path}")
            except FileNotFoundError:
                print(f"Warning: {self.json_path} not found. Using FIXED_PERSONAS instead.")
                self._personas = FIXED_PERSONAS
            except json.JSONDecodeError as e:
                print(f"Warning: Error parsing JSON file: {e}. Using FIXED_PERSONAS instead.")
                self._personas = FIXED_PERSONAS
        return self._personas
    
    def generate_one(self, user_id: str = None, index: int = None) -> UserPersona:
        """Generate a single user persona from JSON file.
        
        Args:
            user_id: Optional user ID (defaults to user_001, user_002, etc.)
            index: Optional index to select specific persona
        
        Returns:
            UserPersona object with persona data from JSON
        """
        personas = self._load_personas()
        
        # Use index if provided, otherwise random selection
        if index is not None and 0 <= index < len(personas):
            persona_data = personas[index]
        else:
            persona_data = random.choice(personas)
        
        # Generate user_id if not provided
        user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        # Handle financial_goals which might be a string (from LLM validation)
        financial_goals = persona_data.get("financial_goals", [])
        if isinstance(financial_goals, str):
            # It's a validation string, use default goals
            financial_goals = ["Retirement planning", "Financial stability"]
        
        # Handle risk_tolerance which might be a string (from LLM validation)
        risk_tolerance = persona_data.get("risk_tolerance", "Moderate")
        if not isinstance(risk_tolerance, str) or " " in risk_tolerance:
            # It's a validation message, use default
            risk_tolerance = "Moderate"
        
        # Handle income_level which might contain validation messages
        income_level = persona_data.get("income_level", "$50,000")
        if isinstance(income_level, str) and not income_level.startswith("$"):
            income_level = f"${persona_data.get('annual_income', 50000):,}"
        
        return UserPersona(
            user_id=user_id,
            name=persona_data.get("name", "Unknown"),
            gender=persona_data.get("gender", "Male"),
            age=persona_data.get("age", 35),
            occupation=persona_data.get("occupation", "Engineer"),
            income_level=income_level,
            family_status=persona_data.get("family_status", "Single"),
            financial_knowledge_level=persona_data.get("financial_knowledge_level", "Intermediate"),
            financial_goals=financial_goals,
            risk_tolerance=risk_tolerance
        )
    
    def generate_batch(self, num_personas: int = None) -> List[UserPersona]:
        """Generate a batch of user personas from JSON file.
        
        Args:
            num_personas: Number of personas to generate (defaults to all in JSON)
        
        Returns:
            List of UserPersona objects
        """
        personas = self._load_personas()
        
        if num_personas is None:
            num_personas = len(personas)
        
        personas_list = []
        for i in tqdm(range(num_personas), desc="Loading personas from JSON"):
            # Cycle through available personas if needed
            persona = self.generate_one(user_id=f"user_{i+1:03d}", index=i % len(personas))
            personas_list.append(persona)
        
        return personas_list
    
    def get_total_count(self) -> int:
        """Get total number of personas available in JSON file."""
        return len(self._load_personas())


class TimelineGenerator:
    """Generates timeline events for a persona using LLM for personalized generation."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        random.seed(self.config.seed)
    
    def _generate_random_date(self, start_year: int = 2020, end_year: int = 2026) -> str:
        """Generate a random date within a range."""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        days_range = (end_date - start_date).days
        random_days = random.randint(0, days_range)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")
    
    def _generate_event_amount(self, subtype: Dict) -> float:
        """Generate a random amount for an event."""
        if subtype.get("amount", None) is not None and subtype["amount"] > 0:
            return subtype["amount"]
        if subtype.get("min_amount") and subtype.get("max_amount"):
            return round(random.uniform(subtype["min_amount"], subtype["max_amount"]), 2)
        return subtype.get("amount", 0)
    
    def _generate_events_llm(self, persona: UserPersona, num_events: int = None) -> List[TimelineEvent]:
        """Generate personalized timeline events using LLM based on user profile.
        
        This method considers the user's:
        - Age and life stage
        - Occupation and income level
        - Family status
        - Financial goals
        - Risk tolerance
        - Financial knowledge level
        - Personalized timeline range based on profile
        """
        if not self.llm_client:
            return self._generate_events_template(persona, num_events)
        
        num_events = num_events or random.randint(8, 15)
        
        # Get personalized timeline range based on user profile
        timeline_range = self._get_timeline_range(persona)
        start_year = timeline_range["start_year"]
        end_year = timeline_range["end_year"]
        span_years = timeline_range["span_years"]
        
        # Build detailed persona context
        persona_info = (
            f"Name: {persona.name}\n"
            f"Age: {persona.age}\n"
            f"Occupation: {persona.occupation}\n"
            f"Income: {persona.income_level}\n"
            f"Family Status: {persona.family_status}\n"
            f"Financial Knowledge: {persona.financial_knowledge_level}\n"
            f"Financial Goals: {', '.join(persona.financial_goals)}\n"
            f"Risk Tolerance: {persona.risk_tolerance}"
        )
        
        # Determine life stage and relevant events
        life_stage_events = self._get_lifecycle_stage_events(persona)
        
        system_prompt = """You are a financial timeline generator. Generate realistic financial events 
that are SPECIFIC to this user's profile and life situation. The events should be personalized 
and reflect their actual financial journey."""

        user_prompt = f"""Generate {num_events} personalized financial timeline events for the following user.

{persona_info}

Life Stage Context: {life_stage_events}

TIMELINE RANGE (IMPORTANT - use these specific years):
- Start Year: {start_year}
- End Year: {end_year}
- Time Span: approximately {span_years} years of financial history

Generate events as a JSON array with this exact format:
[
  {{"timestamp": "YYYY-MM-DD", "event_type": "...", "description": "...", "amount": 1234.56, "impact": "positive/negative/mixed/neutral"}},
  ...
]

Requirements:
1. Timestamps MUST be between {start_year}-01-01 and {end_year}-12-31, in chronological order
2. The time span of events should reflect approximately {span_years} years of history
3. Event types should be realistic based on the user's profile:
   - "income_change": salary changes, bonuses, promotions, new jobs
   - "expense_increase": medical emergencies, home/car repairs, rent increases, tuition
   - "expense_decrease": debt payoff, loan completion, reduced expenses
   - "investment_event": stock gains/losses, dividends, property sales, portfolio changes
   - "savings_change": account openings, contribution changes, transfers
   - "behavior_change": budgeting apps, expense tracking, spending habit changes
   - "financial_learning": courses, books, advisor consultations
   - "life_event": marriage, children, retirement, relocation (with financial impact)
4. Amounts should be realistic (positive = income/gain, negative = expense/loss)
5. Events should be PERSONALIZED - a young professional has different events than a retired person
6. Consider their specific occupation, income, family status, and goals
7. Events should be spread out naturally over the {span_years} year period
8. Return ONLY valid JSON, no other text."""

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            # Parse JSON response
            events_data = json.loads(response)
            events = []
            
            for e in events_data:
                event = TimelineEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    timestamp=e.get("timestamp", self._generate_random_date(start_year, end_year)),
                    event_type=e.get("event_type", "income_change"),
                    description=e.get("description", "Financial event"),
                    amount=e.get("amount", 0),
                    impact=e.get("impact", "neutral")
                )
                events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            return events
            
        except Exception as e:
            print(f"LLM timeline generation failed: {e}, using template fallback")
            return self._generate_events_template(persona, num_events)
    
    def _get_lifecycle_stage_events(self, persona: UserPersona) -> str:
        """Determine lifecycle stage and relevant events based on persona."""
        age = persona.age
        
        # Determine life stage
        if age < 30:
            stage = "Early Career"
            context = "Young professional starting career, may have student loans, building emergency fund, first investments"
        elif age < 45:
            stage = "Family Accumulation"
            context = "Family formation, home buying, children expenses, career advancement, wealth building"
        elif age < 60:
            stage = "Peak Wealth"
            context = "Peak earning years, retirement planning, education funding, estate planning"
        else:
            stage = "Retirement"
            context = "Retirement living, estate distribution, wealth preservation, healthcare costs"
        
        # Add occupation-specific context
        if "student" in persona.occupation.lower() or "intern" in persona.occupation.lower():
            context += ", internship income, tuition payments"
        elif "manager" in persona.occupation.lower() or "director" in persona.occupation.lower() or "executive" in persona.occupation.lower():
            context += ", bonus income, stock options, higher retirement contributions"
        elif "owner" in persona.occupation.lower() or "entrepreneur" in persona.occupation.lower():
            context += ", business income variability, business expenses, potential sale"
        elif "retired" in persona.occupation.lower():
            context += ", pension income, social security, retirement withdrawals"
        
        # Add family-specific context
        family = persona.family_status.lower()
        if "married" in family:
            context += ", joint finances, wedding costs"
        if "child" in family or "children" in family:
            context += ", childcare/education expenses, life insurance needs"
        
        # Add goal-specific context
        for goal in persona.financial_goals:
            goal_lower = goal.lower()
            if "retirement" in goal_lower:
                context += ", increased retirement contributions"
            if "home" in goal_lower or "house" in goal_lower:
                context += ", down payment savings, mortgage"
            if "education" in goal_lower or "college" in goal_lower:
                context += ", 529 plan, education savings"
            if "debt" in goal_lower or "pay off" in goal_lower:
                context += ", debt repayment"
        
        return f"{stage}: {context}"
    
    def _get_timeline_range(self, persona: UserPersona) -> Dict[str, int]:
        """Calculate timeline range (start_year and end_year) based on user profile.
        
        Different users have different timeline ranges based on:
        - Age: Older users have longer financial history
        - Career stage: Early career vs late career
        - Occupation: Students have different timeline than retirees
        - Net worth context: High net worth individuals may have longer investment history
        
        Returns:
            Dict with 'start_year' and 'end_year' keys
        """
        current_year = datetime.now().year
        age = persona.age
        
        # Determine timeline span based on age
        if age < 25:
            # Young adult just starting - 1-3 years of history
            span_years = random.randint(1, 3)
        elif age < 35:
            # Early career - 3-5 years
            span_years = random.randint(3, 5)
        elif age < 45:
            # Mid career - 8-12 years
            span_years = random.randint(5, 8)
        elif age < 55:
            # Late career - 12-18 years
            span_years = random.randint(8, 10)
        elif age < 65:
            # Pre-retirement - 15-25 years
            span_years = random.randint(10, 15)
        else:
            # Retired - 20-35 years of history
            span_years = random.randint(15, 20)
        
        # Adjust based on occupation
        occupation = persona.occupation.lower()
        if "student" in occupation or "intern" in occupation:
            # Students have shorter history
            span_years = min(span_years, 4)
        elif "retired" in occupation:
            # Retirees have longer history
            span_years = max(span_years, 20)
        
        # Adjust based on financial knowledge (more experience = longer history)
        knowledge = persona.financial_knowledge_level.lower()
        if "professional" in knowledge or "advanced" in knowledge:
            span_years = int(span_years * 1.2)  # 20% longer
        
        # Calculate end_year (current year or slightly in past for dataset)
        # For realistic data, end somewhere in recent 1-2 years
        end_year = current_year - random.randint(0, 2)
        start_year = end_year - span_years
        
        # Ensure reasonable bounds
        start_year = max(2000, start_year)
        end_year = min(current_year, end_year)
        
        return {
            "start_year": start_year,
            "end_year": end_year,
            "span_years": span_years
        }
    
    def _generate_events_template(self, persona: UserPersona, num_events: int = None) -> List[TimelineEvent]:
        """Generate timeline events using template (fallback when LLM unavailable).
        
        Uses personalized timeline range based on user profile.
        """
        num_events = num_events or random.randint(8, 15)
        
        # Get personalized timeline range based on user profile
        timeline_range = self._get_timeline_range(persona)
        start_year = timeline_range["start_year"]
        end_year = timeline_range["end_year"]
        
        # Start from beginning of the timeline range
        current_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        total_days = (end_date - current_date).days
        
        events = []
        event_types = list(TIMELINE_EVENT_TYPES.keys())
        
        for i in range(num_events):
            event_type = random.choice(event_types)
            event_data = TIMELINE_EVENT_TYPES[event_type]
            subtype = random.choice(event_data["subtypes"])
            
            # Generate date (increasing over time) - spread events evenly across timeline
            if i == 0:
                days_forward = random.randint(0, total_days // (num_events + 1))
            else:
                # Spread events across the remaining time period
                remaining_events = num_events - i
                max_days = total_days - (current_date - datetime(start_year, 1, 1)).days
                if remaining_events > 0:
                    days_forward = random.randint(max_days // (remaining_events + 1), max_days // max(remaining_events - 1, 1) if remaining_events > 1 else max_days)
                else:
                    days_forward = random.randint(0, max(1, max_days // 10))
            
            current_date = current_date + timedelta(days=days_forward)
            
            # Ensure we don't go past end_year
            if current_date > end_date:
                current_date = end_date
            
            # Generate amount (can be negative for expenses)
            amount = self._generate_event_amount(subtype)
            
            event = TimelineEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                timestamp=current_date.strftime("%Y-%m-%d"),
                event_type=event_type,
                description=subtype["description"],
                amount=amount,
                impact=event_data["impact"]
            )
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        return events
    
    def generate_for_persona(self, persona: UserPersona, num_events: int = None) -> List[TimelineEvent]:
        """Generate timeline events for a persona using LLM when available.
        
        This method now uses LLM to generate personalized events based on the user's
        profile, falling back to template-based generation only when LLM is unavailable.
        """
        if self.llm_client:
            return self._generate_events_llm(persona, num_events)
        else:
            return self._generate_events_template(persona, num_events)
    
    def generate_batch(self, personas: List[UserPersona], events_per_persona: int = None) -> Dict[str, List[TimelineEvent]]:
        """Generate timeline events for multiple personas."""
        timelines = {}
        for persona in tqdm(personas, desc="Generating personalized timelines"):
            timelines[persona.user_id] = self.generate_for_persona(persona, events_per_persona)
        return timelines


class DialogueSessionGenerator:
    """Generates multi-turn dialogue sessions."""
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        random.seed(self.config.seed)
    
    def _generate_turns_llm(self, persona: UserPersona, topic: Dict, timeline: List[TimelineEvent]) -> List[DialogueTurn]:
        """Generate dialogue turns using LLM."""
        if not self.llm_client:
            return self._generate_turns_template(persona, topic)
        
        # Build context from timeline
        timeline_summary = "\n".join([
            f"- {e.timestamp}: {e.description} (${e.amount if e.amount else 0})"
            for e in timeline[-5:]
        ])
        
        persona_info = (
            f"Name: {persona.name}\n"
            f"Age: {persona.age}\n"
            f"Occupation: {persona.occupation}\n"
            f"Income: {persona.income_level}\n"
            f"Family: {persona.family_status}\n"
            f"Knowledge Level: {persona.financial_knowledge_level}\n"
            f"Goals: {', '.join(persona.financial_goals)}\n"
            f"Risk Tolerance: {persona.risk_tolerance}\n"
        )
        
        system_prompt = """You are a professional financial assistant having a natural conversation with a client.
Generate realistic multi-turn dialogue that:
1. Shows the user asking questions naturally based on their profile
2. Provides helpful, accurate financial advice
3. References the user's financial situation when appropriate
4. Maintains a conversational, professional tone
5. Includes follow-up questions and clarifications
6. Has 10-15 turns per session"""

        user_prompt = f"""Generate a {self.config.turns_per_session}-turn dialogue session about {topic['topic']}.

Persona Information:
{persona_info}

Recent Financial History:
{timeline_summary}

Topic: {topic['topic']}
Topic keywords: {', '.join(topic['keywords'])}
Question examples: {', '.join(topic['question_types'][:3])}

Please generate the dialogue as a JSON array of turns with this format:
[
  {{"turn_number": 1, "speaker": "user", "message": "..."}},
  {{"turn_number": 2, "speaker": "assistant", "message": "..."}},
  ...
]

Only output the JSON, no other text."""

        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            # Parse JSON response
            import json
            turns_data = json.loads(response)
            turns = []
            for t in turns_data:
                turns.append(DialogueTurn(
                    turn_number=t["turn_number"],
                    speaker=t["speaker"],
                    message=t["message"]
                ))
            return turns
        except Exception as e:
            print(f"LLM generation failed, using template: {e}")
            return self._generate_turns_template(persona, topic)
    
    def _generate_turns_template(self, persona: UserPersona, topic: Dict) -> List[DialogueTurn]:
        """Generate dialogue turns using templates (fallback)."""
        turns = []
        num_turns = self.config.turns_per_session
        
        # Generate some contextual questions based on persona
        questions = topic["question_types"][:num_turns // 2 + 1]
        answers = [
            f"Thank you for asking about {topic['topic']}. Based on your profile as a {persona.age}-year-old {persona.occupation}, I'd recommend considering {persona.risk_tolerance} investment strategies.",
            f"That's a great question. Given your financial goals of {persona.financial_goals[0] if persona.financial_goals else 'financial stability'}, let me provide some guidance.",
            f"I understand your concern. With your current income level of {persona.income_level}, we should create a plan that balances your short-term needs with long-term goals.",
            f"Let me analyze your situation. As someone with {persona.financial_knowledge_level} financial knowledge, I'll explain this step by step.",
            f"Based on recent financial events and your {persona.family_status} situation, this is what I recommend...",
            f"I recommend starting with an emergency fund of 3-6 months expenses, then focusing on tax-advantaged accounts.",
            f"For your age and risk tolerance, a diversified portfolio with 60% stocks and 40% bonds could work well.",
            f"Don't forget to take advantage of any employer 401(k) match - it's essentially free money!",
            f"Would you like me to create a detailed action plan for you to get started?",
            f"Let's schedule a follow-up conversation to review your progress in a few weeks."
        ]
        
        turn_num = 1
        for i in range(num_turns):
            if i % 2 == 0:
                # User turn
                if i // 2 < len(questions):
                    message = questions[i // 2]
                else:
                    message = random.choice([
                        "Can you elaborate more on that point?",
                        "How would this apply to my specific situation?",
                        "What are the next steps I should take?",
                        "Can you explain that in simpler terms?"
                    ])
                speaker = "user"
            else:
                # Assistant turn
                message = answers[(i - 1) // 2 % len(answers)]
                speaker = "assistant"
            
            turns.append(DialogueTurn(
                turn_number=turn_num,
                speaker=speaker,
                message=message
            ))
            turn_num += 1
        
        return turns
    
    def generate_one(self, persona: UserPersona, topic: Dict, timeline: List[TimelineEvent], 
                      session_id: str = None) -> DialogueSession:
        """Generate a single dialogue session."""
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        
        # Generate turns
        if self.llm_client:
            turns = self._generate_turns_llm(persona, topic, timeline)
        else:
            turns = self._generate_turns_template(persona, topic)
        
        # Generate timestamps
        start_date = datetime(2024, 3, 1)
        duration_days = len(turns) * 2  # 2 days per turn on average
        end_date = start_date + timedelta(days=duration_days)
        
        return DialogueSession(
            session_id=session_id,
            topic=topic["topic"],
            topic_description=topic["keywords"][0] if topic["keywords"] else topic["topic"],
            turns=turns,
            start_timestamp=start_date.strftime("%Y-%m-%d"),
            end_timestamp=end_date.strftime("%Y-%m-%d")
        )
    
    def generate_batch(self, persona: UserPersona, timeline: List[TimelineEvent], 
                       num_sessions: int = 5) -> List[DialogueSession]:
        """Generate multiple dialogue sessions for a persona."""
        sessions = []
        selected_topics = random.sample(DIALOGUE_TOPICS, min(num_sessions, len(DIALOGUE_TOPICS)))
        
        for i, topic in enumerate(selected_topics):
            session = self.generate_one(
                persona=persona,
                topic=topic,
                timeline=timeline,
                session_id=f"session_{persona.user_id}_{i+1:02d}"
            )
            sessions.append(session)
        
        return sessions


# ============== QA Quality Validator ==============
class QAQalityValidator:
    """Validates QA pairs to ensure answer quality.
    
    This class checks if generated QA pairs are valid by:
    1. Checking if answer is relevant to the question
    2. Checking if answer uses consistent terminology from dialogue
    3. Checking if answer is not too short or generic
    4. Checking factual consistency with dialogue content
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
    
    def validate(self, qa_pair: QAPair, session: DialogueSession, 
                 persona: UserPersona) -> bool:
        """Validate a QA pair.
        
        Args:
            qa_pair: The QA pair to validate
            session: The dialogue session this QA came from
            persona: The user persona
            
        Returns:
            True if QA pair is valid, False otherwise
        """
        # Basic validation checks
        if not self._basic_validation(qa_pair):
            return False
        
        # LLM-based validation if available
        if self.llm_client:
            return self._llm_validation(qa_pair, session, persona)
        
        return True
    
    def _basic_validation(self, qa_pair: QAPair) -> bool:
        """Perform basic validation checks."""
        # Check if question is not empty
        if not qa_pair.query or len(qa_pair.query.strip()) < 10:
            return False
        
        # Check if answer is not empty
        if not qa_pair.response or len(qa_pair.response.strip()) < 5:
            return False
        
        # Check if answer is not too generic
        generic_answers = [
            "Based on the dialogue...",
            "Here's a financial plan based on your situation...",
            "Let me calculate that for you...",
            "I don't know",
            "Sorry, I can't help with that"
        ]
        
        for generic in generic_answers:
            if generic.lower() in qa_pair.response.lower():
                return False
        
        return True
    
    def _llm_validation(self, qa_pair: QAPair, session: DialogueSession,
                       persona: UserPersona) -> bool:
        """Use LLM to validate if answer is correct and relevant."""
        # Get dialogue context
        dialogue_context = "\n".join([
            f"{'User' if t.speaker == 'user' else 'Assistant'}: {t.message}"
            for t in session.turns[:8]  # Use first 8 turns for context
        ])
        
        system_prompt = """You are a QA quality validator. Evaluate if the answer is correct and 
relevant to the question based on the dialogue context. 

Check:
1. Does the answer directly address the question?
2. Is the answer consistent with the dialogue context?
3. Does the answer use appropriate financial terminology?
4. Is the answer helpful and informative?

Respond with ONLY a JSON object:
{"valid": true/false, "reason": "brief explanation"}"""
        
        user_prompt = f"""Question: {qa_pair.query}

Answer: {qa_pair.response}

Dialogue Context:
{dialogue_context}

User Profile: {persona.name}, {persona.age}, {persona.occupation}, Risk tolerance: {persona.risk_tolerance}

Is this a valid QA pair?"""
        
        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            result = json.loads(response)
            return result.get("valid", True)  # Default to True if parsing fails
        except Exception as e:
            print(f"LLM validation failed: {e}, using basic validation")
            return self._basic_validation(qa_pair)
    
    def validate_batch(self, qa_pairs: List[QAPair], sessions: List[DialogueSession],
                      persona: UserPersona) -> List[QAPair]:
        """Validate a batch of QA pairs and return only valid ones.
        
        Args:
            qa_pairs: List of QA pairs to validate
            sessions: List of dialogue sessions
            persona: User persona
            
        Returns:
            List of valid QA pairs
        """
        valid_qa_pairs = []
        
        # Create session lookup
        session_dict = {s.session_id: s for s in sessions}
        
        for qa_pair in qa_pairs:
            session = session_dict.get(qa_pair.context_session_id)
            if not session:
                # For cross-session questions, use first session
                session = sessions[0] if sessions else None
            
            if session and self.validate(qa_pair, session, persona):
                valid_qa_pairs.append(qa_pair)
        
        return valid_qa_pairs


# ============== Question Type Definitions ==============
# Question types with templates for generation (removed extended and cross_session)
QUESTION_TYPE_TEMPLATES = {
    # User preference (choice questions) - 用户偏好（选择题）
    "preference": {
        "description": "Multiple choice questions about user preferences based on dialogue",
        "templates": [
            "Given my situation, which option would be better: {option_a} or {option_b}?",
            "Should I choose {option_a} or {option_b} for my {goal}?",
            "Between {option_a} and {option_b}, which is more suitable for someone with my risk tolerance?",
            "What do you think about {option_a} vs {option_b} for {purpose}?"
        ]
    },
    # Financial planning (answer questions) - 财务规划（回答）
    "planning": {
        "description": "Questions about financial planning strategies",
        "templates": [
            "How should I plan my {goal} given my current financial situation?",
            "What's the best approach to achieve {goal} in {timeframe}?",
            "Can you help me create a plan for {goal}?",
            "What steps should I take to prepare for {goal}?",
            "How do I prioritize {goal} among my other financial objectives?"
        ]
    },
    # Investment advice (multiple choice) - 投资建议（选择题）
    "investment_advice": {
        "description": "Multiple choice investment recommendations",
        "templates": [
            "For my portfolio, should I allocate more to {option_a} or {option_b}?",
            "Which investment strategy aligns better with my goals: {option_a} or {option_b}?",
            "Given my risk tolerance, would you recommend {option_a} or {option_b}?",
            "Should I invest in {option_a} or {option_b} for {purpose}?"
        ]
    },
    # Financial summary (answer) - 财务总结（回答）
    "summary": {
        "description": "Questions requiring financial summary or analysis",
        "templates": [
            "Can you summarize my current financial position based on our conversation?",
            "What are the key takeaways from our discussion about {topic}?",
            "Based on my financial history, what's my current financial health?",
            "Can you provide a brief overview of my financial situation we've discussed?",
            "What should I focus on based on our conversation about {topic}?"
        ]
    },
    # Financial calculation (answer with numbers) - 财务计算（回答）
    "calculation": {
        "description": "Questions requiring numerical calculations",
        "templates": [
            "How much do I need to save monthly to reach {goal} in {years} years?",
            "What's the expected return if I invest {amount} at {rate}% for {years} years?",
            "Can you calculate how much I'll have if I contribute {amount} monthly with {rate}% return?",
            "What's the total cost of {item} including {factor}?",
            "How much should I allocate to {category} based on the 50/30/20 rule?"
        ]
    }
}


class EnhancedQAPairGenerator:
    """Enhanced QA pair generator that creates diverse questions using LLM.
    
    This generator creates questions that are:
    - NOT direct copies of original dialogue
    - Related to but different from original content
    - Cover multiple question types: preference, planning, investment, summary, calculation
    - Answers are concise and use original dialogue vocabulary
    - Validated for quality before being kept
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        self.validator = QAQalityValidator(llm_client, config) if llm_client else None
        random.seed(self.config.seed)
    
    def _build_context(self, sessions: List[DialogueSession], persona: UserPersona,
                      timeline: List[TimelineEvent]) -> Dict:
        """Build comprehensive context from all sessions, persona, and timeline."""
        # Persona summary
        persona_info = (
            f"Name: {persona.name}, Age: {persona.age}, Occupation: {persona.occupation}\n"
            f"Income: {persona.income_level}, Family: {persona.family_status}\n"
            f"Risk Tolerance: {persona.risk_tolerance}, Knowledge: {persona.financial_knowledge_level}\n"
            f"Financial Goals: {', '.join(persona.financial_goals)}"
        )
        
        # Timeline summary
        timeline_summary = "\n".join([
            f"- {e.timestamp}: {e.description} (${e.amount if e.amount else 0})"
            for e in timeline[-10:]
        ]) if timeline else "No timeline events"
        
        # Session summaries - keep original dialogue vocabulary
        session_summaries = []
        for session in sessions:
            turns_text = "\n".join([
                f"{'User' if t.speaker == 'user' else 'Assistant'}: {t.message[:150]}..."
                for t in session.turns[:6]  # First 6 turns
            ])
            session_summaries.append(f"Session [{session.topic}]:\n{turns_text}")
        
        all_sessions_text = "\n\n".join(session_summaries)
        
        return {
            "persona_info": persona_info,
            "timeline_summary": timeline_summary,
            "sessions_summary": all_sessions_text,
            "session_topics": [s.topic for s in sessions]
        }
    
    def _get_dialogue_vocabulary(self, session: DialogueSession) -> List[str]:
        """Extract key vocabulary/phrases from dialogue for answer generation."""
        vocab = []
        for turn in session.turns:
            if turn.speaker == "assistant":
                # Extract key phrases (financial terms, recommendations)
                words = turn.message.split()
                for i, word in enumerate(words):
                    # Keep meaningful phrases (3-6 words)
                    if i < len(words) - 2:
                        phrase = " ".join(words[i:i+3])
                        if any(kw in phrase.lower() for kw in 
                               ["recommend", "suggest", "should", "consider", "important", 
                                "portfolio", "investment", "savings", "retirement"]):
                            vocab.append(phrase)
        return vocab[:10]  # Return top 10 phrases
    
    def _generate_question_with_llm(self, context: Dict, question_type: str, 
                                   session: DialogueSession = None) -> Optional[Dict]:
        """Generate a specific type of question using LLM."""
        if not self.llm_client:
            return self._generate_question_template(context,question_type, session)
        
        # Get dialogue vocabulary for answer generation
        dialogue_vocab = self._get_dialogue_vocabulary(session) if session else []
        vocab_text = "\n".join([f"- {v}" for v in dialogue_vocab]) if dialogue_vocab else "Use general financial terms"
        
        # Build type-specific prompts
        type_prompts = {
            "preference": """Generate a 4-option multiple-choice question about financial product/strategy preference 
based on the dialogue. The question should ask the user to choose between FOUR different options (A, B, C, D).""",
            
            "planning": """Generate a question about financial planning strategy 
based on the dialogue content. Ask about creating or improving a financial plan.""",
            
            "investment_advice": """Generate an investment allocation question with FOUR options (A, B, C, D)
that requires the assistant to recommend one option over the others.""",
            
            "summary": """Generate a question asking for a summary or analysis 
of the user's financial situation based on the conversation.""",
            
            "calculation": """Generate a financial calculation question with specific numbers 
(e.g., amounts, percentages, time periods) based on the dialogue."""
        }
        
        system_prompt = f"""You are a financial QA dataset generator. Generate diverse, high-quality 
questions based on the dialogue content. Questions should be RELATED to but NOT identical to 
the original dialogue. Make them practical and realistic.

IMPORTANT requirements:
1. For multiple-choice questions (preference, investment_advice), you MUST generate exactly 4 options labeled A, B, C, D
2. The answer should be CONCISE - keep it short (1-3 sentences max)
3. The answer should USE PHRASES from the dialogue vocabulary provided below when possible
4. The answer should indicate which option is correct (e.g., "A" or "Option A")

"""
#Dialogue vocabulary (USE THESE PHRASES IN ANSWER when relevant): {vocab_text}        
        user_prompt = f"""Based on the following dialogue and user profile, generate a {question_type} question.

User Profile:
{context['persona_info']}

Recent Financial Events:
{context['timeline_summary']}

Dialogue Sessions:
{context['sessions_summary']}

Generate ONE {question_type} question that:
1. Is related to the dialogue content but NOT a direct copy
2. Requires LONG-TERM memory: the question must depend on at least 3 specific facts mentioned earlier in the dialogue/profile/timeline (multi-hop), and those facts should be separated across different parts of the context
3. IMPORTANT (no-leak): the question MUST NOT explicitly restate the key facts needed to answer it (no exact numbers, no explicit risk tolerance labels, no repeating event details). Refer to them only indirectly (e.g., "as we discussed earlier", "based on your earlier constraints")
4. Includes a real-world trade-off or constraint conflict (e.g., liquidity vs growth, debt payoff vs investing, near-term expense vs long-term goal) so it cannot be answered with generic advice
5. Is realistic and practical for a financial assistant to answer
6. Forces a non-generic answer: the answer must contain a concrete prioritized plan (steps or timeline) AND at least one quantitative allocation (percentages or amount ranges), justified using the remembered facts
7. The answer should be CONCISE and USE dialogue vocabulary (avoid broad platitudes)

{"8. For multiple-choice: Include exactly 4 options labeled A, B, C, D. Each option must be plausible and reflect different uses of the remembered facts (not obviously wrong)." if question_type in ["preference", "investment_advice"] else ""}
{"9. The answer should indicate the correct option (e.g., 'A' or 'Option A') AND give a one-sentence justification grounded in the earlier context." if question_type in ["preference", "investment_advice"] else ""}

Format as JSON:
{{"question": "...", "answer": "...", "question_type": "{question_type}"}}

Return ONLY valid JSON."""
        
        try:
            response = self.llm_client.generate(system_prompt, user_prompt)
            result = json.loads(response)
            return result
        except Exception as e:
            try:
                response = self.llm_client.generate(system_prompt, user_prompt)
                result = json.loads(response)
                return result
            except Exception as e:
                try:
                    response = self.llm_client.generate(system_prompt, user_prompt)
                    result = json.loads(response)
                    return result
                except Exception as e:
                    print(f"LLM question generation failed for {question_type}: {e}")
    
    def _generate_question_template(self, context: Dict,question_type: str, 
                                    session: DialogueSession = None) -> Optional[Dict]:
        """Fallback template-based question generation."""
        templates = QUESTION_TYPE_TEMPLATES.get(question_type, {})
        template_list = templates.get("templates", [])
        
        if not template_list:
            return None
        
        template = random.choice(template_list)
        
        # Fill in template with session-specific info
        if session:
            topic = session.topic.replace("_", " ")
            template = template.format(
                topic=topic,
                topic_a=random.choice(context.get("session_topics", ["investments"])).replace("_", " "),
                topic_b=random.choice(context.get("session_topics", ["budgeting"])).replace("_", " "),
                goal=random.choice(["retirement", "emergency fund", "home purchase", "education"]),
                timeframe=random.choice(["5 years", "10 years", "20 years"]),
                years=random.choice([5, 10, 15, 20]),
                amount="$10,000",
                rate="7",
                item="investment",
                factor="compounding",
                category="retirement savings",
                option_a=random.choice(["stocks", "bonds", "index funds"]),
                option_b=random.choice(["cash", "real estate", "cryptocurrency"]),
                purpose=random.choice(["retirement", "wealth building", "income"]),
                point=random.choice(["diversification", "risk management", "long-term growth"]),
                aspect=random.choice(["tax implications", "liquidity", "volatility"])
            )
        
        # Generate a concise answer using dialogue vocabulary
        answers = {
            "preference": "Based on your profile and the factors we've discussed, I'd recommend option A.",
            "planning": "Here's a financial plan based on your situation.",
            "investment_advice": "For your portfolio, I'd suggest allocating more to stocks.",
            "summary": "Based on our conversations, your financial position is stable.",
            "calculation": "Let me calculate that for you."
        }
        
        return {
            "question": template,
            "answer": answers.get(question_type, "Based on the dialogue."),
            "question_type": question_type
        }
    
    def generate_from_sessions(self, sessions: List[DialogueSession], persona: UserPersona,
                             timeline: List[TimelineEvent]) -> List[QAPair]:
        """Generate diverse QA pairs from all sessions.
        
        This method creates questions that are:
        1. Extracted/inferred from dialogue content (NOT direct copies)
        2. Various types: preference, planning, investment, summary, calculation
        3. Concise answers using dialogue vocabulary
        4. Validated for quality before being kept
        """
        if not sessions:
            return []
        
        # Build context
        context = self._build_context(sessions, persona, timeline)
        all_qa_pairs = []
        
        # Main question types (removed extended and cross_session)
        main_types = ["preference", "planning", "investment_advice", "summary", "calculation"]
        
        # Generate questions for each session
        for session in sessions:
            # Generate 2-3 different types of questions per session
            num_questions = random.randint(2, 3)
            selected_types = random.sample(main_types, min(num_questions, len(main_types)))
            
            for qtype in selected_types:
                result = self._generate_question_with_llm(context, qtype, session)
                
                if result:
                    qa_pair = QAPair(
                        query_id=f"qa_{uuid.uuid4().hex[:8]}",
                        query=result["question"],
                        response=result["answer"],
                        context_session_id=session.session_id,
                        timestamp=session.start_timestamp,
                        question_type=result["question_type"],
                        dialogue_history=self._get_relevant_history(session, result["question"]),
                        referenced_timeline_events=[e.to_dict() for e in 
                            self._find_relevant_events(result["question"], timeline)]
                    )
                    all_qa_pairs.append(qa_pair)
        
        # Validate QA pairs if validator is available
        if self.validator:
            valid_qa_pairs = self.validator.validate_batch(all_qa_pairs, sessions, persona)
            return valid_qa_pairs
        
        return all_qa_pairs
    
    def _get_relevant_history(self, session: DialogueSession, query: str) -> List[Dict]:
        """Get relevant dialogue history for a query."""
        # Find assistant responses that might be relevant
        relevant = []
        query_keywords = set(query.lower().split())
        
        for turn in session.turns:
            if turn.speaker == "assistant":
                # Keep assistant responses as they're most relevant
                relevant.append({
                    "turn_number": turn.turn_number,
                    "speaker": turn.speaker,
                    "message": turn.message
                })
        
        return relevant[:3]  # Max 3 turns
    
    def _get_cross_session_history(self, sessions: List[DialogueSession]) -> List[Dict]:
        """Get history from multiple sessions for cross-session questions."""
        history = []
        
        # Get 1-2 turns from each session
        for session in sessions[:3]:  # Max 3 sessions
            if session.turns:
                # Get middle assistant turn
                assistant_turns = [t for t in session.turns if t.speaker == "assistant"]
                if assistant_turns:
                    mid_idx = len(assistant_turns) // 2
                    history.append({
                        "session_id": session.session_id,
                        "topic": session.topic,
                        "turn_number": assistant_turns[mid_idx].turn_number,
                        "speaker": "assistant",
                        "message": assistant_turns[mid_idx].message[:200]  # Truncate for context
                    })
        
        return history
    
    def _find_relevant_events(self, query: str, timeline: List[TimelineEvent]) -> List[TimelineEvent]:
        """Find timeline events relevant to the query."""
        query_lower = query.lower()
        relevant = []
        
        event_keywords = {
            "income_change": ["income", "salary", "bonus", "raise", "promotion", "job", "money"],
            "expense_increase": ["expense", "bill", "cost", "medical", "repair", "fee", "spending"],
            "expense_decrease": ["paid off", "debt", "loan", "save", "reduce", "budget"],
            "investment_event": ["investment", "stock", "dividend", "portfolio", "crypto", "return"],
            "savings_change": ["savings", "401k", "fund", "account", "emergency"],
            "life_event": ["married", "baby", "child", "retired", "moved", "family"]
        }
        
        for event in timeline:
            for event_type, keywords in event_keywords.items():
                if event.event_type == event_type and any(kw in query_lower for kw in keywords):
                    relevant.append(event)
                    break
        
        return relevant[:3]  # Max 3 events


class QAPairGenerator:
    """Legacy QA pair generator - extracts from dialogue sessions.
    
    DEPRECATED: Use EnhancedQAPairGenerator for better quality questions.
    This class is kept for backward compatibility.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None, config: Config = None):
        self.llm_client = llm_client
        self.config = config or Config()
        self.enhanced_generator = EnhancedQAPairGenerator(llm_client, config)
    
    def extract_from_session(self, session: DialogueSession, persona: UserPersona,
                            all_timeline: List[TimelineEvent] = None) -> List[QAPair]:
        """Extract QA pairs from a dialogue session with RELEVANT dialogue history context only."""
        qa_pairs = []
        all_timeline = all_timeline or []
        
        # Keywords to identify relevant context
        topic_keywords = set()
        for kw_list in DIALOGUE_TOPICS:
            topic_keywords.update(kw_list.get("keywords", []))
        
        for idx, turn in enumerate(session.turns):
            if turn.speaker == "user":
                # Find the next assistant response
                next_turns = [t for t in session.turns if t.turn_number > turn.turn_number and t.speaker == "assistant"]
                if next_turns:
                    response = next_turns[0].message
                    
                    # Collect RELEVANT dialogue history only (last 2-3 turns that are related)
                    dialogue_history = []
                    query_keywords = set(turn.message.lower().split())
                    
                    # Get turns before this question
                    prev_turns = [t for t in session.turns if t.turn_number < turn.turn_number]
                    prev_turns.reverse()  # Start from most recent
                    
                    # Only keep turns that share keywords with query or are assistant explanations
                    related_count = 0
                    for prev_turn in prev_turns:
                        if related_count >= 3:  # Max 3 related turns
                            break
                        
                        prev_keywords = set(prev_turn.message.lower().split())
                        is_related = (
                            prev_turn.speaker == "assistant" or  # Keep assistant responses
                            bool(query_keywords & prev_keywords) or  # Shares keywords with query
                            any(kw in prev_turn.message.lower() for kw in query_keywords)
                        )
                        
                        if is_related:
                            dialogue_history.append({
                                "turn_number": prev_turn.turn_number,
                                "speaker": prev_turn.speaker,
                                "message": prev_turn.message
                            })
                            related_count += 1
                    
                    dialogue_history.reverse()  # Restore chronological order
                    
                    # Find referenced timeline events based on keywords in query
                    referenced_events = self._find_referenced_events(turn.message, all_timeline)
                    
                    qa_pair = QAPair(
                        query_id=f"qa_{uuid.uuid4().hex[:8]}",
                        query=turn.message,
                        response=response,
                        context_session_id=session.session_id,
                        timestamp=session.start_timestamp,
                        question_type=self._classify_question(turn.message),
                        dialogue_history=dialogue_history,
                        referenced_timeline_events=[e.to_dict() for e in referenced_events]
                    )
                    qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def _find_referenced_events(self, query: str, timeline: List[TimelineEvent]) -> List[TimelineEvent]:
        """Find timeline events that are RELEVANT to the query (keyword matching)."""
        query_lower = query.lower()
        referenced = []
        
        # Define keywords that might link to timeline events
        event_keywords = {
            "income_change": ["income", "salary", "bonus", "raise", "promotion", "job"],
            "expense_increase": ["expense", "bill", "cost", "medical", "repair", "fee"],
            "expense_decrease": ["paid off", "debt", "loan", "save", "reduce"],
            "investment_event": ["investment", "stock", "dividend", "portfolio", "crypto"],
            "savings_change": ["savings", "401k", "fund", "account"],
            "life_event": ["married", "baby", "child", "retired", "moved"]
        }
        
        for event in timeline:
            # Check if any keyword matches
            for event_type, keywords in event_keywords.items():
                if event.event_type == event_type and any(kw in query_lower for kw in keywords):
                    referenced.append(event)
                    break
        
        return referenced[:3]  # Max 3 referenced events
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        if any(kw in question_lower for kw in ["how much", "amount", "calculate"]):
            return "calculation"
        elif any(kw in question_lower for kw in ["should", "recommend", "best", "advice"]):
            return "recommendation"
        elif any(kw in question_lower for kw in ["what is", "explain", "difference"]):
            return "explanation"
        elif any(kw in question_lower for kw in ["when", "timeline", "how long"]):
            return "temporal"
        elif any(kw in question_lower for kw in ["can i", "able", "possible"]):
            return "possibility"
        else:
            return "general"
    
    def generate_batch(self, sessions: List[DialogueSession], persona: UserPersona,
                      all_timeline: List[TimelineEvent] = None) -> List[QAPair]:
        """Generate QA pairs from multiple sessions with timeline context.
        
        Uses EnhancedQAPairGenerator when LLM is available for better quality questions.
        Falls back to legacy extraction when LLM is not available.
        """
        # Use enhanced generator when LLM is available
        if self.llm_client:
            return self.enhanced_generator.generate_from_sessions(sessions, persona, all_timeline)
        
        # Fallback to legacy extraction
        all_qa_pairs = []
        for session in sessions:
            qa_pairs = self.extract_from_session(session, persona, all_timeline)
            all_qa_pairs.extend(qa_pairs)
        return all_qa_pairs


class DatasetFormatter:
    """Formats and exports the dataset."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def format_single_entry(self, persona: UserPersona, timeline: List[TimelineEvent],
                           sessions: List[DialogueSession], qa_pairs: List[QAPair]) -> Dict:
        """Format a single dataset entry."""
        return {
            "persona": persona.to_dict(),
            "timeline": [e.to_dict() for e in timeline],
            "sessions": [s.to_dict() for s in sessions],
            "qa_pairs": [qa.to_dict() for qa in qa_pairs],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_sessions": len(sessions),
                "total_qa_pairs": len(qa_pairs),
                "total_timeline_events": len(timeline)
            }
        }
    
    def create_long_context_format(self, persona: UserPersona, timeline: List[TimelineEvent],
                                    sessions: List[DialogueSession], qa_pairs: List[QAPair],
                                    max_version: str = "32k") -> Dict:
        """Create a long-context format for training."""
        # Build conversation history
        conversation_history = []
        
        for session in sessions:
            conversation_history.append(f"\n[Session: {session.topic}]")
            for turn in session.turns:
                prefix = "User: " if turn.speaker == "user" else "Assistant: "
                conversation_history.append(f"{prefix}{turn.message}")
        
        history_text = "\n".join(conversation_history)
        
        # Build timeline summary
        timeline_summary = "\n".join([
            f"{e.timestamp} - {e.description}: ${e.amount if e.amount else 'N/A'}"
            for e in timeline
        ])
        
        return {
            "persona_id": persona.user_id,
            "persona_summary": f"{persona.name}, {persona.age}, {persona.occupation}, {persona.income_level}",
            "financial_goals": persona.financial_goals,
            "risk_tolerance": persona.risk_tolerance,
            "knowledge_level": persona.financial_knowledge_level,
            "timeline": timeline_summary,
            "conversation_history": history_text,
            "qa_pairs": [
                {
                    "query": qa.query,
                    "response": qa.response,
                    "query_type": qa.question_type
                }
                for qa in qa_pairs
            ],
            "format_version": max_version
        }
    
    def export(self, dataset: List[Dict], output_path: str, format_type: str = "json"):
        """Export the dataset to a file."""
        import json
        import gzip
        
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        elif format_type == "jsonl":
            with open(output_path, 'w') as f:
                for entry in dataset:
                    f.write(json.dumps(entry) + '\n')
        elif format_type == "gz":
            with gzip.open(output_path, 'wt') as f:
                json.dump(dataset, f)
        else:
            raise ValueError(f"Unknown format: {format_type}")


# ============== Main Generator Orchestrator ==============
class FinanceDialogueDatasetGenerator:
    """Main orchestrator for dataset generation."""
    
    def __init__(self, config: Config = None, use_json_personas: bool = True, 
                 json_path: str = "./generated_personas.json"):
        self.config = config or Config()
        self.llm_client = LLMClient(self.config) if self.config.api_key else None
        
        # Use JsonPersonaGenerator by default for consistency
        if use_json_personas:
            self.persona_generator = JsonPersonaGenerator(
                self.llm_client, self.config, json_path
            )
            print(f"Using JsonPersonaGenerator with {json_path}")
        else:
            self.persona_generator = FixedPersonaGenerator(self.llm_client, self.config)
            print("Using FixedPersonaGenerator (5 seed personas)")
        
        self.timeline_generator = TimelineGenerator(self.llm_client, self.config)
        self.session_generator = DialogueSessionGenerator(self.llm_client, self.config)
        self.qa_generator = QAPairGenerator(self.llm_client, self.config)
        self.formatter = DatasetFormatter(self.config)
        
        # Initialize image generator if enabled
        self.image_generator = None
        if self.config.generate_images:
            self.image_generator = RealImageGenerator(
                output_dir=self.config.image_output_dir,
                config=self.config
            )
            if self.image_generator.matplotlib_available:
                print(f"Real image generation enabled. Images will be saved to: {self.config.image_output_dir}")
            else:
                print("Warning: matplotlib not available. Real image generation disabled.")
                self.image_generator = None
    
    def generate_dataset(self, num_personas: int = None, sessions_per_persona: int = None,
                         output_versions: List[str] = None) -> List[Dict]:
        """
        Generate the complete dataset.
        
        Args:
            num_personas: Number of user personas to generate
            sessions_per_persona: Number of dialogue sessions per persona
            output_versions: Output versions to generate ["32k", "128k"]
        
        Returns:
            List of formatted dataset entries
        """
        num_personas = num_personas or self.config.num_personas
        sessions_per_persona = sessions_per_persona or self.config.sessions_per_persona
        output_versions = output_versions or ["32k"]
        
        print(f"Generating dataset with {num_personas} personas, {sessions_per_persona} sessions each...")
        if self.image_generator:
            print(f"Image generation enabled with {self.config.image_probability*100:.0f}% probability per session")
        
        dataset = []
        
        for persona in tqdm(self.persona_generator.generate_batch(num_personas), desc="Processing personas"):
            # Generate timeline
            timeline = self.timeline_generator.generate_for_persona(persona)
            
            # Generate dialogue sessions
            sessions = self.session_generator.generate_batch(persona, timeline, sessions_per_persona)
            
            # Optionally generate images for sessions
            session_images = {}
            if self.image_generator and self.config.image_probability > 0:
                for session in sessions:
                    if random.random() < self.config.image_probability:
                        # Try to generate topic-specific image
                        image_result = self.image_generator.generate_for_topic(
                            session.topic, session.session_id
                        )
                        if image_result is None:
                            # Fallback to random image
                            image_result = self.image_generator.generate_random(session.session_id)
                        if image_result:
                            session_images[session.session_id] = image_result
            
            # Generate QA pairs with dialogue history and timeline context
            qa_pairs = self.qa_generator.generate_batch(sessions, persona, timeline)
            
            # Add image information to QA pairs if available
            for qa in qa_pairs:
                session_id = qa.context_session_id
                if session_id in session_images:
                    qa.image_description = session_images[session_id].get("image_description")
                    qa.image_type = session_images[session_id].get("image_type")
            
            # Format entry
            entry = self.formatter.format_single_entry(persona, timeline, sessions, qa_pairs)
            
            # Add image metadata if any images were generated
            if session_images:
                entry["metadata"]["generated_images"] = len(session_images)
                entry["images"] = list(session_images.values())
            
            dataset.append(entry)
        
        print(f"Generated {len(dataset)} dataset entries")
        return dataset
    
    def export_dataset(self, dataset: List[Dict], output_dir: str = None, 
                       formats: List[str] = None):
        """Export dataset in multiple formats."""
        output_dir = output_dir or self.config.output_dir
        formats = formats or ["json", "jsonl"]
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            output_path = os.path.join(output_dir, f"finance_dialogue.{fmt}")
            self.formatter.export(dataset, output_path, fmt)
            print(f"Exported to {output_path}")


# ============== Entry Point ==============
def main():
    """Main entry point for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Finance Dialogue Dataset")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--base-url", type=str, default="", help="Custom API endpoint (e.g., for private LLM services)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--num-personas", type=int, default=1, help="Number of personas")
    parser.add_argument("--sessions-per-persona", type=int, default=20, help="Sessions per persona")
    parser.add_argument("--turns-per-session", type=int, default=30, help="Turns per session")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--output-formats", type=str, default="json,jsonl", help="Output formats")
    parser.add_argument("--use-json-personas", action="store_true", default=True, 
                        help="Use personas from JSON file (default: True)")
    parser.add_argument("--json-path", type=str, default="./generated_personas.json", 
                        help="Path to JSON file with personas")
    parser.add_argument("--generate-images", action="store_true", default=False,
                        help="Enable real image generation (requires matplotlib)")
    parser.add_argument("--image-output-dir", type=str, default="./output/images",
                        help="Directory to save generated images")
    parser.add_argument("--image-probability", type=float, default=0.3,
                        help="Probability of generating an image for a session (0.0-1.0)")
    
    args = parser.parse_args()
    
    config = Config(
        api_key=args.api_key or "",
        base_url=args.base_url or "",
        model=args.model,
        num_personas=args.num_personas,
        sessions_per_persona=args.sessions_per_persona,
        turns_per_session=args.turns_per_session,
        output_dir=args.output_dir,
        generate_images=args.generate_images,
        image_output_dir=args.image_output_dir,
        image_probability=args.image_probability
    )
    
    generator = FinanceDialogueDatasetGenerator(
        config, 
        use_json_personas=args.use_json_personas,
        json_path=args.json_path
    )
    dataset = generator.generate_dataset()
    generator.export_dataset(dataset, formats=args.output_formats.split(","))


if __name__ == "__main__":
    main()
