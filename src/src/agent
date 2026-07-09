# agent.py
# AthleteIQ — LangGraph ReAct Agent
# Connects LLM + tools + knowledge base into one working agent

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from knowledge_base import get_documents
from tools import (
    calculate_acwr,
    log_athlete_data,
    flag_professional_referral,
    athlete_data_log
)
import os

# ============================================
# LLM SETUP
# ============================================
def get_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        model_name="openrouter/free",
        max_tokens=500,
        temperature=0.1
    )

# ============================================
# VECTOR DATABASE SETUP
# ============================================
def build_vectorstore():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = get_documents()
    vectorstore = Chroma.from_documents(documents, embeddings)
    print(f"Vector database ready — {len(documents)} entries")
    return vectorstore

# ============================================
# RAG SEARCH TOOL — connected to real vectorstore
# ============================================
def create_search_tool(retriever):
    @tool
    def search_sports_science(query: str) -> str:
        """Search the sports science knowledge base for evidence-based
        information about training load, injury prevention, recovery,
        and performance. Always use this first for sports science questions."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the knowledge base."
        return "\n\n".join(doc.page_content for doc in docs)
    return search_sports_science

# ============================================
# AGENT SETUP
# ============================================
def build_agent():
    llm = get_llm()
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    search_tool = create_search_tool(retriever)

    tools = [
        search_tool,
        calculate_acwr,
        log_athlete_data,
        flag_professional_referral
    ]

    system_prompt = """You are AthleteIQ — an expert AI sports science assistant
built to help coaches monitor athlete health and prevent injuries.

You have 4 tools:
- search_sports_science: use FIRST for any sports science question
- calculate_acwr: use when coach gives training load numbers
- log_athlete_data: use when coach mentions specific measurements  
- flag_professional_referral: use IMMEDIATELY for any medical emergency

Rules:
1. Always search knowledge base before answering
2. Always log any specific athlete data mentioned
3. For chest pain, collapse, heat stroke — flag IMMEDIATELY
4. Be specific and evidence-based
5. Never diagnose — assist coaches, not replace doctors"""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    return agent

# ============================================
# RUN FUNCTION
# ============================================
def run_agent(agent, user_input: str) -> str:
    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    return result["messages"][-1].content
