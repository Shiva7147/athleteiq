# AthleteIQ — AI Sports Science Agent

An AI-powered sports science assistant that helps coaches monitor 
athlete health, prevent injuries, and make evidence-based decisions.

## What it does
- Answers sports science questions from a 50-entry evidence-based knowledge base
- Automatically calculates ACWR and flags injury risk zones
- Logs athlete metrics with timestamps
- Escalates medical emergencies to professional referral immediately

## Tech Stack
- **LLM:** OpenRouter API (LangChain compatible)
- **Framework:** LangChain LCEL + LangGraph ReAct agent
- **Vector DB:** ChromaDB with HuggingFace embeddings
- **Tools:** 4 custom tools with safety guardrails
- **Safety:** 3-tier escalation system (IMMEDIATE / URGENT / ROUTINE)

## Architecture
User query → LangGraph agent → selects tool → 
RAG retrieval / ACWR calculation / data logging / safety flag → 
evidence-based response

## Built by
Shiva Shankar S — AI/ML Engineer
github.com/Shiva7147
