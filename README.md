# AthleteIQ — AI Sports Science Agent

> An AI-powered sports science assistant that helps coaches monitor 
> athlete health, prevent injuries, and make evidence-based decisions.

[![Accuracy](https://img.shields.io/badge/Eval%20Accuracy-95%25-brightgreen)]()
[![Knowledge Base](https://img.shields.io/badge/Knowledge%20Base-56%20entries-blue)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)]()
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-purple)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct%20Agent-orange)]()

---

## What AthleteIQ does

AthleteIQ is a RAG-powered AI agent built for sports coaches. It answers 
evidence-based sports science questions, automatically calculates injury 
risk from training load data, logs athlete metrics, and escalates medical 
emergencies to the appropriate level of professional care.

**Key capabilities:**
- Answers sports science questions from a 56-entry evidence-based knowledge base
- Automatically calculates ACWR and flags injury risk zones (safe / caution / danger)
- Logs athlete metrics with timestamps — HRV, RPE, sleep hours, training load
- Escalates medical emergencies with 3-tier referral system (IMMEDIATE / URGENT / ROUTINE)
- Refuses to give advice when professional medical care is required

---

## Evaluation Results

Evaluated on 20 ground truth test cases across 9 categories.  
Uses a hybrid evaluation approach: rule-based scoring for safety-critical 
questions, LLM-as-judge for general knowledge questions.

| Category | Score | Notes |
|---|---|---|
| Overtraining | 100% | All markers correctly identified |
| Hydration | 100% | Specific volumes and protocols accurate |
| Injury | 100% | Return-to-play criteria correct |
| Youth Athletes | 100% | Age thresholds and load limits accurate |
| Mental Performance | 100% | Rehearsal protocols and focus theory correct |
| Training Load | 87% | ACWR, RPE, periodisation |
| Recovery | 95% | Sleep, HRV, cold water immersion |
| Performance | 87% | VO2 max, lactate threshold, caffeine |
| Safety | 87% | Emergency protocols, concussion, heat stroke |
| **Overall** | **95%** | **95/100 across all 20 test cases** |

---

## Architecture


Coach query
↓
LangGraph ReAct Agent
↓ decides which tool to call
├── search_sports_science()  → ChromaDB similarity search → RAG answer
├── calculate_acwr()         → Python math → guaranteed accurate ACWR
├── log_athlete_data()       → timestamped metric storage
└── flag_professional_referral() → 3-tier emergency escalation
↓
Evidence-based response



**Why RAG instead of fine-tuning:**  
The knowledge base updates frequently as new research emerges. RAG allows 
instant updates without retraining. Fine-tuning would also risk hallucination 
on specific numeric values — RAG retrieves exact text, grounding every answer 
in real content.

**Why a separate calculate_acwr tool:**  
LLMs predict tokens — they do not compute. For a health-critical calculation 
where ACWR above 1.5 doubles injury risk, guaranteed mathematical accuracy 
is non-negotiable. Python division is exact. LLM arithmetic is not.

**Why hybrid evaluation:**  
LLM-as-judge is inconsistent on safety-critical questions. Rule-based keyword 
checking ensures safety answers are evaluated on whether they contain required 
emergency action steps — not on stylistic similarity to a reference answer.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenRouter API (LangChain compatible) |
| Agent Framework | LangGraph ReAct agent |
| RAG Pipeline | LangChain LCEL |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Evaluation | Hybrid: rule-based + LLM-as-judge |

---

## Project Structure

athleteiq/
├── README.md
├── requirements.txt
├── athleteiq.py          ← complete single-file version
└── src/
├── knowledge_base.py ← 56 evidence-based entries
├── tools.py          ← 4 agent tools
├── agent.py          ← LangGraph ReAct agent
└── main.py           ← entry point

---

## Key Engineering Decisions

**1. Safety-first agent design**  
The agent is instructed to call `flag_professional_referral` with IMMEDIATE 
urgency before any other tool when detecting emergency symptoms. This means 
the knowledge base is never consulted for chest pain, collapse, or heat stroke 
— the system escalates first, always.

**2. Responsible AI guardrails**  
AthleteIQ explicitly does not diagnose medical conditions. It assists coaches 
with evidence-based information and escalates when professional judgment is 
required. This aligns with EU AI Act principles for AI systems operating in 
high-risk health-adjacent domains.

**3. Iterative evaluation**  
Evaluation was run in multiple rounds. Initial accuracy: 85%. After targeted 
knowledge base improvements and hybrid evaluation methodology: 95%. The 
evaluation harness identified specific failure categories, enabling precise 
knowledge base additions rather than broad rewrites.

---

## Built by

**Shiva Shankar S**  
AI/ML Engineer  
[GitHub](https://github.com/Shiva7147)
