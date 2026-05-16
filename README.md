# 🧠 Multi-Agent Orchestration System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.1+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1.10+-ff9900.svg)](https://python.langchain.com/v0.1/docs/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An advanced, event-driven architecture demonstrating state-of-the-art Multi-Agent orchestration, dynamic routing, and real-time streaming capabilities using Server-Sent Events (SSE).

---

##  Overview

The **Real-Time Multi-Agent LLM Orchestration & Evaluation System** is a cutting-edge platform designed to handle complex queries by dynamically coordinating a swarm of specialized AI agents. Moving beyond linear chains, this system employs a sophisticated **Hub-and-Spoke** topology built on **LangGraph**. A central Orchestrator LLM intelligently evaluates the shared state and autonomously delegates tasks to expert sub-agents, iterating until the problem is solved and verified.

Designed with production readiness in mind, it features real-time telemetry streaming via **FastAPI** and **SSE**, robust state management with **Pydantic**, and comprehensive self-reflection and critique loops.

---

## ✨ Key Features & Architecture

### 🕸️ Dynamic Orchestration (Hub-and-Spoke)
Instead of hardcoded pathways, a central **Orchestrator Node** acts as the cognitive engine. It analyzes the current `AgentState`, evaluates budgets, and makes routing decisions dynamically. Sub-agents execute their tasks and return control to the Orchestrator, ensuring centralized governance and minimizing infinite loops.

### 🤖 Specialized Agent Swarm
- **Decomposition Agent:** Breaks down complex user queries into manageable, parallelizable sub-tasks.
- **RAG (Retrieval-Augmented Generation) Agent:** Dynamically searches and retrieves relevant external information.
- **Critique & Reflection Agent:** Evaluates generated outputs, scoring confidence and flagging hallucinations or incomplete answers.
- **Synthesis Agent:** Combines completed sub-tasks and verified information into a cohesive final response.
- **Compression Agent:** Manages context windows by summarizing older state history when token budgets are tight.

### ⚡ Real-Time Telemetry & Streaming
Built on **FastAPI**, the system exposes asynchronous endpoints that stream granular execution events via Server-Sent Events (SSE). Clients can visualize the exact moment an agent starts, a tool is invoked, or tokens are generated.

### 🛡️ Robust State & Budget Management
Utilizes strict **Pydantic** schemas (`AgentState`, `SubTask`, `Critique`) to maintain type safety across agent boundaries. The state tracks execution history, tool logs, latency, and manages token `context_budget` limits dynamically.

---

## 🛠️ Technology Stack

- **Core Frameworks:** Python 3.12, FastAPI, Uvicorn, SSE-Starlette
- **LLM & Orchestration:** LangGraph, LangChain (Core, OpenAI, Groq)
- **Data Validation & Schemas:** Pydantic, Pydantic-Settings
- **Environment & Dependency Management:** `uv`, `python-dotenv`

---

## 📂 Project Structure

```text
├── src/
│   ├── agent/             # Core agent logic and LangGraph implementation
│   │   ├── graph.py       # StateGraph definition & edge routing
│   │   ├── orchestrator.py# Central dynamic routing logic
│   │   ├── decomposition.py # Task breakdown logic
│   │   ├── rag.py         # Retrieval augmentation
│   │   ├── critique.py    # Self-reflection and quality gating
│   │   ├── synthesis.py   # Final answer generation
│   │   └── compression.py # Token budget management
│   ├── api/               # FastAPI application and routing
│   │   └── main.py        # SSE Streaming endpoints
│   ├── config/            # Application settings and prompts
│   ├── core/              # Shared utilities (LLM factory, Context Management)
│   ├── schemas/           # Pydantic data models for AgentState
│   └── tools/             # Executable tools for agents (Web Search, Python Sandbox, SQL)
├── pyproject.toml         # Project dependencies and metadata
└── main.py                # Entry point
```

---

## 🚦 Getting Started

### Prerequisites
- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer and resolver)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/real-time-multi-agent-orchestrator.git
   cd real-time-multi-agent-orchestrator
   ```

2. **Install dependencies using uv:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

### Running the Application

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload --port 8000
```

### Testing the Streaming API

You can test the real-time SSE streaming endpoint using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query/stream" \
     -H "Content-Type: application/json" \
     -d '{"query": "Research the latest advancements in solid-state batteries and summarize the key players."}'
```
*You will see a real-time stream of agent transitions, tool invocations, and generated tokens.*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Designed and developed to push the boundaries of autonomous AI agents.*
