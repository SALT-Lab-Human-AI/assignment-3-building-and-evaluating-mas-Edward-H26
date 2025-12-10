[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/r1tAQ0HC)
# Multi-Agent Research System - Assignment 3

A multi-agent system for deep research on HCI topics, featuring orchestrated agents, safety guardrails, and LLM-as-a-Judge evaluation.

## Overview

This template provides a starting point for building a multi-agent research assistant system. The system uses multiple specialized agents to:
- Plan research tasks
- Gather evidence from academic papers and web sources
- Synthesize findings into coherent responses
- Evaluate quality and verify accuracy
- Ensure safety through guardrails

## Project Structure

```
.
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Base agent class
│   │   ├── planner_agent.py # Task planning agent
│   │   ├── researcher_agent.py # Evidence gathering agent
│   │   ├── critic_agent.py  # Quality verification agent
│   │   └── writer_agent.py  # Response synthesis agent
│   ├── guardrails/          # Safety guardrails
│   │   ├── safety_manager.py # Main safety coordinator
│   │   ├── input_guardrail.py # Input validation
│   │   └── output_guardrail.py # Output validation
│   ├── tools/               # Research tools
│   │   ├── web_search.py    # Web search integration
│   │   ├── paper_search.py  # Academic paper search
│   │   └── citation_tool.py # Citation formatting
│   ├── evaluation/          # Evaluation system
│   │   ├── judge.py         # LLM-as-a-Judge implementation
│   │   └── evaluator.py     # System evaluator
│   ├── ui/                  # User interfaces
│   │   ├── cli.py           # Command-line interface
│   │   └── streamlit_app.py # Web interface
│   └── orchestrator.py      # Agent orchestration
├── data/
│   └── example_queries.json # Example test queries
├── logs/                    # Log files (created at runtime)
├── outputs/                 # Evaluation results (created at runtime)
├── config.yaml              # System configuration
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── main.py                 # Main entry point
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`
- Virtual environment

### 2. Installation

#### Installing uv (Recommended)

`uv` is a fast Python package installer and resolver. Install it first:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Using pip
pip install uv
```

#### Setting up the Project

Clone the repository and navigate to the project directory:

```bash
cd is-492-assignment-3
```

**Option A: Using uv (Recommended - Much Faster)**

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

**Option B: Using standard pip**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # On macOS/Linux
# OR
venv\Scripts\activate      # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Security Setup (Important!)

**Before committing any code**, set up pre-commit hooks to prevent API key leaks:

```bash
# Quick setup - installs hooks and runs security checks
./scripts/install-hooks.sh

# Or manually
pre-commit install
```

This will automatically scan for hardcoded API keys and secrets before each commit. See `SECURITY_SETUP.md` for full details.

### 4. API Keys Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required: At least one LLM API
GROQ_API_KEY=your_groq_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Recommended: At least one search API
TAVILY_API_KEY=your_tavily_api_key_here
# OR
BRAVE_API_KEY=your_brave_api_key_here

# Optional: For academic paper search
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

#### Getting API Keys

- **Groq** (Recommended for students): [https://console.groq.com](https://console.groq.com) - Free tier available
- **OpenAI**: [https://platform.openai.com](https://platform.openai.com) - Paid, requires credits
- **Tavily**: [https://www.tavily.com](https://www.tavily.com) - Student free quota available
- **Brave Search**: [https://brave.com/search/api](https://brave.com/search/api)
- **Semantic Scholar**: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) - Free tier available

### 5. Configuration

Edit `config.yaml` to customize your system:

- Choose your research topic
- **Configure agent prompts** (see below)
- Set model preferences (Groq vs OpenAI)
- Define safety policies
- Configure evaluation criteria

#### Customizing Agent Prompts

You can customize agent behavior by setting the `system_prompt` in `config.yaml`:

```yaml
agents:
  planner:
    system_prompt: |
      You are an expert research planner specializing in HCI.
      Focus on recent publications and seminal works.
      After creating the plan, say "PLAN COMPLETE".
```

**Important**: Custom prompts must include handoff signals:
- **Planner**: Must include `"PLAN COMPLETE"`
- **Researcher**: Must include `"RESEARCH COMPLETE"`  
- **Writer**: Must include `"DRAFT COMPLETE"`
- **Critic**: Must include `"APPROVED - RESEARCH COMPLETE"` or `"NEEDS REVISION"`

Leave `system_prompt: ""` (empty) to use the default prompts.

## Implementation Guide

This template provides the structure - you need to implement the core functionality. Here's what needs to be done:

### Phase 1: Core Agent Implementation

1. **Implement Agent Logic** (in `src/agents/`)
   - [ ] Complete `planner_agent.py` - Integrate LLM to break down queries
   - [ ] Complete `researcher_agent.py` - Integrate search APIs (Tavily, Semantic Scholar)
   - [ ] Complete `critic_agent.py` - Implement quality evaluation logic
   - [ ] Complete `writer_agent.py` - Implement synthesis with proper citations

2. **Implement Tools** (in `src/tools/`)
   - [ ] Complete `web_search.py` - Integrate Tavily or Brave API
   - [ ] Complete `paper_search.py` - Integrate Semantic Scholar API
   - [ ] Complete `citation_tool.py` - Implement APA citation formatting

### Phase 2: Orchestration

Choose your preferred framework to implement the multi-agent system. The current assignment template code uses AutoGen, but you can also choose to use other frameworks as you prefer (e.g., LangGraph and Crew.ai).


3. **Update `orchestrator.py`**
   - Integrate your chosen framework
   - Implement the workflow: plan → research → write → critique → revise
   - Add error handling

### Phase 3: Safety Guardrails

4. **Implement Guardrails** (in `src/guardrails/`)
   - [ ] Choose framework: Guardrails AI or NeMo Guardrails
   - [ ] Define safety policies in `safety_manager.py`
   - [ ] Implement input validation in `input_guardrail.py`
   - [ ] Implement output validation in `output_guardrail.py`
   - [ ] Set up safety event logging

### Phase 4: Evaluation

5. **Implement LLM-as-a-Judge** (in `src/evaluation/`)
   - [ ] Complete `judge.py` - Integrate LLM API for judging
   - [ ] Define evaluation rubrics for each criterion
   - [ ] Implement score parsing and aggregation

6. **Create Test Dataset**
   - [ ] Add more test queries to `data/example_queries.json`
   - [ ] Define expected outputs or ground truths where possible
   - [ ] Cover different query types and topics

### Phase 5: User Interface

7. **Complete UI** (choose one or both)
   - [ ] Finish CLI implementation in `src/ui/cli.py`
   - [ ] Finish web UI in `src/ui/streamlit_app.py`
   - [ ] Display agent traces clearly
   - [ ] Show citations and sources
   - [ ] Indicate safety events

## Running the System

### Command Line Interface

```bash
python main.py --mode cli
```

### Web Interface

```bash
python main.py --mode web
# OR directly:
streamlit run src/ui/streamlit_app.py
```

### Running Evaluation

```bash
python main.py --mode evaluate
```

This will:
- Load test queries from `data/example_queries.json`
- Run each query through your system
- Evaluate outputs using LLM-as-a-Judge
- Generate report in `outputs/`

## Testing

Run tests (if you create them):

```bash
pytest tests/
```

## Resources

### Documentation
- [uv Documentation](https://docs.astral.sh/uv/) - Fast Python package installer
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)
- [Tavily API](https://docs.tavily.com/)
- [Semantic Scholar API](https://api.semanticscholar.org/)
