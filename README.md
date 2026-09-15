# LangGraph Email Assistant
Small CLI assistant for IMAP inbox workflows using LangGraph.
## Features
- List unread emails (UID, subject, sender)
- Summarize a specific email by UID
- Simple command loop in terminal
## Requirements
- Python 3.10+
- IMAP account access (for Gmail, use an app password)
Install dependencies:
```bash
pip install -r requirements.txt
```
## Configuration
Create a `.env` file in `/home/runner/work/langchain_init/langchain_init/LangGraph`:
```env
IMAP_HOST=imap.gmail.com
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_app_password
```
Used by `main.py`:
- `IMAP_HOST`
- `IMAP_USER`
- `IMAP_PASSWORD`
## Run
```bash
python main.py
```
## Supported commands
- `list unread emails`
- `summarize <UID>`
- `exit`
Example:
```text
You: list unread emails
You: summarize 12345
You: exit
```
## Troubleshooting
- **Authentication failed**: verify IMAP is enabled and app password is correct.
- **No unread messages shown**: check the configured folder (`INBOX`) and mailbox state.
- **Connection errors**: confirm host, network access, and credentials in `.env`.
## Screenshots
![LangGraph screenshot 1](image.png)
![LangGraph screenshot 2](image-1.png)
‎README.md‎
+117
Lines changed: 117 additions & 0 deletions


Original file line number	Diff line number	Diff line change
# langchain_init
This repository contains multiple small LangChain/LangGraph/Ollama experiments.  
Each folder is an independent project with its own dependencies and runtime.
## Repository structure
- `/home/runner/work/langchain_init/langchain_init/AgentAI`  
  Local assistant that routes questions to:
  - a population CSV query engine
  - a PDF-backed Canada knowledge engine
  - a simple note-saving tool
- `/home/runner/work/langchain_init/langchain_init/LangGraph`  
  CLI email assistant using IMAP + LangGraph for:
  - listing unread emails
  - summarizing an email by UID
- `/home/runner/work/langchain_init/langchain_init/Mistral`  
  Retrieval-augmented QA project with:
  - FastAPI endpoint (`app.py`)
  - Streamlit UI (`app_ui.py`)
  - document processing and embedding helpers (`document_loader.py`)
- `/home/runner/work/langchain_init/langchain_init/langchain_local`  
  Local RAG example over restaurant reviews using Chroma + Ollama.
## Prerequisites
- Python 3.10+ (LangGraph `pyproject.toml` currently declares 3.13)
- [Ollama](https://ollama.com/) running locally
- Required Ollama models pulled locally before running each project (examples in code include `llama3.2`, `llama3.2:1b`, `mistral`, `mxbai-embed-large`, `nomic-embed-text`)
## Quick start
From the repository root:
1. Open the project folder you want to run.
2. Create a virtual environment.
3. Install that folder's requirements file.
4. Run the corresponding entrypoint.
Example workflow:
```bash
cd /home/runner/work/langchain_init/langchain_init/<project-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python <entrypoint>.py
```
## Project run commands
### 1) AgentAI
```bash
cd /home/runner/work/langchain_init/langchain_init/AgentAI
pip install -r requirements.txt
python main.py
```
Notes:
- Expects local files under `data/`, including `population.csv` and `canada.pdf`.
- CLI supports population/country questions and note-saving commands.
### 2) LangGraph
```bash
cd /home/runner/work/langchain_init/langchain_init/LangGraph
pip install -r requirements.txt
python main.py
```
Before running, create a `.env` file in this folder with:
```env
IMAP_HOST=imap.gmail.com
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_app_password
```
### 3) Mistral
API:
```bash
cd /home/runner/work/langchain_init/langchain_init/Mistral
pip install -r requirements.txt
uvicorn app:app --reload
```
UI (in another terminal):
```bash
cd /home/runner/work/langchain_init/langchain_init/Mistral
streamlit run app_ui.py
```
### 4) langchain_local
```bash
cd /home/runner/work/langchain_init/langchain_init/langchain_local
pip install -r requirement.txt
python vector.py
python main.py
```
Notes:
- `vector.py` initializes the Chroma store from `realistic_restaurant_reviews.csv`.
- `main.py` starts an interactive question-answer CLI.
## Current limitations
- Dependency files are not fully standardized (`requirements.txt` vs `requirement.txt`).
- Some scripts rely on relative paths, so run commands from each project folder.
- This repo is experimental; interfaces and model names may change.
