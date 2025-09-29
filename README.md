# HealthBridge Medical Chatbot

A production-ready retrieval-augmented medical assistant built with Flask, LangChain, Gemini (Google Generative AI), and Pinecone. The application ingests medical PDFs, stores embeddings in Pinecone, and serves a chat UI that remembers conversation history using SQLite.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create a Python environment](#2-create-a-python-environment)
  - [3. Install dependencies](#3-install-dependencies)
  - [4. Configure environment variables](#4-configure-environment-variables)
  - [5. Prepare the knowledge base](#5-prepare-the-knowledge-base)
  - [6. Build or refresh the Pinecone index](#6-build-or-refresh-the-pinecone-index)
  - [7. Run the Flask application](#7-run-the-flask-application)
  - [8. Persisted chat history](#8-persisted-chat-history)
- [Running with Docker](#running-with-docker)
- [CI/CD Deployment to AWS](#cicd-deployment-to-aws)
  - [GitHub Secrets](#github-secrets)
  - [Set up the self-hosted runner](#set-up-the-self-hosted-runner)
  - [Deployment flow](#deployment-flow)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features
- **Retrieval-Augmented Generation (RAG):** Ingest PDFs, chunk them, and store embeddings in Pinecone for fast similarity search.【F:store_index.py†L1-L40】【F:src/helper.py†L1-L45】
- **Gemini-powered chat:** Uses `ChatGoogleGenerativeAI` with configurable model, temperature, and safety settings supplied via environment variables.【F:app.py†L1-L83】
- **Persistent conversation history:** Chat history is automatically written to SQLite and restored on refresh; the storage path is configurable via `CHAT_HISTORY_DB_PATH`.【F:app.py†L85-L124】【F:src/history.py†L1-L40】
- **Flask web UI:** Simple chat interface served from `templates/chat.html` at `http://localhost:8080`.【F:app.py†L94-L136】
- **Containerized deployment:** Dockerfile provided; GitHub Actions workflow builds and pushes images to Amazon ECR and deploys to an EC2-hosted runner.【F:Dockerfile†L1-L8】【F:.github/workflows/cicd.yaml†L1-L54】

## Prerequisites
- Python 3.10+
- pip (or conda)
- Pinecone account and API key
- Google Cloud project with access to Gemini API and API key
- (Optional) AWS account for CI/CD deployment

## Project Structure
```
.
├── app.py                # Flask application entrypoint
├── data/                 # Place source PDFs here; stores chat history SQLite DB
├── src/
│   ├── helper.py         # PDF loading, splitting, and embedding utilities
│   ├── history.py        # Persistent chat history helpers
│   └── prompt.py         # Prompt templates
├── store_index.py        # Script to build Pinecone index from PDFs
├── templates/
│   └── chat.html         # Web UI
├── Dockerfile            # Container build definition
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
cd Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS
```

### 2. Create a Python environment
Create and activate an isolated environment (choose one):

**Conda**
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

**venv**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the example file and update it with your credentials:
```bash
cp .env.example .env
```
Edit `.env` to include your keys:
```ini
PINECONE_API_KEY=your-pinecone-api-key
GOOGLE_API_KEY=your-google-api-key
# Optional overrides
# GEMINI_MODEL=gemini-1.5-flash
# GEMINI_TEMPERATURE=0.2
# GEMINI_SAFETY_SETTINGS={"HARASSMENT":"BLOCK_NONE"}
# FLASK_SECRET_KEY=replace-with-a-secret-value
# CHAT_HISTORY_DB_PATH=/absolute/path/to/chat_history.sqlite
```

### 5. Prepare the knowledge base
Place the medical PDFs you want to ingest inside the `data/` directory. The defaults in the repository are used if you skip this step.【F:store_index.py†L17-L24】

### 6. Build or refresh the Pinecone index
Run the embedding pipeline to create (or update) the Pinecone index named `medical-chatbot`:
```bash
python store_index.py
```
The script will create the index if it does not exist and upsert the embeddings for all PDFs in `data/`.【F:store_index.py†L17-L40】

### 7. Run the Flask application
Start the chat server:
```bash
python app.py
```
The app listens on `http://localhost:8080` by default. Visit the URL in your browser to interact with the chatbot.【F:app.py†L129-L139】

### 8. Persisted chat history
- Conversations are saved to `data/chat_history.sqlite`. Delete this file to reset history or change the location with `CHAT_HISTORY_DB_PATH` in `.env`.
- Each browser session receives its own conversation ID so refreshes will show prior context.【F:app.py†L94-L124】【F:src/history.py†L1-L40】

## Running with Docker
1. Ensure `.env` is populated (as described above).
2. Build the image:
   ```bash
   docker build -t medical-chatbot .
   ```
3. Populate Pinecone (one-time or whenever PDFs change):
   ```bash
   docker run --rm --env-file .env medical-chatbot python store_index.py
   ```
4. Start the containerized app:
   ```bash
   docker run --rm --env-file .env -p 8080:8080 medical-chatbot
   ```
   Navigate to `http://localhost:8080`.

## CI/CD Deployment to AWS
This project ships with a GitHub Actions workflow that builds a Docker image, pushes it to Amazon Elastic Container Registry (ECR), and deploys it on an EC2-based self-hosted runner.【F:.github/workflows/cicd.yaml†L1-L54】

### GitHub Secrets
Configure the following repository secrets before enabling the workflow:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO` – ECR repository name (e.g., `medicalbot`)
- `PINECONE_API_KEY`
- `GOOGLE_API_KEY`
- `OPENAI_API_KEY` – required by the workflow run command even if unused by the app

### Set up the self-hosted runner
1. Launch an Ubuntu EC2 instance with access to your VPC and security groups.
2. Install Docker on the instance:
   ```bash
   sudo apt-get update -y
   sudo apt-get install -y docker.io
   sudo usermod -aG docker $USER
   newgrp docker
   ```
3. Register the EC2 machine as a self-hosted GitHub Actions runner (`Settings → Actions → Runners → New self-hosted runner`).

### Deployment flow
1. Push to the `main` branch to trigger the workflow.
2. The CI job builds the Docker image and pushes it to the configured ECR repository.
3. The CD job (running on the EC2 runner) pulls the latest image from ECR and starts the container, exposing port `8080` to serve the chatbot.【F:.github/workflows/cicd.yaml†L9-L54】

## Troubleshooting
- **Pinecone authentication errors:** Ensure `PINECONE_API_KEY` is set in `.env` and the index region matches the Serverless spec (`aws`, `us-east-1`).【F:store_index.py†L17-L37】
- **Gemini authorization errors:** Confirm your Google Cloud project has Gemini API access and that `GOOGLE_API_KEY` is valid.【F:app.py†L31-L55】
- **Chat history not persisting:** Verify the process can write to `data/` or override `CHAT_HISTORY_DB_PATH` to a writable location.【F:src/history.py†L7-L22】
- **Docker cannot reach Pinecone or Gemini:** Pass environment variables via `--env-file .env` or individual `-e` flags when running containers.

## License
This project is licensed under the terms of the [MIT License](LICENSE).
