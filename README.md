# AI Voice Calling Platform

An enterprise AI voice calling and customer management platform featuring automated outbound calling, real-time speech-to-text (STT), local LLM generation (Ollama Gemma 3), multi-language neural text-to-speech (Piper TTS & Edge-TTS), human escalation routing, and MySQL persistence.

---

## 🛠️ Architecture & Core Components

1. **Backend**: FastAPI & Uvicorn (REST API, WebSockets, static file serving)
2. **Database**: MySQL 8.0 via SQLAlchemy & PyMySQL ORM
3. **LLM Engine**: Ollama running locally at `http://localhost:11434` (`gemma3:4b`)
4. **Speech-to-Text (STT)**: Faster-Whisper (`models/whisper-base`)
5. **Text-to-Speech (TTS)**: 
   - English: Local Piper ONNX TTS (`models/piper/en_US-lessac-medium.onnx`)
   - Hindi & Marathi: High-fidelity neural voice synthesis (Edge-TTS + PyAV)
6. **Frontend**: Single Page Dashboard in [frontend/index.html](file:///c:/RSL%20Solutions/AI%20VoiceAgent/ai_voice_calling_F/ai_voice_calling_F/frontend/index.html) served directly at `/`

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: Version 3.12 or 3.13 (64-bit)
- **MySQL**: MySQL Server 8.0 running locally on port `3306`
- **Ollama**: Installed from [ollama.com](https://ollama.com)

---

### 2. Configure Environment (`.env`)

Create a `.env` file in the project root (you can copy [.env.example](file:///c:/RSL%20Solutions/AI%20VoiceAgent/ai_voice_calling_F/ai_voice_calling_F/.env.example)):

```ini
# MySQL connection string
# Format: mysql+pymysql://<username>:<password>@localhost:3306/<database_name>
DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/ai_voice_calling

# JWT Authentication Secret Key
SECRET_KEY=ai-voice-calling-secret-key-2026-secure-session
```

> **Note**: Create the database in MySQL if it doesn't already exist:
> ```sql
> CREATE DATABASE IF NOT EXISTS ai_voice_calling;
> ```

---

### 3. Setup Python Virtual Environment

In PowerShell:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Setup Ollama (Fix for "The model is currently unreachable")

The AI assistant uses **Ollama** running locally on port 11434 with model `gemma3:4b`.

1. **Verify Ollama is installed**:
   Download and install from [ollama.com/download](https://ollama.com/download).

2. **Start the Ollama service**:
   Open a separate terminal and run:
   ```powershell
   ollama serve
   ```
   *(Or ensure the Ollama icon is running in your Windows system tray)*

3. **Pull the model**:
   ```powershell
   ollama pull gemma3:4b
   ```

4. **Verify Ollama is reachable**:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   You should see `gemma3:4b` in the list of installed models.

---

### 5. Initialize Database & Create Admin User

Run the admin creation script to verify database connectivity and create the default login:

```powershell
python scripts/create_admin.py -u admin -p AdminPass123! -r admin
```

- **Username**: `admin`
- **Password**: `AdminPass123!`

---

### 6. Start the Application fast

Start the FastAPI application with Uvicorn:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🔍 Verification & Testing

To test and verify the running system:

```powershell
# Run backend audit suite
python scripts/verify_audit_suite.py

# Run comprehensive strict 14-point functional verification
python scripts/verify_all_14_strict.py
```

---

## ⚠️ Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Error: The model is currently unreachable`** | Ollama is not running or model is missing | 1. Run `ollama serve`<br>2. Run `ollama pull gemma3:4b`<br>3. Verify via `curl http://localhost:11434/api/tags` |
| **`DATABASE_URL environment variable is missing`** | No `.env` file found | Create a `.env` file with your `DATABASE_URL` as shown in `.env.example` |
| **`Access denied for user 'root'@'localhost'`** | Wrong MySQL credentials | Update the username and password in `.env` to match your MySQL server |
| **`No Python at 'C:\Users\Siddhi\...'`** | Old copied virtualenv | Delete the old `.venv` folder and run `python -m venv .venv` followed by `pip install -r requirements.txt` |
