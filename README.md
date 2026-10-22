# AI Voice Calling Platform with Modular Telephony Layer

An enterprise AI voice calling and customer operations platform featuring real-time telephone calling (Exotel Connect Voice AI & AgentStream), web microphone conversation, speech-to-text (STT via Faster-Whisper), local LLM generation (Ollama Gemma 3), multi-language neural text-to-speech (Piper TTS for English, Edge-TTS for Hindi & Marathi), DTMF language routing, human escalation management, and MySQL / SQLite persistence.

---

## 🛠️ Telephony & System Architecture

```
Telephone Network (PSTN / Mobile)
       ▲                     │
       │ Audio Stream        │ Caller Audio (slin Linear PCM)
       │                     ▼
+---------------------------------------------------------+
|     Telephony Provider Gateway (Exotel Connect Voice AI)|
+---------------------------------------------------------+
       │                     ▲
       │ Webhooks            │ AgentStream media / clear WebSocket
       ▼                     │
+---------------------------------------------------------+
|        FastAPI Backend Platform (Port 8000 / 8001)       |
|                                                         |
|  • Webhooks:   POST /api/telephony/webhook/exotel       |
|  • Inbound:    POST /api/telephony/inbound              |
|  • Outbound:   POST /api/telephony/outbound             |
|  • Streaming:  WebSocket /api/telephony/stream/{call_id}|
+---------------------------------------------------------+
       │                     ▲
       ▼                     │
+---------------------------------------------------------+
|                  Telephony Service Layer                |
|  • TelephonyAdapter (ExotelTelephonyAdapter, MockAdapter) |
|  • Concurrency Controller (MAX_CONCURRENT_CALLS)        |
|  • Duration Guard (MAX_CALL_DURATION_SECONDS)           |
|  • VAD & Utterance Accumulator (RMS speech/silence)     |
|  • Barge-In / Interruption Handler ('clear' event)      |
+---------------------------------------------------------+
       │                     ▲
       ▼                     │
+---------------------------------------------------------+
|                 AI Conversation Engine                  |
|  1. STT Service: Faster-Whisper (16kHz Mono 16-bit PCM) |
|  2. Context Service: Multi-turn history formatting       |
|  3. LLM Service: Ollama Gemma 3                         |
|  4. Escalation Service: Hindi/Marathi & keyword queue   |
|  5. TTS Service: Piper (en) & Edge-TTS (hi, mr)         |
|  6. Audio Converter: PyAV Resampler to Telephony PCM    |
+---------------------------------------------------------+
       │
       ▼
+---------------------------------------------------------+
|              Persistence & Dashboard Layer              |
|  • MySQL 8.0 / SQLite (voice_agent.db)                  |
|  • Single-Page Operations Dashboard (frontend/index.html|
+---------------------------------------------------------+
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python**: Version 3.12 or 3.13 (64-bit)
- **Database**: Local SQLite (zero setup out of the box) or MySQL 8.0 running on port `3306`
- **Ollama**: Installed from [ollama.com](https://ollama.com) with model `gemma3:4b`
- **Exotel Account (Optional for real PSTN calls)**: Exotel Account SID, API Key, API Token, and Virtual Number. For local development, the built-in **Mock Telephony Adapter** runs immediately without paid provider credentials.

---

### 2. Environment Configuration (`.env`)

Copy `.env.example` to `.env`:

```ini
# Database Configuration
DATABASE_URL=sqlite:///./voice_agent.db
# Or MySQL: mysql+pymysql://root:password@localhost:3306/ai_voice_calling

# JWT Authentication Secret Key
SECRET_KEY=ai-voice-calling-secret-key-2026-secure-session

# Ollama LLM Configuration
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=gemma3:4b

# ==============================================================================
# Telephony Integration Settings
# ==============================================================================
# Provider selection: 'mock' (safe local simulator) or 'exotel'
TELEPHONY_PROVIDER=mock
TELEPHONY_ENABLED=false

# Public Base URL and WebSocket URL (for webhooks and real-time audio streams via ngrok tunnel)
PUBLIC_BASE_URL=https://your-domain.ngrok-free.app
TELEPHONY_WS_URL=wss://your-domain.ngrok-free.app

# Exotel Credentials (Required only when TELEPHONY_PROVIDER=exotel)
# Subdomain: api.in.exotel.com (Mumbai cluster) or api.exotel.com (Singapore)
EXOTEL_SUBDOMAIN=api.in.exotel.com
EXOTEL_ACCOUNT_SID=your_exotel_account_sid
EXOTEL_API_KEY=your_exotel_api_key
EXOTEL_API_TOKEN=your_exotel_api_token
EXOTEL_CALLER_ID=your_exotel_virtual_number
EXOTEL_WEBHOOK_SECRET=your_webhook_validation_secret

# Human Escalation Forwarding Number (PSTN destination for transfers)
HUMAN_ESCALATION_FORWARD_NUMBER=+919876543210

# Call Concurrency & Duration Safety Limits
MAX_CONCURRENT_CALLS=5
MAX_CALL_DURATION_SECONDS=600

# Telephony Audio Sample Rate (8000 or 16000 Hz)
TELEPHONY_SAMPLE_RATE=16000
```

---

### 3. Setup Python Virtual Environment

In PowerShell:

```powershell
# Create and activate virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
pip install -r requirements.txt
```

---

### 4. Initialize Admin User

```powershell
python scripts/create_admin.py -u admin -p AdminPass123! -r admin
```

---

### 5. Start the Application

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- **Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📱 Telephony Layer Capabilities & Features

### 1. Provider Abstraction
The system uses `TelephonyAdapter` (ABC) to isolate all telecom provider details from the AI engine:
- `MockTelephonyAdapter`: Complete local mock provider enabling end-to-end testing, webhook validation, and audio streaming without placing real phone calls or consuming carrier credits.
- `ExotelTelephonyAdapter`: Production integration with Exotel's official **Connect Voice AI API** and **AgentStream** real-time media WebSocket protocol.
- Factory: `get_telephony_adapter()` selects provider automatically and falls back gracefully to Mock mode if credentials are unconfigured.

### 2. Outbound Calling
- **Endpoint**: `POST /api/telephony/outbound`
- **Security**: Requires JWT authentication (`Authorization: Bearer <token>`).
- **Validation**: Enforces strict phone number normalization via `app/utils/phone_utils.py` (E.164, Indian 10-digit).
- **Concurrency & Duration Guard**: Rejects requests exceeding `MAX_CONCURRENT_CALLS` with HTTP 429; automatically terminates calls exceeding `MAX_CALL_DURATION_SECONDS`.

### 3. Bidirectional Audio Streaming & AgentStream Protocol
- **Endpoint**: `WebSocket /api/telephony/stream/{call_id}`
- **Protocol**: Exotel AgentStream (`connected`, `start`, `media`, `clear`, `dtmf`, `stop`).
- **VAD (Voice Activity Detection)**: Real-time RMS calculation on 20ms frames detects speech begin and silence threshold (750ms).
- **Barge-in / Interruption**: When caller speaks while the assistant is talking, playback is immediately aborted and a `clear` event (`{"event": "clear", "streamSid": "..."}`) is transmitted to wipe the telecom buffer.
- **Utterance Processing**: Gathers caller audio, converts to 16kHz WAV, invokes Faster-Whisper, sends prompt to Ollama Gemma 3, generates speech via Piper/Edge-TTS, resamples to Linear PCM, and streams 20ms chunks back to the caller.

### 4. Language & Human Escalation Routing
- **Greeting**: Plays IVR prompt ("Press 1 for English, 2 for Hindi, 3 for Marathi, or simply start speaking").
- **Option 1 (English)**: Full multi-turn conversation with automatic structured requirement extraction.
- **Option 2 (Hindi) & Option 3 (Marathi)**: Plays native Devanagari assistance message, creates a `HumanAssistanceRequest` in the queue, and initiates provider call transfer to `HUMAN_ESCALATION_FORWARD_NUMBER`.

### 5. Inbound Calling Flow
- **Endpoint**: `POST /api/telephony/inbound`
- Accepts incoming calls on your Exotel virtual number, recovers or registers customer records, binds a call session, and responds with AgentStream connection instructions.

### 6. Local Development via Secure Tunnel (e.g. ngrok)
To test live phone calls with Exotel during development:
```powershell
# In a separate terminal:
ngrok http 8000
```
Copy the generated forwarding URL (e.g. `https://abc-123.ngrok-free.app`) into your `.env`:
```ini
PUBLIC_BASE_URL=https://abc-123.ngrok-free.app
TELEPHONY_WS_URL=wss://abc-123.ngrok-free.app
```
Then configure your Exotel virtual number or App Bazaar Voicebot applet to target this URL.

---

## 🧪 Comprehensive Automated Test Suites

Run the automated test suites using the local Python virtualenv:

```powershell
# 1. Telephony Layer Suite (14 points: Outbound, Auth, Concurrency, Webhooks, State Machine, Streaming, VAD, Hangup)
.\.venv\Scripts\python tests/test_telephony.py

# 2. Complete 13-Point Regression Suite (Auth, CRUD, English AI, Hindi/Marathi DTMF, Escalations, Automation)
.\.venv\Scripts\python tests/test_complete_regression_13.py

# 3. Backend Integration Suite (12 points)
.\.venv\Scripts\python tests/test_backend.py

# 4. Multi-Turn Microphone Conversation Test
.\.venv\Scripts\python tests/test_microphone_conversation.py

# 5. Full End-to-End Pipeline Test
.\.venv\Scripts\python tests/test_full_e2e.py

# 6. Real Runtime Flows Verification
.\.venv\Scripts\python tests/test_real_runtime_flows.py
```

All 6 test suites execute with mocked providers without placing real phone calls or incurring paid carrier costs.

---

## 🔒 Security Best Practices

1. **Credentials**: Never commit `.env` or provider API tokens to Git. Real API keys are read securely via environment variables.
2. **Webhook Verification**: Webhooks validate tokens / HMAC signatures before accepting state updates.
3. **Session Binding**: Media WebSockets bind strictly to existing, authenticated call IDs; arbitrary sessions cannot hijack active calls.
4. **Data Privacy**: Phone numbers are masked in logs and public UI views (`mask_phone_number()`).
5. **State Machine Guards**: Stale callbacks or duplicate webhook events cannot transition a terminated call back into an active state.

---

## ⚠️ Troubleshooting Guide

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Concurrency limit reached (429)`** | Active calls equal or exceed `MAX_CONCURRENT_CALLS` | Increase `MAX_CONCURRENT_CALLS` in `.env` or hang up completed calls |
| **`Phone number validation failed (400)`** | Number has invalid characters or fewer than 10 digits | Provide standard 10-digit Indian number or international E.164 (e.g. `+919876543210`) |
| **`Exotel credentials not configured (502)`** | `TELEPHONY_PROVIDER=exotel` selected but `.env` lacks SID or keys | Set `TELEPHONY_PROVIDER=mock` for testing or add real Exotel credentials |
| **`WebSocket connection closed (4004)`** | Media stream connected with an invalid or non-existent `call_id` | Ensure call session is initiated via `/api/telephony/outbound` before connecting socket |
| **`Audio choppy or silent`** | Sample rate mismatch or buffer underflow | Verify `TELEPHONY_SAMPLE_RATE=16000` and ensure audio is 16-bit mono Linear PCM |
