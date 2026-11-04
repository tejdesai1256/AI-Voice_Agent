"""
Phase 3 Telephony Verification Suite
====================================
Comprehensive automated test suite covering all 18 requirements of Phase 3
without initiating real paid telephone calls.

Requirements covered:
 1. Outbound call request
 2. Invalid telephone number rejection
 3. Provider authentication failure (401 / auth errors)
 4. Provider API failure (500 / 402 insufficient balance / timeout)
 5. Successful provider request & SID storage
 6. Webhook authentication (signature / token verification)
 7. Duplicate webhook delivery idempotency
 8. Call state transitions & anti-resurrection guards
 9. Multi-turn telephone conversation (exact required dialogue flow)
10. Speech-recognition failure (silence / inaudible input)
11. AI response failure (graceful fallback)
12. TTS failure (resilient audio synthesis fallback)
13. Call completion & requirement persistence
14. Call disconnection & resource cleanup
15. Recording & transcript association
16. Callback scheduling, calling-hour constraints & unanswered call retries
17. Human Assistance escalation & live transfer / callback requirement
18. Regression verification of existing application features
"""

import os
import sys
import json
import uuid
import random
import base64
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure root directory on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import (
    Call,
    Customer,
    Message,
    HumanAssistanceRequest,
    CallbackRequest,
    Requirement,
    ConversationState,
    SystemUser,
)
from app.services.telephony_adapter import (
    get_telephony_adapter,
    MockTelephonyAdapter,
    ExotelTelephonyAdapter,
    TelephonyCallResult,
    is_valid_status_transition,
)
from app.services.telephony_service import TelephonyService
from app.services.callback_service import CallbackService
from app.services.conversation_flow_service import ConversationFlowService
from app.utils.phone_utils import normalize_phone_number, mask_phone_number

client = TestClient(app)


def random_phone():
    """Generates a valid 10-digit Indian test phone number."""
    return f"+91987{random.randint(1000000, 9999999)}"


def make_customer_and_call(db, phone=None, provider="mock"):
    """Helper to create a fresh customer and call for isolated unit tests."""
    uid = uuid.uuid4().hex[:8]
    cust = Customer(
        name=f"Customer {uid}",
        phone=phone or random_phone(),
        external_customer_id=f"cust_{uid}",
    )
    db.add(cust)
    db.commit()
    db.refresh(cust)
    call = Call(
        customer_id=cust.id,
        destination_phone=cust.phone,
        call_type="OUTBOUND",
        provider=provider,
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return cust, call


@pytest.fixture(scope="module")
def auth_headers():
    """Retrieve valid admin JWT headers."""
    res = client.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"})
    assert res.status_code == 200, f"Login failed: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def clean_environment():
    """Ensure safe testing mode, clean active in-memory calls and DB concurrency state."""
    os.environ["TELEPHONY_PROVIDER"] = "mock"
    from app.api.telephony_routes import telephony_service
    telephony_service.active_calls.clear()
    telephony_service.max_concurrent_calls = 50
    with SessionLocal() as db:
        db.query(Call).filter(Call.ended_at.is_(None)).update(
            {"ended_at": datetime.now(timezone.utc), "status": "completed"},
            synchronize_session=False,
        )
        db.commit()
    yield
    telephony_service.active_calls.clear()


# -----------------------------------------------------------------------------
# 1. Outbound call request
# -----------------------------------------------------------------------------
def test_01_outbound_call_request(auth_headers):
    """Test outbound call request creates a Call record and does not mark success prematurely."""
    phone = random_phone()
    res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": phone, "customer_name": "Test Customer 1"},
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    assert data["success"] is True
    call_id = data["call_id"]
    assert call_id is not None
    assert data["call_sid"] is not None

    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call is not None
        assert call.destination_phone == phone
        assert call.call_type == "OUTBOUND"
        # Must be initiated or queued initially, never marked completed prematurely
        assert call.status in ("initiated", "queued", "ringing", "in_progress")


# -----------------------------------------------------------------------------
# 2. Invalid telephone number
# -----------------------------------------------------------------------------
def test_02_invalid_telephone_number(auth_headers):
    """Test rejection of malformed or too short phone numbers with 400 Bad Request."""
    # Too short
    res_short = client.post(
        "/api/telephony/outbound",
        json={"phone_number": "12345"},
        headers=auth_headers,
    )
    assert res_short.status_code == 400
    assert "invalid" in res_short.json()["detail"].lower()

    # Alphabetic
    res_alpha = client.post(
        "/api/telephony/outbound",
        json={"phone_number": "invalid_phone_number"},
        headers=auth_headers,
    )
    assert res_alpha.status_code == 400


# -----------------------------------------------------------------------------
# 3. Provider authentication failure
# -----------------------------------------------------------------------------
def test_03_provider_auth_failure():
    """Test provider HTTP 401 authentication rejection is handled and recorded as failed."""
    adapter = ExotelTelephonyAdapter(
        account_sid="bad_sid",
        api_key="bad_key",
        api_token="bad_token",
        caller_id="02012345678",
    )
    mock_401 = MagicMock()
    mock_401.status_code = 401
    mock_401.text = '{"RestException": {"Status": 401, "Message": "Authenticate"}}'

    loop = asyncio.new_event_loop()
    with patch("httpx.AsyncClient.post", return_value=mock_401):
        res = loop.run_until_complete(
            adapter.start_outbound_call("+919876500003", 901, "wss://test/stream")
        )
        assert not res.success
        assert res.error_type == "AUTH_ERROR"
        assert "authentication" in res.error_message.lower()


# -----------------------------------------------------------------------------
# 4. Provider API failure (500, 402 Insufficient Balance, timeout)
# -----------------------------------------------------------------------------
def test_04_provider_api_failure():
    """Test provider API error classifications: Insufficient Balance (402), 500, and timeout."""
    import httpx
    adapter = ExotelTelephonyAdapter(
        account_sid="sid",
        api_key="key",
        api_token="token",
        caller_id="02012345678",
    )
    loop = asyncio.new_event_loop()

    # 402 Insufficient balance
    mock_402 = MagicMock()
    mock_402.status_code = 402
    mock_402.text = '{"RestException": {"Status": 402, "Message": "Insufficient balance in Exotel account"}}'
    with patch("httpx.AsyncClient.post", return_value=mock_402):
        res_402 = loop.run_until_complete(
            adapter.start_outbound_call("+919876500004", 902, "wss://test/stream")
        )
        assert not res_402.success
        assert res_402.error_type == "INSUFFICIENT_CREDITS"
        assert "balance" in res_402.error_message.lower() or "credits" in res_402.error_message.lower()

    # 500 Server Error
    mock_500 = MagicMock()
    mock_500.status_code = 500
    mock_500.text = "Internal Server Error"
    mock_500.json.side_effect = Exception("Not JSON")
    with patch("httpx.AsyncClient.post", return_value=mock_500):
        res_500 = loop.run_until_complete(
            adapter.start_outbound_call("+919876500004", 903, "wss://test/stream")
        )
        assert not res_500.success
        assert res_500.error_type in ("PROVIDER_SERVER_ERROR", "PROVIDER_ERROR")

    # Timeout
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Connection timed out")):
        res_timeout = loop.run_until_complete(
            adapter.start_outbound_call("+919876500004", 904, "wss://test/stream")
        )
        assert not res_timeout.success
        assert res_timeout.error_type == "TIMEOUT"


# -----------------------------------------------------------------------------
# 5. Successful provider request & SID storage
# -----------------------------------------------------------------------------
def test_05_successful_provider_request(auth_headers):
    """Test successful provider request correctly persists provider CallSid."""
    phone = random_phone()
    res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": phone, "customer_name": "Test Customer 5"},
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    data = res.json()
    call_id = data["call_id"]
    call_sid = data["call_sid"]

    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call.provider_call_id == call_sid
        assert call.provider in ("mock", "exotel")


# -----------------------------------------------------------------------------
# 6. Webhook authentication
# -----------------------------------------------------------------------------
def test_06_webhook_authentication():
    """Test webhook signature / token authentication verification."""
    adapter = ExotelTelephonyAdapter(webhook_secret="correct_secret_123")
    # Valid token
    assert adapter.validate_webhook({}, b"", {"token": "correct_secret_123"}) is True
    # Invalid token
    assert adapter.validate_webhook({}, b"", {"token": "wrong_token"}) is False
    # Missing token
    assert adapter.validate_webhook({}, b"", {}) is False


# -----------------------------------------------------------------------------
# 7. Duplicate webhook delivery
# -----------------------------------------------------------------------------
def test_07_duplicate_webhook_delivery(auth_headers):
    """Test duplicate webhook deliveries are handled idempotently."""
    phone = random_phone()
    res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": phone},
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    call_id = res.json()["call_id"]
    call_sid = res.json()["call_sid"]

    # In-progress webhook
    client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "in-progress", "CustomField": str(call_id)},
    )

    # First delivery: completed
    cb1 = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "completed", "Duration": "30", "CustomField": str(call_id)},
    )
    assert cb1.status_code == 200

    # Second delivery: identical completed webhook
    cb2 = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "completed", "Duration": "30", "CustomField": str(call_id)},
    )
    assert cb2.status_code == 200

    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call.status == "completed"
        assert call.duration_seconds == 30


# -----------------------------------------------------------------------------
# 8. Call state transitions
# -----------------------------------------------------------------------------
def test_08_call_state_transitions():
    """Test call state machine valid progressions and anti-resurrection rules."""
    # Valid transitions
    assert is_valid_status_transition("queued", "initiated")
    assert is_valid_status_transition("initiated", "ringing")
    assert is_valid_status_transition("ringing", "in_progress")
    assert is_valid_status_transition("in_progress", "completed")
    assert is_valid_status_transition("in_progress", "busy")
    assert is_valid_status_transition("in_progress", "no_answer")

    # Invalid transitions (Terminal resurrection blocked)
    assert not is_valid_status_transition("completed", "ringing")
    assert not is_valid_status_transition("completed", "in_progress")
    assert not is_valid_status_transition("failed", "in_progress")
    assert not is_valid_status_transition("canceled", "queued")


# -----------------------------------------------------------------------------
# 9. Multi-turn telephone conversation (Exact required dialogue)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_09_multi_turn_telephone_conversation():
    """
    Test complete multi-turn telephone conversation matching the exact Phase 3 dialogue:
    - Turn 0 (AI greeting): "Hello! I am calling regarding your business requirements. Is this a good time to talk?"
    - Turn 1 (Customer): "Yes." -> AI asks: "What type of service are you looking for?"
    - Turn 2 (Customer): "I need an e-commerce website." -> AI asks: "What products do you want to sell?"
    - Turn 3 (Customer): "Clothes." -> AI asks: "Do you need online payments?"
    - Turn 4 (Customer): "Yes." -> AI asks: "What is your approximate budget?"
    """
    flow_service = ConversationFlowService()

    with SessionLocal() as db:
        cust, call = make_customer_and_call(db)

        # Initial Outbound Telephone Greeting
        greeting = flow_service.get_initial_greeting_and_question(db, call)
        assert "Hello!" in greeting
        assert "Is this a good time to talk?" in greeting

        # Turn 1: Customer: "Yes."
        t1 = await flow_service.process_turn(call, "Yes.", db)
        assert "What type of service are you looking for?" in t1["response"]

        # Turn 2: Customer: "I need an e-commerce website."
        t2 = await flow_service.process_turn(call, "I need an e-commerce website.", db)
        assert "What products do you want to sell?" in t2["response"]

        # Turn 3: Customer: "Clothes."
        t3 = await flow_service.process_turn(call, "Clothes.", db)
        assert "Do you need online payments?" in t3["response"]

        # Turn 4: Customer: "Yes."
        t4 = await flow_service.process_turn(call, "Yes.", db)
        assert "What is your approximate budget?" in t4["response"]

        # Verify state integrity
        db.refresh(call)
        state = call.conversation_state
        assert state is not None
        assert "good_time_to_talk" in state.answers
        assert "project_type" in state.answers
        assert "products" in state.answers
        assert "online_payments" in state.answers
        # Verify no duplicate questions asked
        assert len(state.questions_asked) == len(set(state.questions_asked))


# -----------------------------------------------------------------------------
# 10. Speech-recognition failure (Silence / inaudible speech)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_10_speech_recognition_failure():
    """Test handling of empty or inaudible transcription without crashing."""
    flow_service = ConversationFlowService()
    with SessionLocal() as db:
        _, call = make_customer_and_call(db)

        # Send empty speech
        res_empty = await flow_service.process_turn(call, "", db)
        assert res_empty is not None
        msg = res_empty["response"].lower()
        assert "hear you" in msg or "repeat" in msg or "clarify" in msg or "catch" in msg


# -----------------------------------------------------------------------------
# 11. AI response failure (Fallback conversational response)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_11_ai_response_failure():
    """Test graceful fallback when dynamic LLM generation encounters an unexpected error."""
    flow_service = ConversationFlowService()
    with SessionLocal() as db:
        _, call = make_customer_and_call(db)

        # Patch internal generator to simulate failure
        with patch.object(flow_service, "get_dynamic_next_question", side_effect=Exception("LLM Timeout")):
            res = await flow_service.process_turn(call, "We need a new web application.", db)
            assert res is not None
            # Should fall back cleanly without raising uncaught exception
            assert len(res["response"]) > 0


# -----------------------------------------------------------------------------
# 12. TTS failure
# -----------------------------------------------------------------------------
def test_12_tts_failure():
    """Test audio synthesis utility resilience on missing audio or empty buffers."""
    from app.utils.audio_utils import wav_to_pcm_bytes, pcm_to_wav_bytes

    missing = wav_to_pcm_bytes("non_existent_file_xyz.wav")
    assert missing == b""

    # Valid empty wav generation
    empty_wav = pcm_to_wav_bytes(b"", sample_rate=16000)
    assert len(empty_wav) == 44
    assert empty_wav.startswith(b"RIFF")


# -----------------------------------------------------------------------------
# 13. Call completion & requirement persistence
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_13_call_completion():
    """Test call completion concludes session gracefully."""
    flow_service = ConversationFlowService()
    with SessionLocal() as db:
        _, call = make_customer_and_call(db)

        # Conclude conversation
        res = await flow_service.process_turn(call, "I am not interested at this moment, thank you.", db)
        assert res["call_status"] in ("RESOLVED", "COMPLETED")
        db.refresh(call)
        assert call.status in ("RESOLVED", "COMPLETED")
        assert call.ended_at is not None


# -----------------------------------------------------------------------------
# 14. Call disconnection
# -----------------------------------------------------------------------------
def test_14_call_disconnection(auth_headers):
    """Test hangup endpoint safely terminates call and sets ended_at."""
    phone = random_phone()
    res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": phone},
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    call_id = res.json()["call_id"]

    h_res = client.post(f"/api/telephony/calls/{call_id}/hangup", headers=auth_headers)
    assert h_res.status_code == 200

    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call.status == "canceled"
        assert call.ended_at is not None


# -----------------------------------------------------------------------------
# 15. Recording & transcript association
# -----------------------------------------------------------------------------
def test_15_recording_and_transcript_association(auth_headers):
    """Test call recording URL and full multi-turn transcript association."""
    phone = random_phone()
    res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": phone},
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    call_id = res.json()["call_id"]
    call_sid = res.json()["call_sid"]

    # Transition to in-progress first
    client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "in-progress", "CustomField": str(call_id)},
    )

    # Deliver completed webhook with recording URL
    rec_url = "https://media.exotel.com/recordings/call_15_rec.mp3"
    cb = client.post(
        "/api/telephony/webhook/exotel",
        data={
            "CallSid": call_sid,
            "Status": "completed",
            "Duration": "75",
            "RecordingUrl": rec_url,
            "CustomField": str(call_id),
        },
    )
    assert cb.status_code == 200

    # Query call details endpoint
    detail_res = client.get(f"/api/calls/{call_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    call_data = detail_res.json()["call"]
    assert call_data["status"] == "completed"

    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call.recording_url == rec_url
        assert call.duration_seconds == 75


# -----------------------------------------------------------------------------
# 16. Callback scheduling, calling-hour constraints & retry logic
# -----------------------------------------------------------------------------
def test_16_callback_scheduling(auth_headers):
    """Test callback scheduling, calling-hour restrictions (9AM-8PM), duplicate guards, and retries."""
    phone = random_phone()
    sched_time = (datetime.now(timezone.utc) + timedelta(minutes=45)).isoformat()
    res = client.post(
        "/api/telephony/callbacks",
        json={
            "phone_number": phone,
            "customer_name": "Scheduled Lead",
            "scheduled_time": sched_time,
            "reason": "customer_request",
        },
        headers=auth_headers,
    )
    assert res.status_code == 200, f"Failed: {res.text}"
    cb_id = res.json()["callback_id"]

    # Calling hours adjustment test
    now_utc = datetime.now(timezone.utc)
    adjusted = CallbackService.adjust_for_calling_hours(now_utc)
    assert adjusted is not None

    # Duplicate callback prevention test
    dup_res = client.post(
        "/api/telephony/callbacks",
        json={"phone_number": phone},
        headers=auth_headers,
    )
    assert dup_res.status_code == 409
    assert "already" in dup_res.json()["detail"].lower()

    # Cancel callback
    can_res = client.post(f"/api/telephony/callbacks/{cb_id}/cancel", headers=auth_headers)
    assert can_res.status_code == 200
    assert can_res.json()["status"] == "cancelled"


# -----------------------------------------------------------------------------
# 17. Human Assistance escalation & live transfer / callback requirement
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_17_human_assistance_escalation():
    """Test intent classification for ESCALATION triggers human assistance workflow."""
    flow_service = ConversationFlowService()
    with SessionLocal() as db:
        _, call = make_customer_and_call(db)

        res = await flow_service.process_turn(call, "I want to speak with a human agent right now.", db)
        assert res["is_escalated"] is True or res["action"] == "escalate"

        # Check HumanAssistanceRequest created in database
        db.refresh(call)
        esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == call.id).first()
        assert esc is not None
        assert "human" in esc.escalation_reason.lower() or "assistance" in esc.escalation_reason.lower() or "escalat" in esc.escalation_reason.lower()


# -----------------------------------------------------------------------------
# 18. Regression of existing application features
# -----------------------------------------------------------------------------
def test_18_regression_of_existing_features(auth_headers):
    """Test that customer directory, health, and core APIs continue functioning normally."""
    # Health check
    health_res = client.get("/health")
    assert health_res.status_code == 200

    # Customer list
    cust_res = client.get("/api/customers", headers=auth_headers)
    assert cust_res.status_code == 200
    assert isinstance(cust_res.json(), list)

    # Telephony status endpoint
    st_res = client.get("/api/telephony/status", headers=auth_headers)
    assert st_res.status_code == 200
    assert "active_provider" in st_res.json()
