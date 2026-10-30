import os
import io
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import Customer, Call, Message, ConversationState
from app.services.conversation_flow_service import ConversationFlowService
from app.services.voice_response_service import VoiceResponseService
from scripts.create_admin import create_or_update_admin

client = TestClient(app)


def get_auth_headers():
    create_or_update_admin("phase1_test_admin", "TestPassPhase1!", "admin")
    login_res = client.post(
        "/api/auth/login",
        json={"username": "phase1_test_admin", "password": "TestPassPhase1!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_dummy_wav_file(filename="test.wav"):
    content = b"RIFF1234WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data" + b"\x00" * 320
    return (filename, io.BytesIO(content), "audio/wav")


def test_is_plausible_name_validation():
    """Verify is_plausible_name accurately distinguishes real human names from requirements and system placeholders."""
    flow = ConversationFlowService()
    
    # Valid names
    assert flow.is_plausible_name("Rahul") is True
    assert flow.is_plausible_name("Rahul Sharma") is True
    assert flow.is_plausible_name("Tej Desai") is True
    assert flow.is_plausible_name("Priya K.") is True
    assert flow.is_plausible_name("Jean-Luc Picard") is True
    
    # Non-names (requirement sentences, placeholders, digits, questions)
    assert flow.is_plausible_name("Hello, I need an e-commerce website for my clothing business.") is False
    assert flow.is_plausible_name("I need 50 laptops with 16GB RAM next month. Budget is 45000 dollars.") is False
    assert flow.is_plausible_name("Customer") is False
    assert flow.is_plausible_name("unknown") is False
    assert flow.is_plausible_name("N/A") is False
    assert flow.is_plausible_name("Caller +919876543210") is False
    assert flow.is_plausible_name("123456") is False
    assert flow.is_plausible_name("") is False
    assert flow.is_plausible_name(None) is False
    assert flow.is_plausible_name("not interested") is False


def test_name_extraction_does_not_corrupt_with_requirement_statement():
    """Verify that a customer stating business requirements when asked for name is handled gracefully without corrupting customer name."""
    headers = get_auth_headers()
    ext_id = f"REG_NAME_CORRUPT_TEST_{int(os.getpid())}"
    
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    assert call_res.status_code == 200
    call_id = call_res.json()["call_id"]
    
    req_statement = "Hello, I need an e-commerce website for my clothing business. My budget is 50000 rupees and I need it in two months."
    
    with patch("app.services.stt_service.STTService.transcribe", return_value=req_statement), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()
        
        # Must NOT treat the statement as their name
        assert data["conversation_state"]["current_question_id"] == "q1_name"
        assert "name" not in data["conversation_state"]["answers"]
        # AI acknowledges requirement and asks for name
        assert "name" in data["response"].lower()
        
        # Verify customer in DB does NOT have the requirement sentence as name
        with SessionLocal() as db:
            call = db.query(Call).filter(Call.id == call_id).first()
            assert call.customer.name != req_statement


def test_case_insensitive_terminal_status_rejection():
    """Verify that turns are rejected on calls in all terminal statuses regardless of case (lowercase from telephony or uppercase)."""
    headers = get_auth_headers()
    ext_id = f"REG_TERM_TEST_{int(os.getpid())}"
    
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]
    
    # Simulate a lowercase terminal status set by telephony adapter (e.g. 'completed' with ended_at is None)
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        call.status = "completed"
        call.ended_at = None
        db.commit()
        
    turn_res = client.post(
        f"/api/calls/{call_id}/turn",
        files={"file": create_dummy_wav_file()},
        headers=headers,
    )
    assert turn_res.status_code == 400
    assert "already ended" in turn_res.text.lower()


def test_concurrency_lock_cleanup_on_exception():
    """Verify that if an exception occurs during turn processing, the database lock is reliably released."""
    headers = get_auth_headers()
    ext_id = f"REG_LOCK_CLEANUP_{int(os.getpid())}"
    
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]
    
    # Cause an artificial unexpected exception during conversation flow
    with patch("app.services.stt_service.STTService.transcribe", return_value="Rahul Sharma"), \
         patch("app.services.conversation_flow_service.ConversationFlowService.process_turn", side_effect=RuntimeError("Simulated LLM pipeline failure")):
        failed_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert failed_res.status_code in (400, 500)
        
    # Verify lock is freed (is_processing == 0) and does not hang in processing state
    with SessionLocal() as db:
        state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
        assert state is not None
        assert state.is_processing == 0
        assert state.conversation_status in ("AWAITING_RESPONSE", "INIT")
        
    # Subsequent turn can proceed without 409 Conflict
    with patch("app.services.stt_service.STTService.transcribe", return_value="My name is Rahul"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        recov_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert recov_res.status_code == 200
