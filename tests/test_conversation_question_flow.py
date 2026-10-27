import os
import io
import sys
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import Customer, Call, Message, ConversationState, HumanAssistanceRequest
from app.config.conversation_questions import (
    get_configured_questions,
    set_configured_questions,
    reset_configured_questions,
)
from app.services.conversation_flow_service import ConversationFlowService
from scripts.create_admin import create_or_update_admin

client = TestClient(app)


def get_auth_headers():
    create_or_update_admin("flow_test_admin", "TestFlowPass123!", "admin")
    login_res = client.post(
        "/api/auth/login",
        json={"username": "flow_test_admin", "password": "TestFlowPass123!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_dummy_wav_file(filename="test.wav", content=b"RIFF1234WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data" + b"\x00" * 320):
    return (filename, io.BytesIO(content), "audio/wav")


@pytest.fixture(autouse=True)
def ensure_default_questions():
    reset_configured_questions()
    with patch("app.services.requirement_service.RequirementService.extract_requirements", return_value={}), \
         patch("app.services.llm_service.generate_response", return_value="I am an AI assistant."):
        yield
    reset_configured_questions()



# -----------------------------------------------------------------------------
# Scenario 1 & 2: Customer answers first question -> AI automatically asks next question
# -----------------------------------------------------------------------------
def test_scenario_1_and_2_first_question_and_auto_next():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_{int(time.time() * 1000)}"

    # Create English call session
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    assert call_res.status_code == 200
    call_data = call_res.json()
    call_id = call_data["call_id"]

    # Initial question should be question 1 (Name)
    assert "May I know your name?" in call_data["response"]
    assert call_data.get("conversation_state") is not None
    assert call_data["conversation_state"]["current_question_id"] == "q1_name"

    # Mock speech input: Customer answers "Rahul Sharma"
    with patch("app.services.stt_service.STTService.transcribe", return_value="My name is Rahul Sharma"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):

        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()

        # 1. Answer is captured and acknowledged
        assert "Rahul Sharma" in data["response"] or "Rahul" in data["response"]
        # 2. AI automatically asks next configured question (Company)
        assert "Which company do you represent?" in data["response"]
        assert data["conversation_state"]["current_question_id"] == "q2_company"
        assert data["conversation_state"]["answers"]["name"] == "Rahul Sharma"


# -----------------------------------------------------------------------------
# Scenario 3 & 4: Consecutive answers & Context preservation throughout call
# -----------------------------------------------------------------------------
def test_scenario_3_and_4_consecutive_answers_and_context_retention():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_MULTI_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Turn 1: Name -> Rahul
    with patch("app.services.stt_service.STTService.transcribe", return_value="Rahul"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        t1 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t1.status_code == 200
        assert "Which company do you represent?" in t1.json()["response"]

    # Turn 2: Company -> ABC Technologies
    with patch("app.services.stt_service.STTService.transcribe", return_value="ABC Technologies"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        t2 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t2.status_code == 200
        data2 = t2.json()
        assert "What industry does your company operate in?" in data2["response"]
        assert data2["conversation_state"]["answers"]["company"] == "ABC Technologies"
        assert data2["conversation_state"]["answers"]["name"] == "Rahul"

    # Turn 3: Industry -> Software and Cloud Services
    with patch("app.services.stt_service.STTService.transcribe", return_value="Software and Cloud Services"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        t3 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t3.status_code == 200
        data3 = t3.json()
        assert "What is your role in the company?" in data3["response"]

        # Verify all previous answers remain retained in conversation context
        answers = data3["conversation_state"]["answers"]
        assert answers["name"] == "Rahul"
        assert answers["company"] == "ABC Technologies"
        assert "Software" in answers["industry"]

    # Verify Database state
    with SessionLocal() as db:
        state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
        assert state is not None
        assert state.answers["name"] == "Rahul"
        assert state.answers["company"] == "ABC Technologies"


# -----------------------------------------------------------------------------
# Scenario 5: Already answered questions are not repeated
# -----------------------------------------------------------------------------
def test_scenario_5_already_answered_not_repeated():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_MULTIINFO_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # In turn 1, customer provides both their name and company together:
    # "I am Rahul from ABC Technologies"
    with patch("app.services.stt_service.STTService.transcribe", return_value="I am Rahul from ABC Technologies"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        t1 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t1.status_code == 200
        data = t1.json()

        # Both name and company should be recorded
        assert "name" in data["conversation_state"]["answers"]
        assert "company" in data["conversation_state"]["answers"]

        # Since company was already answered, AI should SKIP asking "Which company do you represent?"
        # and jump directly to Question 3 (Industry)!
        assert "Which company do you represent?" not in data["response"]
        assert "What industry does your company operate in?" in data["response"]
        assert data["conversation_state"]["current_question_id"] == "q3_industry"


# -----------------------------------------------------------------------------
# Scenario 6: Empty speech recognition does not advance conversation
# -----------------------------------------------------------------------------
def test_scenario_6_empty_speech_does_not_advance():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_EMPTY_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Speech recognition returns empty / inaudible marker
    with patch("app.services.stt_service.STTService.transcribe", return_value="[No clear speech detected]"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()

        # AI asks to repeat
        assert "couldn't hear you clearly" in data["response"].lower() or "repeat" in data["response"].lower()
        # Question stage must NOT advance!
        assert data["conversation_state"]["current_question_id"] == "q1_name"
        assert "name" not in data["conversation_state"]["answers"]


# -----------------------------------------------------------------------------
# Scenario 7: Unclear answer triggers clarification without advancing
# -----------------------------------------------------------------------------
def test_scenario_7_unclear_answer_triggers_clarification():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_UNCLEAR_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Customer says something confused/unclear
    with patch("app.services.stt_service.STTService.transcribe", return_value="I don't know what you mean, what did you say?"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()

        # Clarification prompt returned
        assert "clarify" in data["response"].lower() or "catch that" in data["response"].lower() or "name" in data["response"].lower()
        # Did not advance past question 1
        assert data["conversation_state"]["current_question_id"] == "q1_name"
        assert "name" not in data["conversation_state"]["answers"]


# -----------------------------------------------------------------------------
# Scenario 8: Customer requests Human Assistance
# -----------------------------------------------------------------------------
def test_scenario_8_human_assistance_escalation():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_ESC_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Customer asks for human agent
    with patch("app.services.stt_service.STTService.transcribe", return_value="Please transfer me to a human representative"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()

        assert data["is_escalated"] is True
        assert data["call_status"] == "ESCALATED"
        assert "human representative" in data["response"].lower()

        # Verify escalation record created in DB
        with SessionLocal() as db:
            esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == call_id).first()
            assert esc is not None


# -----------------------------------------------------------------------------
# Scenario 9: Customer says they are not interested
# -----------------------------------------------------------------------------
def test_scenario_9_customer_not_interested():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_NOTINT_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    with patch("app.services.stt_service.STTService.transcribe", return_value="I am not interested, stop calling me"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()

        # Graciously ends conversation
        assert "understand completely" in data["response"].lower()
        assert data["call_status"] == "RESOLVED"

        with SessionLocal() as db:
            call = db.query(Call).filter(Call.id == call_id).first()
            assert call.ended_at is not None


# -----------------------------------------------------------------------------
# Scenario 10: Call ends while AI is processing or call already ended
# -----------------------------------------------------------------------------
def test_scenario_10_call_ended_guard():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_ENDED_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Explicitly end call
    end_res = client.post(f"/api/calls/{call_id}/end", headers=headers)
    assert end_res.status_code == 200

    # Sending another turn on an ended call must be rejected
    turn_res = client.post(
        f"/api/calls/{call_id}/turn",
        files={"file": create_dummy_wav_file()},
        headers=headers,
    )
    assert turn_res.status_code == 400
    assert "already ended" in turn_res.text.lower()


# -----------------------------------------------------------------------------
# Scenario 11: Duplicate concurrent requests do not produce race conditions
# -----------------------------------------------------------------------------
def test_scenario_11_concurrency_lock_prevents_duplicate():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_LOCK_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Simulate active lock in database
    with SessionLocal() as db:
        state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
        state.is_processing = 1
        db.commit()

    # Turn attempt while locked must be rejected with 409 Conflict
    turn_res = client.post(
        f"/api/calls/{call_id}/turn",
        files={"file": create_dummy_wav_file()},
        headers=headers,
    )
    assert turn_res.status_code == 409
    assert "already being processed" in turn_res.text.lower()

    # Releasing lock permits turn again
    with SessionLocal() as db:
        state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
        state.is_processing = 0
        db.commit()

    with patch("app.services.stt_service.STTService.transcribe", return_value="Rahul"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        recov_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert recov_res.status_code == 200


# -----------------------------------------------------------------------------
# Scenario 12: Mock voice conversation complete multi-turn flow
# -----------------------------------------------------------------------------
def test_scenario_12_mock_voice_full_conversation_completion():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_FULL_{int(time.time() * 1000)}"

    # Custom short qualification sequence for end-to-end full completion test
    test_questions = [
        {"id": "q1_name", "key": "name", "question": "May I know your name?", "order": 1},
        {"id": "q2_company", "key": "company", "question": "Which company do you represent?", "order": 2},
        {"id": "q3_ai_interest", "key": "ai_interest", "question": "Are you interested in our AI solutions?", "order": 3},
    ]
    set_configured_questions(test_questions)

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    answers = [
        ("Rahul", "Which company do you represent?"),
        ("ABC Technologies", "Are you interested in our AI solutions?"),
        ("Yes definitely", "Thank you so much"),
    ]

    for ans_text, expected_phrase in answers:
        with patch("app.services.stt_service.STTService.transcribe", return_value=ans_text), \
             patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
            res = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
            assert res.status_code == 200
            data = res.json()
            assert expected_phrase.lower() in data["response"].lower()

    # Final check: call should be resolved and completed
    assert data["call_status"] == "RESOLVED"
    assert data["conversation_state"]["conversation_status"] == "COMPLETED"

    # Verify call details endpoint
    details_res = client.get(f"/api/calls/{call_id}", headers=headers)
    assert details_res.status_code == 200
    details = details_res.json()
    assert details["call"]["status"] == "RESOLVED"
    assert details["conversation_state"]["answers"]["name"] == "Rahul"
    assert details["conversation_state"]["answers"]["company"] == "ABC Technologies"
    assert "Yes" in details["conversation_state"]["answers"]["ai_interest"]


# -----------------------------------------------------------------------------
# Additional Scenario: Customer refuses to answer a question (Respect & Continue)
# -----------------------------------------------------------------------------
def test_scenario_refusal_respect_and_continue():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_REFUSAL_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Customer refuses to give their name
    with patch("app.services.stt_service.STTService.transcribe", return_value="I'd rather not say, please skip this question"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        res = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert res.status_code == 200
        data = res.json()

        # Respects refusal politely and asks next question
        assert "no problem" in data["response"].lower() or "skip" in data["response"].lower()
        assert "Which company do you represent?" in data["response"]
        assert data["conversation_state"]["current_question_id"] == "q2_company"
        assert "[Skipped" in data["conversation_state"]["answers"]["name"]


# -----------------------------------------------------------------------------
# Additional Scenario: Customer asks a question (Answers briefly and continues)
# -----------------------------------------------------------------------------
def test_scenario_customer_asks_question():
    headers = get_auth_headers()
    ext_id = f"FLOW_CUST_ASK_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Customer asks "What kind of AI solutions do you provide?"
    with patch("app.services.stt_service.STTService.transcribe", return_value="What kind of AI solutions do you provide?"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/mock_tts.wav"):
        res = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert res.status_code == 200
        data = res.json()

        # AI answers customer question and returns to current qualification stage
        assert "conversational" in data["response"].lower() or "voice" in data["response"].lower()
        assert "Returning to our conversation" in data["response"]
        assert "May I know your name?" in data["response"]
        # Question stage did not advance because question was not answered yet
        assert data["conversation_state"]["current_question_id"] == "q1_name"


# -----------------------------------------------------------------------------
# Additional Scenario: Configurable Questions API (GET & POST)
# -----------------------------------------------------------------------------
def test_configurable_questions_api():
    headers = get_auth_headers()

    # GET configured questions
    get_res = client.get("/api/conversation-questions", headers=headers)
    assert get_res.status_code == 200
    questions = get_res.json()
    assert isinstance(questions, list)
    assert len(questions) >= 5

    # POST new custom questions sequence
    custom_qs = [
        {"id": "custom_1", "key": "need", "question": "What business need brings you to us today?", "order": 1},
        {"id": "custom_2", "key": "budget", "question": "What is your approximate budget for this project?", "order": 2},
    ]
    post_res = client.post("/api/conversation-questions", json=custom_qs, headers=headers)
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"

    # Verify active questions updated
    get2_res = client.get("/api/conversation-questions", headers=headers)
    assert get2_res.status_code == 200
    active = get2_res.json()
    assert len(active) == 2
    assert active[0]["question"] == "What business need brings you to us today?"

    # Reset
    reset_configured_questions()

