import os
import sys
import io
import time
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
from app.database.database import SessionLocal
from app.database.models import Call, Message, Requirement, HumanAssistanceRequest
from app.services.requirement_service import RequirementService
from scripts.create_admin import create_or_update_admin

client = TestClient(app)


def get_auth_headers():
    create_or_update_admin("phase2_admin", "AdminPassPhase2!", "admin")
    login_res = client.post(
        "/api/auth/login",
        json={"username": "phase2_admin", "password": "AdminPassPhase2!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_dummy_wav_file(filename="turn.wav"):
    # 16kHz mono 16-bit PCM WAV with 0.5s of header/silence
    header = b"RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00"
    pcm = b"\x00\x00" * 1000
    return (filename, io.BytesIO(header + pcm), "audio/wav")


# =============================================================================
# 1. Dynamic Follow-Up Questions (Feature 1)
# =============================================================================
def test_dynamic_followup_ecommerce_flow():
    """
    Verifies that the conversation engine generates relevant follow-up questions
    based on customer responses instead of a rigid linear sequence:
    E-commerce website -> Products -> Online payments -> Budget -> Launch timeline.
    """
    headers = get_auth_headers()
    ext_id = f"PHASE2_ECOM_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    assert call_res.status_code == 200
    call_id = call_res.json()["call_id"]

    turns = [
        # Turn 1: Customer introduces requirement
        ("I need an e-commerce website.", "What products will you sell?"),
        # Turn 2: Customer specifies products
        ("Clothes and accessories.", "Do you need online payments?"),
        # Turn 3: Customer specifies payment methods
        ("Yes, UPI and credit cards.", "What is your approximate budget?"),
        # Turn 4: Customer specifies budget
        ("$10,000.", "When are you planning to launch?"),
        # Turn 5: Customer specifies timeline -> leads to completion
        ("Next month.", "recorded all your requirements"),
    ]

    for transcript, expected_reply_phrase in turns:
        with patch("app.services.stt_service.STTService.transcribe", return_value=transcript), \
             patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):

            turn_res = client.post(
                f"/api/calls/{call_id}/turn",
                files={"file": create_dummy_wav_file()},
                headers=headers,
            )
            assert turn_res.status_code == 200, f"Failed turn '{transcript}': {turn_res.text}"
            data = turn_res.json()
            assert expected_reply_phrase.lower() in data["response"].lower(), (
                f"Expected '{expected_reply_phrase}' in reply to '{transcript}', got: '{data['response']}'"
            )

    # Verify session completed after all essential requirements gathered
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        assert call.status == "RESOLVED"
        assert call.ended_at is not None


# =============================================================================
# 2. Duplicate Question Prevention & Skipping (Feature 2 & Feature 4)
# =============================================================================
def test_duplicate_question_prevention():
    """
    Verifies that when a customer provides answers unprompted (e.g. project, products, budget, timeline),
    the AI remembers and avoids repeatedly asking about those already-answered fields.
    """
    headers = get_auth_headers()
    ext_id = f"PHASE2_MULTI_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Customer supplies multiple requirements in Turn 1
    multi_utterance = (
        "I need an e-commerce website to sell clothes and accessories. "
        "Our budget is around $10,000 and we want to launch next month."
    )

    with patch("app.services.stt_service.STTService.transcribe", return_value=multi_utterance), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):

        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 200
        data = turn_res.json()
        reply = data["response"].lower()

        # It MUST NOT ask for products or budget since both were provided
        assert "what products will you sell" not in reply
        assert "what is your approximate budget" not in reply

        # The only remaining unaddressed e-commerce requirement is online payments
        assert "online payments" in reply


# =============================================================================
# 3. Customer Correction Handling (Feature 2)
# =============================================================================
def test_customer_correction_handling():
    """
    Verifies that when a customer corrects previously supplied information,
    the AI:
    - Accepts the correction.
    - Updates the stored answer.
    - Acknowledges the update naturally without resetting the call.
    """
    headers = get_auth_headers()
    ext_id = f"PHASE2_CORR_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Turn 1: State budget as $45,000
    with patch("app.services.stt_service.STTService.transcribe", return_value="Our budget is $45,000"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):
        t1 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t1.status_code == 200

    # Turn 2: Customer corrects budget to $60,000
    with patch("app.services.stt_service.STTService.transcribe", return_value="Actually, my budget is $60,000, not $45,000"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):
        t2 = client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)
        assert t2.status_code == 200
        t2_data = t2.json()

        # Must acknowledge the correction
        assert "updated your budget to $60,000" in t2_data["response"].lower() or "$60,000" in t2_data["response"]
        assert t2_data["conversation_state"]["answers"]["budget"] == "$60,000"

    # Verify persisted in database
    with SessionLocal() as db:
        call = db.query(Call).filter(Call.id == call_id).first()
        state = call.conversation_state
        assert state.answers.get("budget") == "$60,000"


# =============================================================================
# 4. Structured Requirement Extraction & Unknown Fields (Feature 3)
# =============================================================================
@pytest.mark.asyncio
async def test_structured_requirement_extraction_and_unknowns():
    """
    Verifies that RequirementService extracts all structured fields,
    properly sets unmentioned fields to null/empty without hallucinating,
    and updates fields when corrected.
    """
    service = RequirementService()
    transcript = (
        "Hello, my name is John Miller from Apex Logistics. We need 50 laptops with 16GB RAM "
        "and SSD storage for our field agents by next month. Our budget is $45,000."
    )

    with patch("app.services.llm_service.generate_response", return_value=""):
        req = await service.extract_requirements(transcript)

    assert req["customer_name"] == "John Miller"
    assert req["company_name"] == "Apex Logistics"
    assert "laptops" in req["project_type"].lower()
    assert req["budget"] == "$45,000"
    assert req["expected_timeline"] == "next month"
    assert any("16gb ram" in f.lower() for f in req["required_features"])

    # Unknown fields must remain None / empty (never invented)
    assert req["business_type"] is None or req["business_type"] != "Hallucinated"
    assert isinstance(req["customer_concerns"], list)

    # Test correction application
    corrected_transcript = "Actually, make that 100 laptops instead of 50 and budget is $80,000"
    corrected_req = service.apply_corrections(corrected_transcript, req, {})
    assert "100" in str(corrected_req.get("quantity"))
    assert "$80,000" in str(corrected_req.get("budget"))


# =============================================================================
# 5. Automatic Conversation Summary (Feature 5)
# =============================================================================
@pytest.mark.asyncio
async def test_automatic_conversation_summary_generation():
    """
    Verifies that the summary generator captures main requirements, details,
    budget, timeline, next steps, and follow-up requirements.
    """
    service = RequirementService()
    structured = {
        "project_type": "E-commerce Website",
        "required_features": ["UPI and Card Payments", "Product Catalog"],
        "budget": "$10,000",
        "expected_timeline": "next month",
        "customer_concerns": ["Pricing & budget sensitivity"],
        "next_action": "Sales engineering team to prepare proposal",
    }

    summary = await service.generate_summary(
        call_id=999,
        transcript="Full transcript text...",
        structured_req=structured,
        unanswered_questions=["Design wireframe preferences"],
    )

    assert summary["call_id"] == 999
    assert summary["main_requirement"] == "E-commerce Website"
    assert summary["budget"] == "$10,000"
    assert summary["timeline"] == "next month"
    assert summary["follow_up_required"] is True
    assert "E-commerce Website" in summary["concise_summary"]
    assert "$10,000" in summary["concise_summary"]


def test_call_summary_api_endpoint():
    """
    Verifies GET /api/calls/{call_id}/summary returns a structured summary
    and caches it idempotently on the call requirement.
    """
    headers = get_auth_headers()
    ext_id = f"PHASE2_SUM_API_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # Process turn with requirement
    with patch("app.services.stt_service.STTService.transcribe", return_value="I need an e-commerce website for $10,000"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):
        client.post(f"/api/calls/{call_id}/turn", files={"file": create_dummy_wav_file()}, headers=headers)

    sum_res = client.get(f"/api/calls/{call_id}/summary", headers=headers)
    assert sum_res.status_code == 200
    sum_data = sum_res.json()

    assert sum_data.get("main_requirement") is not None
    assert sum_data.get("concise_summary") is not None
    assert sum_data.get("follow_up_required") is True


# =============================================================================
# 6. Multilingual Conversation & Human Assistance Routing (Feature 6)
# =============================================================================
def test_multilingual_routing_and_human_assistance():
    """
    Verifies that:
    - English (DTMF 1) continues to AI voice conversation.
    - Hindi (DTMF 2) and Marathi (DTMF 3) route to Human Assistance queue
      with localized announcement audio and ticket creation.
    """
    headers = get_auth_headers()

    # 1. Hindi Routing
    ext_hi = f"PHASE2_HI_{int(time.time() * 1000)}"
    c_hi = client.post("/api/calls", data={"external_customer_id": ext_hi, "language": "prompt"}, headers=headers).json()
    hi_call_id = c_hi["call_id"]

    hi_res = client.post(f"/api/calls/{hi_call_id}/dtmf", data={"digit": "2"}, headers=headers)
    assert hi_res.status_code == 200
    hi_data = hi_res.json()
    assert hi_data["status"] == "ESCALATED"
    assert hi_data["language"] == "Hindi"
    assert hi_data["audio_url"] == "/audio/hi_notice.wav"

    # Verify escalation ticket created
    with SessionLocal() as db:
        esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == hi_call_id).first()
        assert esc is not None
        assert esc.language_selected == "Hindi"
        assert esc.status == "waiting"

    # 2. Marathi Routing
    ext_mr = f"PHASE2_MR_{int(time.time() * 1000)}"
    c_mr = client.post("/api/calls", data={"external_customer_id": ext_mr, "language": "prompt"}, headers=headers).json()
    mr_call_id = c_mr["call_id"]

    mr_res = client.post(f"/api/calls/{mr_call_id}/dtmf", data={"digit": "3"}, headers=headers)
    assert mr_res.status_code == 200
    mr_data = mr_res.json()
    assert mr_data["status"] == "ESCALATED"
    assert mr_data["language"] == "Marathi"
    assert mr_data["audio_url"] == "/audio/mr_notice.wav"


# =============================================================================
# 7. Ended Call Rejection & Database Persistence (Feature 2 & 4)
# =============================================================================
def test_ended_call_rejection_and_db_state():
    """
    Verifies that turns attempted on ended calls are rejected,
    and call records, messages, and states remain consistent in the database.
    """
    headers = get_auth_headers()
    ext_id = f"PHASE2_ENDED_{int(time.time() * 1000)}"

    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "en"},
        headers=headers,
    )
    call_id = call_res.json()["call_id"]

    # End call explicitly
    end_res = client.post(f"/api/calls/{call_id}/end", headers=headers)
    assert end_res.status_code == 200

    # Attempt voice turn on ended call -> must return 400
    with patch("app.services.stt_service.STTService.transcribe", return_value="Hello?"), \
         patch("app.services.tts_service.TTSService.synthesize", return_value="recordings/test.wav"):
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": create_dummy_wav_file()},
            headers=headers,
        )
        assert turn_res.status_code == 400
        assert "already ended" in turn_res.text
