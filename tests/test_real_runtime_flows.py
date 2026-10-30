import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import HumanAssistanceRequest, Call, Message, Requirement
import httpx

try:
    _live = httpx.get("http://127.0.0.1:8001/health", timeout=1.0)
    if _live.status_code == 200:
        client = httpx.Client(base_url="http://127.0.0.1:8001", timeout=120.0)
    else:
        from main import app
        client = TestClient(app)
except Exception:
    from main import app
    client = TestClient(app)

def run_real_runtime_verification():
    print("\n========================================================")
    print("RUNNING REAL RUNTIME FLOW VERIFICATION")
    print("========================================================")

    # 1. Login as system user
    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "AdminPass123!"},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[PASS] Step 1: System user authenticated.")

    # 2. Get real customer
    cust_res = client.get("/api/customers", headers=headers)
    assert cust_res.status_code == 200 and len(cust_res.json()) > 0
    customer = cust_res.json()[0]
    ext_id = customer["external_customer_id"]
    print(f"[PASS] Step 2: Using real customer: {customer['name']} ({ext_id})")

    # =========================================================================
    # TEST 1: IVR Language Prompt -> Option 1 (English) -> AI Conversation -> Requirements Saved
    # =========================================================================
    print("\n--- TEST 1: Option 1 (English AI Pipeline & Persistence) ---")
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "prompt"},
        headers=headers,
    )
    assert call_res.status_code == 200, f"Call initiation failed: {call_res.text}"
    call_data = call_res.json()
    call_id = call_data["call_id"]
    assert call_data["status"] == "AWAITING_LANGUAGE", f"Unexpected status: {call_data['status']}"
    assert call_data["audio_url"] == "/audio/ivr_prompt.wav"
    assert "Press 1 for English" in call_data["response"]

    # Verify IVR prompt audio file is accessible
    ivr_audio_res = client.get(call_data["audio_url"])
    assert ivr_audio_res.status_code == 200
    assert len(ivr_audio_res.content) > 100000

    # Verify IVR prompt is saved in DB messages
    with SessionLocal() as db:
        db_call = db.query(Call).filter(Call.id == call_id).first()
        assert db_call is not None
        assert db_call.status == "AWAITING_LANGUAGE"
        msgs = db.query(Message).filter(Message.call_id == call_id).all()
        assert len(msgs) == 1
        assert msgs[0].speaker == "assistant"
        assert "Press 1 for English" in msgs[0].text
    print(f"[PASS] IVR initialized for Call #{call_id}: Prompt audio verified ({len(ivr_audio_res.content)} bytes) and saved to MySQL messages table.")

    # Customer enters DTMF 1 for English
    dtmf_res = client.post(
        f"/api/calls/{call_id}/dtmf",
        data={"digit": "1"},
        headers=headers,
    )
    assert dtmf_res.status_code == 200, f"DTMF 1 failed: {dtmf_res.text}"
    dtmf_data = dtmf_res.json()
    assert dtmf_data["status"] == "IN_PROGRESS"
    assert dtmf_data["language"] == "en"
    assert dtmf_data["is_escalated"] is False
    assert dtmf_data["audio_url"] == "/audio/en_greeting.wav"
    assert "I am your AI voice assistant" in dtmf_data["response"]

    with SessionLocal() as db:
        msgs = db.query(Message).filter(Message.call_id == call_id).order_by(Message.id).all()
        assert len(msgs) == 3  # [IVR prompt, DTMF customer choice, assistant greeting]
        assert msgs[1].speaker == "customer" and "[DTMF Keypad: 1 - English]" in msgs[1].text
        assert msgs[2].speaker == "assistant" and "I am your AI voice assistant" in msgs[2].text
    print(f"[PASS] DTMF 1 routed call #{call_id} to English: Greeting audio verified and saved to MySQL messages table.")

    # Process voice turn with customer requirement
    test_audio_path = "recordings/timing_test_out.wav"
    if not os.path.exists(test_audio_path):
        # synthesize audio
        from app.services.tts_service import TTSService
        tts = TTSService()
        tts.synthesize("I need 50 laptops with 16GB RAM for our department next month. Our budget is 45000 dollars.", test_audio_path)

    with open(test_audio_path, "rb") as f:
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": ("speech.wav", f, "audio/wav")},
            headers=headers,
        )
    assert turn_res.status_code == 200, f"Voice turn failed: {turn_res.text}"
    turn_data = turn_res.json()
    print(f"[PASS] Voice turn processed. Transcribed customer audio: '{turn_data['transcript']}'")
    print(f"[PASS] AI Response generated: '{turn_data['response']}'")

    # End call
    end_res = client.post(f"/api/calls/{call_id}/end", headers=headers)
    assert end_res.status_code == 200

    # Check Call Details API
    details_res = client.get(f"/api/calls/{call_id}", headers=headers)
    assert details_res.status_code == 200
    details = details_res.json()

    # Verify transcript is non-blank
    assert len(details["transcript"]) >= 4, f"Expected at least 4 messages in transcript, got {len(details['transcript'])}"
    print(f"[PASS] Transcript verified: {len(details['transcript'])} chronological messages saved and retrieved.")

    # Verify requirements are collected and non-blank
    assert details["requirements"] is not None, "Requirements object is missing!"
    req_structured = details["requirements"]["structured"]
    assert req_structured is not None, "Structured requirements are None!"
    print(f"[PASS] Structured Requirements Extracted & Persisted in DB:")
    for k, v in req_structured.items():
        if k != "raw_response":
            print(f"       - {k}: {v}")
    assert any(req_structured.get(k) is not None for k in ["requirements", "products_or_services", "quantity", "budget"]), "All requirement fields are empty!"

    # =========================================================================
    # TEST 2: IVR Language Prompt -> Option 2 (Hindi) -> Human Assistance
    # =========================================================================
    print("\n--- TEST 2: Option 2 (Hindi Human Assistance Routing) ---")
    hi_call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "prompt"},
        headers=headers,
    )
    assert hi_call_res.status_code == 200
    hi_call_id = hi_call_res.json()["call_id"]

    # Customer presses 2 for Hindi
    hi_dtmf_res = client.post(
        f"/api/calls/{hi_call_id}/dtmf",
        data={"digit": "2"},
        headers=headers,
    )
    assert hi_dtmf_res.status_code == 200
    hi_dtmf = hi_dtmf_res.json()
    assert hi_dtmf["status"] == "ESCALATED"
    assert hi_dtmf["is_escalated"] is True
    assert hi_dtmf["language"] == "Hindi"
    assert "हमारी टीम का एक सदस्य जल्द ही आपसे संपर्क करेगा" in hi_dtmf["response"]
    assert hi_dtmf["audio_url"] == "/audio/hi_notice.wav"

    # Verify audio HTTP 200
    hi_audio_check = client.get(hi_dtmf["audio_url"])
    assert hi_audio_check.status_code == 200
    assert len(hi_audio_check.content) > 100000

    # Verify MySQL database records directly
    with SessionLocal() as db:
        hi_db_call = db.query(Call).filter(Call.id == hi_call_id).first()
        assert hi_db_call.status == "ESCALATED"
        assert hi_db_call.ended_at is not None
        assert hi_db_call.language == "Hindi"

        hi_msgs = db.query(Message).filter(Message.call_id == hi_call_id).order_by(Message.id).all()
        assert len(hi_msgs) == 3
        assert hi_msgs[1].speaker == "customer" and "Hindi" in hi_msgs[1].text
        assert hi_msgs[2].speaker == "assistant" and "हमारी टीम का एक सदस्य जल्द ही आपसे संपर्क करेगा" in hi_msgs[2].text

        hi_esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == hi_call_id).first()
        assert hi_esc is not None
        assert hi_esc.language_selected == "Hindi"
        assert hi_esc.status == "waiting"
        assert "Customer selected Hindi" in hi_esc.escalation_reason

    # Verify Frontend Escalations API
    escs_res = client.get("/api/escalations", headers=headers)
    assert escs_res.status_code == 200
    escs_list = escs_res.json()
    hi_found = any(e["call_id"] == hi_call_id and e["language"] == "Hindi" and e["status"] == "waiting" for e in escs_list)
    assert hi_found, "Hindi Human Assistance request not found in /api/escalations"
    print(f"[PASS] Hindi (2) routing verified: Call #{hi_call_id} escalated, audio verified ({len(hi_audio_check.content)} bytes), Human Assistance request created with status='waiting'.")

    # =========================================================================
    # TEST 3: IVR Language Prompt -> Option 3 (Marathi) -> Human Assistance
    # =========================================================================
    print("\n--- TEST 3: Option 3 (Marathi Human Assistance Routing) ---")
    mr_call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_id, "language": "prompt"},
        headers=headers,
    )
    assert mr_call_res.status_code == 200
    mr_call_id = mr_call_res.json()["call_id"]

    # Customer presses 3 for Marathi
    mr_dtmf_res = client.post(
        f"/api/calls/{mr_call_id}/dtmf",
        data={"digit": "3"},
        headers=headers,
    )
    assert mr_dtmf_res.status_code == 200
    mr_dtmf = mr_dtmf_res.json()
    assert mr_dtmf["status"] == "ESCALATED"
    assert mr_dtmf["is_escalated"] is True
    assert mr_dtmf["language"] == "Marathi"
    assert "आमच्या टीममधील एक सदस्य लवकरच आपल्याशी संपर्क साधेल" in mr_dtmf["response"]
    assert mr_dtmf["audio_url"] == "/audio/mr_notice.wav"

    # Verify audio HTTP 200
    mr_audio_check = client.get(mr_dtmf["audio_url"])
    assert mr_audio_check.status_code == 200
    assert len(mr_audio_check.content) > 100000

    # Verify MySQL database records directly
    with SessionLocal() as db:
        mr_db_call = db.query(Call).filter(Call.id == mr_call_id).first()
        assert mr_db_call.status == "ESCALATED"
        assert mr_db_call.ended_at is not None
        assert mr_db_call.language == "Marathi"

        mr_msgs = db.query(Message).filter(Message.call_id == mr_call_id).order_by(Message.id).all()
        assert len(mr_msgs) == 3
        assert mr_msgs[1].speaker == "customer" and "Marathi" in mr_msgs[1].text
        assert mr_msgs[2].speaker == "assistant" and "आमच्या टीममधील एक सदस्य लवकरच आपल्याशी संपर्क साधेल" in mr_msgs[2].text

        mr_esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == mr_call_id).first()
        assert mr_esc is not None
        assert mr_esc.language_selected == "Marathi"
        assert mr_esc.status == "waiting"
        assert "Customer selected Marathi" in mr_esc.escalation_reason

    # Verify Frontend Escalations API
    escs_res = client.get("/api/escalations", headers=headers)
    assert escs_res.status_code == 200
    escs_list = escs_res.json()
    mr_found = any(e["call_id"] == mr_call_id and e["language"] == "Marathi" and e["status"] == "waiting" for e in escs_list)
    assert mr_found, "Marathi Human Assistance request not found in /api/escalations"
    print(f"[PASS] Marathi (3) routing verified: Call #{mr_call_id} escalated, audio verified ({len(mr_audio_check.content)} bytes), Human Assistance request created with status='waiting'.")

    print("\n========================================================")
    print("ALL REAL RUNTIME VERIFICATIONS PASSED SUCCESSFULLY!")
    print("========================================================\n")

def test_real_runtime_verification():
    run_real_runtime_verification()


if __name__ == "__main__":
    run_real_runtime_verification()
