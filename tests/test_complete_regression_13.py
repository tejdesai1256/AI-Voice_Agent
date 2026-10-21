import os
import sys
import time
from fastapi.testclient import TestClient

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import Customer, Call, Message, Requirement, HumanAssistanceRequest, SystemUser

client = TestClient(app)


def run_full_13_regression_suite():
    print("\n========================================================")
    print("STARTING 13-POINT MANDATORY REGRESSION TEST SUITE")
    print("========================================================")

    # 1. Login / Logout
    print("\n[1/13] Testing Authentication (Login/Logout)...")
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"})
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200 and me_res.json()["username"] == "admin"
    print("  [PASS] Login & Token auth verified successfully")

    # 2. Customer CRUD
    print("\n[2/13] Testing Customer Management (CRUD)...")
    test_ext_id = f"REG_{int(time.time())}"
    # Create
    c_create = client.post("/api/customers", data={"external_customer_id": test_ext_id, "name": "Regression Test Customer", "phone": "9898989898"}, headers=headers)
    assert c_create.status_code == 200, f"Customer create failed: {c_create.text}"
    new_cust = c_create.json()
    new_cust_id = new_cust["id"]
    # Read
    c_list = client.get(f"/api/customers?q={test_ext_id}", headers=headers)
    assert c_list.status_code == 200 and len(c_list.json()) > 0
    assert c_list.json()[0]["name"] == "Regression Test Customer"
    # Update
    c_update = client.put(
        f"/api/customers/{new_cust_id}",
        data={"external_customer_id": test_ext_id, "name": "Updated Regression Customer", "phone": "9999999999"},
        headers=headers
    )
    assert c_update.status_code == 200 and c_update.json()["name"] == "Updated Regression Customer"
    print("  [PASS] Customer CRUD (Create, Read, Update) verified")

    # 3. English AI Call & 4. Transcript Persistence & 5. Requirement Persistence
    print("\n[3/13, 4/13, 5/13] Testing English AI Call, Transcript Persistence & Requirement Extraction...")
    # Initialize IVR call
    call_res = client.post("/api/calls", data={"external_customer_id": test_ext_id, "language": "prompt"}, headers=headers)
    assert call_res.status_code == 200
    call_id = call_res.json()["call_id"]

    # Customer chooses 1 for English
    dtmf_res = client.post(f"/api/calls/{call_id}/dtmf", data={"digit": "1"}, headers=headers)
    assert dtmf_res.status_code == 200 and dtmf_res.json()["language"] == "en"

    # Use existing test audio for speech turn
    speech_wav = "recordings/timing_test_out.wav"
    if not os.path.exists(speech_wav):
        from app.services.tts_service import TTSService
        tts = TTSService()
        tts.synthesize("I need 50 laptops with 16GB RAM for our department next month. Our budget is 45000 dollars.", speech_wav)

    with open(speech_wav, "rb") as f:
        turn_res = client.post(f"/api/calls/{call_id}/turn", files={"file": ("turn.wav", f, "audio/wav")}, headers=headers)
    assert turn_res.status_code == 200, f"Turn failed: {turn_res.text}"
    turn_data = turn_res.json()
    assert turn_data["response"], "Assistant response missing"
    assert turn_data["audio_url"], "Voice audio missing"
    print("  [PASS] English AI Voice Turn processed successfully")

    # Check MySQL Transcripts
    with SessionLocal() as db:
        msgs = db.query(Message).filter(Message.call_id == call_id).order_by(Message.id).all()
        assert len(msgs) >= 4, f"Expected at least 4 messages in transcript, got {len(msgs)}"
        print(f"  [PASS] Transcript persistence verified: {len(msgs)} chronological messages stored in DB")

        # Check MySQL Requirements
        req = db.query(Requirement).filter(Requirement.call_id == call_id).first()
        assert req is not None, "Requirement record missing from DB"
        assert req.structured_requirements, "Structured requirements empty"
        sr = req.structured_requirements
        print(f"  [PASS] Requirement persistence verified: Budget={sr.get('budget')}, Quantity={sr.get('quantity')}, Products={sr.get('products_or_services')}")
        assert sr.get("quantity") or sr.get("budget") or sr.get("requirements"), "Requirement fields unexpectedly empty"

    # 6. Hindi Language Escalation & Audio
    print("\n[6/13] Testing Hindi Language Escalation & Assistance Audio...")
    hi_call = client.post("/api/calls", data={"external_customer_id": test_ext_id, "language": "prompt"}, headers=headers).json()
    hi_dtmf = client.post(f"/api/calls/{hi_call['call_id']}/dtmf", data={"digit": "2"}, headers=headers).json()
    assert hi_dtmf["is_escalated"] is True
    assert hi_dtmf["language"] == "Hindi"
    assert hi_dtmf["audio_url"] == "/audio/hi_notice.wav"
    hi_audio = client.get(hi_dtmf["audio_url"])
    assert hi_audio.status_code == 200 and len(hi_audio.content) > 100000
    print(f"  [PASS] Hindi escalation verified: Audio={hi_dtmf['audio_url']} ({len(hi_audio.content)} bytes, HTTP 200)")

    # 7. Marathi Language Escalation & Audio
    print("\n[7/13] Testing Marathi Language Escalation & Assistance Audio...")
    mr_call = client.post("/api/calls", data={"external_customer_id": test_ext_id, "language": "prompt"}, headers=headers).json()
    mr_dtmf = client.post(f"/api/calls/{mr_call['call_id']}/dtmf", data={"digit": "3"}, headers=headers).json()
    assert mr_dtmf["is_escalated"] is True
    assert mr_dtmf["language"] == "Marathi"
    assert mr_dtmf["audio_url"] == "/audio/mr_notice.wav"
    mr_audio = client.get(mr_dtmf["audio_url"])
    assert mr_audio.status_code == 200 and len(mr_audio.content) > 100000
    print(f"  [PASS] Marathi escalation verified: Audio={mr_dtmf['audio_url']} ({len(mr_audio.content)} bytes, HTTP 200)")

    # 8. Human Assistance Display & 9. New Callback Option
    print("\n[8/13, 9/13] Testing Human Assistance Display & Callback Action...")
    escs_res = client.get("/api/escalations", headers=headers)
    assert escs_res.status_code == 200
    escs = escs_res.json()
    target_esc = next((e for e in escs if e["call_id"] == hi_call["call_id"]), None)
    assert target_esc is not None, "Hindi escalation record missing from /api/escalations"
    assert target_esc["callback_status"] == "Waiting for Callback"
    print(f"  [PASS] Human Assistance request retrieved: ID={target_esc['id']}, Status='{target_esc['callback_status']}'")

    # Initiate Callback (Requirement 1)
    cb_init = client.post(f"/api/escalations/{target_esc['id']}/callback", json={"notes": "Support operator callback test"}, headers=headers)
    assert cb_init.status_code == 200, f"Callback init failed: {cb_init.text}"
    cb_data = cb_init.json()
    assert cb_data["status"] == "callback_initiated"
    assert cb_data["callback_status"] == "Callback Initiated"
    assert cb_data["telephony_active"] is False, "Distinguished from actual carrier connection"
    assert "pending configuration" in cb_data["message"]
    print(f"  [PASS] Callback initiated successfully: Status='{cb_data['callback_status']}', Telephony Active={cb_data['telephony_active']}")

    # Complete Callback (Requirement 1)
    cb_comp = client.put(f"/api/escalations/{target_esc['id']}/callback-complete", json={"notes": "Resolved customer inquiry"}, headers=headers)
    assert cb_comp.status_code == 200
    assert cb_comp.json()["status"] == "callback_completed"
    print("  [PASS] Callback completed successfully: Status='callback_completed'")

    # 10. Automated Calling Start & 11. Automated Calling Stop
    print("\n[10/13, 11/13] Testing Automated Calling Start / Stop Controls...")
    auto_start = client.post("/api/automation/start", json={"customer_ids": [new_cust_id]}, headers=headers)
    assert auto_start.status_code == 200
    start_status = auto_start.json()["status"]
    assert start_status in ["started", "already_running"]

    status_check = client.get("/api/automation/status", headers=headers)
    assert status_check.status_code == 200 and "automation_status" in status_check.json()
    print(f"  [PASS] Automated calling started: Status='{status_check.json()['automation_status']}'")

    auto_stop = client.post("/api/automation/stop", headers=headers)
    assert auto_stop.status_code == 200
    print("  [PASS] Automated calling stopped successfully")

    # 12. Call History
    print("\n[12/13] Testing Call History & Transcript Detail API...")
    hist_res = client.get("/api/calls", headers=headers)
    assert hist_res.status_code == 200 and len(hist_res.json()) > 0
    detail_res = client.get(f"/api/calls/{call_id}", headers=headers)
    assert detail_res.status_code == 200
    call_detail = detail_res.json()
    assert len(call_detail["transcript"]) > 0
    assert call_detail["requirements"] is not None
    print(f"  [PASS] Call History verified: Call #{call_id} retrieved with {len(call_detail['transcript'])} messages and requirements")

    # 13. Frontend Actions / Index Endpoint
    print("\n[13/13] Testing Frontend Delivery & Static Assets...")
    fe_res = client.get("/")
    assert fe_res.status_code == 200
    assert "Voice Operations Hub" in fe_res.text
    assert "modal-callback-confirm" in fe_res.text
    assert "waiting_1.wav" in fe_res.text
    print("  [PASS] Frontend HTML verified: Contains Call Back workflow, refined design tokens, and waiting audio integrity")

    # Clean up test customer
    client.delete(f"/api/customers/{new_cust_id}", headers=headers)
    print("\n========================================================")
    print("ALL 13 REGRESSION AREAS VERIFIED AND PASSED 100%!")
    print("========================================================\n")


def test_full_regression_suite():
    """Pytest-discoverable wrapper for the 13-point mandatory regression suite."""
    run_full_13_regression_suite()


if __name__ == "__main__":
    run_full_13_regression_suite()
