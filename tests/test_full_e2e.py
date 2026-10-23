import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.services.tts_service import TTSService
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


def test_full_pipeline():
    print("\n==========================================")
    print("STARTING FULL END-TO-END PIPELINE TEST")
    print("==========================================")

    # 1. Login
    login_res = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "AdminPass123!"}
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[OK] 1. System user authentication successful")

    # 2. Customer Management: Use real customer from database
    cust_res = client.get("/api/customers", headers=headers)
    assert cust_res.status_code == 200 and len(cust_res.json()) > 0
    customer = cust_res.json()[0]
    ext_cust_id = customer["external_customer_id"]
    print(f"[OK] 2. Verified real MySQL customer: {customer['name']} (ID: {customer['id']})")

    # 3. Start Live Call (English)
    call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_cust_id, "language": "en"},
        headers=headers,
    )
    assert call_res.status_code == 200, f"Failed to start call: {call_res.text}"
    call_data = call_res.json()
    call_id = call_data["call_id"]
    print(f"[OK] 3. Call session initialized in MySQL: Call ID #{call_id} (Language: {call_data['language']})")

    # 4. Generate customer speech audio file for testing
    os.makedirs("scratch", exist_ok=True)
    test_audio_path = "scratch/e2e_customer_speech.wav"
    tts = TTSService()
    tts.synthesize("Hello, I need an e-commerce website for my clothing business. My budget is 50000 rupees and I need it in two months.", test_audio_path)
    assert os.path.exists(test_audio_path), "Test customer audio not generated"
    print("[OK] 4. Generated real customer speech input audio via Piper TTS")

    # 5. Process voice turn with full AI pipeline (STT -> Gemma 3 LLM -> Piper TTS -> MySQL)
    with open(test_audio_path, "rb") as audio_file:
        turn_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": ("speech.wav", audio_file, "audio/wav")},
            headers=headers,
        )
    assert turn_res.status_code == 200, f"Turn failed: {turn_res.text}"
    turn_data = turn_res.json()
    assert turn_data.get("transcript"), "Transcript was empty"
    assert turn_data.get("response"), "AI response was empty"
    assert turn_data.get("audio_url"), "TTS audio URL missing"
    print(f"[OK] 5. Voice pipeline completed:")
    print(f"   [Transcript]: {turn_data['transcript']}")
    print(f"   [AI Response]: {turn_data['response']}")
    print(f"   [Audio URL]: {turn_data['audio_url']}")

    # 6. Verify Call Details, Messages & Requirements Persistence in DB
    details_res = client.get(f"/api/calls/{call_id}", headers=headers)
    assert details_res.status_code == 200, f"Details failed: {details_res.text}"
    details = details_res.json()
    assert len(details["transcript"]) >= 2, "Messages not stored in MySQL"
    print(f"[OK] 6. Database persistence verified: {len(details['transcript'])} messages stored in MySQL")
    if details.get("requirements") and details["requirements"].get("structured"):
        print(f"   [Extracted Requirements]: {details['requirements']['structured']}")

    # 7. End the call
    end_res = client.post(f"/api/calls/{call_id}/end", headers=headers)
    assert end_res.status_code == 200
    print(f"[OK] 7. Call #{call_id} safely ended")

    # 8. Test Language Fallback to Human Assistance (Hindi)
    hi_call_res = client.post(
        "/api/calls",
        data={"external_customer_id": ext_cust_id, "language": "hi"},
        headers=headers,
    )
    assert hi_call_res.status_code == 200
    hi_data = hi_call_res.json()
    assert hi_data["is_escalated"] is True, "Hindi selection should trigger escalation"
    print(f"[OK] 8. Phase 7 Language Fallback verified: Call #{hi_data['call_id']} escalated for Hindi routing")

    # 9. Test Human Assistance: Verify escalation in API and resolve
    esc_res = client.get("/api/escalations?status=PENDING", headers=headers)
    assert esc_res.status_code == 200
    escs = esc_res.json()
    assert len(escs) > 0, "No pending escalations found"
    target_esc = next(e for e in escs if e["call_id"] == hi_data["call_id"])
    assert target_esc["language_selected"] in ["hi", "Hindi"]

    # Resolve escalation
    resolve_res = client.put(
        f"/api/escalations/{target_esc['id']}/resolve",
        json={"notes": "Assigned to Hindi-speaking executive Rahul for callback."},
        headers=headers,
    )
    assert resolve_res.status_code == 200
    print(f"[OK] 9. Phase 5 Human Assistance verified: Escalation #{target_esc['id']} marked resolved with notes")

    stop_auto = client.post("/api/automation/stop", headers=headers)
    assert stop_auto.status_code == 200
    print("[OK] 10. Phase 4 Automated Calling safely stopped by user")

    # Clean up test call so test does not leave artifacts in database
    from app.database.database import SessionLocal
    from app.database.models import Call, Message, Requirement, HumanAssistanceRequest
    _db = SessionLocal()
    try:
        _db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == call_id).delete()
        _db.query(Requirement).filter(Requirement.call_id == call_id).delete()
        _db.query(Message).filter(Message.call_id == call_id).delete()
        _db.query(Call).filter(Call.id == call_id).delete()
        _db.commit()
    finally:
        _db.close()

    client.close()

    print("\n==========================================")
    print("ALL 10 END-TO-END PIPELINE TESTS PASSED!")
    print("==========================================")


if __name__ == "__main__":
    test_full_pipeline()
