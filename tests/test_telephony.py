import os
import sys
import json
import base64
import unittest
import asyncio
from datetime import datetime, timezone
from fastapi.testclient import TestClient

# Ensure root directory on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from app.database.database import SessionLocal
from app.database.models import Call, Customer, Message, HumanAssistanceRequest, SystemUser
from app.utils.phone_utils import normalize_phone_number, mask_phone_number
from app.utils.audio_utils import pcm_to_wav_bytes, calculate_pcm_rms, wav_to_pcm_bytes
from app.services.telephony_adapter import (
    get_telephony_adapter,
    MockTelephonyAdapter,
    ExotelTelephonyAdapter,
    is_valid_status_transition,
)
from app.services.telephony_service import TelephonyService

client = TestClient(app)


def test_telephony_suite():
    print("\n========================================================")
    print("STARTING TELEPHONY LAYER COMPREHENSIVE TEST SUITE")
    print("========================================================")

    # -------------------------------------------------------------------------
    # 1. Phone number validation & formatting
    # -------------------------------------------------------------------------
    print("\n[1/14] Testing Phone Number Validation & Masking...")
    valid, norm, err = normalize_phone_number("9876543210")
    assert valid and norm == "+919876543210", f"10-digit normal failed: {norm}"

    valid, norm, err = normalize_phone_number("+91 98765-43210")
    assert valid and norm == "+919876543210", f"Formatted normal failed: {norm}"

    valid, norm, err = normalize_phone_number("09876543210")
    assert valid and norm == "+919876543210", f"0-prefixed normal failed: {norm}"

    valid, norm, err = normalize_phone_number("123")  # too short
    assert not valid, "Short number unexpectedly validated"

    valid, norm, err = normalize_phone_number("invalid_phone!@#")
    assert not valid, "Alphanumeric string unexpectedly validated"

    masked = mask_phone_number("+919876543210")
    assert masked.startswith("+91") and masked.endswith("3210") and "*" in masked
    print(f"  [PASS] Phone validation and masking verified (+919876543210 -> {masked})")

    # -------------------------------------------------------------------------
    # 2. Audio decoding, format conversion & RMS VAD calculation
    # -------------------------------------------------------------------------
    print("\n[2/14] Testing Audio Conversion & RMS VAD...")
    pcm_silence = b"\x00\x00" * 320  # 20ms at 16kHz
    rms_silence = calculate_pcm_rms(pcm_silence)
    assert rms_silence == 0.0, f"Expected 0.0 RMS for silence, got {rms_silence}"

    # Generate synthetic tone
    import math
    pcm_tone = bytearray()
    for i in range(320):
        val = int(3000 * math.sin(2 * math.pi * 440 * i / 16000))
        pcm_tone.extend(val.to_bytes(2, byteorder="little", signed=True))
    rms_tone = calculate_pcm_rms(bytes(pcm_tone))
    assert rms_tone > 1500, f"Expected tone RMS > 1500, got {rms_tone}"

    wav_bytes = pcm_to_wav_bytes(bytes(pcm_tone), sample_rate=16000)
    assert len(wav_bytes) == 44 + len(pcm_tone), "WAV container header size mismatch"
    assert wav_bytes[:4] == b"RIFF"
    print(f"  [PASS] Audio PCM/WAV conversion and RMS VAD verified (Silence: {rms_silence}, Tone RMS: {rms_tone:.1f})")

    # -------------------------------------------------------------------------
    # 3. Telephony state machine transitions & protection against stale callbacks
    # -------------------------------------------------------------------------
    print("\n[3/14] Testing State Machine Validations & Stale Protection...")
    assert is_valid_status_transition("queued", "initiated")
    assert is_valid_status_transition("initiated", "ringing")
    assert is_valid_status_transition("ringing", "in_progress")
    assert is_valid_status_transition("in_progress", "completed")
    assert is_valid_status_transition("in_progress", "human_transferred")

    # Stale callbacks must NOT resurrect completed/terminated calls
    assert not is_valid_status_transition("completed", "ringing"), "Resurrection completed -> ringing was permitted!"
    assert not is_valid_status_transition("failed", "in_progress"), "Resurrection failed -> in_progress was permitted!"
    assert not is_valid_status_transition("canceled", "queued"), "Resurrection canceled -> queued was permitted!"
    print("  [PASS] State machine rules verified (Stale callbacks blocked from resurrecting calls)")

    # -------------------------------------------------------------------------
    # 4. Authentication: Login to acquire JWT bearer token
    # -------------------------------------------------------------------------
    print("\n[4/14] Authenticating Dashboard User...")
    login_res = client.post("/api/auth/login", json={"username": "admin", "password": "AdminPass123!"})
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("  [PASS] System admin authenticated for telephony actions")

    # -------------------------------------------------------------------------
    # 5. Outbound request authorization & rejection of unauthenticated callers
    # -------------------------------------------------------------------------
    print("\n[5/14] Testing Unauthorized Outbound Rejection...")
    unauth_res = client.post("/api/telephony/outbound", json={"phone_number": "+919876543210"})
    assert unauth_res.status_code == 401, f"Expected 401 Unauthorized, got {unauth_res.status_code}"
    print("  [PASS] Outbound endpoint rejected unauthenticated request with 401")

    # -------------------------------------------------------------------------
    # 6. Invalid phone number rejection
    # -------------------------------------------------------------------------
    print("\n[6/14] Testing Invalid Outbound Phone Rejection...")
    inv_res = client.post("/api/telephony/outbound", json={"phone_number": "123"}, headers=headers)
    assert inv_res.status_code == 400, f"Expected 400 Bad Request, got {inv_res.status_code}"
    print("  [PASS] Outbound rejected invalid phone number '123' with 400 Bad Request")

    # -------------------------------------------------------------------------
    # 7. Valid Outbound Call Initiation (Mock Provider)
    # -------------------------------------------------------------------------
    print("\n[7/14] Testing Valid Outbound Call Initiation...")
    out_res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": "+919876543210", "customer_name": "Telephony Test Caller"},
        headers=headers,
    )
    assert out_res.status_code == 200, f"Outbound initiation failed: {out_res.text}"
    out_data = out_res.json()
    assert out_data["success"] is True
    call_id = out_data["call_id"]
    call_sid = out_data["call_sid"]
    assert call_id and call_sid
    print(f"  [PASS] Outbound call initiated successfully: Call ID #{call_id}, SID '{call_sid}'")

    # 7b. Duplicate active call rejection (idempotency guard)
    print("\n[7b/15] Testing Duplicate In-Progress Call Rejection...")
    dup_res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": "+919876543210"},
        headers=headers,
    )
    assert dup_res.status_code == 409, f"Expected 409 Conflict for duplicate call, got {dup_res.status_code}: {dup_res.text}"
    print("  [PASS] Duplicate outbound call to active number was correctly blocked with 409 Conflict")

    # -------------------------------------------------------------------------
    # 8. Concurrency guard limit test
    # -------------------------------------------------------------------------
    print("\n[8/14] Testing Concurrency Guard Limit...")
    svc = TelephonyService()
    svc.max_concurrent_calls = 1
    svc.active_calls[999] = {"call_id": 999}
    assert svc.is_concurrency_limit_reached() is True
    print("  [PASS] Concurrency guard correctly triggers when active calls reach limit")

    # -------------------------------------------------------------------------
    # 9. Provider Status Callbacks & State Updates
    # -------------------------------------------------------------------------
    print("\n[9/14] Testing Provider Status Callback Webhook...")
    # Ringing callback
    cb_ringing = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "ringing", "CustomField": str(call_id)},
    )
    assert cb_ringing.status_code == 200
    with SessionLocal() as db:
        c = db.query(Call).filter(Call.id == call_id).first()
        assert c.status == "ringing"

    # In-Progress callback
    cb_progress = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "in-progress", "CustomField": str(call_id)},
    )
    assert cb_progress.status_code == 200
    with SessionLocal() as db:
        c = db.query(Call).filter(Call.id == call_id).first()
        assert c.status == "in_progress"

    # Completed callback with duration
    cb_completed = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "completed", "Duration": "45", "CustomField": str(call_id)},
    )
    assert cb_completed.status_code == 200
    with SessionLocal() as db:
        c = db.query(Call).filter(Call.id == call_id).first()
        assert c.status == "completed"
        assert c.duration_seconds == 45
        assert c.ended_at is not None

    # Stale out-of-order callback attempt (trying to move completed -> ringing)
    cb_stale = client.post(
        "/api/telephony/webhook/exotel",
        data={"CallSid": call_sid, "Status": "ringing", "CustomField": str(call_id)},
    )
    assert cb_stale.status_code == 200
    with SessionLocal() as db:
        c = db.query(Call).filter(Call.id == call_id).first()
        assert c.status == "completed", "Stale callback corrupted completed call status!"
    print("  [PASS] Webhook callbacks verified (ringing -> in_progress -> completed; stale callback ignored)")

    # -------------------------------------------------------------------------
    # 10. Webhook authentication validation
    # -------------------------------------------------------------------------
    print("\n[10/14] Testing Webhook Signature / Token Security...")
    exotel_adapter = ExotelTelephonyAdapter(webhook_secret="super_secret_webhook_key")
    # Missing token
    assert not exotel_adapter.validate_webhook({}, b"", {"token": "wrong_secret"})
    # Valid token
    assert exotel_adapter.validate_webhook({}, b"", {"token": "super_secret_webhook_key"})
    print("  [PASS] Webhook security check successfully rejected unauthorized payloads")

    # -------------------------------------------------------------------------
    # 11. Inbound Call Webhook
    # -------------------------------------------------------------------------
    print("\n[11/14] Testing Inbound Telephony Webhook...")
    inbound_res = client.post(
        "/api/telephony/inbound",
        data={"From": "+919811122233", "CallSid": "ex_inbound_test_123", "To": "+918044455566"},
    )
    assert inbound_res.status_code == 200, f"Inbound failed: {inbound_res.text}"
    in_data = inbound_res.json()
    assert in_data["success"] is True
    in_call_id = in_data["call_id"]
    with SessionLocal() as db:
        c_in = db.query(Call).filter(Call.id == in_call_id).first()
        assert c_in.call_type == "INBOUND"
        assert c_in.caller_phone == "+919811122233"
    print(f"  [PASS] Inbound call accepted: Call ID #{in_call_id}, Caller +919811122233")

    # -------------------------------------------------------------------------
    # 12. Authenticated Hangup Endpoint
    # -------------------------------------------------------------------------
    print("\n[12/14] Testing Authenticated Hangup Endpoint...")
    hangup_res = client.post(f"/api/telephony/calls/{in_call_id}/hangup", headers=headers)
    assert hangup_res.status_code == 200, f"Hangup failed: {hangup_res.text}"
    with SessionLocal() as db:
        c_h = db.query(Call).filter(Call.id == in_call_id).first()
        assert c_h.status == "canceled" and c_h.ended_at is not None
    print(f"  [PASS] Hangup endpoint terminated Call #{in_call_id} cleanly")

    # -------------------------------------------------------------------------
    # 13. Telephony Media WebSocket Protocol Simulation
    # -------------------------------------------------------------------------
    print("\n[13/14] Testing Media WebSocket Protocol (Connect, Start, Media, Clear, DTMF, Stop)...")
    # Create fresh call for WebSocket stream test
    ws_call_res = client.post(
        "/api/telephony/outbound",
        json={"phone_number": "+919877788899", "customer_name": "Stream Test Caller"},
        headers=headers,
    )
    assert ws_call_res.status_code == 200
    ws_call_id = ws_call_res.json()["call_id"]

    with client.websocket_connect(f"/api/telephony/stream/{ws_call_id}") as ws:
        # Handshake
        ws.send_text(json.dumps({"event": "connected"}))

        # Start event
        ws.send_text(json.dumps({
            "event": "start",
            "start": {
                "stream_sid": f"str_{ws_call_id}",
                "call_sid": f"call_sid_{ws_call_id}",
            }
        }))

        # Send simulated audio chunk (silence)
        b64_audio = base64.b64encode(b"\x00\x00" * 320).decode("utf-8")
        ws.send_text(json.dumps({
            "event": "media",
            "media": {"payload": b64_audio}
        }))

        # Send DTMF 2 (Hindi escalation)
        ws.send_text(json.dumps({
            "event": "dtmf",
            "dtmf": {"digit": "2"}
        }))

        # Allow async event handler to process
        import time
        time.sleep(0.5)

        # Send stop event
        ws.send_text(json.dumps({"event": "stop"}))

    with SessionLocal() as db:
        c_ws = db.query(Call).filter(Call.id == ws_call_id).first()
        assert c_ws.ended_at is not None
        assert c_ws.language == "Hindi"
        # Check human assistance escalation created
        esc = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.call_id == ws_call_id).first()
        assert esc is not None, "Escalation record was not created on DTMF 2"
        assert "Hindi" in esc.language_selected
    print("  [PASS] WebSocket stream lifecycle & DTMF Hindi escalation verified")

    # -------------------------------------------------------------------------
    # 14. Telephony Status & History Observability
    # -------------------------------------------------------------------------
    print("\n[14/14] Testing Telephony Observability & Status Endpoint...")
    st_res = client.get("/api/telephony/status", headers=headers)
    assert st_res.status_code == 200
    st_data = st_res.json()
    assert "active_provider" in st_data
    assert "max_concurrent_calls" in st_data
    assert "active_call_count" in st_data

    hist_res = client.get("/api/telephony/calls", headers=headers)
    assert hist_res.status_code == 200
    hist_calls = hist_res.json()
    assert len(hist_calls) >= 2
    print(f"  [PASS] Telephony status & call history verified ({len(hist_calls)} telephony calls tracked)")

    # -------------------------------------------------------------------------
    # 15. Provider Credential Self-Check (Zero-Cost Verification)
    # -------------------------------------------------------------------------
    print("\n[15/15] Testing Provider Credentials Self-Check Endpoint...")
    v_res = client.get("/api/telephony/verify-provider", headers=headers)
    assert v_res.status_code == 200
    v_data = v_res.json()
    assert "valid" in v_data
    assert "provider" in v_data
    assert "message" in v_data
    print(f"  [PASS] Provider self-check verified: {v_data['provider']} (valid: {v_data['valid']})")

    print("\n========================================================")
    print("ALL 15 TELEPHONY LAYER TESTS PASSED 100% SUCCESSFULLY!")
    print("========================================================")


if __name__ == "__main__":
    test_telephony_suite()
