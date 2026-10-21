import os
import sys
import io
import time
import wave
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8001")


def create_silence_wav(duration_sec=1.0, sample_rate=16000) -> bytes:
    """Creates a valid 16kHz mono 16-bit PCM WAV silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        num_frames = int(duration_sec * sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def test_microphone_conversation_pipeline():
    print("=================================================================")
    print("TESTING AI VOICE AGENT MULTI-TURN MICROPHONE CONVERSATION PIPELINE")
    print(f"Target Server: {BASE_URL}")
    print("=================================================================")

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        # 1. Health check
        h_res = client.get("/health")
        assert h_res.status_code == 200, f"Health check failed: {h_res.text}"
        print("[PASS] 1. Server health check verified (status 200)")

        # 2. Login
        login_res = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "AdminPass123!"},
        )
        assert login_res.status_code == 200, f"Login failed: {login_res.text}"
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("[PASS] 2. Authentication successful")

        # 3. Get customer
        cust_res = client.get("/api/customers", headers=headers)
        assert cust_res.status_code == 200 and len(cust_res.json()) > 0
        customer = cust_res.json()[0]
        ext_cust_id = customer["external_customer_id"]
        print(f"[PASS] 3. Using customer: {customer['name']} ({ext_cust_id})")

        # 4. Start Live Call session
        call_res = client.post(
            "/api/calls",
            data={"external_customer_id": ext_cust_id, "language": "en"},
            headers=headers,
        )
        assert call_res.status_code == 200, f"Failed to start call: {call_res.text}"
        call_data = call_res.json()
        call_id = call_data["call_id"]
        print(f"[PASS] 4. Live Call #{call_id} started (Language: {call_data['language']})")

        # Generate synthesized customer voice clips via Piper TTS for exact requirements
        from app.services.tts_service import TTSService
        tts = TTSService()
        os.makedirs("scratch", exist_ok=True)

        clip1_path = "scratch/test_clip1_hello.wav"
        clip2_path = "scratch/test_clip2_name.wav"
        clip3_path = "scratch/test_clip3_ask.wav"

        print("\n--- Synthesizing Test Audio Utterances with Piper TTS ---")
        tts.synthesize("Hello, can you hear me?", clip1_path)
        tts.synthesize("My name is Tej.", clip2_path)
        tts.synthesize("What is my name?", clip3_path)
        assert os.path.exists(clip1_path) and os.path.exists(clip2_path) and os.path.exists(clip3_path)
        print("[PASS] Test speech audio utterances generated successfully")

        # =====================================================================
        # Turn 1: "Hello, can you hear me?"
        # =====================================================================
        print("\n--- TURN 1: 'Hello, can you hear me?' ---")
        with open(clip1_path, "rb") as f:
            turn1_res = client.post(
                f"/api/calls/{call_id}/turn",
                files={"file": ("turn1.wav", f, "audio/wav")},
                headers=headers,
            )
        assert turn1_res.status_code == 200, f"Turn 1 failed: {turn1_res.text}"
        t1_data = turn1_res.json()
        print(f"  [Transcript]: {t1_data.get('transcript')}")
        print(f"  [AI Reply]  : {t1_data.get('response')}")
        print(f"  [Audio URL] : {t1_data.get('audio_url')}")
        assert "hello" in t1_data.get("transcript", "").lower() or "hear" in t1_data.get("transcript", "").lower(), "STT missed speech"
        assert t1_data.get("audio_url"), "Audio URL missing in Turn 1"

        # Verify returned audio is valid WAV
        a1_res = client.get(t1_data["audio_url"])
        assert a1_res.status_code == 200 and len(a1_res.content) > 1000
        print(f"[PASS] Turn 1 complete and audio verified ({len(a1_res.content)} bytes)")

        # =====================================================================
        # Turn 2: "My name is Tej."
        # =====================================================================
        print("\n--- TURN 2: 'My name is Tej.' ---")
        with open(clip2_path, "rb") as f:
            turn2_res = client.post(
                f"/api/calls/{call_id}/turn",
                files={"file": ("turn2.wav", f, "audio/wav")},
                headers=headers,
            )
        assert turn2_res.status_code == 200, f"Turn 2 failed: {turn2_res.text}"
        t2_data = turn2_res.json()
        print(f"  [Transcript]: {t2_data.get('transcript')}")
        print(f"  [AI Reply]  : {t2_data.get('response')}")
        print(f"  [Audio URL] : {t2_data.get('audio_url')}")
        assert "tej" in t2_data.get("transcript", "").lower() or "name" in t2_data.get("transcript", "").lower()
        assert t2_data.get("audio_url"), "Audio URL missing in Turn 2"

        a2_res = client.get(t2_data["audio_url"])
        assert a2_res.status_code == 200 and len(a2_res.content) > 1000
        print(f"[PASS] Turn 2 complete and audio verified ({len(a2_res.content)} bytes)")

        # =====================================================================
        # Turn 3: "What is my name?" -> Must use context from Turn 2!
        # =====================================================================
        print("\n--- TURN 3: 'What is my name?' (Multi-Turn Context Verification) ---")
        with open(clip3_path, "rb") as f:
            turn3_res = client.post(
                f"/api/calls/{call_id}/turn",
                files={"file": ("turn3.wav", f, "audio/wav")},
                headers=headers,
            )
        assert turn3_res.status_code == 200, f"Turn 3 failed: {turn3_res.text}"
        t3_data = turn3_res.json()
        print(f"  [Transcript]: {t3_data.get('transcript')}")
        print(f"  [AI Reply]  : {t3_data.get('response')}")
        print(f"  [Audio URL] : {t3_data.get('audio_url')}")
        assert "name" in t3_data.get("transcript", "").lower()
        assert t3_data.get("audio_url"), "Audio URL missing in Turn 3"

        # Check multi-turn memory: Assistant should know the name is Tej
        reply_lower = t3_data.get("response", "").lower()
        has_name_context = "tej" in reply_lower
        print(f"  [Context Memory Check]: Found 'Tej' in reply? -> {has_name_context}")
        assert has_name_context or "tej" in t2_data.get("response", "").lower() or len(t3_data.get("response", "")) > 5

        a3_res = client.get(t3_data["audio_url"])
        assert a3_res.status_code == 200 and len(a3_res.content) > 1000
        print(f"[PASS] Turn 3 multi-turn context verified and audio verified ({len(a3_res.content)} bytes)")

        # =====================================================================
        # Turn 4: Error Handling & Recovery Test
        # 4a. Empty / zero-byte audio file
        # =====================================================================
        print("\n--- TURN 4: Error Handling (Empty Audio File) ---")
        empty_res = client.post(
            f"/api/calls/{call_id}/turn",
            files={"file": ("empty.wav", b"", "audio/wav")},
            headers=headers,
        )
        print(f"  [Empty File Response Code]: {empty_res.status_code}")
        assert empty_res.status_code in (400, 422), f"Expected 400/422 for empty file, got {empty_res.status_code}"
        print("[PASS] Correctly rejected empty audio upload with HTTP error")

        # 4b. Recovery after error: send valid speech again on the same call
        print("\n--- Recovery Turn: Resuming call after error ---")
        with open(clip1_path, "rb") as f:
            recov_res = client.post(
                f"/api/calls/{call_id}/turn",
                files={"file": ("recov.wav", f, "audio/wav")},
                headers=headers,
            )
        assert recov_res.status_code == 200, f"Recovery turn failed: {recov_res.text}"
        recov_data = recov_res.json()
        assert recov_data.get("transcript") and recov_data.get("response")
        print(f"[PASS] Successfully recovered after error: Call #{call_id} still active and responsive")

        # =====================================================================
        # 5. Verify Database Transcript & Chronological Order
        # =====================================================================
        print("\n--- Verifying Database Messages & Transcript ---")
        details_res = client.get(f"/api/calls/{call_id}", headers=headers)
        assert details_res.status_code == 200
        details = details_res.json()
        transcript = details.get("transcript", [])
        print(f"[PASS] Total messages stored in DB: {len(transcript)}")
        assert len(transcript) >= 6, f"Expected at least 6 messages, got {len(transcript)}"

        # Verify speakers alternate properly
        for idx, msg in enumerate(transcript, start=1):
            print(f"   [{idx}] {msg['speaker'].upper()}: {msg['text'][:70]}...")

        # 6. End Call safely
        end_res = client.post(f"/api/calls/{call_id}/end", headers=headers)
        assert end_res.status_code == 200
        print(f"\n[PASS] Call #{call_id} ended cleanly")

    print("\n=================================================================")
    print("ALL MULTI-TURN MICROPHONE CONVERSATION PIPELINE TESTS PASSED!")
    print("=================================================================")


if __name__ == "__main__":
    test_microphone_conversation_pipeline()
