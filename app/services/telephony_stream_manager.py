import os
import io
import time
import json
import base64
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from app.database.database import SessionLocal
from app.database.models import Call, Message
from app.services.voice_response_service import VoiceResponseService
from app.services.escalation_service import EscalationService
from app.services.telephony_adapter import get_telephony_adapter
from app.utils.audio_utils import pcm_to_wav_bytes, wav_to_pcm_bytes, calculate_pcm_rms

RECORDINGS_DIR = "recordings"

IVR_STREAM_GREETING = "Press 1 for English, 2 for Hindi, 3 for Marathi, or simply start speaking."
HINDI_ASSISTANCE_TEXT = "हमारी टीम का एक सदस्य जल्द ही आपसे संपर्क करेगा। कृपया कुछ समय बाद हमारे कॉल का इंतजार करें। धन्यवाद।"
MARATHI_ASSISTANCE_TEXT = "आमच्या टीममधील एक सदस्य लवकरच आपल्याशी संपर्क साधेल. कृपया काही वेळाने आमच्या कॉलची वाट पाहा. धन्यवाद."


class TelephonyStreamManager:
    """
    Manages bidirectional audio streaming between telephony WebSocket clients
    (e.g., Exotel AgentStream) and the AI Voice Agent conversation engine.
    """

    def __init__(self, voice_service: Optional[VoiceResponseService] = None):
        self.voice_service = voice_service or VoiceResponseService()
        self.sample_rate = int(os.getenv("TELEPHONY_SAMPLE_RATE", "16000"))
        # Exotel AgentStream specification:
        # Minimum chunk size: 3.2 KB (3200 bytes)
        # Must be exact multiple of 320 bytes
        self.chunk_size_bytes = 3200
        self.speech_rms_threshold = 280.0
        self.silence_threshold_sec = 0.75

    async def handle_stream(self, websocket: WebSocket, call_id: int):
        await websocket.accept()

        with SessionLocal() as db:
            call = db.query(Call).filter(Call.id == call_id).first()
            if not call:
                await websocket.close(code=4004, reason="Call session not found")
                return

        adapter = get_telephony_adapter(call.provider)
        stream_sid: Optional[str] = None
        is_call_active = True
        is_playing_assistant_audio = False
        current_playback_task: Optional[asyncio.Task] = None

        # Inspect query parameters for requested sample rate (e.g. ?sample-rate=16000)
        qp_rate = websocket.query_params.get("sample-rate") or websocket.query_params.get("sample_rate")
        stream_sample_rate = int(qp_rate) if qp_rate and qp_rate.isdigit() else self.sample_rate

        # Audio buffering state
        audio_buffer = bytearray()
        last_speech_time = 0.0
        has_spoken = False
        speech_frame_count = 0

        async def send_audio_file(wav_path: str):
            nonlocal is_playing_assistant_audio
            if not os.path.exists(wav_path) or not stream_sid:
                return

            # Resample audio cleanly to the negotiated stream sample rate (16kHz or 8kHz)
            pcm_bytes = wav_to_pcm_bytes(wav_path, target_sample_rate=stream_sample_rate)
            if not pcm_bytes:
                return

            is_playing_assistant_audio = True
            chunk_size = self.chunk_size_bytes  # 3200 bytes (meets Exotel >= 3.2KB and % 320 == 0)
            chunk_duration = (chunk_size / 2) / stream_sample_rate

            try:
                start_time = time.monotonic()
                total_chunks = (len(pcm_bytes) + chunk_size - 1) // chunk_size

                for chunk_idx in range(total_chunks):
                    if not is_call_active or not is_playing_assistant_audio:
                        break

                    offset = chunk_idx * chunk_size
                    chunk = pcm_bytes[offset : offset + chunk_size]

                    # Zero pad trailing partial frame to guarantee exact 3200 bytes
                    if len(chunk) < chunk_size:
                        chunk = chunk + b"\x00" * (chunk_size - len(chunk))

                    b64_payload = base64.b64encode(chunk).decode("utf-8")
                    msg = adapter.format_media_message(stream_sid=stream_sid, pcm_base64=b64_payload)
                    await websocket.send_text(json.dumps(msg))

                    # Pre-buffer: Send first 2 chunks quickly (5ms) to prime Exotel's jitter buffer
                    if chunk_idx < 2:
                        await asyncio.sleep(0.005)
                        continue

                    # Drift-compensated pacing: Keeps audio perfectly synchronized with real-time playback
                    expected_elapsed = (chunk_idx - 1) * chunk_duration
                    actual_elapsed = time.monotonic() - start_time
                    sleep_delay = expected_elapsed - actual_elapsed
                    if sleep_delay > 0:
                        await asyncio.sleep(sleep_delay)
            except Exception as e:
                print(f"[Stream Warning] Audio streaming interrupted: {e}")
            finally:
                is_playing_assistant_audio = False

        async def handle_caller_utterance(utterance_bytes: bytes):
            nonlocal current_playback_task
            if not is_call_active:
                return
            min_bytes = int(stream_sample_rate * 2 * 0.4)  # < 0.4s
            if not utterance_bytes or len(utterance_bytes) < min_bytes:
                return

            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            wav_bytes = pcm_to_wav_bytes(bytes(utterance_bytes), sample_rate=stream_sample_rate)
            temp_in = os.path.join(RECORDINGS_DIR, f"tel_in_{call_id}_{int(time.time()*1000)}.wav")
            temp_out = os.path.join(RECORDINGS_DIR, f"tel_out_{call_id}_{int(time.time()*1000)}.wav")

            with open(temp_in, "wb") as f:
                f.write(wav_bytes)

            try:
                with SessionLocal() as db:
                    active_call = db.query(Call).filter(Call.id == call_id).first()
                    if not active_call or active_call.ended_at is not None:
                        return
                    if active_call.status in ("human_transferred", "escalated", "completed", "failed", "canceled"):
                        return

                    # Transcribe with Faster-Whisper
                    transcript = self.voice_service.stt_service.transcribe(temp_in)
                    if not transcript or not transcript.strip():
                        return

                    t_lower = transcript.lower()

                    # Language / DTMF speech fallback
                    if active_call.status == "AWAITING_LANGUAGE" or active_call.language == "pending":
                        if any(k in t_lower for k in ["2", "hindi", "हिन्दी"]):
                            await execute_language_escalation(db, active_call, "Hindi", HINDI_ASSISTANCE_TEXT, "hi")
                            return
                        elif any(k in t_lower for k in ["3", "marathi", "मराठी"]):
                            await execute_language_escalation(db, active_call, "Marathi", MARATHI_ASSISTANCE_TEXT, "mr")
                            return
                        else:
                            active_call.language = "en"
                            active_call.status = "in_progress"
                            db.commit()

                    # Process voice turn via existing VoiceResponseService (reusing the already computed transcript)
                    turn_res = await self.voice_service.process_existing_turn(
                        call_id=call_id,
                        audio_path=temp_in,
                        output_audio_path=temp_out,
                        db=db,
                        transcript_override=transcript,
                    )

                    # Stream assistant's spoken reply
                    if os.path.exists(temp_out) and is_call_active:
                        current_playback_task = asyncio.create_task(send_audio_file(temp_out))
            except Exception as ex:
                print(f"[Stream Error] Turn processing failed: {ex}")
            finally:
                if os.path.exists(temp_in):
                    try:
                        os.remove(temp_in)
                    except OSError:
                        pass

        async def execute_language_escalation(db, call_record, lang_name: str, message_text: str, lang_code: str):
            nonlocal is_call_active, current_playback_task
            is_call_active = False  # Terminate AI response loop upon escalation
            call_record.language = lang_name
            call_record.status = "human_transferred"
            call_record.ended_at = datetime.now(timezone.utc)

            self.voice_service.conversation_service.add_message(
                db, call_id=call_record.id, speaker="customer", text=f"Selected {lang_name}"
            )
            self.voice_service.conversation_service.add_message(
                db, call_id=call_record.id, speaker="assistant", text=message_text
            )
            EscalationService.create_escalation(
                db=db,
                call_id=call_record.id,
                reason=f"Customer selected {lang_name}",
                language_selected=lang_name,
                status="waiting",
                notes="Routed to Human Assistance queue via telephony stream.",
            )
            db.commit()

            # Synthesize notice audio and stream it
            notice_wav = os.path.join(RECORDINGS_DIR, f"{lang_code}_notice.wav")
            if not os.path.exists(notice_wav):
                self.voice_service.tts_service.synthesize(message_text, notice_wav, language=lang_code)

            await send_audio_file(notice_wav)

            # Initiate transfer to human forward number if configured
            forward_phone = os.getenv("HUMAN_ESCALATION_FORWARD_NUMBER")
            if forward_phone and call_record.provider_call_id:
                try:
                    await adapter.transfer_call(call_record.provider_call_id, forward_phone)
                except Exception as ex:
                    print(f"[Stream Warning] Transfer failed: {ex}")

        # Main WebSocket message loop
        try:
            while is_call_active:
                raw_text = await websocket.receive_text()
                if not raw_text:
                    continue

                try:
                    msg = json.parse(raw_text) if hasattr(json, "parse") else json.loads(raw_text)
                except Exception:
                    continue

                event_type = msg.get("event", "").lower()

                if event_type == "connected":
                    pass

                elif event_type == "start":
                    start_data = msg.get("start") or {}
                    media_format = start_data.get("mediaFormat") or msg.get("mediaFormat") or {}
                    detected_rate = media_format.get("sampleRate") or media_format.get("sample_rate")
                    if detected_rate and str(detected_rate).isdigit():
                        stream_sample_rate = int(detected_rate)
                    print(f"[Telephony Stream] Active stream sample rate negotiated: {stream_sample_rate}Hz")

                    stream_sid = (
                        msg.get("stream_sid")
                        or msg.get("streamSid")
                        or start_data.get("stream_sid")
                        or start_data.get("streamSid")
                        or f"str_{call_id}"
                    )
                    call_sid = (
                        msg.get("call_sid")
                        or msg.get("callSid")
                        or start_data.get("call_sid")
                        or start_data.get("callSid")
                    )

                    with SessionLocal() as db:
                        c = db.query(Call).filter(Call.id == call_id).first()
                        if c:
                            c.status = "in_progress"
                            if call_sid and not c.provider_call_id:
                                c.provider_call_id = call_sid
                            db.commit()

                    # Synthesize and play initial greeting
                    greeting_path = os.path.join(RECORDINGS_DIR, "ivr_prompt.wav")
                    if not os.path.exists(greeting_path):
                        self.voice_service.tts_service.synthesize(IVR_STREAM_GREETING, greeting_path, language="en")
                    current_playback_task = asyncio.create_task(send_audio_file(greeting_path))

                elif event_type == "media":
                    media_data = msg.get("media") or {}
                    b64_pcm = media_data.get("payload") or msg.get("payload")
                    if not b64_pcm:
                        continue

                    chunk = base64.b64decode(b64_pcm)
                    rms = calculate_pcm_rms(chunk)
                    now_ts = time.time()

                    # Voice Activity Detection (VAD) & Barge-in
                    if rms >= self.speech_rms_threshold:
                        speech_frame_count += 1
                        last_speech_time = now_ts

                        # If user speaks for 3 consecutive frames while assistant audio is playing, trigger BARGE-IN!
                        if is_playing_assistant_audio and speech_frame_count >= 3:
                            is_playing_assistant_audio = False
                            if current_playback_task and not current_playback_task.done():
                                current_playback_task.cancel()
                            # Send clear event to provider
                            if stream_sid:
                                clear_msg = adapter.format_clear_message(stream_sid)
                                await websocket.send_text(json.dumps(clear_msg))

                        has_spoken = True
                        audio_buffer.extend(chunk)
                    else:
                        speech_frame_count = 0
                        if has_spoken:
                            audio_buffer.extend(chunk)
                            # Check silence threshold
                            if (now_ts - last_speech_time) >= self.silence_threshold_sec:
                                # Utterance completed
                                to_process = bytes(audio_buffer)
                                audio_buffer.clear()
                                has_spoken = False
                                asyncio.create_task(handle_caller_utterance(to_process))

                    # Safety guard: prevent single utterance exceeding 25 seconds
                    if len(audio_buffer) > (stream_sample_rate * 2 * 25):
                        to_process = bytes(audio_buffer)
                        audio_buffer.clear()
                        has_spoken = False
                        asyncio.create_task(handle_caller_utterance(to_process))

                elif event_type == "dtmf":
                    dtmf_data = msg.get("dtmf") or {}
                    digit = str(dtmf_data.get("digit") or "")
                    if digit in ["1", "2", "3"]:
                        with SessionLocal() as db:
                            c = db.query(Call).filter(Call.id == call_id).first()
                            if c and c.ended_at is None:
                                if digit == "2":
                                    await execute_language_escalation(db, c, "Hindi", HINDI_ASSISTANCE_TEXT, "hi")
                                elif digit == "3":
                                    await execute_language_escalation(db, c, "Marathi", MARATHI_ASSISTANCE_TEXT, "mr")
                                else:
                                    c.language = "en"
                                    c.status = "in_progress"
                                    db.commit()

                elif event_type in ("stop", "hangup"):
                    is_call_active = False
                    break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[Stream Warning] WebSocket connection error: {e}")
        finally:
            is_call_active = False
            is_playing_assistant_audio = False
            if current_playback_task and not current_playback_task.done():
                current_playback_task.cancel()

            # Guarantee call completion status in DB
            with SessionLocal() as db:
                c = db.query(Call).filter(Call.id == call_id).first()
                if c and c.ended_at is None:
                    c.ended_at = datetime.now(timezone.utc)
                    if c.status not in ("completed", "failed", "human_transferred"):
                        c.status = "completed"
                    db.commit()
