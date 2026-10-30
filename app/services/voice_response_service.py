import time
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.services.stt_service import STTService
from app.services.response_service import ResponseService
from app.services.tts_service import TTSService
from app.services.conversation_service import ConversationService
from app.services.conversation_context_service import ConversationContextService
from app.services.requirement_service import RequirementService
from app.services.escalation_service import EscalationService
from app.services.conversation_flow_service import ConversationFlowService
from app.database.models import Requirement, Message, ConversationState


class VoiceResponseService:
    def __init__(
        self,
        stt_service: STTService = None,
        response_service: ResponseService = None,
        tts_service: TTSService = None,
        conversation_service: ConversationService = None,
        context_service: ConversationContextService = None,
        requirement_service: RequirementService = None,
        conversation_flow_service: ConversationFlowService = None,
    ):
        """
        Initializes VoiceResponseService by reusing STTService, ResponseService, TTSService,
        ConversationService, ConversationContextService, RequirementService, and ConversationFlowService.
        """
        self.stt_service = stt_service if stt_service is not None else STTService()
        self.response_service = (
            response_service if response_service is not None else ResponseService()
        )
        self.tts_service = tts_service if tts_service is not None else TTSService()
        self.conversation_service = (
            conversation_service if conversation_service is not None else ConversationService()
        )
        self.context_service = (
            context_service if context_service is not None else ConversationContextService()
        )
        self.requirement_service = (
            requirement_service if requirement_service is not None else RequirementService()
        )
        self.conversation_flow_service = (
            conversation_flow_service if conversation_flow_service is not None else ConversationFlowService()
        )

    def _print_performance_summary(
        self,
        stt_time: float,
        context_db_time: float,
        llm_time: float,
        tts_time: float,
        db_persistence_time: float,
        total_time: float,
    ):
        print("=== PERFORMANCE ===")
        print(f"STT: {stt_time:.2f} sec")
        print(f"Context/DB: {context_db_time:.2f} sec")
        print(f"LLM: {llm_time:.2f} sec")
        print(f"TTS: {tts_time:.2f} sec")
        print(f"DB Persistence: {db_persistence_time:.2f} sec")
        print(f"Total: {total_time:.2f} sec")
        print("===================")

    def _schedule_background_requirements(self, call_id: int, full_text: str, is_resolved: bool):
        """Dispatches requirements extraction in a non-blocking background task."""
        async def _bg_worker():
            from app.database.database import SessionLocal
            bg_db = SessionLocal()
            try:
                await self._update_requirements(bg_db, call_id, full_text, is_resolved)
            except Exception as e:
                print(f"Background requirements extraction error for call {call_id}: {e}")
            finally:
                bg_db.close()

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_bg_worker())
        except RuntimeError:
            pass

    async def _update_requirements(
        self,
        db: Session,
        call_id: int,
        full_text: str,
        is_resolved: bool,
        extra_answers: Optional[Dict[str, Any]] = None,
    ):
        """Extracts and persists customer requirements into requirements table with smart field preservation."""
        try:
            req_data = await self.requirement_service.extract_requirements(full_text)
            if extra_answers:
                # Merge qualification answers into structured requirements
                for k, v in extra_answers.items():
                    if v and not req_data.get(k):
                        req_data[k] = v

            now = datetime.now(timezone.utc)
            req_record = db.query(Requirement).filter(Requirement.call_id == call_id).first()
            if not req_record:
                req_record = Requirement(
                    call_id=call_id,
                    status="COLLECTED" if is_resolved else "IN_PROGRESS",
                    structured_requirements=req_data,
                    created_at=now,
                    updated_at=now,
                )
                db.add(req_record)
            else:
                existing = req_record.structured_requirements or {}
                merged = dict(req_data)
                for key in ["requirements", "products_or_services", "quantity", "budget", "deadline"]:
                    if not merged.get(key) and existing.get(key):
                        merged[key] = existing[key]
                if not merged.get("important_details") and existing.get("important_details"):
                    merged["important_details"] = existing["important_details"]

                if extra_answers:
                    for k, v in extra_answers.items():
                        if v and not merged.get(k):
                            merged[k] = v

            if is_resolved:
                existing = req_record.structured_requirements or {}
                if not existing.get("conversation_summary"):
                    try:
                        summary = await self.requirement_service.generate_summary(
                            call_id=call_id,
                            transcript=full_text,
                            structured_req=existing,
                            unanswered_questions=extra_answers.get("missing_info") if isinstance(extra_answers, dict) else None,
                        )
                        existing["conversation_summary"] = summary
                        req_record.structured_requirements = dict(existing)
                    except Exception as s_err:
                        print(f"Warning: Summary generation failed for call {call_id}: {s_err}")

            db.commit()
        except Exception as e:
            print(f"Warning: Failed to update requirements for call {call_id}: {e}")

    async def process_existing_turn(
        self,
        call_id: int,
        audio_path: str,
        output_audio_path: str,
        db: Session,
        transcript_override: Optional[str] = None,
    ) -> dict:
        """
        Processes an additional voice turn for an already active call.
        Enforces:
        - Call termination check (cannot speak on ended calls)
        - Concurrency lock (prevents duplicate simultaneous requests)
        - Empty speech handling (does not advance question flow)
        - Automatic question flow progression with natural conversation
        - Context & answers persistence throughout call
        """
        total_start = time.perf_counter()

        # 1. Check if call has ended
        call = self.conversation_service.get_call(db, call_id)
        call_status_upper = (call.status or "").upper()
        terminal_statuses = {"RESOLVED", "COMPLETED", "ESCALATED", "FAILED", "CANCELED", "CANCELLED", "BUSY", "NO_ANSWER", "HUMAN_TRANSFERRED"}
        if call.ended_at is not None or call_status_upper in terminal_statuses:
            raise ValueError(f"Call with id {call_id} has already ended.")

        # 2. Concurrency lock check
        conv_state = self.conversation_flow_service.get_or_create_state(db, call_id)
        if conv_state.is_processing == 1:
            raise ValueError(f"A voice turn is already being processed for call {call_id}.")

        conv_state.is_processing = 1
        conv_state.conversation_status = "PROCESSING"
        db.commit()

        try:
            customer = call.customer
            all_calls = self.conversation_service.get_call_history(db, customer.id)
            context_db_time = 0.0

            # 3. Transcribe with Faster-Whisper (or use pre-computed transcript)
            stt_start = time.perf_counter()
            if transcript_override is not None:
                transcript = transcript_override
            else:
                call_lang = call.language if getattr(call, "language", None) not in ("pending", None) else None
                transcript = self.stt_service.transcribe(audio_path, language=call_lang)
            if not transcript or not transcript.strip():
                transcript = "[No clear speech detected]"
            stt_time = time.perf_counter() - stt_start

            # 4. Handle Empty / Inaudible speech (do not advance question)
            t_clean = transcript.strip()
            if t_clean in ("[No clear speech detected]", "[Inaudible]", "[Inaudible speech]", "") or len(t_clean) < 2:
                clarification_text = "I couldn't hear you clearly. Could you please repeat that?"
                # Save customer message
                self.conversation_service.add_message(
                    db, call_id=call.id, speaker="customer", text=transcript
                )
                # Save assistant clarification
                self.conversation_service.add_message(
                    db, call_id=call.id, speaker="assistant", text=clarification_text
                )
                conv_state.conversation_status = "AWAITING_RESPONSE"
                db.commit()

                tts_start = time.perf_counter()
                generated_audio_path = self.tts_service.synthesize(clarification_text, output_audio_path)
                tts_time = time.perf_counter() - tts_start
                total_time = time.perf_counter() - total_start

                return {
                    "transcript": transcript,
                    "response": clarification_text,
                    "audio_path": generated_audio_path,
                    "customer_id": customer.external_customer_id,
                    "call_id": call.id,
                    "call_status": call.status,
                    "is_escalated": False,
                    "conversation_state": {
                        "current_question_id": conv_state.current_question_id,
                        "questions_asked": conv_state.questions_asked or [],
                        "answers": conv_state.answers or {},
                        "missing_info": conv_state.missing_info or [],
                        "conversation_status": "AWAITING_RESPONSE",
                    },
                    "action": "repeat",
                    "timings": {
                        "stt_sec": round(stt_time, 2),
                        "llm_sec": 0.0,
                        "tts_sec": round(tts_time, 2),
                        "context_db_sec": round(context_db_time, 2),
                        "total_sec": round(total_time, 2),
                    },
                }

            # 5. Save customer message
            p1_start = time.perf_counter()
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="customer", text=transcript
            )
            p1_time = time.perf_counter() - p1_start

            # 6. Process turn via ConversationFlowService
            llm_start = time.perf_counter()
            flow_result = await self.conversation_flow_service.process_turn(
                call=call, transcript=transcript, db=db
            )
            llm_time = time.perf_counter() - llm_start

            response_text = flow_result["response"]
            call_status = flow_result["call_status"]
            is_escalated = flow_result["is_escalated"]

            # 7. Save assistant response
            p2_start = time.perf_counter()
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            p2_time = time.perf_counter() - p2_start

            # 8. Update Call status & End call if needed
            if call_status in ("RESOLVED", "COMPLETED"):
                self.conversation_service.end_call(db, call.id, status=call_status)
            elif is_escalated:
                self.conversation_service.end_call(db, call.id, status="ESCALATED")
            else:
                self.conversation_service.update_call_status(db, call.id, status=call_status)

            db_persistence_time = p1_time + p2_time

            # 9. Synthesize TTS response immediately
            tts_start = time.perf_counter()
            try:
                safe_text = (response_text or "").strip() or "Thank you. How can I help you today?"
                generated_audio_path = self.tts_service.synthesize(safe_text, output_audio_path)
            except Exception as tts_err:
                print(f"[TTS Warning] Audio synthesis failed: {tts_err}. Creating emergency fallback audio.")
                from app.utils.audio_utils import pcm_to_wav_bytes
                os.makedirs(os.path.dirname(output_audio_path) or ".", exist_ok=True)
                with open(output_audio_path, "wb") as f:
                    f.write(pcm_to_wav_bytes(b"\x00\x00" * 8000, sample_rate=16000))
                generated_audio_path = output_audio_path
            tts_time = time.perf_counter() - tts_start

            total_time = time.perf_counter() - total_start

            # 10. Update requirements with customer answers & extracted specs
            customer_msgs = (
                db.query(Message)
                .filter(Message.call_id == call.id, Message.speaker == "customer")
                .order_by(Message.id)
                .all()
            )
            full_customer_transcript = "\n".join(m.text for m in customer_msgs if m.text)
            if not full_customer_transcript.strip():
                full_customer_transcript = transcript

            try:
                await self._update_requirements(
                    db,
                    call.id,
                    full_customer_transcript,
                    is_resolved=(call_status in ("RESOLVED", "COMPLETED")),
                    extra_answers=conv_state.answers,
                )
            except Exception as e:
                print(f"Error extracting requirements for call {call.id}: {e}")

            self._print_performance_summary(
                stt_time=stt_time,
                context_db_time=context_db_time,
                llm_time=llm_time,
                tts_time=tts_time,
                db_persistence_time=db_persistence_time,
                total_time=total_time,
            )

            return {
                "transcript": transcript,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": call_status,
                "is_escalated": is_escalated,
                "conversation_state": flow_result.get("conversation_state"),
                "action": flow_result.get("action"),
                "timings": {
                    "stt_sec": round(stt_time, 2),
                    "llm_sec": round(llm_time, 2),
                    "tts_sec": round(tts_time, 2),
                    "context_db_sec": round(context_db_time, 2),
                    "total_sec": round(total_time, 2),
                },
            }
        finally:
            # Guarantee lock release even if an exception invalidated transaction
            try:
                db.rollback()
            except Exception:
                pass
            try:
                state = db.query(ConversationState).filter(ConversationState.call_id == call_id).first()
                if state:
                    state.is_processing = 0
                    if state.conversation_status == "PROCESSING":
                        state.conversation_status = "AWAITING_RESPONSE"
                    db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass

    async def process_voice_turn(
        self,
        audio_path: str,
        output_audio_path: str,
        external_customer_id: str,
        db: Session,
        language: str = "en",
    ) -> dict:
        """
        Processes a single-turn voice interaction with persistent database memory.
        """
        # Create customer & call session
        customer = self.conversation_service.get_or_create_customer(
            db, external_customer_id=external_customer_id
        )
        call = self.conversation_service.create_call(db, customer.id)
        call.language = language
        db.commit()

        # Initialize conversation state
        self.conversation_flow_service.get_or_create_state(db, call.id)

        # Delegate turn execution to unified process_existing_turn pipeline
        return await self.process_existing_turn(
            call_id=call.id,
            audio_path=audio_path,
            output_audio_path=output_audio_path,
            db=db,
        )
