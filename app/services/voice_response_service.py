import time
import asyncio
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.services.stt_service import STTService
from app.services.response_service import ResponseService
from app.services.tts_service import TTSService
from app.services.conversation_service import ConversationService
from app.services.conversation_context_service import ConversationContextService
from app.services.requirement_service import RequirementService
from app.services.escalation_service import EscalationService
from app.database.models import Requirement, Message


class VoiceResponseService:
    def __init__(
        self,
        stt_service: STTService = None,
        response_service: ResponseService = None,
        tts_service: TTSService = None,
        conversation_service: ConversationService = None,
        context_service: ConversationContextService = None,
        requirement_service: RequirementService = None,
    ):
        """
        Initializes VoiceResponseService by reusing STTService, ResponseService, TTSService,
        ConversationService, ConversationContextService, and RequirementService.
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

    async def _update_requirements(self, db: Session, call_id: int, full_text: str, is_resolved: bool):
        """Extracts and persists customer requirements into requirements table with smart field preservation."""
        try:
            req_data = await self.requirement_service.extract_requirements(full_text)
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
                # Merge existing requirements to preserve non-null values
                existing = req_record.structured_requirements or {}
                merged = dict(req_data)
                for key in ["requirements", "products_or_services", "quantity", "budget", "deadline"]:
                    if not merged.get(key) and existing.get(key):
                        merged[key] = existing[key]
                if not merged.get("important_details") and existing.get("important_details"):
                    merged["important_details"] = existing["important_details"]

                req_record.structured_requirements = merged
                req_record.status = "COLLECTED" if is_resolved else "IN_PROGRESS"
                req_record.updated_at = now
            db.commit()
        except Exception as e:
            print(f"Warning: Failed to update requirements for call {call_id}: {e}")

    async def process_existing_turn(
        self,
        call_id: int,
        audio_path: str,
        output_audio_path: str,
        db: Session,
    ) -> dict:
        """
        Processes an additional voice turn for an already active call.
        Optimized pipeline: Transcribe -> LLM Response -> Immediate TTS Synthesis -> Return audio.
        Requirements are extracted asynchronously in the background.
        """
        total_start = time.perf_counter()

        # 1-4. DB Retrieval & Context Building
        db_start = time.perf_counter()
        call = self.conversation_service.get_call(db, call_id)
        if call.ended_at is not None:
            raise ValueError(f"Call with id {call_id} has already ended.")
        customer = call.customer
        all_calls = self.conversation_service.get_call_history(db, customer.id)
        history_context = self.context_service.build_context(all_calls)
        context_db_time = time.perf_counter() - db_start

        # 5. Transcribe with Faster-Whisper
        stt_start = time.perf_counter()
        transcript = self.stt_service.transcribe(audio_path)
        if not transcript or not transcript.strip():
            transcript = "[No clear speech detected]"
        stt_time = time.perf_counter() - stt_start

        # 6. Save customer message
        p1_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="customer", text=transcript
        )
        p1_time = time.perf_counter() - p1_start

        # Check Human Escalation
        should_esc, esc_reason = EscalationService.should_escalate(transcript)
        if should_esc:
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason,
                language_selected=call.language or "en",
            )
            response_text = (
                "I completely understand. I am transferring you to a human representative right away. "
                "Please stay on the line while we connect you."
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            self.conversation_service.end_call(db, call.id, status="ESCALATED")

            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start
            total_time = time.perf_counter() - total_start

            return {
                "transcript": transcript,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": "ESCALATED",
                "is_escalated": True,
                "timings": {
                    "stt_sec": round(stt_time, 2),
                    "llm_sec": 0.0,
                    "tts_sec": round(tts_time, 2),
                    "context_db_sec": round(context_db_time, 2),
                    "total_sec": round(total_time, 2),
                },
            }

        # 7. Generate assistant response
        llm_start = time.perf_counter()
        response_text = await self.response_service.generate_response(
            transcript=transcript, conversation_context=history_context
        )
        llm_time = time.perf_counter() - llm_start

        # 8. Save assistant response
        p2_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="assistant", text=response_text
        )
        p2_time = time.perf_counter() - p2_start

        # 8b. Fast resolution check
        is_resolved = await self.response_service.check_resolution(transcript, history_context)
        call_status = "RESOLVED" if is_resolved else "IN_PROGRESS"

        if is_resolved:
            self.conversation_service.end_call(db, call.id, status="RESOLVED")
        else:
            self.conversation_service.update_call_status(db, call.id, status="IN_PROGRESS")

        db_persistence_time = p1_time + p2_time

        # 9. Synthesize TTS immediately so the customer gets audio as fast as possible
        tts_start = time.perf_counter()
        generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
        tts_time = time.perf_counter() - tts_start

        total_time = time.perf_counter() - total_start

        # 10. Schedule structured requirements extraction in the background (non-blocking)
        # Gather all customer statements so requirement extraction has full call context
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
            await self._update_requirements(db, call.id, full_customer_transcript, is_resolved)
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
            "is_escalated": False,
            "timings": {
                "stt_sec": round(stt_time, 2),
                "llm_sec": round(llm_time, 2),
                "tts_sec": round(tts_time, 2),
                "context_db_sec": round(context_db_time, 2),
                "total_sec": round(total_time, 2),
            },
        }

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
        total_start = time.perf_counter()

        # a-d. Customer & Call creation + history context retrieval
        db_start = time.perf_counter()
        customer = self.conversation_service.get_or_create_customer(
            db, external_customer_id=external_customer_id
        )
        prior_calls = self.conversation_service.get_call_history(db, customer.id)
        history_context = self.context_service.build_context(prior_calls)
        call = self.conversation_service.create_call(db, customer.id)
        call.language = language
        db.commit()
        context_db_time = time.perf_counter() - db_start

        # e. Transcribe current audio
        stt_start = time.perf_counter()
        transcript = self.stt_service.transcribe(audio_path)
        stt_time = time.perf_counter() - stt_start

        # f. Save customer transcript
        p1_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="customer", text=transcript
        )
        p1_time = time.perf_counter() - p1_start

        # Check Human Escalation
        should_esc, esc_reason = EscalationService.should_escalate(transcript)
        if should_esc:
            EscalationService.create_escalation(
                db=db,
                call_id=call.id,
                reason=esc_reason,
                language_selected=language,
            )
            response_text = (
                "I understand your request. I am immediately connecting you to our human support team."
            )
            self.conversation_service.add_message(
                db, call_id=call.id, speaker="assistant", text=response_text
            )
            self.conversation_service.end_call(db, call.id, status="ESCALATED")

            tts_start = time.perf_counter()
            generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
            tts_time = time.perf_counter() - tts_start

            return {
                "transcript": transcript,
                "response": response_text,
                "audio_path": generated_audio_path,
                "customer_id": customer.external_customer_id,
                "call_id": call.id,
                "call_status": "ESCALATED",
                "is_escalated": True,
            }

        # g. Generate AI response using ResponseService with history
        llm_start = time.perf_counter()
        response_text = await self.response_service.generate_response(
            transcript=transcript,
            conversation_context=history_context,
        )
        llm_time = time.perf_counter() - llm_start

        # h. Save assistant response
        p2_start = time.perf_counter()
        self.conversation_service.add_message(
            db, call_id=call.id, speaker="assistant", text=response_text
        )
        p2_time = time.perf_counter() - p2_start

        # i. Generate TTS audio
        tts_start = time.perf_counter()
        generated_audio_path = self.tts_service.synthesize(response_text, output_audio_path)
        tts_time = time.perf_counter() - tts_start

        # j. End the Call
        p3_start = time.perf_counter()
        self.conversation_service.end_call(db, call.id)
        p3_time = time.perf_counter() - p3_start

        db_persistence_time = p1_time + p2_time + p3_time
        total_time = time.perf_counter() - total_start

        # Schedule requirements extraction in the background
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
            await self._update_requirements(db, call.id, full_customer_transcript, is_resolved=True)
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
            "call_status": "RESOLVED",
            "is_escalated": False,
            "timings": {
                "stt_sec": round(stt_time, 2),
                "llm_sec": round(llm_time, 2),
                "tts_sec": round(tts_time, 2),
                "context_db_sec": round(context_db_time, 2),
                "total_sec": round(total_time, 2),
            },
        }
