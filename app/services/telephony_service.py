import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.database import SessionLocal
from app.database.models import Call, Customer, Message
from app.services.telephony_adapter import (
    get_telephony_adapter,
    is_valid_status_transition,
    TERMINAL_CALL_STATUSES,
)
from app.services.conversation_service import ConversationService
from app.utils.phone_utils import normalize_phone_number, mask_phone_number


class TelephonyService:
    def __init__(self):
        self.conversation_service = ConversationService()
        self.active_calls: Dict[int, Dict[str, Any]] = {}
        self.max_concurrent_calls = int(os.getenv("MAX_CONCURRENT_CALLS", "5"))
        self.max_call_duration_seconds = int(os.getenv("MAX_CALL_DURATION_SECONDS", "600"))
        self.active_provider_override: Optional[str] = None

    def set_active_provider(self, provider_name: str) -> Dict[str, Any]:
        """Explicitly sets active provider ('mock' or 'exotel') for the session."""
        prov = str(provider_name or "").strip().lower()
        if prov not in ("mock", "exotel"):
            raise ValueError(f"Unsupported provider '{provider_name}'. Must be 'mock' or 'exotel'.")
        self.active_provider_override = prov
        return {
            "success": True,
            "active_provider": self.active_provider_override,
            "message": f"Telephony provider switched to '{self.active_provider_override}'.",
        }

    def get_adapter(self, provider_name: Optional[str] = None):
        target = provider_name or self.active_provider_override
        return get_telephony_adapter(target)

    def get_active_call_count(self) -> int:
        return len(self.active_calls)

    def is_concurrency_limit_reached(self, db: Optional[Session] = None) -> bool:
        if len(self.active_calls) >= self.max_concurrent_calls:
            return True
        if db:
            from datetime import timedelta
            # Reclaim abandoned/stale calls older than max_call_duration_seconds
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.max_call_duration_seconds)
            stale_calls = (
                db.query(Call)
                .filter(
                    Call.ended_at.is_(None),
                    Call.started_at < cutoff,
                    func.lower(Call.status).in_(["queued", "initiated", "ringing", "in_progress", "active", "awaiting_language"]),
                )
                .all()
            )
            if stale_calls:
                now_ts = datetime.now(timezone.utc)
                for sc in stale_calls:
                    sc.ended_at = now_ts
                    if sc.status not in ("completed", "canceled", "failed"):
                        sc.status = "completed"
                db.commit()

            active_db_count = (
                db.query(func.count(Call.id))
                .filter(
                    Call.ended_at.is_(None),
                    func.lower(Call.status).in_(["queued", "initiated", "ringing", "in_progress", "active", "awaiting_language"]),
                )
                .scalar() or 0
            )
            return active_db_count >= self.max_concurrent_calls
        return False

    async def initiate_outbound_call(
        self,
        db: Session,
        destination_phone: str,
        customer_id: Optional[int] = None,
        customer_name: Optional[str] = None,
        provider: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates phone number, enforces concurrency guards, creates pending call
        record, and initiates outbound call via selected telephony provider.
        """
        # 1. Validate and normalize phone number
        is_valid, norm_phone, err = normalize_phone_number(destination_phone)
        if not is_valid:
            return {
                "success": False,
                "error": err,
                "code": "INVALID_PHONE_NUMBER",
            }

        # 2. Check provider configuration if real mode is selected
        selected_provider = (provider or self.active_provider_override or os.getenv("TELEPHONY_PROVIDER", "mock")).strip().lower()
        adapter = self.get_adapter(selected_provider)
        provider_name = getattr(adapter, "provider_name", selected_provider)
        if provider_name == "exotel" and not getattr(adapter, "is_configured", lambda: True)():
            return {
                "success": False,
                "error": "Telephony provider 'exotel' is selected but required credentials (EXOTEL_ACCOUNT_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_CALLER_ID) are missing or incomplete in environment.",
                "code": "PROVIDER_CONFIG_ERROR",
                "provider": "exotel",
            }

        # 3. Check concurrency limits
        if self.is_concurrency_limit_reached(db):
            return {
                "success": False,
                "error": f"Concurrency limit reached ({self.max_concurrent_calls} concurrent calls max). Please wait for active calls to conclude.",
                "code": "CONCURRENCY_LIMIT_EXCEEDED",
            }

        # 3b. Check for duplicate active call to the same phone number (idempotency guard)
        for active in self.active_calls.values():
            if active.get("destination_phone") == norm_phone:
                return {
                    "success": False,
                    "error": f"A call is already active for {mask_phone_number(norm_phone)} (Call ID #{active.get('call_id')}). Please wait for it to complete.",
                    "code": "DUPLICATE_CALL_IN_PROGRESS",
                }

        active_db = (
            db.query(Call)
            .filter(
                Call.destination_phone == norm_phone,
                Call.ended_at.is_(None),
                func.lower(Call.status).in_(["queued", "initiated", "ringing", "in_progress", "active", "awaiting_language"]),
            )
            .first()
        )
        if active_db:
            return {
                "success": False,
                "error": f"A call is already in progress for {mask_phone_number(norm_phone)} (Call ID #{active_db.id}, status: {active_db.status}).",
                "code": "DUPLICATE_CALL_IN_PROGRESS",
            }

        # 4. Resolve or create customer record
        if customer_id:
            customer = db.query(Customer).filter(Customer.id == customer_id).first()
            if not customer:
                return {"success": False, "error": f"Customer ID {customer_id} not found", "code": "CUSTOMER_NOT_FOUND"}
            if not customer.phone:
                customer.phone = norm_phone
                db.commit()
        else:
            ext_cust_id = f"TEL_{norm_phone.replace('+', '')}"
            customer = self.conversation_service.get_or_create_customer(
                db,
                external_customer_id=ext_cust_id,
                name=customer_name or f"Caller {mask_phone_number(norm_phone)}",
                phone=norm_phone,
            )

        # 5. Create pending Call session in DB
        now = datetime.now(timezone.utc)
        call = Call(
            customer_id=customer.id,
            status="queued",
            call_type="OUTBOUND",
            language="pending",
            provider=provider_name,
            destination_phone=norm_phone,
            started_at=now,
        )
        db.add(call)
        db.commit()
        db.refresh(call)

        # 6. Determine public WebSocket URL for bidirectional audio streaming
        ws_base = os.getenv("TELEPHONY_WS_URL")
        if not ws_base:
            public_base = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
            ws_base = public_base.replace("https://", "wss://").replace("http://", "ws://")
        ws_base = ws_base.rstrip("/")
        sample_rate = int(os.getenv("TELEPHONY_SAMPLE_RATE", "16000"))
        stream_url = f"{ws_base}/api/telephony/stream/{call.id}?sample-rate={sample_rate}"

        # 7. Execute provider request
        result = await adapter.start_outbound_call(
            to_phone=norm_phone,
            call_id=call.id,
            stream_url=stream_url,
            metadata=metadata,
        )

        if result.success and result.call_sid:
            call.provider_call_id = result.call_sid
            call.status = result.status or "initiated"
            db.commit()

            # Track in active calls
            self.active_calls[call.id] = {
                "call_id": call.id,
                "provider_call_id": result.call_sid,
                "destination_phone": norm_phone,
                "started_at": now,
            }

            # Spawn call duration limit safety task
            self._schedule_duration_guard(call.id)

            # In mock mode: schedule realistic deterministic call lifecycle simulation
            if provider_name == "mock":
                self._schedule_mock_lifecycle(call.id)

            return {
                "success": True,
                "call_id": call.id,
                "call_sid": result.call_sid,
                "provider": provider_name,
                "status": call.status,
                "destination": norm_phone,
                "stream_url": stream_url,
            }
        else:
            call.status = "failed"
            call.ended_at = datetime.now(timezone.utc)
            call.error_code = result.error_type or "PROVIDER_REJECTED"
            db.commit()
            return {
                "success": False,
                "call_id": call.id,
                "error": result.error_message or "Telephony provider rejected call initiation request.",
                "code": result.error_type or "PROVIDER_REJECTED",
                "provider": provider_name,
            }

    async def handle_status_callback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives and normalizes provider status callback events.
        Enforces state transition rules and prevents terminal resurrection.
        """
        adapter = self.get_adapter()
        event = adapter.parse_status_callback(payload)

        with SessionLocal() as db:
            call = None
            if event.provider_call_id:
                call = db.query(Call).filter(Call.provider_call_id == event.provider_call_id).first()

            if not call:
                custom_field = payload.get("CustomField")
                if custom_field and str(custom_field).isdigit():
                    call = db.query(Call).filter(Call.id == int(custom_field)).first()

            if not call:
                return {"status": "ignored", "reason": "Call not found"}

            if not call.provider_call_id and event.provider_call_id:
                call.provider_call_id = event.provider_call_id

            current_status = call.status
            new_status = event.normalized_status

            if not is_valid_status_transition(current_status, new_status):
                return {
                    "status": "ignored",
                    "reason": f"Invalid transition from {current_status} to {new_status}",
                    "call_id": call.id,
                }

            call.status = new_status
            if event.duration_seconds:
                call.duration_seconds = event.duration_seconds
            if event.recording_url:
                call.recording_url = event.recording_url
            if event.error_code:
                call.error_code = event.error_code

            if new_status in TERMINAL_CALL_STATUSES and not call.ended_at:
                call.ended_at = datetime.now(timezone.utc)
                if call.id in self.active_calls:
                    del self.active_calls[call.id]

            db.commit()
            return {"status": "updated", "call_id": call.id, "call_status": call.status}

    async def handle_inbound_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes incoming telephone call webhook.
        Creates Call session and returns stream connection details.
        """
        with SessionLocal() as db:
            if self.is_concurrency_limit_reached(db):
                return {
                    "success": False,
                    "error": "Concurrency limit reached. Unable to accept inbound call.",
                    "code": "CONCURRENCY_LIMIT_EXCEEDED",
                }

            adapter = self.get_adapter()
            inbound_info = adapter.handle_incoming_call(payload)
            caller_phone = inbound_info.get("customer_phone", "")

            is_valid, norm_phone, _ = normalize_phone_number(caller_phone)
            safe_phone = norm_phone if is_valid else caller_phone

            ext_cust_id = f"TEL_{safe_phone.replace('+', '')}" if safe_phone else f"TEL_UNKNOWN_{int(datetime.now().timestamp())}"
            customer = self.conversation_service.get_or_create_customer(
                db,
                external_customer_id=ext_cust_id,
                name=f"Inbound Caller {mask_phone_number(safe_phone)}",
                phone=safe_phone,
            )

            call = Call(
                customer_id=customer.id,
                status="ringing",
                call_type="INBOUND",
                language="pending",
                provider=getattr(adapter, "provider_name", "mock"),
                provider_call_id=inbound_info.get("external_call_id"),
                caller_phone=safe_phone,
                started_at=datetime.now(timezone.utc),
            )
            db.add(call)
            db.commit()
            db.refresh(call)

            ws_base = os.getenv("TELEPHONY_WS_URL") or os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").replace("https://", "wss://").replace("http://", "ws://")
            sample_rate = int(os.getenv("TELEPHONY_SAMPLE_RATE", "16000"))
            stream_url = f"{ws_base.rstrip('/')}/api/telephony/stream/{call.id}?sample-rate={sample_rate}"

            self.active_calls[call.id] = {
                "call_id": call.id,
                "provider_call_id": call.provider_call_id,
                "destination_phone": safe_phone,
                "started_at": call.started_at,
            }
            self._schedule_duration_guard(call.id)

            return {
                "success": True,
                "call_id": call.id,
                "stream_url": stream_url,
                "customer_id": customer.id,
            }

    async def hangup_call(self, call_id: int) -> Dict[str, Any]:
        """
        Sends hangup command to provider and updates database.
        """
        with SessionLocal() as db:
            call = db.query(Call).filter(Call.id == call_id).first()
            if not call:
                return {"success": False, "error": f"Call ID {call_id} not found"}

            if call.ended_at is not None:
                return {"success": True, "status": "already_ended", "call_id": call_id}

            adapter = self.get_adapter()
            if call.provider_call_id:
                try:
                    await adapter.end_call(call.provider_call_id)
                except Exception as e:
                    print(f"[Telephony Warning] Error in adapter.end_call: {e}")

            call.status = "canceled"
            call.ended_at = datetime.now(timezone.utc)
            db.commit()

            if call_id in self.active_calls:
                del self.active_calls[call_id]

            return {"success": True, "status": "ended", "call_id": call_id}

    def _schedule_mock_lifecycle(self, call_id: int):
        """
        Deterministic simulated call lifecycle for mock telephony.
        Transitions call from queued -> ringing -> in_progress -> completed.
        Respects manual hangups or external events.
        """
        async def _simulate():
            # 1. Wait 1.5s -> ringing
            await asyncio.sleep(1.5)
            if call_id not in self.active_calls:
                return
            with SessionLocal() as db:
                c = db.query(Call).filter(Call.id == call_id).first()
                if c and c.ended_at is None and is_valid_status_transition(c.status, "ringing"):
                    c.status = "ringing"
                    db.commit()

            # 2. Wait 2.0s -> in_progress
            await asyncio.sleep(2.0)
            if call_id not in self.active_calls:
                return
            with SessionLocal() as db:
                c = db.query(Call).filter(Call.id == call_id).first()
                if c and c.ended_at is None and is_valid_status_transition(c.status, "in_progress"):
                    c.status = "in_progress"
                    c.language = "en"
                    db.commit()
                    # Add simulated greeting message to conversation if empty
                    msg_count = db.query(Message).filter(Message.call_id == call_id).count()
                    if msg_count == 0:
                        self.conversation_service.add_message(
                            db, call_id=call_id, speaker="assistant",
                            text="Hello! I am your AI voice assistant. How can I help you today?"
                        )
                        db.commit()

            # 3. Wait 8.0s -> completed
            await asyncio.sleep(8.0)
            if call_id not in self.active_calls:
                return
            with SessionLocal() as db:
                c = db.query(Call).filter(Call.id == call_id).first()
                if c and c.ended_at is None and is_valid_status_transition(c.status, "completed"):
                    now = datetime.now(timezone.utc)
                    c.status = "completed"
                    c.ended_at = now
                    if c.started_at:
                        c_start = c.started_at.replace(tzinfo=timezone.utc) if c.started_at.tzinfo is None else c.started_at
                        c.duration_seconds = max(1, round((now - c_start).total_seconds()))
                    db.commit()
                    if call_id in self.active_calls:
                        del self.active_calls[call_id]

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_simulate())
        except RuntimeError:
            pass

    def _schedule_duration_guard(self, call_id: int):
        """
        Background task automatically terminating calls exceeding MAX_CALL_DURATION_SECONDS.
        """
        async def _guard():
            await asyncio.sleep(self.max_call_duration_seconds)
            if call_id in self.active_calls:
                print(f"[Telephony Guard] Call {call_id} exceeded max duration ({self.max_call_duration_seconds}s). Hanging up.")
                await self.hangup_call(call_id)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_guard())
        except RuntimeError:
            pass

    def get_system_status(self) -> Dict[str, Any]:
        adapter = self.get_adapter()
        is_exotel = getattr(adapter, "provider_name", "mock") == "exotel"
        configured = getattr(adapter, "is_configured", lambda: True)()
        return {
            "telephony_enabled": os.getenv("TELEPHONY_ENABLED", "false").lower() == "true",
            "active_provider": "exotel" if is_exotel else "mock",
            "is_provider_configured": configured,
            "providers_available": ["mock", "exotel"],
            "active_call_count": self.get_active_call_count(),
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_call_duration_seconds": self.max_call_duration_seconds,
        }

    async def verify_provider(self) -> Dict[str, Any]:
        """
        Runs zero-cost credentials verification against the active provider.
        """
        adapter = self.get_adapter()
        return await adapter.verify_credentials()
