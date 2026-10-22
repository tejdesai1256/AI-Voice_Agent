import os
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
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

    def get_adapter(self):
        return get_telephony_adapter()

    def get_active_call_count(self) -> int:
        return len(self.active_calls)

    def is_concurrency_limit_reached(self) -> bool:
        return len(self.active_calls) >= self.max_concurrent_calls

    async def initiate_outbound_call(
        self,
        db: Session,
        destination_phone: str,
        customer_id: Optional[int] = None,
        customer_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates phone number, enforces concurrency guards, creates pending call
        record, and initiates outbound call via selected telephony provider.
        """
        # 1. Check concurrency limits
        if self.is_concurrency_limit_reached():
            return {
                "success": False,
                "error": f"Concurrency limit reached ({self.max_concurrent_calls} concurrent calls max). Please wait for active calls to conclude.",
                "code": "CONCURRENCY_LIMIT_EXCEEDED",
            }

        # 2. Validate and normalize phone number
        is_valid, norm_phone, err = normalize_phone_number(destination_phone)
        if not is_valid:
            return {
                "success": False,
                "error": err,
                "code": "INVALID_PHONE_NUMBER",
            }

        # 3. Resolve or create customer record
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

        # 4. Create pending Call session in DB
        now = datetime.now(timezone.utc)
        adapter = self.get_adapter()
        provider_name = getattr(adapter, "provider_name", type(adapter).__name__.replace("TelephonyAdapter", "").lower())
        if "exotel" in type(adapter).__name__.lower():
            provider_name = "exotel"
        elif "mock" in type(adapter).__name__.lower():
            provider_name = "mock"

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

        # 5. Determine public WebSocket URL for bidirectional audio streaming
        ws_base = os.getenv("TELEPHONY_WS_URL")
        if not ws_base:
            public_base = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
            ws_base = public_base.replace("https://", "wss://").replace("http://", "ws://")
        ws_base = ws_base.rstrip("/")
        stream_url = f"{ws_base}/api/telephony/stream/{call.id}"

        # 6. Execute provider request
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
            call.error_code = result.error_message
            db.commit()
            return {
                "success": False,
                "call_id": call.id,
                "error": result.error_message or "Telephony provider rejected call initiation request.",
                "code": "PROVIDER_REJECTED",
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
        if self.is_concurrency_limit_reached():
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

        with SessionLocal() as db:
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
                provider=type(adapter).__name__.replace("TelephonyAdapter", "").lower(),
                provider_call_id=inbound_info.get("external_call_id"),
                caller_phone=safe_phone,
                started_at=datetime.now(timezone.utc),
            )
            db.add(call)
            db.commit()
            db.refresh(call)

            ws_base = os.getenv("TELEPHONY_WS_URL") or os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").replace("https://", "wss://").replace("http://", "ws://")
            stream_url = f"{ws_base.rstrip('/')}/api/telephony/stream/{call.id}"

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
        is_exotel = "exotel" in type(adapter).__name__.lower()
        configured = getattr(adapter, "is_configured", lambda: True)()
        return {
            "telephony_enabled": os.getenv("TELEPHONY_ENABLED", "false").lower() == "true",
            "active_provider": "exotel" if is_exotel else "mock",
            "is_provider_configured": configured,
            "active_call_count": self.get_active_call_count(),
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_call_duration_seconds": self.max_call_duration_seconds,
        }
