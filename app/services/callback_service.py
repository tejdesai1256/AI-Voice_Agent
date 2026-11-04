import os
import re
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.database.models import CallbackRequest, Call, Customer
from app.utils.phone_utils import normalize_phone_number, mask_phone_number


# Default calling hours in local time (e.g. 09:00 to 20:00)
CALLING_HOURS_START = 9
CALLING_HOURS_END = 20

DEFAULT_MAX_RETRIES = int(os.getenv("MAX_CALL_RETRIES", "3"))
DEFAULT_RETRY_INTERVAL_MINUTES = int(os.getenv("CALL_RETRY_INTERVAL_MINUTES", "30"))


class CallbackService:
    @staticmethod
    def is_within_calling_hours(dt: Optional[datetime] = None) -> bool:
        """
        Enforces customer protection guardrails:
        Calling permitted only between 9:00 AM and 8:00 PM.
        """
        now = dt or datetime.now(timezone.utc)
        # Assuming UTC+5:30 (India Standard Time) or local server hour
        # For timezone safety, inspect local hour
        hour = now.hour
        return CALLING_HOURS_START <= hour < CALLING_HOURS_END

    @staticmethod
    def adjust_for_calling_hours(target_time: datetime) -> datetime:
        """
        If scheduled time falls outside allowable calling hours (9 AM - 8 PM),
        moves target to 10:00 AM the next day.
        """
        if target_time.hour < CALLING_HOURS_START:
            return target_time.replace(hour=CALLING_HOURS_START, minute=0, second=0, microsecond=0)
        elif target_time.hour >= CALLING_HOURS_END:
            next_day = target_time + timedelta(days=1)
            return next_day.replace(hour=CALLING_HOURS_START + 1, minute=0, second=0, microsecond=0)
        return target_time

    @classmethod
    def schedule_callback(
        cls,
        db: Session,
        phone_number: str,
        customer_id: Optional[int] = None,
        customer_name: Optional[str] = None,
        scheduled_time: Optional[datetime] = None,
        reason: str = "customer_requested",
        call_id: Optional[int] = None,
        notes: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_interval_minutes: int = DEFAULT_RETRY_INTERVAL_MINUTES,
    ) -> Dict[str, Any]:
        """
        Schedules a callback safely:
        - Validates and normalizes phone number
        - Checks for existing active scheduled callbacks (duplicate prevention)
        - Adjusts target for legal calling hours
        - Persists CallbackRequest
        """
        is_valid, norm_phone, err = normalize_phone_number(phone_number)
        if not is_valid:
            return {"success": False, "error": f"Invalid phone number: {err}"}

        # Check duplicate pending callback for the same destination phone
        existing = (
            db.query(CallbackRequest)
            .filter(
                CallbackRequest.phone_number == norm_phone,
                CallbackRequest.status.in_(["scheduled", "retry_pending"]),
            )
            .first()
        )
        if existing:
            return {
                "success": False,
                "error": f"A callback is already scheduled for {mask_phone_number(norm_phone)} at {existing.scheduled_time.isoformat()}.",
                "callback_id": existing.id,
                "is_duplicate": True,
            }

        # Determine target scheduled time (defaults to +30 minutes if unspecified)
        now = datetime.now(timezone.utc)
        target = scheduled_time or (now + timedelta(minutes=retry_interval_minutes))
        target = cls.adjust_for_calling_hours(target)

        # Resolve customer name if missing
        c_name = customer_name
        if not c_name and customer_id:
            c = db.query(Customer).filter(Customer.id == customer_id).first()
            if c:
                c_name = c.name

        cb = CallbackRequest(
            customer_id=customer_id,
            phone_number=norm_phone,
            customer_name=c_name or f"Caller {mask_phone_number(norm_phone)}",
            scheduled_time=target,
            status="scheduled",
            retry_count=0,
            max_retries=max_retries,
            retry_interval_minutes=retry_interval_minutes,
            reason=reason,
            call_id=call_id,
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        db.add(cb)
        db.commit()
        db.refresh(cb)

        return {
            "success": True,
            "callback_id": cb.id,
            "phone_number": norm_phone,
            "scheduled_time": cb.scheduled_time.isoformat(),
            "reason": cb.reason,
            "status": cb.status,
            "message": f"Callback scheduled for {mask_phone_number(norm_phone)} at {cb.scheduled_time.strftime('%Y-%m-%d %H:%M:%S UTC')}.",
        }

    @classmethod
    def handle_unanswered_call(
        cls,
        db: Session,
        call: Call,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_interval_minutes: int = DEFAULT_RETRY_INTERVAL_MINUTES,
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically evaluates unanswered / busy calls for retry scheduling:
        - Rejects already exhausted retries (stops infinite loops)
        - Rejects opted-out / DND callers
        - Respects calling hours
        """
        if not call or call.call_type != "OUTBOUND":
            return None

        phone = call.destination_phone or call.caller_phone
        if not phone:
            return None

        # Check if customer has opted out (DND or NOT_INTERESTED in messages or status)
        if call.status in ("NOT_INTERESTED", "opt_out") or call.error_code == "OPT_OUT":
            print(f"[Callback Guard] Number {mask_phone_number(phone)} opted out. No retry scheduled.")
            return None

        # Count previous retries for this phone within the last 24 hours
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        prior_retries = (
            db.query(CallbackRequest)
            .filter(
                CallbackRequest.phone_number == phone,
                CallbackRequest.created_at >= yesterday,
                CallbackRequest.reason.like("unanswered_%"),
            )
            .count()
        )

        if prior_retries >= max_retries:
            print(f"[Callback Guard] Max retry limit ({max_retries}) reached for {mask_phone_number(phone)}. Stopping retries.")
            return None

        # Check existing pending callback
        existing = (
            db.query(CallbackRequest)
            .filter(
                CallbackRequest.phone_number == phone,
                CallbackRequest.status.in_(["scheduled", "retry_pending"]),
            )
            .first()
        )
        if existing:
            return None

        now = datetime.now(timezone.utc)
        target = now + timedelta(minutes=retry_interval_minutes * (prior_retries + 1))
        target = cls.adjust_for_calling_hours(target)

        cust_name = call.customer.name if call.customer else None

        cb = CallbackRequest(
            customer_id=call.customer_id,
            phone_number=phone,
            customer_name=cust_name or f"Customer {mask_phone_number(phone)}",
            scheduled_time=target,
            status="retry_pending",
            retry_count=prior_retries + 1,
            max_retries=max_retries,
            retry_interval_minutes=retry_interval_minutes,
            reason=f"unanswered_{call.status.lower()}",
            call_id=call.id,
            notes=f"Automatic retry #{prior_retries + 1} scheduled after call ended with status '{call.status}'.",
            created_at=now,
            updated_at=now,
        )
        db.add(cb)
        db.commit()
        db.refresh(cb)

        print(f"[Callback Retry] Scheduled retry #{cb.retry_count} for {mask_phone_number(phone)} at {cb.scheduled_time.isoformat()}.")
        return {
            "callback_id": cb.id,
            "retry_count": cb.retry_count,
            "scheduled_time": cb.scheduled_time.isoformat(),
        }

    @staticmethod
    def get_callbacks(
        db: Session,
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Returns list of callbacks with customer details."""
        query = db.query(CallbackRequest)
        if status_filter:
            query = query.filter(CallbackRequest.status == status_filter.strip().lower())
        results = query.order_by(CallbackRequest.scheduled_time.asc()).limit(limit).all()

        return [
            {
                "id": cb.id,
                "customer_id": cb.customer_id,
                "phone_number": cb.phone_number,
                "customer_name": cb.customer_name,
                "scheduled_time": cb.scheduled_time.isoformat() if cb.scheduled_time else None,
                "status": cb.status,
                "retry_count": cb.retry_count,
                "max_retries": cb.max_retries,
                "reason": cb.reason,
                "call_id": cb.call_id,
                "notes": cb.notes,
                "created_at": cb.created_at.isoformat() if cb.created_at else None,
            }
            for cb in results
        ]

    @staticmethod
    def cancel_callback(db: Session, callback_id: int) -> Dict[str, Any]:
        cb = db.query(CallbackRequest).filter(CallbackRequest.id == callback_id).first()
        if not cb:
            raise ValueError(f"Callback ID {callback_id} not found")
        cb.status = "cancelled"
        cb.updated_at = datetime.now(timezone.utc)
        db.commit()
        return {"success": True, "callback_id": cb.id, "status": "cancelled"}

