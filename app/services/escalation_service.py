import re
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from app.database.models import HumanAssistanceRequest, Call, Customer, Message

ESCALATION_KEYWORDS = [
    r"\b(human|agent|operator|representative|person|executive|officer)\b",
    r"\b(talk to (someone|a human|an agent|a person|a representative))\b",
    r"\b(speak with (someone|a human|an agent|a person|a representative))\b",
    r"\b(connect me|transfer me|customer care|helpdesk)\b",
    r"\b(frustrated|angry|terrible|horrible|useless|stupid ai|waste of time)\b",
    r"\b(you don't understand|you do not understand|not helpful|can't help)\b",
]


class EscalationService:
    @staticmethod
    def should_escalate(transcript: str) -> tuple[bool, Optional[str]]:
        """
        Analyzes customer statement to check if human escalation is required.
        Returns (is_escalated, reason).
        """
        text = transcript.lower().strip()
        for pattern in ESCALATION_KEYWORDS:
            if re.search(pattern, text):
                if any(w in text for w in ["frustrated", "angry", "terrible", "horrible", "useless", "waste of time", "don't understand"]):
                    return True, f"Customer expressed dissatisfaction or frustration: \"{transcript[:100]}\""
                return True, f"Customer requested a human representative: \"{transcript[:100]}\""
        return False, None

    @staticmethod
    def create_escalation(
        db: Session,
        call_id: int,
        reason: str,
        language_selected: str = "en",
        notes: Optional[str] = None,
        status: str = "waiting",
    ) -> HumanAssistanceRequest:
        """Creates and persists an escalation in human_assistance_requests table."""
        call = db.query(Call).options(joinedload(Call.customer)).filter(Call.id == call_id).first()
        customer = call.customer if call else None

        # Clean language name
        clean_lang = language_selected
        if language_selected in ["hi", "2"]:
            clean_lang = "Hindi"
        elif language_selected in ["mr", "3"]:
            clean_lang = "Marathi"
        elif language_selected in ["en", "1"]:
            clean_lang = "English"

        now = datetime.now(timezone.utc)
        req = HumanAssistanceRequest(
            call_id=call_id,
            customer_id=customer.id if customer else 0,
            customer_name=customer.name if customer else "Unknown Customer",
            customer_phone=customer.phone if customer else "N/A",
            language_selected=clean_lang,
            escalation_reason=reason,
            status=status.lower(),
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        db.add(req)
        if call:
            call.status = "ESCALATED"
            call.ended_at = now
        db.commit()
        db.refresh(req)
        return req

    @staticmethod
    def get_escalations(db: Session, status_filter: Optional[str] = None) -> List[dict]:
        """Retrieves escalations with call and transcript details."""
        query = db.query(HumanAssistanceRequest).order_by(HumanAssistanceRequest.created_at.desc())
        if status_filter:
            s = status_filter.strip().lower()
            if s in ["pending", "waiting"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["waiting", "PENDING"]))
            elif s in ["resolved"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["resolved", "RESOLVED"]))
            elif s in ["contacted"]:
                query = query.filter(HumanAssistanceRequest.status.in_(["contacted", "CONTACTED"]))
            else:
                query = query.filter(HumanAssistanceRequest.status.ilike(f"%{s}%"))
        items = query.all()

        results = []
        for item in items:
            call = db.query(Call).options(joinedload(Call.messages)).filter(Call.id == item.call_id).first()
            transcript_msgs = []
            if call and call.messages:
                sorted_msgs = sorted(call.messages, key=lambda m: (m.timestamp or datetime.min, m.id or 0))
                transcript_msgs = [
                    {
                        "speaker": m.speaker,
                        "text": m.text,
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    }
                    for m in sorted_msgs
                ]

            # Derive a human-readable callback_status from the raw status
            status_lower = (item.status or "").lower()
            if status_lower in ["callback_completed"]:
                cb_status = "Callback Completed"
            elif status_lower in ["callback_initiated"]:
                cb_status = "Callback Initiated"
            elif status_lower in ["contacted"]:
                cb_status = "Contacted"
            elif status_lower in ["resolved"]:
                cb_status = "Resolved"
            else:
                cb_status = "Waiting for Callback"

            results.append({
                "id": item.id,
                "call_id": item.call_id,
                "customer_id": item.customer_id,
                "customer_name": item.customer_name or (call.customer.name if call and call.customer else "N/A"),
                "customer_phone": item.customer_phone or (call.customer.phone if call and call.customer else "N/A"),
                "language": item.language_selected,
                "language_selected": item.language_selected,
                "reason": item.escalation_reason,
                "escalation_reason": item.escalation_reason,
                "status": item.status,
                "callback_status": cb_status,
                "notes": item.notes,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                "transcript": transcript_msgs,
            })
        return results

    @staticmethod
    def mark_contacted(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks an escalation as contacted."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        item.status = "contacted"
        if notes:
            item.notes = f"{item.notes}\n[Contacted]: {notes}".strip() if item.notes else f"[Contacted]: {notes}"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {"status": "contacted", "id": item.id, "notes": item.notes}

    @staticmethod
    def resolve_escalation(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks an escalation as resolved with resolution notes."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        item.status = "resolved"
        if notes:
            item.notes = f"{item.notes}\n[Resolved]: {notes}".strip() if item.notes else f"[Resolved]: {notes}"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {"status": "resolved", "id": item.id, "notes": item.notes}

    @staticmethod
    def schedule_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Records a scheduled callback for human support team and marks contacted."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        callback_note = f"[Callback Queued at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]: Operator scheduled phone callback."
        if notes:
            callback_note += f" Notes: {notes}"
        item.notes = f"{item.notes}\n{callback_note}".strip() if item.notes else callback_note
        item.status = "contacted"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {"status": "contacted", "id": item.id, "notes": item.notes}

    @staticmethod
    def initiate_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Initiates an operator callback for the escalated customer."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        callback_note = f"[Callback Initiated at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        if notes:
            callback_note += f": {notes}"
        item.notes = f"{item.notes}\n{callback_note}".strip() if item.notes else callback_note
        item.status = "callback_initiated"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {
            "status": "callback_initiated",
            "callback_status": "Callback Initiated",
            "id": item.id,
            "notes": item.notes,
            "telephony_active": False,
            "message": "Callback initiated for customer. Actual telephony dialing is pending configuration of a carrier integration.",
        }

    @staticmethod
    def complete_callback(db: Session, escalation_id: int, notes: str = "") -> dict:
        """Marks an operator callback as completed."""
        item = db.query(HumanAssistanceRequest).filter(HumanAssistanceRequest.id == escalation_id).first()
        if not item:
            raise ValueError(f"Escalation request {escalation_id} not found")

        callback_note = f"[Callback Completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        if notes:
            callback_note += f": {notes}"
        item.notes = f"{item.notes}\n{callback_note}".strip() if item.notes else callback_note
        item.status = "callback_completed"
        item.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(item)
        return {
            "status": "callback_completed",
            "callback_status": "Callback Completed",
            "id": item.id,
            "notes": item.notes,
        }
