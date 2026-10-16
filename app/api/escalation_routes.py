from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import SystemUser
from app.services.escalation_service import EscalationService
from app.utils.auth import get_current_user

router = APIRouter()


class ResolveRequest(BaseModel):
    notes: Optional[str] = ""


class CreateEscalationRequest(BaseModel):
    call_id: int
    reason: str
    language_selected: Optional[str] = "en"
    notes: Optional[str] = None


@router.get("")
def get_escalations(
    status: Optional[str] = None,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all human assistance requests with transcript and customer info."""
    try:
        return EscalationService.get_escalations(db, status_filter=status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
def create_escalation(
    req: CreateEscalationRequest,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new human assistance escalation request."""
    try:
        esc = EscalationService.create_escalation(
            db=db,
            call_id=req.call_id,
            reason=req.reason,
            language_selected=req.language_selected or "en",
            notes=req.notes,
        )
        return {
            "id": esc.id,
            "call_id": esc.call_id,
            "customer_name": esc.customer_name,
            "reason": esc.escalation_reason,
            "status": esc.status,
            "language": esc.language_selected,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{escalation_id}/contacted")
def mark_escalation_contacted(
    escalation_id: int,
    req: Optional[ResolveRequest] = None,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark an escalation as contacted by an operator."""
    try:
        notes = req.notes if req and req.notes else ""
        return EscalationService.mark_contacted(db, escalation_id, notes)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{escalation_id}/resolve")
def resolve_escalation(
    escalation_id: int,
    req: Optional[ResolveRequest] = None,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark an escalation as resolved by an operator."""
    try:
        notes = req.notes if req and req.notes else ""
        return EscalationService.resolve_escalation(db, escalation_id, notes)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{escalation_id}/callback")
def initiate_callback(
    escalation_id: int,
    req: Optional[ResolveRequest] = None,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Initiates an operator callback for the escalated customer."""
    try:
        notes = req.notes if req and req.notes else ""
        return EscalationService.initiate_callback(db, escalation_id, notes)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{escalation_id}/callback-complete")
def complete_callback(
    escalation_id: int,
    req: Optional[ResolveRequest] = None,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Marks an operator callback as completed."""
    try:
        notes = req.notes if req and req.notes else ""
        return EscalationService.complete_callback(db, escalation_id, notes)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

