from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.scheduler_service import AutomationScheduler
from app.database.models import SystemUser
from app.utils.auth import get_current_user

router = APIRouter()
scheduler = AutomationScheduler()


class StartAutomationRequest(BaseModel):
    customer_ids: Optional[List[int]] = None


@router.post("/start")
async def start_automation(
    req: Optional[StartAutomationRequest] = None,
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Starts the automated outbound calling queue for explicitly selected customer IDs.
    Does NOT start automatically on system boot or page load.
    """
    if scheduler.is_running:
        return {"status": "already_running"}

    selected_ids = req.customer_ids if req else None
    scheduler.start(customer_ids=selected_ids)
    return {"status": "started", "queued_count": len(scheduler.outbound_queue)}


@router.post("/stop")
async def stop_automation(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Safely stops the automated calling queue and stops future calls.
    """
    if not scheduler.is_running:
        return {"status": "not_running"}

    await scheduler.stop()
    return {"status": "stopped"}


@router.post("/stop_call")
async def stop_active_call(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Forcefully cancels the currently active simulated call.
    """
    if not scheduler.is_running:
        return {"status": "not_running"}

    await scheduler.cancel_active_call()
    return {"status": "active_call_cancelled"}


@router.get("/status")
async def get_status(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Returns real-time automation queue statuses (Waiting, In Call, Completed, Failed).
    """
    return scheduler.get_status()


@router.post("/incoming")
async def trigger_incoming_call(
    customer_id: int,
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Simulates an incoming customer priority call into the queue.
    """
    if not scheduler.is_running:
        raise HTTPException(status_code=400, detail="Automated scheduler is not running")

    scheduler.trigger_incoming(customer_id)
    return {"status": "incoming_call_queued", "customer_id": customer_id}
