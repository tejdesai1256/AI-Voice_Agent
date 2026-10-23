import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import SystemUser, Call
from app.utils.auth import get_current_user
from app.services.telephony_service import TelephonyService
from app.services.telephony_stream_manager import TelephonyStreamManager

router = APIRouter()
telephony_service = TelephonyService()
stream_manager = TelephonyStreamManager()


class OutboundCallRequest(BaseModel):
    phone_number: str
    customer_id: Optional[int] = None
    customer_name: Optional[str] = None


@router.post("/outbound")
async def initiate_outbound_telephony_call(
    request: Request,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Authenticated endpoint to initiate an outbound phone call via configured provider.
    Validates phone number and enforces concurrency safeguards.
    """
    # Accept JSON or Form
    phone_number = None
    customer_id = None
    customer_name = None

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            phone_number = body.get("phone_number")
            customer_id = body.get("customer_id")
            customer_name = body.get("customer_name")
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            phone_number = form.get("phone_number")
            cid = form.get("customer_id")
            customer_id = int(cid) if cid and str(cid).isdigit() else None
            customer_name = form.get("customer_name")
        except Exception:
            pass

    if not phone_number:
        raise HTTPException(status_code=400, detail="Field 'phone_number' is required.")

    result = await telephony_service.initiate_outbound_call(
        db=db,
        destination_phone=str(phone_number).strip(),
        customer_id=customer_id,
        customer_name=customer_name,
    )

    if not result.get("success"):
        code = result.get("code")
        status_code = 400
        if code == "CONCURRENCY_LIMIT_EXCEEDED":
            status_code = 429
        elif code == "DUPLICATE_CALL_IN_PROGRESS":
            status_code = 409
        elif code == "CUSTOMER_NOT_FOUND":
            status_code = 404
        elif code == "PROVIDER_REJECTED":
            status_code = 502
        raise HTTPException(status_code=status_code, detail=result.get("error"))

    return result


@router.post("/webhook/exotel")
async def exotel_status_webhook(
    request: Request,
):
    """
    Public webhook receiver for Exotel call-status callbacks.
    Validates webhook authenticity, normalizes status, and guards call state transitions.
    """
    adapter = telephony_service.get_adapter()
    body_bytes = await request.body()
    headers_dict = dict(request.headers)
    query_dict = dict(request.query_params)

    # Validate webhook signature / token
    is_valid = adapter.validate_webhook(headers_dict, body_bytes, query_dict)
    if not is_valid:
        raise HTTPException(status_code=403, detail="Invalid webhook signature or token.")

    # Parse payload (form data or JSON)
    payload = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            payload = {}
    else:
        try:
            form = await request.form()
            payload = dict(form)
        except Exception:
            payload = {}

    result = await telephony_service.handle_status_callback(payload)
    return {"status": "ok", "result": result}


@router.post("/inbound")
async def inbound_telephony_webhook(
    request: Request,
):
    """
    Public webhook receiver for inbound customer phone calls.
    Validates caller, creates Call session, and returns stream connection instructions.
    """
    adapter = telephony_service.get_adapter()
    body_bytes = await request.body()
    headers_dict = dict(request.headers)
    query_dict = dict(request.query_params)

    if not adapter.validate_webhook(headers_dict, body_bytes, query_dict):
        raise HTTPException(status_code=403, detail="Invalid inbound webhook signature.")

    payload = {}
    try:
        form = await request.form()
        payload = dict(form)
    except Exception:
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            payload = {}

    res = await telephony_service.handle_inbound_call(payload)
    if not res.get("success"):
        raise HTTPException(status_code=429 if res.get("code") == "CONCURRENCY_LIMIT_EXCEEDED" else 400, detail=res.get("error"))

    # Return Exotel-compatible response or JSON
    accept = request.headers.get("accept", "")
    if "xml" in accept.lower():
        stream_url = res.get("stream_url", "")
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
        return Response(content=xml_content, media_type="application/xml")

    return res


@router.api_route("/calls/{call_id}/connect", methods=["GET", "POST"])
async def exotel_call_connect_instructions(
    call_id: int,
    request: Request,
):
    """
    Public ExOMl endpoint queried by Exotel when an outbound call connects.
    Returns the <Connect><Stream url="..." /></Connect> instruction to initiate AgentStream.
    """
    import os
    ws_base = os.getenv("TELEPHONY_WS_URL")
    if not ws_base:
        public_base = os.getenv("PUBLIC_BASE_URL", "http://localhost:8001")
        ws_base = public_base.replace("https://", "wss://").replace("http://", "ws://")
    ws_base = ws_base.rstrip("/")
    stream_url = f"{ws_base}/api/telephony/stream/{call_id}"

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>"""
    return Response(content=xml_content, media_type="application/xml")


@router.post("/calls/{call_id}/hangup")
async def hangup_call_endpoint(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Authenticated endpoint to terminate an active telephone call session.
    """
    res = await telephony_service.hangup_call(call_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error"))
    return res


@router.get("/status")
def get_telephony_status(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Returns telemetry and health of provider configuration without exposing secrets.
    """
    return telephony_service.get_system_status()


@router.get("/verify-provider")
async def verify_telephony_provider(
    current_user: SystemUser = Depends(get_current_user),
):
    """
    Safely tests provider credentials and health without initiating a paid call.
    """
    return await telephony_service.verify_provider()


@router.get("/calls")
def get_telephony_call_history(
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieves call history sessions specifically initiated or received via Telephony.
    """
    calls = (
        db.query(Call)
        .filter(Call.provider.in_(["exotel", "mock"]))
        .order_by(Call.started_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": c.id,
            "provider": c.provider,
            "provider_call_id": c.provider_call_id,
            "call_type": c.call_type,
            "status": c.status,
            "destination_phone": c.destination_phone,
            "caller_phone": c.caller_phone,
            "language": c.language,
            "started_at": c.started_at.isoformat() if c.started_at else None,
            "ended_at": c.ended_at.isoformat() if c.ended_at else None,
            "duration_seconds": c.duration_seconds,
            "error_code": c.error_code,
        }
        for c in calls
    ]


@router.websocket("/stream/{call_id}")
async def telephony_media_stream(
    websocket: WebSocket,
    call_id: int,
):
    """
    WebSocket endpoint handling real-time bidirectional media stream
    (Exotel AgentStream format: connected, start, media, clear, dtmf, stop).
    """
    await stream_manager.handle_stream(websocket, call_id)
