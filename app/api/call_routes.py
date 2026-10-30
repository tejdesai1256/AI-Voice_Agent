import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request, Query
from sqlalchemy.orm import Session, joinedload

from app.services.voice_response_service import VoiceResponseService
from app.services.conversation_service import ConversationService
from app.services.escalation_service import EscalationService
from app.services.requirement_service import RequirementService
from app.database.database import get_db
from app.database.models import Customer, Call, Message, Requirement, SystemUser
from app.utils.auth import get_current_user
from app.utils.audio_utils import convert_to_16k_mono_wav

router = APIRouter()
voice_service = VoiceResponseService()
conversation_service = ConversationService()
requirement_service = RequirementService()

RECORDINGS_DIR = "recordings"
GENERATED_AUDIO_DIR = "recordings"

IVR_PROMPT_MESSAGE = (
    "Press 1 for English. Press 2 for Hindi. Press 3 for Marathi."
)
ENGLISH_GREETING_MESSAGE = (
    "Hello! I am your AI voice assistant. How can I help you today?"
)
HINDI_ASSISTANCE_MESSAGE = (
    "हमारी टीम का एक सदस्य जल्द ही आपसे संपर्क करेगा। कृपया कुछ समय बाद हमारे कॉल का इंतजार करें। धन्यवाद।"
)
MARATHI_ASSISTANCE_MESSAGE = (
    "आमच्या टीममधील एक सदस्य लवकरच आपल्याशी संपर्क साधेल. कृपया काही वेळाने आमच्या कॉलची वाट पाहा. धन्यवाद."
)


# ---------------------------------------------------------------------------
# Customers Endpoints (Authenticated)
# ---------------------------------------------------------------------------

@router.get("/customers")
def get_customers(
    q: Optional[str] = Query(None, description="Search query for customer name, phone, or ID"),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve actual customer records from MySQL database with optional search filtering."""
    try:
        query = db.query(Customer).options(joinedload(Customer.calls)).order_by(Customer.id.desc())
        if q and q.strip():
            search_term = f"%{q.strip()}%"
            query = query.filter(
                (Customer.name.ilike(search_term))
                | (Customer.phone.ilike(search_term))
                | (Customer.external_customer_id.ilike(search_term))
            )
        customers = query.all()
        return [
            {
                "id": c.id,
                "external_customer_id": c.external_customer_id,
                "name": c.name or "N/A",
                "phone": c.phone or "N/A",
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "call_count": len(c.calls),
            }
            for c in customers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/customers")
def create_customer(
    external_customer_id: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new customer in MySQL database."""
    try:
        ext_id = external_customer_id.strip()
        existing = db.query(Customer).filter(Customer.external_customer_id == ext_id).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Customer with ID '{ext_id}' already exists",
            )

        customer = conversation_service.get_or_create_customer(
            db,
            external_customer_id=ext_id,
            name=name.strip(),
            phone=phone.strip(),
        )
        return {
            "id": customer.id,
            "external_customer_id": customer.external_customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "created_at": customer.created_at.isoformat() if customer.created_at else None,
            "call_count": len(customer.calls),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/customers/{customer_id}")
def update_customer(
    customer_id: int,
    external_customer_id: str = Form(...),
    name: str = Form(...),
    phone: str = Form(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update existing customer details in MySQL database."""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        ext_id = external_customer_id.strip()
        if ext_id != customer.external_customer_id:
            existing = db.query(Customer).filter(Customer.external_customer_id == ext_id).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail=f"External customer ID '{ext_id}' is already assigned to another customer.",
                )
            customer.external_customer_id = ext_id

        customer.name = name.strip()
        customer.phone = phone.strip()
        db.commit()
        db.refresh(customer)

        return {
            "id": customer.id,
            "external_customer_id": customer.external_customer_id,
            "name": customer.name,
            "phone": customer.phone,
            "created_at": customer.created_at.isoformat() if customer.created_at else None,
            "call_count": len(customer.calls),
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/customers/{customer_id}")
def delete_customer(
    customer_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a customer and all associated records from MySQL database."""
    try:
        customer = db.query(Customer).filter(Customer.id == customer_id).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        db.delete(customer)
        db.commit()
        return {"status": "deleted", "customer_id": customer_id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/customers/{customer_id}/calls")
def get_customer_calls(
    customer_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve call history for a specific customer."""
    try:
        calls = conversation_service.get_call_history(db, customer_id)
        return [
            {
                "id": c.id,
                "status": c.status,
                "call_type": c.call_type,
                "language": c.language,
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                "duration_seconds": round((c.ended_at - c.started_at).total_seconds()) if c.ended_at and c.started_at else None,
                "message_count": len(c.messages),
                "messages": [
                    {
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                    for msg in c.messages
                ],
            }
            for c in calls
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Calls & Call History Endpoints (Authenticated)
# ---------------------------------------------------------------------------

@router.get("/calls")
def get_all_calls(
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve all call history sessions ordered chronologically with customer details."""
    try:
        calls = (
            db.query(Call)
            .options(joinedload(Call.customer), joinedload(Call.messages), joinedload(Call.requirement))
            .order_by(Call.started_at.desc())
            .all()
        )
        result = []
        for c in calls:
            cust = c.customer
            duration = round((c.ended_at - c.started_at).total_seconds()) if c.ended_at and c.started_at else None
            result.append({
                "id": c.id,
                "customer_id": cust.id if cust else None,
                "customer_name": cust.name if cust else "Unknown",
                "customer_phone": cust.phone if cust else "N/A",
                "external_customer_id": cust.external_customer_id if cust else "N/A",
                "status": c.status,
                "call_type": c.call_type,
                "language": c.language or "en",
                "started_at": c.started_at.isoformat() if c.started_at else None,
                "ended_at": c.ended_at.isoformat() if c.ended_at else None,
                "duration_seconds": duration,
                "message_count": len(c.messages),
                "has_requirements": c.requirement is not None,
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def handle_dtmf_routing(db: Session, call: Call, digit_or_lang: str) -> dict:
    """Core logic for DTMF / keypad / language routing."""
    choice = str(digit_or_lang or "").strip().lower()

    if choice in ["2", "hi", "hindi"]:
        call.language = "Hindi"
        call.status = "ESCALATED"
        call.ended_at = datetime.now(timezone.utc)

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 2 - Hindi]",
        )
        # 2. Record native Hindi notice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=HINDI_ASSISTANCE_MESSAGE,
        )
        # 3. Create Human Assistance request
        EscalationService.create_escalation(
            db=db,
            call_id=call.id,
            reason="Customer selected Hindi",
            language_selected="Hindi",
            status="waiting",
            notes="Customer selected Hindi via language selection prompt. Routed to Human Assistance queue.",
        )
        db.commit()

        # Ensure native audio file exists on disk
        hi_path = os.path.join(RECORDINGS_DIR, "hi_notice.wav")
        if not os.path.exists(hi_path):
            try:
                voice_service.tts_service.synthesize(HINDI_ASSISTANCE_MESSAGE, hi_path, language="hi")
            except Exception as e:
                print(f"Warning generating Hindi audio: {e}")

        return {
            "call_id": call.id,
            "language": "Hindi",
            "status": "ESCALATED",
            "is_escalated": True,
            "escalation_reason": "Customer selected Hindi",
            "audio_url": "/audio/hi_notice.wav",
            "response": HINDI_ASSISTANCE_MESSAGE,
        }

    elif choice in ["3", "mr", "marathi"]:
        call.language = "Marathi"
        call.status = "ESCALATED"
        call.ended_at = datetime.now(timezone.utc)

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 3 - Marathi]",
        )
        # 2. Record native Marathi notice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=MARATHI_ASSISTANCE_MESSAGE,
        )
        # 3. Create Human Assistance request
        EscalationService.create_escalation(
            db=db,
            call_id=call.id,
            reason="Customer selected Marathi",
            language_selected="Marathi",
            status="waiting",
            notes="Customer selected Marathi via language selection prompt. Routed to Human Assistance queue.",
        )
        db.commit()

        # Ensure native audio file exists on disk
        mr_path = os.path.join(RECORDINGS_DIR, "mr_notice.wav")
        if not os.path.exists(mr_path):
            try:
                voice_service.tts_service.synthesize(MARATHI_ASSISTANCE_MESSAGE, mr_path, language="mr")
            except Exception as e:
                print(f"Warning generating Marathi audio: {e}")

        return {
            "call_id": call.id,
            "language": "Marathi",
            "status": "ESCALATED",
            "is_escalated": True,
            "escalation_reason": "Customer selected Marathi",
            "audio_url": "/audio/mr_notice.wav",
            "response": MARATHI_ASSISTANCE_MESSAGE,
        }

    else:
        # Option 1: English
        call.language = "en"
        call.status = "IN_PROGRESS"

        greeting_text = voice_service.conversation_flow_service.get_initial_greeting_and_question(db, call)

        # 1. Record customer DTMF choice in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="customer",
            text="[DTMF Keypad: 1 - English]",
        )
        # 2. Record assistant greeting in messages
        conversation_service.add_message(
            db,
            call_id=call.id,
            speaker="assistant",
            text=greeting_text,
        )
        db.commit()

        # Generate English greeting audio
        en_path = os.path.join(RECORDINGS_DIR, "en_greeting.wav")
        try:
            voice_service.tts_service.synthesize(greeting_text, en_path, language="en")
        except Exception as e:
            print(f"Warning generating English audio: {e}")

        conv_state = voice_service.conversation_flow_service.get_or_create_state(db, call.id)

        return {
            "call_id": call.id,
            "language": "en",
            "status": "IN_PROGRESS",
            "is_escalated": False,
            "escalation_reason": None,
            "audio_url": "/audio/en_greeting.wav",
            "response": greeting_text,
            "conversation_state": {
                "current_question_id": conv_state.current_question_id,
                "questions_asked": conv_state.questions_asked or [],
                "answers": conv_state.answers or {},
                "missing_info": conv_state.missing_info or [],
                "conversation_status": conv_state.conversation_status,
            },
        }


@router.post("/calls")
def create_call(
    external_customer_id: str = Form(...),
    language: str = Form("prompt"),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new Call session for a customer.
    By default (or language='prompt'/'pending'), the call starts by asking the customer to select a language:
    Press 1 for English.
    Press 2 for Hindi.
    Press 3 for Marathi.
    """
    try:
        customer = conversation_service.get_or_create_customer(db, external_customer_id=external_customer_id)
        call = conversation_service.create_call(db, customer.id)

        lang_input = str(language or "").strip().lower()

        if lang_input in ["prompt", "pending", "none", ""]:
            # System asks customer to choose language
            call.status = "AWAITING_LANGUAGE"
            call.language = "pending"
            db.commit()

            # Record IVR prompt in MySQL messages table
            conversation_service.add_message(
                db,
                call_id=call.id,
                speaker="assistant",
                text=IVR_PROMPT_MESSAGE,
            )

            return {
                "call_id": call.id,
                "customer_id": customer.id,
                "external_customer_id": customer.external_customer_id,
                "name": customer.name or "N/A",
                "phone": customer.phone or "N/A",
                "language": "pending",
                "status": "AWAITING_LANGUAGE",
                "is_escalated": False,
                "escalation_reason": None,
                "audio_url": "/audio/ivr_prompt.wav",
                "response": IVR_PROMPT_MESSAGE,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": None,
            }
        else:
            # Caller already provided language input (1, 2, or 3)
            routing = handle_dtmf_routing(db, call, lang_input)
            return {
                "call_id": call.id,
                "customer_id": customer.id,
                "external_customer_id": customer.external_customer_id,
                "name": customer.name or "N/A",
                "phone": customer.phone or "N/A",
                "language": routing["language"],
                "status": routing["status"],
                "is_escalated": routing["is_escalated"],
                "escalation_reason": routing.get("escalation_reason"),
                "audio_url": routing["audio_url"],
                "response": routing["response"],
                "conversation_state": routing.get("conversation_state"),
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            }
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calls/{call_id}/dtmf")
@router.post("/calls/{call_id}/select-language")
async def process_dtmf_or_language_selection(
    call_id: int,
    request: Request,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Unified DTMF / Language Selection endpoint.
    Option 1 -> English AI conversation continues.
    Option 2 -> Hindi voice notice played and customer queued to Human Assistance.
    Option 3 -> Marathi voice notice played and customer queued to Human Assistance.
    """
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

    # Read digit/key/language from form or json
    digit_val = None
    try:
        form = await request.form()
        digit_val = form.get("digit") or form.get("key") or form.get("language")
    except Exception:
        pass

    if not digit_val:
        try:
            body = await request.json()
            if isinstance(body, dict):
                digit_val = body.get("digit") or body.get("key") or body.get("language")
        except Exception:
            pass

    if not digit_val:
        digit_val = "1"

    return handle_dtmf_routing(db, call, str(digit_val))


@router.get("/calls/{call_id}")
async def get_call_details(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve comprehensive details of a specific call session:
    Customer Information -> Call Information -> Full Transcript -> Requirements Collected.
    """
    try:
        call = (
            db.query(Call)
            .options(
                joinedload(Call.customer),
                joinedload(Call.messages),
                joinedload(Call.requirement),
                joinedload(Call.escalations),
                joinedload(Call.conversation_state),
            )
            .filter(Call.id == call_id)
            .first()
        )
        if not call:
            raise HTTPException(status_code=404, detail="Call session not found")

        customer = call.customer
        sorted_messages = sorted(
            call.messages,
            key=lambda m: (m.timestamp or m.id, m.id or 0),
        )

        # Retrieve requirements
        requirements_data = None
        if call.requirement:
            requirements_data = {
                "id": call.requirement.id,
                "status": call.requirement.status,
                "structured": call.requirement.structured_requirements,
                "confirmed_at": call.requirement.confirmed_at.isoformat() if call.requirement.confirmed_at else None,
            }
        elif len(sorted_messages) > 0:
            # If transcript exists but requirements not yet saved, extract and persist now
            full_text = "\n".join(m.text for m in sorted_messages if m.speaker == "customer")
            if full_text.strip():
                try:
                    await voice_service._update_requirements(db, call.id, full_text, is_resolved=True)
                    db.refresh(call)
                    if call.requirement:
                        requirements_data = {
                            "id": call.requirement.id,
                            "status": call.requirement.status,
                            "structured": call.requirement.structured_requirements,
                            "confirmed_at": None,
                        }
                except Exception as ex:
                    print(f"Error in on-demand requirement extraction: {ex}")

        duration = (
            round((call.ended_at - call.started_at).total_seconds())
            if call.ended_at and call.started_at
            else None
        )

        conv_state_data = None
        if call.conversation_state:
            conv_state_data = {
                "current_question_id": call.conversation_state.current_question_id,
                "current_stage": call.conversation_state.current_stage,
                "questions_asked": call.conversation_state.questions_asked or [],
                "answers": call.conversation_state.answers or {},
                "missing_info": call.conversation_state.missing_info or [],
                "conversation_status": call.conversation_state.conversation_status,
                "is_processing": bool(call.conversation_state.is_processing),
            }

        summary_data = None
        if call.requirement and call.requirement.structured_requirements:
            summary_data = call.requirement.structured_requirements.get("conversation_summary")

        return {
            "call_id": call.id,
            "customer": {
                "id": customer.id if customer else None,
                "external_customer_id": customer.external_customer_id if customer else "N/A",
                "name": customer.name if customer else "Unknown",
                "phone": customer.phone if customer else "N/A",
                "created_at": customer.created_at.isoformat() if customer and customer.created_at else None,
            },
            "call": {
                "id": call.id,
                "status": call.status,
                "call_type": call.call_type,
                "language": call.language or "en",
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
                "duration_seconds": duration,
                "is_escalated": len(call.escalations) > 0 or call.status == "ESCALATED",
            },
            "transcript": [
                {
                    "id": msg.id,
                    "speaker": msg.speaker,
                    "text": msg.text,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in sorted_messages
            ],
            "requirements": requirements_data,
            "summary": summary_data,
            "conversation_state": conv_state_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calls/{call_id}/summary")
async def get_or_generate_call_summary(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the conversation summary for a call, generating it on-demand if not already generated.
    Avoids duplicate generation for already summarized calls.
    """
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call session not found")

    if call.requirement and call.requirement.structured_requirements:
        existing_sum = call.requirement.structured_requirements.get("conversation_summary")
        if existing_sum:
            return existing_sum

    # Generate summary on demand
    msgs = db.query(Message).filter(Message.call_id == call_id).order_by(Message.id).all()
    cust_text = "\n".join(m.text for m in msgs if m.speaker == "customer")
    req_struct = (call.requirement.structured_requirements if call.requirement else {}) or {}
    unanswered = (call.conversation_state.missing_info if call.conversation_state else []) or []

    summary = await requirement_service.generate_summary(
        call_id=call_id,
        transcript=cust_text,
        structured_req=req_struct,
        unanswered_questions=unanswered,
    )

    if call.requirement:
        updated = dict(call.requirement.structured_requirements or {})
        updated["conversation_summary"] = summary
        call.requirement.structured_requirements = updated
        db.commit()

    return summary


@router.post("/calls/{call_id}/turn")
async def add_turn(
    call_id: int,
    request: Request,
    file: UploadFile = File(...),
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Process a voice turn for an existing call session:
    1. If call is AWAITING_LANGUAGE, transcribes speech to route language.
    2. If call is IN_PROGRESS, transcribes speech, generates LLM response, updates requirements, and returns TTS audio.
    """
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))

    call_status_upper = (call.status or "").upper()
    terminal_statuses = {"RESOLVED", "COMPLETED", "ESCALATED", "FAILED", "CANCELED", "CANCELLED", "BUSY", "NO_ANSWER", "HUMAN_TRANSFERRED"}
    if call.ended_at is not None or call_status_upper in terminal_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Call with ID {call_id} has already ended or has been routed to human assistance.",
        )

    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".wav"
    allowed_exts = {".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".aac"}
    if ext not in allowed_exts and not (file.content_type and "audio" in file.content_type):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Please record or upload a valid audio file (.wav, .webm, .mp3, .ogg).",
        )

    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(RECORDINGS_DIR, temp_filename)
    output_filename = f"{uuid.uuid4()}_response.wav"
    output_path = os.path.join(GENERATED_AUDIO_DIR, output_filename)

    converted_path = None
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(temp_path)
        if file_size < 100:
            raise HTTPException(
                status_code=400,
                detail="The recorded audio was empty or silent. Please speak into the microphone and try again.",
            )
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Audio file size exceeds the 25MB limit.",
            )

        # Standardize audio container/sample rate to 16kHz mono WAV for Faster-Whisper
        converted_path = convert_to_16k_mono_wav(temp_path)
        active_audio_path = converted_path if (converted_path and os.path.exists(converted_path)) else temp_path

        # If call is awaiting language selection, handle speech-based language selection
        if call.status == "AWAITING_LANGUAGE":
            transcript = voice_service.stt_service.transcribe(active_audio_path)
            if not transcript or not transcript.strip():
                transcript = "[No clear speech detected]"
            t_lower = transcript.lower()

            if any(k in t_lower for k in ["2", "hindi", "हिन्दी"]):
                routing = handle_dtmf_routing(db, call, "2")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": True,
                }
            elif any(k in t_lower for k in ["3", "marathi", "मराठी"]):
                routing = handle_dtmf_routing(db, call, "3")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": True,
                }
            elif any(k in t_lower for k in ["1", "english", "one"]):
                routing = handle_dtmf_routing(db, call, "1")
                return {
                    "transcript": transcript,
                    "response": routing["response"],
                    "audio_url": routing["audio_url"],
                    "call_id": call.id,
                    "call_status": routing["status"],
                    "is_escalated": False,
                }
            else:
                # Repeat IVR prompt if unrecognized
                return {
                    "transcript": transcript,
                    "response": IVR_PROMPT_MESSAGE,
                    "audio_url": "/audio/ivr_prompt.wav",
                    "call_id": call.id,
                    "call_status": "AWAITING_LANGUAGE",
                    "is_escalated": False,
                }

        # Normal English AI Conversation Pipeline Turn
        try:
            result = await voice_service.process_existing_turn(
                call_id=call_id,
                audio_path=active_audio_path,
                output_audio_path=output_path,
                db=db,
            )
        except ValueError as ve:
            err_msg = str(ve)
            if "already ended" in err_msg:
                raise HTTPException(status_code=400, detail=err_msg)
            elif "already being processed" in err_msg:
                raise HTTPException(status_code=409, detail=err_msg)
            else:
                raise HTTPException(status_code=400, detail=err_msg)

        filename = os.path.basename(output_path)
        audio_url = request.url_for("audio_files", path=filename)
        result["audio_url"] = str(audio_url)
        if "audio_path" in result:
            del result["audio_path"]
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (temp_path, converted_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@router.post("/calls/{call_id}/end")
async def end_call_endpoint(
    call_id: int,
    current_user: SystemUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark an existing call as ended and guarantee requirements are extracted and committed."""
    try:
        call = conversation_service.get_call(db, call_id)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    call_status_upper = (call.status or "").upper()
    terminal_statuses = {"RESOLVED", "COMPLETED", "ESCALATED", "FAILED", "CANCELED", "CANCELLED", "BUSY", "NO_ANSWER", "HUMAN_TRANSFERRED"}
    if call.ended_at is not None or call_status_upper in terminal_statuses:
        return {"status": "already_ended", "call_id": call_id}

    try:
        # Guarantee requirements extraction on call completion if customer spoke
        customer_msgs = (
            db.query(Message)
            .filter(Message.call_id == call_id, Message.speaker == "customer")
            .order_by(Message.id)
            .all()
        )
        full_text = "\n".join(m.text for m in customer_msgs if m.text)
        if full_text.strip():
            try:
                await voice_service._update_requirements(db, call_id, full_text, is_resolved=True)
            except Exception as e:
                print(f"Error finalizing requirements on end_call: {e}")

        conversation_service.end_call(db, call_id)
        return {"status": "ended", "call_id": call_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-questions")
def get_questions_endpoint(
    current_user: SystemUser = Depends(get_current_user),
):
    """Retrieve the configurable customer qualification question sequence."""
    from app.config.conversation_questions import get_configured_questions
    return get_configured_questions()


@router.post("/conversation-questions")
async def update_questions_endpoint(
    request: Request,
    current_user: SystemUser = Depends(get_current_user),
):
    """Update the configurable customer qualification question sequence."""
    from app.config.conversation_questions import set_configured_questions
    try:
        body = await request.json()
        if not isinstance(body, list):
            raise HTTPException(status_code=400, detail="Expected a list of question definitions.")
        set_configured_questions(body, persist=True)
        return {"status": "success", "questions": body}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

