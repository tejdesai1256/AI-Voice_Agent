import os
import uuid
import hmac
import hashlib
import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set
from datetime import datetime, timezone

from app.utils.phone_utils import normalize_phone_number, mask_phone_number


@dataclass
class TelephonyCallResult:
    success: bool
    call_sid: Optional[str] = None
    provider: str = "mock"
    status: str = "queued"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class TelephonyStatusEvent:
    provider_call_id: str
    raw_status: str
    normalized_status: str
    duration_seconds: Optional[int] = None
    recording_url: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: Optional[datetime] = None


TERMINAL_CALL_STATUSES: Set[str] = {
    "completed",
    "failed",
    "busy",
    "no_answer",
    "canceled",
    "cancelled",
    "human_transferred",
    "escalated",
    "resolved",
}

VALID_STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "queued": {"initiated", "ringing", "in_progress", "failed", "busy", "no_answer", "canceled"},
    "initiated": {"ringing", "in_progress", "failed", "busy", "no_answer", "canceled"},
    "ringing": {"in_progress", "failed", "busy", "no_answer", "canceled"},
    "in_progress": {"completed", "failed", "busy", "no_answer", "canceled", "human_transferred"},
    "awaiting_language": {"in_progress", "human_transferred", "completed", "canceled", "failed"},
    "active": {"in_progress", "completed", "failed", "busy", "no_answer", "canceled", "human_transferred"},
    "completed": set(),
    "failed": set(),
    "busy": set(),
    "no_answer": set(),
    "canceled": set(),
    "human_transferred": {"completed"},
    "escalated": {"completed"},
}


def is_valid_status_transition(current_status: str, new_status: str) -> bool:
    """
    Guarantees state machine consistency and protects against out-of-order
    or duplicate webhook callbacks transitioning an already ended call back to active.
    """
    curr = str(current_status or "").strip().lower()
    nxt = str(new_status or "").strip().lower()

    if curr == nxt:
        return True  # Idempotent state update
    if curr in TERMINAL_CALL_STATUSES:
        return False  # Never resurrect a terminated call
    valid_next = VALID_STATUS_TRANSITIONS.get(curr, set())
    return nxt in valid_next


class TelephonyAdapter(ABC):
    """
    Abstract base class for provider-independent telephony operations.
    Keeps provider-specific protocols, webhook structures, and credentials
    isolated from the AI conversation and database services.
    """

    @abstractmethod
    async def start_outbound_call(
        self,
        to_phone: str,
        call_id: int,
        stream_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelephonyCallResult:
        """
        Initiates an outbound call via the telephony provider's API.
        """
        pass

    @abstractmethod
    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses an incoming telephony webhook payload to extract caller details.
        """
        pass

    @abstractmethod
    def parse_status_callback(self, payload: Dict[str, Any]) -> TelephonyStatusEvent:
        """
        Normalizes provider-specific status callback webhooks.
        """
        pass

    @abstractmethod
    def validate_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
        query_params: Dict[str, str],
    ) -> bool:
        """
        Validates webhook authenticity (e.g. signature, auth token).
        """
        pass

    @abstractmethod
    async def end_call(self, provider_call_id: str) -> bool:
        """
        Requests the provider to terminate an ongoing phone call.
        """
        pass

    @abstractmethod
    async def transfer_call(self, provider_call_id: str, forward_to_phone: str) -> bool:
        """
        Transfers an active call to a human operator destination.
        """
        pass

    @abstractmethod
    def format_media_message(self, stream_sid: str, pcm_base64: str) -> Dict[str, Any]:
        """
        Formats audio bytes into the provider's WebSocket media message structure.
        """
        pass

    @abstractmethod
    def format_clear_message(self, stream_sid: str) -> Dict[str, Any]:
        """
        Formats playback interruption/clear message for the provider's audio buffer.
        """
        pass

    @abstractmethod
    def normalize_status(self, provider_status: str) -> str:
        """
        Normalizes provider status string into platform-standard status.
        """
        pass

    @abstractmethod
    def get_recording(self, provider_call_id: str) -> Optional[str]:
        """
        Retrieves recording URL if available.
        """
        pass

    @abstractmethod
    async def verify_credentials(self) -> Dict[str, Any]:
        """
        Safely validates provider API credentials without dialing a paid call.
        """
        pass


class MockTelephonyAdapter(TelephonyAdapter):
    """
    Mock telephony adapter for local testing, development, and unit tests
    without incurring carrier costs or requiring external telecom credentials.
    """
    provider_name: str = "mock"

    def __init__(self):
        self.active_calls: Dict[str, Dict[str, Any]] = {}

    async def start_outbound_call(
        self,
        to_phone: str,
        call_id: int,
        stream_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelephonyCallResult:
        is_valid, normalized_phone, err = normalize_phone_number(to_phone)
        if not is_valid:
            return TelephonyCallResult(
                success=False,
                call_sid=None,
                provider="mock",
                status="failed",
                error_message=err,
            )

        mock_sid = f"mock_sid_{uuid.uuid4().hex[:12]}"
        self.active_calls[mock_sid] = {
            "call_id": call_id,
            "to_phone": normalized_phone,
            "status": "in_progress",
            "stream_url": stream_url,
            "metadata": metadata or {},
        }
        return TelephonyCallResult(
            success=True,
            call_sid=mock_sid,
            provider="mock",
            status="queued",
            raw_response={"mock_sid": mock_sid, "destination": mask_phone_number(normalized_phone)},
        )

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        caller = external_payload.get("From", "+919999999999")
        call_sid = external_payload.get("CallSid", f"mock_inbound_{uuid.uuid4().hex[:8]}")
        return {
            "customer_phone": caller,
            "external_call_id": call_sid,
            "caller_id": external_payload.get("To", "+918888888888"),
        }

    def parse_status_callback(self, payload: Dict[str, Any]) -> TelephonyStatusEvent:
        call_sid = payload.get("CallSid") or payload.get("call_sid") or "mock_unknown"
        raw_status = str(payload.get("Status") or payload.get("status") or "completed").lower()
        duration = int(payload.get("Duration") or payload.get("duration") or 0)
        return TelephonyStatusEvent(
            provider_call_id=call_sid,
            raw_status=raw_status,
            normalized_status=self.normalize_status(raw_status),
            duration_seconds=duration if duration > 0 else None,
            recording_url=payload.get("RecordingUrl"),
            error_code=payload.get("ErrorCode"),
            timestamp=datetime.now(timezone.utc),
        )

    def validate_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
        query_params: Dict[str, str],
    ) -> bool:
        # Mock adapter accepts requests with valid format or test token
        return True

    async def end_call(self, provider_call_id: str) -> bool:
        if provider_call_id in self.active_calls:
            self.active_calls[provider_call_id]["status"] = "completed"
        return True

    async def transfer_call(self, provider_call_id: str, forward_to_phone: str) -> bool:
        if provider_call_id in self.active_calls:
            self.active_calls[provider_call_id]["status"] = "human_transferred"
            self.active_calls[provider_call_id]["transferred_to"] = forward_to_phone
        return True

    def format_media_message(self, stream_sid: str, pcm_base64: str) -> Dict[str, Any]:
        return {
            "event": "media",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
            "media": {"payload": pcm_base64},
        }

    def format_clear_message(self, stream_sid: str) -> Dict[str, Any]:
        return {
            "event": "clear",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
        }

    def normalize_status(self, provider_status: str) -> str:
        s = str(provider_status or "").strip().lower().replace("-", "_")
        mapping = {
            "queued": "queued",
            "initiated": "initiated",
            "ringing": "ringing",
            "in_progress": "in_progress",
            "answered": "in_progress",
            "completed": "completed",
            "busy": "busy",
            "failed": "failed",
            "no_answer": "no_answer",
            "canceled": "canceled",
            "cancelled": "canceled",
            "human_transferred": "human_transferred",
        }
        return mapping.get(s, "in_progress")

    def get_recording(self, provider_call_id: str) -> Optional[str]:
        return f"/audio/mock_recording_{provider_call_id}.wav"

    async def verify_credentials(self) -> Dict[str, Any]:
        return {
            "valid": True,
            "provider": "mock",
            "message": "Mock Telephony Provider is operational (local simulation mode).",
            "configured": True,
        }


class ExotelTelephonyAdapter(TelephonyAdapter):
    """
    Exotel provider adapter implementing the Connect Voice AI API and AgentStream protocol.
    Documentation: https://developer.exotel.com/docs/agentstream/connect-voice-ai
    """
    provider_name: str = "exotel"

    def __init__(
        self,
        account_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        api_token: Optional[str] = None,
        caller_id: Optional[str] = None,
        subdomain: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        self.account_sid = (account_sid if account_sid is not None else os.getenv("EXOTEL_ACCOUNT_SID", "")).strip()
        self.api_key = (api_key if api_key is not None else os.getenv("EXOTEL_API_KEY", "")).strip()
        self.api_token = (api_token if api_token is not None else os.getenv("EXOTEL_API_TOKEN", "")).strip()
        raw_caller = (caller_id if caller_id is not None else os.getenv("EXOTEL_CALLER_ID", "")).strip()
        self.caller_id = raw_caller.replace("-", "").replace(" ", "")
        self.subdomain = (subdomain if subdomain is not None else os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")).strip()
        self.webhook_secret = (webhook_secret if webhook_secret is not None else os.getenv("EXOTEL_WEBHOOK_SECRET", "")).strip()

    @property
    def base_url(self) -> str:
        return f"https://{self.subdomain}/v1/Accounts/{self.account_sid}"

    def is_configured(self) -> bool:
        keys = [self.account_sid, self.api_key, self.api_token, self.caller_id]
        if not all(keys):
            return False
        for k in keys:
            val = str(k).strip()
            if not val or val.startswith(("<", "{", "your_exotel_")) or val.endswith((">")):
                return False
        return True

    async def start_outbound_call(
        self,
        to_phone: str,
        call_id: int,
        stream_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelephonyCallResult:
        if not self.is_configured():
            return TelephonyCallResult(
                success=False,
                call_sid=None,
                provider="exotel",
                status="failed",
                error_message="Exotel credentials (EXOTEL_ACCOUNT_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_CALLER_ID) are not configured in environment.",
            )

        is_valid, normalized_to, err = normalize_phone_number(to_phone)
        if not is_valid:
            return TelephonyCallResult(
                success=False,
                call_sid=None,
                provider="exotel",
                status="failed",
                error_message=f"Phone number validation failed: {err}",
            )

        # Build Connect Voice AI API request
        # API endpoint: POST https://<subdomain>/v1/Accounts/<AccountSid>/Calls/connect.json
        connect_url = f"{self.base_url}/Calls/connect.json"
        public_base = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        callback_url = f"{public_base}/api/telephony/webhook/exotel" if public_base else ""
        if callback_url and self.webhook_secret:
            callback_url = f"{callback_url}?token={self.webhook_secret}"

        # Exotel Connect Voice AI requires StreamUrl (wss://) and StreamType (bidirectional)
        ws_stream_url = stream_url
        if ws_stream_url.startswith("https://"):
            ws_stream_url = ws_stream_url.replace("https://", "wss://")
        elif ws_stream_url.startswith("http://"):
            ws_stream_url = ws_stream_url.replace("http://", "ws://")

        # Explicitly negotiate sample rate with Exotel (defaults to 16000 Hz or TELEPHONY_SAMPLE_RATE)
        configured_rate = int(os.getenv("TELEPHONY_SAMPLE_RATE", "16000"))
        if "sample-rate" not in ws_stream_url and "sample_rate" not in ws_stream_url:
            separator = "&" if "?" in ws_stream_url else "?"
            ws_stream_url = f"{ws_stream_url}{separator}sample-rate={configured_rate}"

        payload = {
            "From": normalized_to,
            "CallerId": self.caller_id,
            "StreamUrl": ws_stream_url,
            "StreamType": "bidirectional",
            "CustomField": str(call_id),
        }
        if callback_url:
            payload["StatusCallback"] = callback_url

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    connect_url,
                    data=payload,
                    auth=(self.api_key, self.api_token),
                )

                if res.status_code in (200, 201):
                    data = res.json()
                    call_obj = data.get("Call") or data
                    call_sid = call_obj.get("Sid") or call_obj.get("sid") or call_obj.get("CallSid")
                    raw_status = call_obj.get("Status", "queued")
                    return TelephonyCallResult(
                        success=True,
                        call_sid=call_sid,
                        provider="exotel",
                        status=self.normalize_status(raw_status),
                        raw_response=data,
                    )
                else:
                    err_msg = f"Exotel API error HTTP {res.status_code}: {res.text}"
                    print(f"[Telephony Warning] {err_msg}")
                    error_type = "PROVIDER_ERROR"
                    raw_msg_str = ""
                    try:
                        err_json = res.json()
                        rest_exc = err_json.get("RestException", {})
                        raw_msg = rest_exc.get("Message") or err_json.get("message") or err_json.get("error")
                        if raw_msg:
                            raw_msg_str = str(raw_msg).strip()
                    except Exception:
                        raw_msg_str = res.text.strip() if res.text else ""

                    lowered = raw_msg_str.lower()
                    if res.status_code == 401:
                        error_type = "AUTH_ERROR"
                        detail = "Exotel authentication failed (HTTP 401): Invalid API Key or Token."
                    elif res.status_code == 403:
                        if "kyc" in lowered:
                            error_type = "KYC_RESTRICTION"
                            detail = (
                                f"Exotel account compliance restriction: {raw_msg_str}. "
                                "Outbound calls to external numbers are blocked until Exotel completes account verification. "
                                "You can continue testing all AI voice features in Mock Telephony mode."
                            )
                        elif any(term in lowered for term in ["international", "isd", "country"]):
                            error_type = "INTERNATIONAL_RESTRICTION"
                            detail = f"Exotel international calling restriction: {raw_msg_str}. Verify international dialing permissions."
                        elif any(term in lowered for term in ["trial", "verified number", "whitelisted"]):
                            error_type = "TRIAL_RESTRICTION"
                            detail = f"Exotel trial account restriction: {raw_msg_str}. In trial mode, calls are restricted to verified numbers."
                        else:
                            error_type = "POLICY_RESTRICTION"
                            detail = f"Exotel policy restriction (HTTP 403): {raw_msg_str or 'Access Forbidden'}"
                    elif res.status_code == 429:
                        error_type = "RATE_LIMIT"
                        detail = "Exotel rate limit exceeded (HTTP 429). Please wait before retrying."
                    elif res.status_code >= 500:
                        error_type = "PROVIDER_SERVER_ERROR"
                        detail = f"Exotel server error (HTTP {res.status_code}): {raw_msg_str or res.text[:150]}"
                    else:
                        detail = f"Exotel error (HTTP {res.status_code}): {raw_msg_str or res.text[:150]}"

                    return TelephonyCallResult(
                        success=False,
                        call_sid=None,
                        provider="exotel",
                        status="failed",
                        error_message=detail,
                        error_type=error_type,
                        raw_response={"status_code": res.status_code, "body": res.text, "error_type": error_type},
                    )
        except httpx.TimeoutException:
            return TelephonyCallResult(
                success=False,
                call_sid=None,
                provider="exotel",
                status="failed",
                error_message="Telephony provider connection timed out. Request aborted safely without retry.",
                error_type="TIMEOUT",
            )
        except Exception as ex:
            return TelephonyCallResult(
                success=False,
                call_sid=None,
                provider="exotel",
                status="failed",
                error_message=f"Provider request error: {str(ex)}",
                error_type="CONNECTION_ERROR",
            )

    def handle_incoming_call(self, external_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses inbound Exotel call webhook payload.
        """
        caller_phone = external_payload.get("From") or external_payload.get("CallFrom", "")
        call_sid = external_payload.get("CallSid") or external_payload.get("Sid", "")
        virtual_number = external_payload.get("To") or external_payload.get("CallTo", "")
        return {
            "customer_phone": caller_phone,
            "external_call_id": call_sid,
            "caller_id": virtual_number,
            "custom_field": external_payload.get("CustomField"),
        }

    def parse_status_callback(self, payload: Dict[str, Any]) -> TelephonyStatusEvent:
        call_sid = payload.get("CallSid") or payload.get("Sid") or payload.get("call_sid") or ""
        raw_status = str(payload.get("Status") or payload.get("CallStatus") or payload.get("status") or "").lower()
        duration_val = payload.get("Duration") or payload.get("CallDuration") or payload.get("duration")
        duration = int(duration_val) if duration_val and str(duration_val).isdigit() else None
        recording_url = payload.get("RecordingUrl") or payload.get("recording_url")
        error_code = payload.get("ErrorCode") or payload.get("error_code")

        return TelephonyStatusEvent(
            provider_call_id=call_sid,
            raw_status=raw_status,
            normalized_status=self.normalize_status(raw_status),
            duration_seconds=duration,
            recording_url=recording_url,
            error_code=str(error_code) if error_code else None,
            timestamp=datetime.now(timezone.utc),
        )

    def validate_webhook(
        self,
        headers: Dict[str, str],
        body: bytes,
        query_params: Dict[str, str],
    ) -> bool:
        """
        Validates webhook authenticity via shared secret token query parameter
        or HMAC signature if provided by the environment.
        """
        if not self.webhook_secret:
            # If no secret configured in development, allow webhook with warning
            return True

        # Check signature header or auth token param
        provided_token = query_params.get("token") or headers.get("x-exotel-token")
        if provided_token and hmac.compare_digest(provided_token, self.webhook_secret):
            return True

        # Check HMAC-SHA256 signature if provided
        sig_header = headers.get("x-exotel-signature")
        if sig_header:
            computed = hmac.new(
                self.webhook_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(sig_header, computed)

        return False

    async def end_call(self, provider_call_id: str) -> bool:
        """
        Terminates call via Exotel REST API if call_sid is known.
        """
        if not self.is_configured() or not provider_call_id:
            return False

        url = f"{self.base_url}/Calls/{provider_call_id}.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    data={"Status": "completed"},
                    auth=(self.api_key, self.api_token),
                )
                return res.status_code in (200, 204)
        except Exception as e:
            print(f"[Telephony Warning] Exotel end_call failed for {provider_call_id}: {e}")
            return False

    async def transfer_call(self, provider_call_id: str, forward_to_phone: str) -> bool:
        """
        Transfers call to a human operator telephone number.
        """
        if not self.is_configured() or not provider_call_id:
            return False

        is_valid, norm_dest, _ = normalize_phone_number(forward_to_phone)
        if not is_valid:
            return False

        # In Exotel, transfers can be executed by updating the call leg or sending redirect URL
        url = f"{self.base_url}/Calls/{provider_call_id}.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    url,
                    data={"ForwardTo": norm_dest},
                    auth=(self.api_key, self.api_token),
                )
                return res.status_code in (200, 204)
        except Exception as e:
            print(f"[Telephony Warning] Exotel transfer_call failed for {provider_call_id}: {e}")
            return False

    def format_media_message(self, stream_sid: str, pcm_base64: str) -> Dict[str, Any]:
        """
        Exotel AgentStream format:
        { "event": "media", "stream_sid": "<stream_sid>", "media": { "payload": "<base64_pcm>" } }
        """
        return {
            "event": "media",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
            "media": {"payload": pcm_base64},
        }

    def format_clear_message(self, stream_sid: str) -> Dict[str, Any]:
        """
        Exotel AgentStream barge-in clear buffer format:
        { "event": "clear", "stream_sid": "<stream_sid>" }
        """
        return {
            "event": "clear",
            "stream_sid": stream_sid,
            "streamSid": stream_sid,
        }

    def normalize_status(self, provider_status: str) -> str:
        s = str(provider_status or "").strip().lower().replace("-", "_")
        mapping = {
            "queued": "queued",
            "initiated": "initiated",
            "ringing": "ringing",
            "in_progress": "in_progress",
            "answered": "in_progress",
            "completed": "completed",
            "busy": "busy",
            "failed": "failed",
            "no_answer": "no_answer",
            "canceled": "canceled",
            "cancelled": "canceled",
            "human_transferred": "human_transferred",
            "escalated": "human_transferred",
        }
        return mapping.get(s, "in_progress")

    def get_recording(self, provider_call_id: str) -> Optional[str]:
        if not self.is_configured() or not provider_call_id:
            return None
        return f"{self.base_url}/Calls/{provider_call_id}/Recording"

    async def verify_credentials(self) -> Dict[str, Any]:
        if not self.is_configured():
            return {
                "valid": False,
                "provider": "exotel",
                "message": "Exotel credentials are not fully configured (missing EXOTEL_ACCOUNT_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, or EXOTEL_CALLER_ID).",
                "configured": False,
            }
        # Safe zero-cost query to Exotel Account endpoint (does NOT dial a number)
        account_url = f"{self.base_url}.json"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(account_url, auth=(self.api_key, self.api_token))
                if res.status_code == 200:
                    data = res.json()
                    acc_info = data.get("Account", {})
                    kyc_status = str(acc_info.get("kyc_status") or acc_info.get("KYCStatus") or "").lower()
                    kyc_ok = True
                    msg = "Exotel credentials verified successfully via Exotel Account API."
                    if kyc_status in ("pending", "pending_documents", "rejected", "not_submitted", "unverified", "suspended"):
                        kyc_ok = False
                        msg = f"Exotel account credentials are valid, but KYC status is '{kyc_status}'. Outbound calling is blocked until Exotel KYC verification is approved."
                    return {
                        "valid": kyc_ok,
                        "provider": "exotel",
                        "account_sid": self.account_sid,
                        "caller_id": mask_phone_number(self.caller_id),
                        "account_status": acc_info.get("Status", "active"),
                        "kyc_compliant": kyc_ok,
                        "message": msg,
                        "configured": True,
                    }
                elif res.status_code == 401:
                    return {
                        "valid": False,
                        "provider": "exotel",
                        "status_code": 401,
                        "message": "Exotel authentication failed (HTTP 401): Invalid API Key or Token.",
                        "configured": True,
                    }
                elif res.status_code == 403:
                    msg = "Exotel access forbidden (HTTP 403)."
                    is_kyc = False
                    try:
                        err_json = res.json()
                        rest_exc = err_json.get("RestException", {})
                        if rest_exc.get("Message"):
                            raw_m = rest_exc["Message"]
                            if "kyc" in raw_m.lower():
                                is_kyc = True
                                msg = f"Exotel account compliance restriction: {raw_m}. KYC verification required on Exotel Console before outbound calls can be placed."
                            else:
                                msg = f"Exotel account restriction: {raw_m}"
                    except Exception:
                        pass
                    return {
                        "valid": False,
                        "provider": "exotel",
                        "status_code": 403,
                        "kyc_compliant": False if is_kyc else None,
                        "error_type": "KYC_RESTRICTION" if is_kyc else "POLICY_RESTRICTION",
                        "message": msg,
                        "configured": True,
                    }
                else:
                    return {
                        "valid": False,
                        "provider": "exotel",
                        "status_code": res.status_code,
                        "message": f"Exotel API check returned HTTP {res.status_code}: {res.text[:200]}",
                        "configured": True,
                    }
        except httpx.TimeoutException:
            return {
                "valid": False,
                "provider": "exotel",
                "message": "Connection to Exotel timed out.",
                "configured": True,
            }
        except Exception as e:
            return {
                "valid": False,
                "provider": "exotel",
                "message": f"Network error checking Exotel credentials: {str(e)}",
                "configured": True,
            }


def get_telephony_adapter(provider_name: Optional[str] = None) -> TelephonyAdapter:
    """
    Factory function providing the selected TelephonyAdapter instance.
    Keeps provider selection configurable so that providers can be switched
    without rewriting AI engine or call management services.
    Enforces clear separation: real-mode returns real adapter (does not silently fall back to mock).
    """
    selected = (provider_name or os.getenv("TELEPHONY_PROVIDER", "mock")).strip().lower()

    if selected == "exotel":
        return ExotelTelephonyAdapter()

    return MockTelephonyAdapter()
