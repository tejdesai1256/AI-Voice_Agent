from abc import ABC, abstractmethod
from typing import Optional


class TelephonyAdapter(ABC):
    """
    Abstract base class for a Telephony Provider (e.g., Exotel, Twilio).
    """

    @abstractmethod
    def start_outbound_call(self, customer_id: int, phone_number: str) -> str:
        """
        Initiates an outbound call via the telephony provider.
        Returns the external provider's unique call/session ID.
        """
        pass

    @abstractmethod
    def handle_incoming_call(self, external_payload: dict) -> dict:
        """
        Processes an incoming webhook from the telephony provider.
        Returns extracted relevant details (e.g., customer_phone).
        """
        pass

    @abstractmethod
    def end_call(self, external_call_id: str) -> bool:
        """
        Sends a command to the telephony provider to hang up the call.
        """
        pass

    @abstractmethod
    def get_recording(self, external_call_id: str) -> Optional[str]:
        """
        Retrieves the recording URL from the telephony provider after the call ends.
        """
        pass


class MockTelephonyAdapter(TelephonyAdapter):
    """
    A local mock implementation of the TelephonyAdapter for testing the
    Automated Calling workflow without making real phone calls.
    """

    def start_outbound_call(self, customer_id: int, phone_number: str) -> str:
        # Simulate initiating a call and returning a fake external call ID
        return f"mock_outbound_{customer_id}"

    def handle_incoming_call(self, external_payload: dict) -> dict:
        # Simulate receiving an incoming call
        return {
            "customer_phone": external_payload.get("From", "+910000000000"),
            "external_call_id": f"mock_inbound_{external_payload.get('CallSid', 'default')}"
        }

    def end_call(self, external_call_id: str) -> bool:
        # Simulate hanging up
        return True

    def get_recording(self, external_call_id: str) -> Optional[str]:
        # Simulate returning a recording URL
        return f"http://localhost:8000/audio/mock_recording_{external_call_id}.wav"
