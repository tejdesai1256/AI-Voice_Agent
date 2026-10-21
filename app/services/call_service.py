from app.services.stt_service import STTService
from app.services.requirement_service import RequirementService


class CallService:
    """
    Lightweight STT-and-Requirement extraction pipeline.
    NOTE: The full multi-turn voice calling pipeline is orchestrated by VoiceResponseService
    (app/services/voice_response_service.py) which adds LLM generation, TTS synthesis,
    DTMF language routing, and persistence.
    """
    def __init__(
        self,
        stt_service: STTService = None,
        requirement_service: RequirementService = None,
    ):
        self.stt_service = stt_service if stt_service is not None else STTService()
        self.requirement_service = (
            requirement_service if requirement_service is not None else RequirementService()
        )

    async def process_call(self, audio_path: str) -> dict:
        """
        Processes an incoming call audio file:
        1. Transcribes the audio using STTService.
        2. Extracts requirements from the transcript using RequirementService.
        3. Returns a dictionary with transcript and extracted requirements.
        """
        transcript = self.stt_service.transcribe(audio_path)
        requirements = await self.requirement_service.extract_requirements(transcript)

        return {
            "transcript": transcript,
            "requirements": requirements,
        }
