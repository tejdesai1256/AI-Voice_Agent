import os
import re
import av
from typing import Optional

# Fix compatibility between faster-whisper and newer PyAV versions where metadata_errors was removed
_orig_av_open = av.open


def _compat_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _compat_av_open

from faster_whisper import WhisperModel


_shared_whisper_model = None

DEFAULT_INITIAL_PROMPT = (
    "Phone conversation with customer for AI voice calling. "
    "Domain terms: company name, business, IT industry, software development, "
    "solutions, founder, CEO, CTO, director, manager, requirements, budget, "
    "timeline, Bangalore, Mumbai, Pune, Delhi, Hyderabad, India."
)

HALLUCINATION_PATTERNS = [
    r"thank you for watching",
    r"please subscribe",
    r"i'?ll be right back",
    r"i'?m going to look really good",
    r"subtitles by",
    r"transcript by",
    r"watch next video",
    r"see you in the next video",
]


class STTService:
    def __init__(
        self,
        model_path: str = "models/whisper-base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Initializes the STTService with a local Faster-Whisper model.
        Caches model in memory to prevent duplicate native MKL memory pool allocations.
        """
        global _shared_whisper_model
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        if _shared_whisper_model is None:
            _shared_whisper_model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        self.model = _shared_whisper_model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        beam_size: int = 5,
    ) -> str:
        """
        Transcribes the audio file at the given audio_path using Faster-Whisper.
        Configured with beam search, domain context prompt, hallucination filters,
        and soft VAD padding to ensure the beginning and end of caller speech are captured.
        """
        if not audio_path or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
            return ""

        prompt = initial_prompt or DEFAULT_INITIAL_PROMPT
        # Clean language code (e.g. 'en', 'hi', 'mr') or None for auto-detect
        lang = str(language).strip().lower() if language and language not in ("pending", "auto") else None

        vad_params = dict(
            threshold=0.25,
            min_speech_duration_ms=100,
            min_silence_duration_ms=500,
            speech_pad_ms=400,
        )

        try:
            segments, _ = self.model.transcribe(
                audio_path,
                beam_size=beam_size,
                best_of=beam_size,
                temperature=0.0,
                language=lang,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                vad_filter=True,
                vad_parameters=vad_params,
            )
            raw_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
            return self._sanitize_transcript(raw_text)
        except Exception as e:
            print(f"[STT Error] Faster-Whisper transcription failed: {e}")
            return ""

    @staticmethod
    def _sanitize_transcript(text: str) -> str:
        clean = (text or "").strip()
        if not clean or len(clean) < 2:
            return ""

        lowered = clean.lower()
        for pat in HALLUCINATION_PATTERNS:
            if re.search(pat, lowered):
                clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

        # Remove repetitive phrase loops
        words = clean.split()
        if len(words) >= 4 and len(set(words)) <= 2:
            return ""

        return clean.strip()
