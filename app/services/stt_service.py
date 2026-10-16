import av

# Fix compatibility between faster-whisper and newer PyAV versions where metadata_errors was removed
_orig_av_open = av.open


def _compat_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _compat_av_open

from faster_whisper import WhisperModel


class STTService:
    def __init__(
        self,
        model_path: str = "models/whisper-base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """
        Initializes the STTService with a local Faster-Whisper model.
        """
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the audio file at the given audio_path using Faster-Whisper.
        Combines all returned segments into one clean transcript string and returns it.
        """
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=1,
            language="en",
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
