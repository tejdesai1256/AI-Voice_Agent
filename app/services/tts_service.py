import os
import wave
import asyncio
from piper.voice import PiperVoice

VOICE_MAP = {
    "en": "en_US-lessac-medium",
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
}

_shared_piper_voice = None


class TTSService:
    def __init__(
        self,
        model_path: str = "models/piper/en_US-lessac-medium.onnx",
        config_path: str = None,
        use_cuda: bool = False,
    ):
        """
        Initializes the TTSService with:
        - English: Local Piper TTS voice model (proven, unchanged, cached in memory)
        - Hindi: hi-IN-SwaraNeural (native Devanagari neural voice)
        - Marathi: mr-IN-AarohiNeural (native Devanagari neural voice)
        """
        global _shared_piper_voice
        self.model_path = model_path
        self.config_path = config_path
        self.use_cuda = use_cuda

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Piper model file not found at: {self.model_path}")

        if _shared_piper_voice is None:
            _shared_piper_voice = PiperVoice.load(
                model_path=self.model_path,
                config_path=self.config_path,
                use_cuda=self.use_cuda,
            )
        self.voice = _shared_piper_voice

    def synthesize(self, text: str, output_path: str, language: str = "en") -> str:
        """
        Synthesizes text to speech with language-specific voice models:
        - English (default): Local Piper TTS engine.
        - Hindi (hi): Native Hindi neural voice (hi-IN-SwaraNeural).
        - Marathi (mr): Native Marathi neural voice (mr-IN-AarohiNeural).

        Returns output_path.
        """
        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        lang_code = str(language or "en").strip().lower()

        if lang_code in ["hi", "hindi", "2"]:
            return self._synthesize_native_indic(text.strip(), "hi-IN-SwaraNeural", output_path)
        elif lang_code in ["mr", "marathi", "3"]:
            return self._synthesize_native_indic(text.strip(), "mr-IN-AarohiNeural", output_path)
        else:
            # English (Option 1) - Existing proven Piper TTS model
            with wave.open(output_path, "wb") as wav_file:
                self.voice.synthesize_wav(text.strip(), wav_file)
            return output_path

    def _synthesize_native_indic(self, text: str, voice_name: str, output_path: str) -> str:
        """
        Synthesizes native Devanagari text into natural voice audio using dedicated
        language-specific neural models, then decodes directly into standard 16-bit PCM WAV.
        """
        temp_mp3 = output_path + ".temp.mp3"
        try:
            import edge_tts
            import av

            async def _generate():
                comm = edge_tts.Communicate(text, voice=voice_name)
                await comm.save(temp_mp3)

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _generate()).result()
            else:
                asyncio.run(_generate())

            # Convert MP3 to standard 24000Hz 16-bit PCM WAV
            with av.open(temp_mp3) as in_container:
                in_stream = in_container.streams.audio[0]
                with wave.open(output_path, "wb") as out_wav:
                    out_wav.setnchannels(1)
                    out_wav.setsampwidth(2)
                    out_wav.setframerate(24000)

                    # Resample / reformat audio frames to 1-channel 24000Hz PCM
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
                    for frame in in_container.decode(in_stream):
                        for resampled in resampler.resample(frame):
                            out_wav.writeframes(resampled.to_ndarray().tobytes())

            return output_path
        finally:
            if os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except OSError:
                    pass
