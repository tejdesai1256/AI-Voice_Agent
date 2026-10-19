import os
import wave
import av

# PyAV compatibility patch for newer av versions where metadata_errors was removed
_orig_av_open = av.open


def _compat_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _compat_av_open


def convert_to_16k_mono_wav(input_path: str, output_path: str = None) -> str:
    """
    Converts incoming audio (WebM/Opus, Ogg, MP3, or arbitrary WAV) into a standardized
    16kHz mono 16-bit PCM WAV file optimized for Faster-Whisper speech-to-text.

    Preserves original speech dynamics and vocal formant energy, avoiding
    stereo-to-mono downmix attenuation and Opus packet boundary issues.
    Returns path to the converted WAV file, or original path on failure.
    """
    if not input_path or not os.path.exists(input_path):
        return input_path

    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_16k_mono.wav"

    try:
        with av.open(input_path) as in_container:
            if not in_container.streams.audio:
                return input_path

            in_stream = in_container.streams.audio[0]
            with wave.open(output_path, "wb") as out_wav:
                out_wav.setnchannels(1)       # Mono
                out_wav.setsampwidth(2)       # 16-bit (2 bytes)
                out_wav.setframerate(16000)   # 16kHz standard for Whisper

                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                for frame in in_container.decode(in_stream):
                    for resampled in resampler.resample(frame):
                        out_wav.writeframes(resampled.to_ndarray().tobytes())

        if os.path.exists(output_path) and os.path.getsize(output_path) > 44:
            return output_path
        return input_path
    except Exception as e:
        print(f"[audio_utils Warning] Audio conversion to 16kHz mono failed: {e}")
        return input_path
