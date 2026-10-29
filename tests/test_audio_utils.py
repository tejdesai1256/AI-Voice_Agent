import os
import sys
import wave

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.audio_utils import convert_to_16k_mono_wav, preprocess_pcm_for_stt
from app.services.stt_service import STTService


def test_audio_normalization():
    print("=== TESTING AUDIO NORMALIZATION UTILITY ===")
    
    test_wav = "recordings/test_resample_src.wav"
    out_wav = "recordings/test_resample_out.wav"
    
    # Create a small dummy 8kHz stereo WAV to test resampling
    with wave.open(test_wav, "wb") as w:
        w.setnchannels(2)      # Stereo
        w.setsampwidth(2)      # 16-bit
        w.setframerate(8000)   # 8kHz
        # Write 0.5s of silence / dummy audio
        w.writeframes(b"\x00\x00" * 8000)
        
    try:
        converted = convert_to_16k_mono_wav(test_wav, out_wav)
        assert os.path.exists(converted), "Converted WAV does not exist"
        
        # Verify target parameters: 16kHz mono 16-bit
        with wave.open(converted, "rb") as r:
            assert r.getnchannels() == 1, f"Expected 1 channel, got {r.getnchannels()}"
            assert r.getsampwidth() == 2, f"Expected sampwidth 2, got {r.getsampwidth()}"
            assert r.getframerate() == 16000, f"Expected 16000Hz, got {r.getframerate()}"
            
        print("PASS: convert_to_16k_mono_wav normalized stereo 8kHz to mono 16kHz 16-bit PCM WAV")
    finally:
        for p in (test_wav, out_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # Test preprocess_pcm_for_stt (DC offset and AGC gain)
    import struct
    import numpy as np

    raw_low_vol = struct.pack("<4h", 1000, -2000, 1500, -500)
    cleaned = preprocess_pcm_for_stt(raw_low_vol, target_peak=26000.0, max_gain=6.0)
    assert len(cleaned) == len(raw_low_vol)
    samples = struct.unpack("<4h", cleaned)
    new_peak = max(abs(s) for s in samples)
    assert new_peak > 2000, f"Expected gain boost, got new peak {new_peak}"

    # Test STT sanitization against known Whisper hallucination loops
    hallucination_loop = "I'm going to look really good, don't worry. I'll be right back. I'm going to be fine. Thank you."
    sanitized = STTService._sanitize_transcript(hallucination_loop)
    assert "i'll be right back" not in sanitized.lower()

    repetition_loop = "Thank you. Thank you. Thank you. Thank you."
    assert STTService._sanitize_transcript(repetition_loop) == ""

    # Legitimate business phrase preserved
    valid_speech = "We are an IT industry company based in Pune."
    assert STTService._sanitize_transcript(valid_speech) == valid_speech

    print("ALL AUDIO UTILS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_audio_normalization()
