import os
import sys
import wave

# Ensure root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.audio_utils import convert_to_16k_mono_wav


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
                    
    print("ALL AUDIO UTILS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_audio_normalization()
