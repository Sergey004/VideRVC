import os
import torch
import librosa
import numpy as np
from rvc_py.rvc_infer import rvc_infer


def run_rvc_test():
    # --- Configuration --- #
    # IMPORTANT: Replace with your actual model and audio file paths
    rvc_model_path = "models\RVC\Atreyu_Alcantar\Atreyu_Alcantar_38e_2546s_best_epoch.pth"  # e.g., "./models/your_model.pth"
    audio_input_path = "Record 2025-08-30 at 14h43m05s.wav"  # e.g., "./test_audio.wav"
    f0_method = "rmvpe"
    pitch_shift = 0
    index_rate = 0.5
    index_path = "models\RVC\Atreyu_Alcantar\added_Atreyu_Alcantar_v2.index"  # set path to .index if available, else leave empty
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(rvc_model_path):
        print(f"Error: RVC model not found at {rvc_model_path}")
        return
    if not os.path.exists(audio_input_path):
        print(f"Error: Input audio file not found at {audio_input_path}")
        return

    print(f"Using device: {device}")

    # --- Load Audio --- #
    print(f"Loading audio from {audio_input_path}...")
    wav, sr = librosa.load(audio_input_path, sr=None, mono=True)
    print(f"Original audio shape: {wav.shape}, sample rate: {sr}")

    # --- Perform Inference --- #
    print("Performing RVC inference...")
    try:
        output_wav, output_sr = rvc_infer(
            wav=wav,
            sr=sr,
            rvc_model_path=rvc_model_path,
            device=device,
            f0_method=f0_method,
            pitch_shift=pitch_shift,
            index_path=index_path if index_path else None,
            index_rate=index_rate,
            use_index=bool(index_path),
        )
        print(f"Inference successful. Output audio shape: {output_wav.shape}, sample rate: {output_sr}")
        # Optionally save output
        import soundfile as sf
        sf.write("output_rvc.wav", output_wav, output_sr)
    except Exception as e:
        print(f"Error during RVC inference: {e}")


if __name__ == "__main__":
    run_rvc_test()