import numpy as np
import soundfile as sf
from rvc_py.rvc_infer import rvc_infer

# Assuming a sample audio file 'sample.wav' exists, otherwise create a dummy
# For testing, create a dummy audio
dummy_wav = np.random.rand(16000 * 10) - 0.5  # 10 seconds at 16kHz
dummy_sr = 16000
sf.write('sample.wav', dummy_wav, dummy_sr)

# Test parameters
hubert_model_path = 'path/to/hubert/model.pth'  # Replace with actual
rvc_model_path = 'path/to/rvc/model.pth'  # Replace with actual
f0_method = 'rmvpe'
output_path = 'output.wav'

rvc_infer(hubert_model_path, rvc_model_path, 'sample.wav', output_path, f0_method=f0_method, device='cpu')
print('Inference completed. Check output.wav')