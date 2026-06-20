import torch
import transformers
import datasets
import soundfile as sf
import librosa
import kagglehub

print("Environment Verification Successful!")
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Datasets Version: {datasets.__version__}")
print("All audio decoding binaries (libsndfile/soundfile) loaded cleanly.")