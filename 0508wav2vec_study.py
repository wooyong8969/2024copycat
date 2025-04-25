import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt


torchaudio.set_audio_backend("soundfile")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def load_and_resample(file_path, new_sampling_rate=16000):
    speech_array, sampling_rate = torchaudio.load(file_path)

    # 리샘플링
    if sampling_rate != new_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=new_sampling_rate)
        speech_array = resampler(speech_array)
    return speech_array

original_audio = load_and_resample(r"D:\W00Y0NG\PRGM2\2024copycat\Dataset\or_who_r_u.wav")
imitation_audio = load_and_resample(r"D:\W00Y0NG\PRGM2\2024copycat\Dataset\im_who_r_u.wav")

# 특징 벡터 추출
def extract_features(audio, sampling_rate=16000):
    input_values = processor(audio, return_tensors="pt", sampling_rate=sampling_rate).input_values
    with torch.no_grad():
        features = model(input_values).last_hidden_state
    return features.mean(dim=1).squeeze().numpy()

original_features = extract_features(original_audio)
imitation_features = extract_features(imitation_audio)

plt.figure(figsize=(12, 6))
plt.plot(original_features, label='Original', color='blue')
plt.plot(imitation_features, label='Imitation', color='red')
plt.xlabel('Feature Dimensions')
plt.ylabel('Feature Values')
plt.legend()
plt.show()
