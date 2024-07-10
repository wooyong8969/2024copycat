import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
import time
import queue

q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(indata.copy())

def play_audio(file_path):
    data, fs = sf.read(file_path, dtype='float32')
    sd.play(data, fs)
    sd.wait()

def record_audio(duration, fs=44100):
    print("녹음을 시작하려면 s를 입력해 주세요: ")
    while True:
        if input().lower() == 's':
            break

    print("녹음 중...")
    with sd.InputStream(samplerate=fs, channels=2, callback=callback):
        print("녹음을 종료하려면 e를 입력해 주세요: ")
        while True:
            if input().lower() == 'e':
                break
    
    print("녹음 완료")
    recorded_data = []
    while not q.empty():
        recorded_data.append(q.get())

    recording = np.concatenate(recorded_data, axis=0)
    return recording, fs

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    #print(f"Loaded {file_path} with sampling rate {sr}")
    return y, sr

def preprocess_recording(recording, fs):
    y = librosa.resample(recording.T[0], orig_sr=fs, target_sr=16000)
    sr = 16000
    #print(f"Processed recording with sampling rate {sr}")
    return y, sr

def extract_features(y, sr):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    #print(f"Extracted features with shape {features.shape}")
    return features.squeeze(0).numpy()

def calculate_similarity(features1, features2):
    distance, _ = fastdtw(features1, features2, dist=cosine)
    print(f"오차율: {distance}")
    similarity_score = max(0, 100 - distance)
    return similarity_score

def main():
    start_time = time.time()

    original_audio = r"D:\W00Y0NG\PRGM2\2024copycat\Dataset\trimmed_or_who_r_u.wav"
    duration = 10  # 녹음 시간 (초 단위)

    # 원본 음성 파일 재생
    print("원본 음성 재생")
    play_audio(original_audio)

    print("성대모사 녹음")
    recording, fs = record_audio(duration)

    # 음성 파일 전처리
    print("음성 처리 중")
    preprocess_start_time = time.time()
    y1, sr1 = preprocess_audio(original_audio)
    y2, sr2 = preprocess_recording(recording, fs)
    preprocess_end_time = time.time()

    print("특징 추출 중")
    extract_start_time = time.time()
    features1 = extract_features(y1, sr1)
    features2 = extract_features(y2, sr2)
    extract_end_time = time.time()

    print("점수 계산 중")
    similarity_start_time = time.time()
    score = calculate_similarity(features1, features2)
    similarity_end_time = time.time()

    end_time = time.time()

    print(f'따라쟁이 점수: {score:.2f}/100')
    print(f'소요 시간: {end_time - start_time:.2f} seconds')
    print(f'음성 처리 시간: {preprocess_end_time - preprocess_start_time:.2f} seconds')
    print(f'특징 추출 시간: {extract_end_time - extract_start_time:.2f} seconds')
    print(f'유사도 분석 시간: {similarity_end_time - similarity_start_time:.2f} seconds')


if __name__ == "__main__":
    main()
