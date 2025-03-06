import os
import cv2
import dlib
import json
import numpy as np
from stable_baselines3 import PPO

# 경로 설정
MODEL_PATH = "data\PPOrl_noise.zip"
JSON_PATH = "data\landmarks.json"
IMAGE_PATH = ""  # 이미지 입력
OUTPUT_PATH = ""  # 이미지 출력

# 출력 폴더 생성
if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
    os.makedirs(os.path.dirname(OUTPUT_PATH))

# dlib 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data\shape_predictor_68_face_landmarks.dat")

# PPO 모델 로드
model = PPO.load(MODEL_PATH)

# 랜드마크 데이터 로드
with open(JSON_PATH, 'r') as f:
    landmarks_data = json.load(f)

# 이미지 로드
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise ValueError(f"Unable to load image: {IMAGE_PATH}")

filename = os.path.basename(IMAGE_PATH)
if filename not in landmarks_data:
    raise ValueError(f"Landmarks for {filename} not found in JSON data.")

original_landmarks = np.array(landmarks_data[filename], dtype=np.float32)

# PPO 모델로부터 노이즈 예측
action, _ = model.predict(original_landmarks)
noisy_landmarks = original_landmarks + action[:, :2]  # 위치 변화 적용

# 이미지에 랜드마크 그리기
for (x, y) in noisy_landmarks.astype(int):
    cv2.circle(image, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

# 결과 저장
cv2.imwrite(OUTPUT_PATH, image)
print(f"Saved modified image: {OUTPUT_PATH}")
