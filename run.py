import cv2
import dlib
import numpy as np
from stable_baselines3 import PPO
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def apply_circular_noise_patch(image, landmarks, patch_radius=10, noise_intensity=30):
    noisy_image = image.copy()
    for (x, y) in landmarks:
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), patch_radius, (255, 255, 255), -1)
        noise = np.random.normal(0, noise_intensity, image.shape).astype(np.uint8)
        noisy_image = cv2.bitwise_and(noisy_image, 255 - mask) + cv2.bitwise_and(noise, mask)
    return noisy_image

def test_model_on_image(image_path, model_path):
    model = PPO.load(model_path)
    print("Model loaded successfully.")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from path {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return
    
    for face in faces:
        landmarks = predictor(gray, face)
        original_landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # 모델 예측 (노이즈 패치 적용 위치, 크기, 강도 결정)
        action, _ = model.predict(original_landmarks.flatten(), deterministic=True)
        landmark_idx = int(action[0] * 67)
        patch_x, patch_y = original_landmarks[landmark_idx]
        patch_radius = int(action[1] * 20) + 5
        noise_intensity = int(action[2] * 50) + 10

        noisy_image = apply_circular_noise_patch(image, [(patch_x, patch_y)], patch_radius, noise_intensity)

        print(f"Testing on Image - Landmark: {landmark_idx}, Patch Pos: ({patch_x}, {patch_y}), Radius: {patch_radius}, Intensity: {noise_intensity}")

        cv2.imwrite("Noisy_Image_경로.jpg", noisy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    
test_model_on_image("WIN_20250306_01_19_02_Pro.jpg", "ppo_face_noise.zip")
