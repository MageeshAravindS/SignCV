import cv2
import mediapipe as mp
import numpy as np
import os
import json
import torch
import collections
from model import SignLanguageMLP

def calculate_distance_3d(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    shifted = [[pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]] for pt in landmarks]
    index_mcp = shifted[5]
    hand_size = calculate_distance_3d([0, 0, 0], index_mcp)
    if hand_size == 0:
        hand_size = 1.0
    normalized = [[pt[0] / hand_size, pt[1] / hand_size, pt[2] / hand_size] for pt in shifted]
    flattened = []
    for pt in normalized:
        flattened.extend(pt)
    return flattened

def main():
    data_dir = 'data'
    weights_path = os.path.join(data_dir, 'model.pth')
    mapping_path = os.path.join(data_dir, 'labels.json')
    
    # Check model files
    if not (os.path.exists(weights_path) and os.path.exists(mapping_path)):
        print("Error: PyTorch model or labels mapping file not found.")
        print("Please train a model first using: python python/train.py")
        return
        
    # Load labels
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    classes = [mapping[str(i)] for i in range(len(mapping))]
    
    # Initialize PyTorch Model
    print("Loading PyTorch model...")
    model = SignLanguageMLP(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    # Smoothing Queue
    queue_size = 10
    prediction_queue = collections.deque(maxlen=queue_size)
    confidence_threshold = 0.70
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("\nStarting local PyTorch webcam inference...")
    print("Press 'q' in the camera window to exit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        # Flip frame horizontally for mirror view
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # Process landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        predicted_label = "-"
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw overlay elements
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Feature extraction
            raw_pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            features = normalize_landmarks(raw_pts)
            
            # PyTorch inference
            features_tensor = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1).numpy()[0]
                pred_idx = np.argmax(probabilities)
                
                raw_pred_label = classes[pred_idx]
                raw_confidence = probabilities[pred_idx]
                
                # Apply confidence gating and queue smoothing
                if raw_confidence >= confidence_threshold:
                    prediction_queue.append(raw_pred_label)
                    
                    # Mode selection
                    counts = collections.Counter(prediction_queue)
                    most_common, frequency = counts.most_common(1)[0]
                    if frequency >= int(queue_size * 0.5):
                        predicted_label = most_common
                        confidence = raw_confidence
                else:
                    prediction_queue.clear()
                    
        # GUI rendering
        # Draw status card at the top left corner
        cv2.rectangle(image, (10, 10), (320, 130), (15, 11, 10), -1)
        cv2.rectangle(image, (10, 10), (320, 130), (99, 102, 241), 2)
        
        cv2.putText(image, "SignCV PyTorch Runner", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Prediction: {predicted_label}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2)
        cv2.putText(image, f"Confidence: {confidence*100:.1f}%", (20, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if confidence > 0 else (128, 128, 128), 2)
                    
        cv2.imshow('SignCV Desktop Client', image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Inference runner stopped.")

if __name__ == '__main__':
    main()
