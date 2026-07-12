import cv2
import mediapipe as mp
import numpy as np
import os
import csv

def calculate_distance_3d(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)

def normalize_landmarks(landmarks):
    # landmarks is list of 21 points, each [x, y, z]
    wrist = landmarks[0]
    
    # 1. Translation: Shift wrist to origin
    shifted = [[pt[0] - wrist[0], pt[1] - wrist[1], pt[2] - wrist[2]] for pt in landmarks]
    
    # 2. Scaling: Calculate distance between wrist (0) and index MCP (5)
    index_mcp = shifted[5]
    hand_size = calculate_distance_3d([0, 0, 0], index_mcp)
    if hand_size == 0:
        hand_size = 1.0
        
    # Divide all points by hand size
    normalized = [[pt[0] / hand_size, pt[1] / hand_size, pt[2] / hand_size] for pt in shifted]
    
    # 3. Flatten
    flattened = []
    for pt in normalized:
        flattened.extend(pt)
        
    return flattened

def main():
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Setup CSV Output Directory
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'sign_landmarks.csv')
    
    # Check if header needs to be written
    file_exists = os.path.exists(csv_path)
    
    print("\n" + "="*50)
    print("  SignCV Landmark Data Collector")
    print("="*50)
    label = input("Enter gesture label name (e.g. HELLO, A, YES): ").strip().upper()
    if not label:
        print("Invalid label name. Exiting.")
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print(f"\nRecording samples for label: '{label}'")
    print("Controls:")
    print("  Press 'SPACE' to collect a single landmark sample")
    print("  Press 'r' to toggle Continuous Recording mode")
    print("  Press 'q' to quit data collection")
    print("-" * 50)
    
    sample_count = 0
    continuous_mode = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
            
        # Flip image horizontally for natural mirror display
        image = cv2.flip(image, 1)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        hand_detected = False
        features = None
        
        # Draw hand annotations
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0] # Focus on primary hand
            
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract and normalize landmarks
            raw_pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            features = normalize_landmarks(raw_pts)
            
        # UI overlays
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(image, f"Hand Detected: {'YES' if hand_detected else 'NO'}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, f"Label: {label}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Samples: {sample_count}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        mode_text = "Continuous [ON]" if continuous_mode else "Single [SPACE]"
        mode_color = (0, 255, 0) if continuous_mode else (255, 200, 0)
        cv2.putText(image, f"Mode: {mode_text}", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Continuous recording handler
        if continuous_mode and hand_detected:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    # Write header
                    header = ['label'] + [f'coord_{i}' for i in range(63)]
                    writer.writerow(header)
                    file_exists = True
                writer.writerow([label] + features)
            sample_count += 1
            
        cv2.imshow('SignCV - Data Collection', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '): # Space bar
            if hand_detected:
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        header = ['label'] + [f'coord_{i}' for i in range(63)]
                        writer.writerow(header)
                        file_exists = True
                    writer.writerow([label] + features)
                sample_count += 1
                print(f"Captured sample {sample_count}")
            else:
                print("No hand detected. Cannot save sample.")
        elif key == ord('r'):
            continuous_mode = not continuous_mode
            print(f"Continuous mode: {continuous_mode}")
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished collection! Saved {sample_count} new samples to {csv_path}")

if __name__ == '__main__':
    main()
