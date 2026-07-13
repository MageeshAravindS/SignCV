# Real-Time Sign Language Recognition System

A real-time computer vision and deep learning application that recognizes sign language gestures from a live webcam feed, translates them into text, and reads them aloud using Text-to-Speech (TTS).

This system provides two independent ways of execution:
1. **Interactive Web Application (Zero Setup)**: Runs MediaPipe Hands directly in the browser via WebGL-accelerated JS CDN, paired with a custom TensorFlow.js trainer to record and train custom gestures on the fly in under 5 seconds.
2. **Python Deep Learning Pipeline**: A command-line suite for dataset collection, model definition, training, evaluation, and desktop-native OpenCV landmark inference utilizing PyTorch.

---

## Technical Architecture Overview

```
                      +-------------------+
                      |   Camera Feed     |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |  MediaPipe Hands  |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Normalization   |
                      |  (Translation &   |
                      |   Scale Invariant)|
                      +---------+---------+
                                |
                                v
               +---------------------------------+
               |      Classifier Dispatch        |
               +--------+----------------+-------+
                        |                |
         (Built-in Mode) |                | (Custom MLP Mode)
                        v                v
            +-------------------+  +-------------------+
            |  Geometric Rules  |  |  TensorFlow.js /  |
            |      Engine       |  |   PyTorch Model   |
            +-----------+-------+  +---------+---------+
                        |                    |
                        +--------+-----------+
                                 |
                                 v
                      +-------------------+
                      | Prediction Queue  |
                      |   & Smoothing     |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      | Sentence Builder  |
                      |   & Web Speech    |
                      +-------------------+
```

---

## Getting Started: Interactive Web App (Recommended)

The Web Application runs at 30+ FPS client-side with zero heavy dependencies!

### 1. Launch the Server
Ensure you have Python installed, then run the server script in your terminal:
```bash
python server.py
```
This script starts a local web server (on port `8000`) and automatically opens your default browser at `http://localhost:8000`.

### 2. Using the App
- **Live Recognition Tab**: Select "Built-in Geometric Engine" to immediately recognize standard ASL characters (A-Z) and numbers (1-9) using your hand shape.
- **Custom Trainer Tab**:
  1. Add a label (e.g., `HELLO`, `YES`, `NO`).
  2. Select the label from the list.
  3. Hold down the **Record** button while holding your hand in different positions in front of the camera (collect at least 30-40 samples).
  4. Repeat for other classes.
  5. Click **Train Neural Network (MLP)**. The browser will train a 3-layer neural network using TensorFlow.js in ~2 seconds.
  6. The app will automatically switch your model to the trained network, enabling real-time classification of your new gestures!
- **Text-to-Speech (TTS)**: Toggle "Auto Text-to-Speech" to speak words aloud upon gesture finalize.

---

## Virtual Audio Driver Integration (Zoom, Discord, Teams)

This system features a built-in **Virtual Audio Routing** channel. When enabled, your sign-to-speech output is sent directly into a Virtual Microphone input rather than your local speakers. This allows you to "speak" with sign language inside meeting and call software like Zoom, Microsoft Teams, Discord, Google Meet, and Skype.

### Step 1: Install VB-CABLE Driver
1. Download the free virtual audio cable driver: **[VB-CABLE Driver Download](https://vb-audio.com/Cable/)**.
2. Extract the downloaded folder on Windows.
3. Right-click `VBCABLE_Setup_x64.exe` and select **Run as administrator**.
4. Click **Install Driver**.
5. Restart your PC to finalize driver registry configurations.

### Step 2: Configure & Launch SignCV
1. Run the local Python server:
   ```bash
   python server.py
   ```
2. Navigate to `http://localhost:8000`. Under Settings, you should now see:
   * **Virtual Mic Status**: `CONNECTED` (green)
3. Turn **ON** the checkbox toggle **Route to Virtual Mic (VB-Cable)**.

### Step 3: Choose Input inside Zoom/Discord/Teams
1. Open your meeting application (e.g. Discord, Zoom, or Teams).
2. Go to **Audio Settings / Input Device**.
3. Select **CABLE Output (VB-Audio Virtual Cable)** as your default microphone input.
4. Set your webcam in the meeting app. Now, when you perform signs on the SignCV viewfinder, the synthesized speech output will stream directly into the meeting call as your live microphone voice!

---

## Python Deep Learning Pipeline

For offline developers who want to train and evaluate custom neural networks.

### 1. Setup Environment
Install python dependencies:
```bash
pip install -r python/requirements.txt
```

### 2. Collect Hand Landmark Data
Collect hand landmark features from your webcam feed:
```bash
python python/data_collection.py
```
- Enter a label (e.g. `HELLO`).
- Press `SPACE` to save individual frames or `r` to record continuous frames.
- Press `q` to exit. Data is saved in a unified dataset at `python/data/sign_landmarks.csv`.

### 3. Train PyTorch Model
To train the Fully Connected Multi-Layer Perceptron (MLP) on the landmarks dataset:
```bash
python python/train.py
```
*Note: If no dataset is found, it will automatically generate a mock dataset of common gestures so you can test the training script immediately.*

### 4. Evaluate Classifier
To generate F1, Precision, Recall scores, and print a Confusion Matrix:
```bash
python python/evaluate.py
```
This prints the metrics to the terminal and saves a detailed plot in `python/data/confusion_matrix.png`.

### 5. Run Local Desktop Inference
To perform real-time local webcam predictions using the PyTorch model:
```bash
python python/inference.py
```

---

## Feature Comparison: Web App vs Python Pipeline

| Feature | Browser Web App | Python Pipeline |
|:---|:---|:---|
| **Frame Rate** | 30+ FPS (WebGL) | ~25 FPS (CPU/GPU) |
| **Inference Engine** | Heuristics / TensorFlow.js | PyTorch (`torch`) |
| **Custom Gesture Training** | Browser MLP (instant) | PyTorch script (`train.py`) |
| **Audio Synthesis** | HTML5 Web Speech API | Optional `pyttsx3` |
| **Interface** | Responsive UI (CSS Glassmorphism) | OpenCV Native Window |
| **Dependencies** | None (Runs via CDNs) | PyTorch, OpenCV, MediaPipe |

---

## File Directory

- `index.html` - Premium UI webpage.
- `style.css` - Custom styling theme.
- `app.js` - Javascript core logic (MediaPipe setup, normalization, algorithms, TF.js).
- `server.py` - Flask/HTTP server code.
- `python/`
  - `requirements.txt` - Python package list.
  - `data_collection.py` - Collects webcam landmarks to CSV.
  - `model.py` - PyTorch MLP & LSTM models definitions.
  - `train.py` - Splits data and trains model.
  - `evaluate.py` - Performance evaluation & confusion matrix chart.
  - `inference.py` - Real-time PyTorch camera tester.
  - `Sign_Language_Training.ipynb` - Jupyter tutorial notebook.
