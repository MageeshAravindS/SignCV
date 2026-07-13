// Global State and Constants
let handsEngine = null;
let cameraHelper = null;
let activeTab = 'recognition-tab';
let lastHandLandmarks = null;
let pretrainedWeights = null;
let pretrainedModel = null;
let coordsBuffer = [];
let customModelInputShape = 2520; // Auto-detected from loaded models (supports 63, 126, and 2520 features)

// UI Element Selectors
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const engineStatus = document.getElementById('engine-status');
const fpsCounter = document.getElementById('fps-counter');
const latencyCounter = document.getElementById('latency-counter');
const loadingSpinner = document.getElementById('loading-spinner');
const leftHandPill = document.getElementById('left-hand-pill');
const rightHandPill = document.getElementById('right-hand-pill');

const predictedLetterText = document.getElementById('predicted-letter');
const predictedConfidenceText = document.getElementById('predicted-confidence');
const confidenceBar = document.getElementById('confidence-bar');

const modelSelector = document.getElementById('model-selector');
const confidenceThresholdSlider = document.getElementById('confidence-threshold');
const thresholdValSpan = document.getElementById('threshold-val');
const smoothingToggle = document.getElementById('smoothing-toggle');
const ttsToggle = document.getElementById('tts-toggle');
const autospaceToggle = document.getElementById('autospace-toggle');

const outputSentenceDiv = document.getElementById('output-sentence');
const activeCharPreview = document.getElementById('active-char-preview');
const speakBtn = document.getElementById('speak-btn');
const backspaceBtn = document.getElementById('backspace-btn');
const clearTextBtn = document.getElementById('clear-text-btn');

// Trainer Elements
const newGestureNameInput = document.getElementById('new-gesture-name');
const addGestureBtn = document.getElementById('add-gesture-btn');
const gestureLabelsUl = document.getElementById('gesture-labels-ul');
const recorderPanel = document.getElementById('recorder-panel');
const recordingActiveLabel = document.getElementById('recording-active-label');
const recordedSamplesCount = document.getElementById('recorded-samples-count');
const recordBtn = document.getElementById('record-btn');
const clearSamplesBtn = document.getElementById('clear-samples-btn');
const trainingStatusText = document.getElementById('training-status-text');
const trainingEpochText = document.getElementById('training-epoch-text');
const trainingLossText = document.getElementById('training-loss-text');
const trainModelBtn = document.getElementById('train-model-btn');
const exportModelBtn = document.getElementById('export-model-btn');
const customModelOption = document.getElementById('custom-model-option');
const pretrainedModelOption = document.getElementById('pretrained-model-option');
const virtualMicToggle = document.getElementById('virtual-mic-toggle');
const virtualMicStatusText = document.getElementById('virtual-mic-status-text');
const wordRecToggle = document.getElementById('word-rec-toggle');
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeSunIcon = document.getElementById('theme-sun-icon');
const themeMoonIcon = document.getElementById('theme-moon-icon');

// Inspector Elements
const coordInspector = document.getElementById('coord-inspector');

// Performance Tracking
let lastFrameTime = performance.now();
let fpsInterval = 0;
let fpsHistory = [];
let latencyHistory = [];
let inferenceStartTime = 0;

// Recognition State
let predictedQueue = [];
const QUEUE_SIZE = 15;
let currentSentence = "";
let currentWord = "";
let lastPredictedChar = "";
let charStreakCount = 0;
const STREAK_THRESHOLD = 25; // number of frames to lock in a letter (at ~30fps, ~0.8s)
let noHandFramesCount = 0;
const NO_HAND_TIMEOUT = 45;  // 1.5 seconds of no hands to finalize word

// Custom Gesture Trainer Data
let customLabels = [];
let customData = {
  X: [], // sequence coordinates
  y: []  // class indices
};
let tfModel = null;
let isRecording = false;
let tempRecordFrames = [];
let selectedTrainerLabelIndex = -1;

// Initialize Web Speech Synthesis
const synth = window.speechSynthesis;

// Theme Toggle (Dark / Light Mode)
const savedTheme = localStorage.getItem('theme') || 'dark';
if (savedTheme === 'light') {
  document.documentElement.setAttribute('data-theme', 'light');
  themeSunIcon.style.display = 'none';
  themeMoonIcon.style.display = 'block';
}

themeToggleBtn.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'light');
    themeSunIcon.style.display = 'none';
    themeMoonIcon.style.display = 'block';
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeSunIcon.style.display = 'block';
    themeMoonIcon.style.display = 'none';
    localStorage.setItem('theme', 'dark');
  }
});

// -------------------------------------------------------------
// 1. Tab Navigation & UI Controls Setup
// -------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    btn.classList.add('active');
    activeTab = btn.getAttribute('data-tab');
    document.getElementById(activeTab).classList.add('active');
  });
});

confidenceThresholdSlider.addEventListener('input', (e) => {
  thresholdValSpan.textContent = parseFloat(e.target.value).toFixed(2);
});

// Clear, Delete, and Speak Actions
clearTextBtn.addEventListener('click', () => {
  currentSentence = "";
  currentWord = "";
  outputSentenceDiv.textContent = "";
  activeCharPreview.textContent = "-";
});

backspaceBtn.addEventListener('click', () => {
  if (currentSentence.length > 0) {
    currentSentence = currentSentence.trimEnd();
    currentSentence = currentSentence.slice(0, -1);
    outputSentenceDiv.textContent = currentSentence;
  }
});

speakBtn.addEventListener('click', () => {
  speakText(outputSentenceDiv.textContent || "No text generated yet.");
});

function speakText(text) {
  if (!text || text.trim() === "") return;
  
  if (virtualMicToggle.checked) {
    // Route TTS to python local server endpoint which streams to VB-Cable Channel
    fetch('/api/speak', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    })
    .then(res => res.json())
    .then(data => {
      if (!data.success) {
        console.error("Virtual Mic speech routing failed:", data.error);
        fallbackSpeak(text);
      }
    })
    .catch(err => {
      console.error("Error calling Speak API:", err);
      fallbackSpeak(text);
    });
  } else {
    fallbackSpeak(text);
  }
}

function fallbackSpeak(text) {
  if (!synth || synth.speaking) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  synth.speak(utterance);
}

const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

// Poll Virtual Mic connection state
function checkVirtualMicConnection() {
  if (!isLocalhost) {
    virtualMicStatusText.textContent = 'LOCAL ONLY';
    virtualMicStatusText.style.color = '#888888';
    virtualMicToggle.checked = false;
    virtualMicToggle.disabled = true;
    return;
  }
  
  fetch('/api/devices')
    .then(res => {
      if (!res.ok) throw new Error("HTTP Error");
      return res.json();
    })
    .then(data => {
      if (data.connected) {
        virtualMicStatusText.textContent = `CONNECTED`;
        virtualMicStatusText.style.color = '#00d180'; // Hulu Green
        virtualMicToggle.disabled = false;
      } else {
        virtualMicStatusText.textContent = 'NOT DETECTED';
        virtualMicStatusText.style.color = '#cf0056'; // LG Red
        virtualMicToggle.checked = false;
        virtualMicToggle.disabled = true;
        localStorage.setItem('virtualMicEnabled', 'false');
      }
    })
    .catch(err => {
      virtualMicStatusText.textContent = 'SERVER OFFLINE';
      virtualMicStatusText.style.color = '#cf0056'; // LG Red
      virtualMicToggle.checked = false;
      virtualMicToggle.disabled = true;
    });
}

// Persist user preferences
virtualMicToggle.addEventListener('change', (e) => {
  localStorage.setItem('virtualMicEnabled', e.target.checked ? 'true' : 'false');
});

wordRecToggle.addEventListener('change', async (e) => {
  const isChecked = e.target.checked;
  if (isChecked) {
    // Dynamically query latest custom model in case it was just compiled in other tab
    let loaded = await loadDeployedCustomModel();
    if (!loaded && !tfModel) {
      customModelOption.textContent = "Training Model...";
      await autoTrainFromDataset();
      loaded = tfModel !== null;
    }
    modelSelector.value = loaded ? 'custom' : 'pretrained';
  } else {
    modelSelector.value = 'similarity';
  }
  localStorage.setItem('wordRecEnabled', isChecked ? 'true' : 'false');
});

modelSelector.addEventListener('change', (e) => {
  const val = e.target.value;
  if (val === 'custom' || val === 'pretrained') {
    wordRecToggle.checked = true;
    localStorage.setItem('wordRecEnabled', 'true');
  } else {
    wordRecToggle.checked = false;
    localStorage.setItem('wordRecEnabled', 'false');
  }
});

// -------------------------------------------------------------
// 2. Custom Gesture Trainer Logic (TensorFlow.js)
// -------------------------------------------------------------
addGestureBtn.addEventListener('click', () => {
  const name = newGestureNameInput.value.trim().toUpperCase();
  if (!name) return;
  if (customLabels.includes(name)) {
    alert("Label already exists!");
    return;
  }
  
  customLabels.push(name);
  newGestureNameInput.value = "";
  updateLabelsList();
});

function updateLabelsList() {
  if (customLabels.length === 0) {
    gestureLabelsUl.innerHTML = '<li class="empty-list-note">No custom labels added yet. Add one above!</li>';
    recorderPanel.style.display = 'none';
    trainModelBtn.disabled = true;
    return;
  }
  
  gestureLabelsUl.innerHTML = '';
  customLabels.forEach((label, idx) => {
    const li = document.createElement('li');
    if (idx === selectedTrainerLabelIndex) li.classList.add('selected');
    
    // Calculate sample count for this label
    const sampleCount = customData.y.filter(val => val === idx).length;
    
    li.innerHTML = `
      <span class="label-name">${label}</span>
      <span class="label-count">${sampleCount} samples</span>
    `;
    
    li.addEventListener('click', () => {
      selectedTrainerLabelIndex = idx;
      updateLabelsList();
      showRecorder(label);
    });
    gestureLabelsUl.appendChild(li);
  });
  
  // Enable training if we have at least 2 labels and some samples for each
  const labelCounts = customLabels.map((_, idx) => customData.y.filter(val => val === idx).length);
  const canTrain = customLabels.length >= 2 && labelCounts.every(count => count >= 10);
  trainModelBtn.disabled = !canTrain;
}

function showRecorder(label) {
  recorderPanel.style.display = 'block';
  recordingActiveLabel.textContent = `Selected: ${label}`;
  updateSampleCounter();
}

function updateSampleCounter() {
  if (selectedTrainerLabelIndex === -1) return;
  const count = customData.y.filter(val => val === selectedTrainerLabelIndex).length;
  recordedSamplesCount.textContent = count;
}

// Recording handler
recordBtn.addEventListener('mousedown', () => startRecording());
recordBtn.addEventListener('mouseup', () => stopRecording());
recordBtn.addEventListener('mouseleave', () => stopRecording());
recordBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener('touchend', () => stopRecording());

function startRecording() {
  if (selectedTrainerLabelIndex === -1) return;
  isRecording = true;
  tempRecordFrames = [];
  recordBtn.textContent = "Recording...";
  recordBtn.style.backgroundColor = "var(--danger)";
}

function stopRecording() {
  isRecording = false;
  recordBtn.textContent = "Hold to Record Samples";
  recordBtn.style.backgroundColor = "var(--accent)";
  updateLabelsList();
}

clearSamplesBtn.addEventListener('click', () => {
  if (selectedTrainerLabelIndex === -1) return;
  // Remove samples matching current label
  const newX = [];
  const newY = [];
  for (let i = 0; i < customData.y.length; i++) {
    if (customData.y[i] !== selectedTrainerLabelIndex) {
      newX.push(customData.X[i]);
      // Adjust class index if needed, but it's easier to recreate labels
      newY.push(customData.y[i]);
    }
  }
  customData.X = newX;
  customData.y = newY;
  updateLabelsList();
  updateSampleCounter();
});

// Train TF.js Model
trainModelBtn.addEventListener('click', async () => {
  if (customLabels.length < 2) return;
  
  trainingStatusText.textContent = "Preparing data...";
  trainModelBtn.disabled = true;
  
  // Convert customData to tensors
  const xs = tf.tensor2d(customData.X);
  // One-hot encode targets
  const ys = tf.oneHot(tf.tensor1d(customData.y, 'int32'), customLabels.length);
  
  // Define MLP Architecture matching the sequence complexity
  tfModel = tf.sequential();
  tfModel.add(tf.layers.dense({inputShape: [2520], units: 128, activation: 'relu'}));
  tfModel.add(tf.layers.dropout({rate: 0.1}));
  tfModel.add(tf.layers.dense({units: 64, activation: 'relu'}));
  tfModel.add(tf.layers.dense({units: customLabels.length, activation: 'softmax'}));
  
  tfModel.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  trainingStatusText.textContent = "Training...";
  
  await tfModel.fit(xs, ys, {
    epochs: 60,
    batchSize: 16,
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        trainingEpochText.textContent = `${epoch + 1}/60`;
        trainingLossText.textContent = logs.loss.toFixed(4);
      }
    }
  });
  
  trainingStatusText.textContent = "Ready!";
  customModelOption.disabled = false;
  customModelOption.textContent = "TensorFlow.js Custom MLP (Model Ready)";
  modelSelector.value = "custom";
  exportModelBtn.disabled = false; // Enable model export
  alert("Model training completed successfully! Switched to Custom MLP mode.");
  
  // Clean up tensors
  xs.dispose();
  ys.dispose();
});

// Reconstruct Keras MLP Model structure using direct JSON weights
function buildModelFromWeights(weightsData) {
  const model = tf.sequential();
  
  // Layer 1: Dense (63 -> 128)
  model.add(tf.layers.dense({
    units: 128,
    inputShape: [63],
    activation: 'relu',
    weights: [
      tf.tensor2d(weightsData.dense.kernel),
      tf.tensor1d(weightsData.dense.bias)
    ]
  }));
  
  // Layer 2: Dense (128 -> 64)
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu',
    weights: [
      tf.tensor2d(weightsData.dense_1.kernel),
      tf.tensor1d(weightsData.dense_1.bias)
    ]
  }));
  
  // Layer 3: Dense (64 -> 28)
  model.add(tf.layers.dense({
    units: 28,
    activation: 'softmax',
    weights: [
      tf.tensor2d(weightsData.dense_2.kernel),
      tf.tensor1d(weightsData.dense_2.bias)
    ]
  }));
  
  return model;
}

// Fetch and load pre-trained weights on startup
fetch('asl_model_weights.json')
  .then(res => {
    if (!res.ok) throw new Error("File not found");
    return res.json();
  })
  .then(data => {
    pretrainedWeights = data;
    pretrainedModel = buildModelFromWeights(data);
    pretrainedModelOption.disabled = false;
    pretrainedModelOption.textContent = "Pre-trained ASL Model (A-Z, Space, Del)";
    // Default active model selection based on Word Recognition toggle preference
    const wordRecEnabled = localStorage.getItem('wordRecEnabled') !== 'false';
    if (wordRecEnabled) {
      modelSelector.value = tfModel ? 'custom' : 'pretrained';
    } else {
      modelSelector.value = 'similarity';
    }
    console.log("Pre-trained ASL weights loaded and model compiled.");
  })
  .catch(err => {
    console.error("Could not load pre-trained weights:", err);
    pretrainedModelOption.textContent = "Pre-trained ASL Model (Load Error)";
  });

// Function to dynamically load the deployed developer custom sequence model
async function loadDeployedCustomModel() {
  try {
    const labelsRes = await fetch('model/labels.json');
    if (!labelsRes.ok) throw new Error("No custom model label map deployed");
    const labels = await labelsRes.json();
    
    const loadedModel = await tf.loadLayersModel('model/model.json');
    
    // Assign to globals
    customLabels = labels;
    tfModel = loadedModel;
    
    // Auto-detect input shape from model layers!
    if (loadedModel.inputs && loadedModel.inputs[0] && loadedModel.inputs[0].shape) {
      customModelInputShape = loadedModel.inputs[0].shape[1] || 2520;
    }
    
    customModelOption.disabled = false;
    customModelOption.textContent = "Developer Custom Model (Model Loaded)";
    console.log(`Successfully loaded custom developer model (${customModelInputShape}-dim input).`);
    return true;
  } catch (err) {
    console.log("No custom developer model active on startup:", err.message);
    return false;
  }
}

// Initial load request
// Function to fetch the dataset JSON and train the custom model dynamically in the browser
async function autoTrainFromDataset() {
  try {
    const res = await fetch('asl_custom_dataset.json');
    if (!res.ok) throw new Error("No custom dataset found on server");
    const datasetData = await res.json();
    
    const labels = Object.keys(datasetData);
    if (labels.length < 2) return;
    
    console.log("Auto-training background model using dataset categories:", labels);
    
    const inputData = [];
    const outputData = [];
    
    labels.forEach((label, index) => {
      datasetData[label].forEach(features => {
        if (features.length === 2520) {
          inputData.push(features);
          const oneHot = Array(labels.length).fill(0);
          oneHot[index] = 1;
          outputData.push(oneHot);
        }
      });
    });
    
    if (inputData.length === 0) return;
    
    const xs = tf.tensor2d(inputData);
    const ys = tf.tensor2d(outputData);
    
    const model = tf.sequential();
    model.add(tf.layers.dense({
      units: 128,
      activation: 'relu',
      inputShape: [2520]
    }));
    model.add(tf.layers.dropout({rate: 0.1}));
    model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));
    model.add(tf.layers.dense({
      units: labels.length,
      activation: 'softmax'
    }));
    
    model.compile({
      optimizer: tf.train.adam(0.005),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    await model.fit(xs, ys, {
      epochs: 35,
      batchSize: 16,
      verbose: 0
    });
    
    // Assign to globals
    customLabels = labels;
    tfModel = model;
    
    customModelOption.disabled = false;
    customModelOption.textContent = "Developer Custom Model (Model Ready)";
    
    if (wordRecToggle.checked) {
      modelSelector.value = "custom";
    }
    
    console.log("Background auto-training completed successfully. Custom sequence model active.");
    xs.dispose();
    ys.dispose();
  } catch (err) {
    console.log("Dataset auto-training bypass:", err.message);
  }
}

// Initial load request
loadDeployedCustomModel().then(loaded => {
  if (!loaded) {
    autoTrainFromDataset();
  }
});

// Export TF.js Model and Labels
exportModelBtn.addEventListener('click', async () => {
  if (!tfModel) return;
  try {
    trainingStatusText.textContent = "Exporting...";
    // Save model topology and weights (browser downloads files: signcv-custom-mlp-model.json and signcv-custom-mlp-model.weights.bin)
    await tfModel.save('downloads://signcv-custom-mlp-model');
    
    // Also generate and download labels.json
    const labelsBlob = new Blob([JSON.stringify(customLabels, null, 2)], {type: 'application/json'});
    const labelsUrl = URL.createObjectURL(labelsBlob);
    const link = document.createElement('a');
    link.href = labelsUrl;
    link.download = 'labels.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(labelsUrl);
    
    trainingStatusText.textContent = "Ready!";
    alert("Model exported successfully! Check your browser's Downloads directory for the exported 'signcv-custom-mlp-model.json', 'signcv-custom-mlp-model.weights.bin', and 'labels.json' files.");
  } catch (err) {
    console.error("Failed to export model: ", err);
    trainingStatusText.textContent = "Export Error";
    alert("Error: Failed to export model files.");
  }
});

// -------------------------------------------------------------
// 3. Coordinate Normalization Engine
// -------------------------------------------------------------
function normalizeLandmarks(landmarks) {
  // Landmarks: Array of 21 elements with {x, y, z}
  // 1. Shift wrist (index 0) to origin (0, 0, 0)
  const wrist = landmarks[0];
  const shifted = landmarks.map(pt => ({
    x: pt.x - wrist.x,
    y: pt.y - wrist.y,
    z: pt.z - wrist.z
  }));
  
  // 2. Scale landmarks by hand size (Euclidean distance from wrist 0 to Index MCP 5)
  const scaleRef = shifted[5];
  const handSize = Math.sqrt(scaleRef.x**2 + scaleRef.y**2 + scaleRef.z**2) || 1.0;
  
  const normalized = shifted.map(pt => ({
    x: pt.x / handSize,
    y: pt.y / handSize,
    z: pt.z / handSize
  }));
  
  // 3. Flatten to 63-dimensional array
  const flattened = [];
  normalized.forEach(pt => {
    flattened.push(pt.x, pt.y, pt.z);
  });
  
  return { flattened, rawNormalized: normalized, handSize };
}

function extractTwoHandFeatures(multiHandLandmarks, multiHandedness) {
  let leftHandFeatures = Array(63).fill(0);
  let rightHandFeatures = Array(63).fill(0);
  
  if (multiHandLandmarks && multiHandedness) {
    for (let i = 0; i < multiHandLandmarks.length; i++) {
      const handLabel = multiHandedness[i].label;
      const landmarks = multiHandLandmarks[i];
      const { flattened } = normalizeLandmarks(landmarks);
      
      if (handLabel === 'Left') {
        leftHandFeatures = flattened;
      } else if (handLabel === 'Right') {
        rightHandFeatures = flattened;
      }
    }
  }
  return leftHandFeatures.concat(rightHandFeatures);
}

// Calculate 3D distance
function dist3D(pt1, pt2) {
  return Math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 + (pt1.z - pt2.z)**2);
}

// -------------------------------------------------------------
// 4. Default Sign Classification Rules (Alphabet & Numbers)
// -------------------------------------------------------------
function classifyASL(landmarks, normalized) {
  // Extract key points from normalized coordinates
  const wrist = normalized[0];
  const thumbTip = normalized[4], thumbIP = normalized[3], thumbMCP = normalized[2], thumbCMC = normalized[1];
  const indexTip = normalized[8], indexDIP = normalized[7], indexPIP = normalized[6], indexMCP = normalized[5];
  const middleTip = normalized[12], middleDIP = normalized[11], middlePIP = normalized[10], middleMCP = normalized[9];
  const ringTip = normalized[16], ringDIP = normalized[15], ringPIP = normalized[14], ringMCP = normalized[13];
  const pinkyTip = normalized[20], pinkyDIP = normalized[19], pinkyPIP = normalized[18], pinkyMCP = normalized[17];

  // Compute finger wrist distance ratios (finger extended if Tip-Wrist distance > PIP-Wrist distance)
  const indexExt = dist3D(indexTip, wrist) > dist3D(indexPIP, wrist) * 1.08;
  const middleExt = dist3D(middleTip, wrist) > dist3D(middlePIP, wrist) * 1.08;
  const ringExt = dist3D(ringTip, wrist) > dist3D(ringPIP, wrist) * 1.08;
  const pinkyExt = dist3D(pinkyTip, wrist) > dist3D(pinkyPIP, wrist) * 1.08;
  
  // Thumb extension (relative to Index MCP)
  const thumbExt = dist3D(thumbTip, indexMCP) > 1.25;

  // Curled state indicators (fist-like coordinates)
  const indexCurled = indexTip.y > indexMCP.y;
  const middleCurled = middleTip.y > middleMCP.y;
  const ringCurled = ringTip.y > ringMCP.y;
  const pinkyCurled = pinkyTip.y > pinkyMCP.y;

  // Finger touching states (thumb tip proximity to finger tips)
  const indexTouchThumb = dist3D(indexTip, thumbTip) < 0.45;
  const middleTouchThumb = dist3D(middleTip, thumbTip) < 0.45;
  const ringTouchThumb = dist3D(ringTip, thumbTip) < 0.45;
  const pinkyTouchThumb = dist3D(pinkyTip, thumbTip) < 0.45;

  // Spread variables
  const indexMiddleDist = dist3D(indexTip, middleTip);

  // Classification Logic Rules
  // 5 (All extended, spread)
  if (indexExt && middleExt && ringExt && pinkyExt && thumbExt && indexMiddleDist > 0.4) {
    return { label: '5', confidence: 0.95 };
  }
  
  // 4 (Four extended, thumb tucked)
  if (indexExt && middleExt && ringExt && pinkyExt && !thumbExt) {
    return { label: '4', confidence: 0.90 };
  }
  
  // W (Index, Middle, Ring extended, Pinky curled, Thumb tucked)
  if (indexExt && middleExt && ringExt && !pinkyExt && !thumbExt) {
    return { label: 'W', confidence: 0.88 };
  }

  // 3 (Thumb, Index, Middle extended, Ring and Pinky folded)
  if (indexExt && middleExt && !ringExt && !pinkyExt && thumbExt) {
    return { label: '3', confidence: 0.92 };
  }
  
  // V / 2 (Index & Middle extended and spread, others folded)
  if (indexExt && middleExt && !ringExt && !pinkyExt && indexMiddleDist > 0.35) {
    return { label: 'V', confidence: 0.90 };
  }
  
  // U (Index & Middle extended and close together, others folded)
  if (indexExt && middleExt && !ringExt && !pinkyExt && indexMiddleDist <= 0.35) {
    return { label: 'U', confidence: 0.88 };
  }

  // D / 1 (Index extended, middle/ring/pinky folded tight)
  if (indexExt && !middleExt && !ringExt && !pinkyExt && !thumbExt) {
    return { label: 'D', confidence: 0.92 };
  }

  // Y (Thumb & Pinky extended, others folded tight)
  if (thumbExt && pinkyExt && !indexExt && !middleExt && !ringExt) {
    return { label: 'Y', confidence: 0.95 };
  }

  // L (Thumb & Index extended, others folded)
  if (thumbExt && indexExt && !middleExt && !ringExt && !pinkyExt) {
    return { label: 'L', confidence: 0.94 };
  }

  // I / 9 (Pinky extended, others folded)
  if (pinkyExt && !indexExt && !middleExt && !ringExt && !thumbExt) {
    return { label: 'I', confidence: 0.92 };
  }

  // B (All fingers extended straight up, thumb folded in front)
  if (indexExt && middleExt && ringExt && pinkyExt && dist3D(thumbTip, pinkyMCP) < 0.6) {
    return { label: 'B', confidence: 0.92 };
  }

  // F (Index and thumb touching, middle, ring, pinky extended straight up)
  if (indexTouchThumb && middleExt && ringExt && pinkyExt) {
    return { label: 'F', confidence: 0.93 };
  }

  // 9 (Alternative check for 9: ring and pinky extended, middle touch thumb etc.)
  if (indexTouchThumb && middleExt && ringExt && !pinkyExt) {
    return { label: '9', confidence: 0.85 };
  }
  
  // 8 (Middle and thumb touching, others extended)
  if (middleTouchThumb && indexExt && ringExt && pinkyExt) {
    return { label: '8', confidence: 0.90 };
  }

  // 7 (Ring and thumb touching, others extended)
  if (ringTouchThumb && indexExt && middleExt && pinkyExt) {
    return { label: '7', confidence: 0.90 };
  }

  // 6 (Pinky and thumb touching, others extended)
  if (pinkyTouchThumb && indexExt && middleExt && ringExt) {
    return { label: '6', confidence: 0.90 };
  }

  // C (Curved hand shape - fingers semi-extended but not flat)
  const allCurved = [indexTip, middleTip, ringTip, pinkyTip].every(pt => pt.y < indexMCP.y && pt.y > wrist.y);
  if (allCurved && thumbExt && dist3D(indexTip, thumbTip) > 0.4 && dist3D(indexTip, thumbTip) < 0.8) {
    return { label: 'C', confidence: 0.82 };
  }

  // A (Fist, thumb extended alongside index MCP)
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && thumbExt && thumbTip.y < indexMCP.y) {
    return { label: 'A', confidence: 0.88 };
  }

  // E (Fist, fingers curled tightly, thumb folded across fingers)
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && dist3D(thumbTip, ringMCP) < 0.4) {
    return { label: 'E', confidence: 0.86 };
  }

  // S (Fist, thumb locked over index & middle fingers)
  if (indexCurled && middleCurled && ringCurled && pinkyCurled && dist3D(thumbTip, middlePIP) < 0.3) {
    return { label: 'S', confidence: 0.85 };
  }

  return { label: 'None', confidence: 0.0 };
}

// -------------------------------------------------------------
// 5. MediaPipe Result Handler & Prediction pipeline
// -------------------------------------------------------------
function onResults(results) {
  // Hide loading spinner on first results frame
  if (loadingSpinner.style.opacity !== '0') {
    loadingSpinner.style.opacity = '0';
    setTimeout(() => { loadingSpinner.style.display = 'none'; }, 500);
    engineStatus.textContent = "Engine Ready";
  }

  // 1. Performance calculation (FPS & Latency)
  const now = performance.now();
  const fps = Math.round(1000 / (now - lastFrameTime));
  lastFrameTime = now;
  fpsHistory.push(fps);
  if (fpsHistory.length > 30) fpsHistory.shift();
  const avgFps = Math.round(fpsHistory.reduce((a,b) => a+b, 0) / fpsHistory.length);
  fpsCounter.textContent = avgFps < 10 ? `0${avgFps}` : avgFps;

  // Live hand detection processing

  let handDetected = false;
  // We calculate inference latency starting from the dispatch timestamp
  let activeHandLandmarks = null;
  let activeHandLabel = "Right"; // Default
  // Reset hand indicator pills
  leftHandPill.classList.remove('active');
  rightHandPill.classList.remove('active');

  if (results.multiHandLandmarks) {
    handDetected = true;
    noHandFramesCount = 0; // reset timeout
    lastHandLandmarks = results.multiHandLandmarks; // Cache all hands for onFrame rendering

    // We process the first hand detected for predictions
    activeHandLandmarks = results.multiHandLandmarks[0] || null;
    
    if (results.multiHandedness) {
      results.multiHandedness.forEach(handedness => {
        if (handedness.label === "Left") leftHandPill.classList.add('active');
        if (handedness.label === "Right") rightHandPill.classList.add('active');
      });
    }

    // Inspect first 5 coordinates for layout
    updateCoordinateInspector(activeHandLandmarks);
  } else {
    lastHandLandmarks = null; // Clear cached landmarks
    noHandFramesCount++;
    if (noHandFramesCount > NO_HAND_TIMEOUT) {
      handleNoHandsTimeout();
    }
  }

  // 2. Queue sequence frames
  const twoHandFeatures = handDetected ? extractTwoHandFeatures(results.multiHandLandmarks, results.multiHandedness) : Array(126).fill(0);
  coordsBuffer.push(twoHandFeatures);
  if (coordsBuffer.length > 20) {
    coordsBuffer.shift();
  }

  // 3. Gesture Prediction execution
  if (handDetected) {
    const { flattened, rawNormalized } = activeHandLandmarks ? normalizeLandmarks(activeHandLandmarks) : { flattened: Array(63).fill(0), rawNormalized: [] };
    
    // If training is active, capture sequence sample (matching sequence length of developer model)
    if (isRecording && selectedTrainerLabelIndex !== -1) {
      tempRecordFrames.push(twoHandFeatures);
      if (tempRecordFrames.length === 20) {
        const flattenedSequence = [];
        tempRecordFrames.forEach(frame => {
          flattenedSequence.push(...frame);
        });
        customData.X.push(flattenedSequence);
        customData.y.push(selectedTrainerLabelIndex);
        updateSampleCounter();
        
        // Sliding window for collection data speed
        tempRecordFrames = tempRecordFrames.slice(10);
      }
    }

    const activeModel = modelSelector.value;
    let prediction = { label: 'None', confidence: 0.0 };

    if (activeModel === 'similarity' && activeHandLandmarks) {
      // Run Geometric Rules Engine
      prediction = classifyASL(activeHandLandmarks, rawNormalized);
    } else if (activeModel === 'pretrained' && pretrainedModel) {
      // Run pre-trained Keras model using direct Tensor weights (63-dim)
      tf.tidy(() => {
        const inputTensor = tf.tensor2d([flattened]);
        const output = pretrainedModel.predict(inputTensor);
        const probabilities = output.dataSync();
        const maxProbIdx = output.argMax(-1).dataSync()[0];
        
        const labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space'];
        
        prediction = {
          label: labels[maxProbIdx],
          confidence: probabilities[maxProbIdx]
        };
      });
    } else if (activeModel === 'custom' && tfModel) {
      // Run TensorFlow.js MLP Inference using dynamic input dimension mapping
      let inputDataArray = null;
      
      if (customModelInputShape === 63) {
        inputDataArray = flattened;
      } else if (customModelInputShape === 126) {
        inputDataArray = twoHandFeatures;
      } else if (customModelInputShape === 2520) {
        if (coordsBuffer.length === 20) {
          inputDataArray = [];
          coordsBuffer.forEach(frame => {
            inputDataArray.push(...frame);
          });
        }
      }
      
      if (inputDataArray) {
        tf.tidy(() => {
          const inputTensor = tf.tensor2d([inputDataArray]);
          const output = tfModel.predict(inputTensor);
          const probabilities = output.dataSync();
          const maxProbIdx = output.argMax(-1).dataSync()[0];
          
          prediction = {
            label: customLabels[maxProbIdx],
            confidence: probabilities[maxProbIdx]
          };
        });
      }
    }

    const inferenceTime = Math.round(performance.now() - inferenceStartTime);
    latencyCounter.textContent = `${inferenceTime} ms`;

    // Process output buffer smoothing
    let finalLabel = prediction.label;
    let finalConfidence = prediction.confidence;

    const threshold = parseFloat(confidenceThresholdSlider.value);

    if (finalConfidence >= threshold && finalLabel !== 'None') {
      if (smoothingToggle.checked) {
        predictedQueue.push(finalLabel);
        if (predictedQueue.length > QUEUE_SIZE) predictedQueue.shift();
        
        // Find mode (most frequent item) in prediction queue
        const frequencyMap = {};
        let maxCount = 0;
        let modeLabel = finalLabel;
        
        predictedQueue.forEach(label => {
          frequencyMap[label] = (frequencyMap[label] || 0) + 1;
          if (frequencyMap[label] > maxCount) {
            maxCount = frequencyMap[label];
            modeLabel = label;
          }
        });
        
        // Only accept if mode comprises at least 45% of the queue
        if (maxCount >= QUEUE_SIZE * 0.45) {
          finalLabel = modeLabel;
        } else {
          finalLabel = '-';
        }
      }

      // Live update prediction HUD elements
      predictedLetterText.textContent = finalLabel;
      predictedConfidenceText.textContent = `${Math.round(finalConfidence * 100)}%`;
      confidenceBar.style.width = `${finalConfidence * 100}%`;

      // Word/Sentence compilation streak tracking
      if (finalLabel !== '-' && finalLabel !== 'None') {
        activeCharPreview.textContent = finalLabel;
        
        if (finalLabel === lastPredictedChar) {
          charStreakCount++;
          if (wordRecToggle.checked) {
            if (charStreakCount === STREAK_THRESHOLD) {
              appendCharacter(finalLabel);
              charStreakCount = 0;
            }
          } else {
            // Direct typing mode: fast letter typing on stable hold
            const typingThreshold = Math.max(4, Math.round(STREAK_THRESHOLD * 0.6));
            if (charStreakCount === typingThreshold) {
              if (finalLabel === 'space') {
                currentSentence += " ";
                speakText("space");
              } else if (finalLabel === 'del') {
                currentSentence = currentSentence.slice(0, -1);
                speakText("delete");
              } else {
                currentSentence += finalLabel;
                speakText(finalLabel.toLowerCase());
              }
              outputSentenceDiv.textContent = currentSentence;
              charStreakCount = 0;
            }
          }
        } else {
          lastPredictedChar = finalLabel;
          charStreakCount = 0;
        }
      }
    } else {
      // Clear live prediction displays if under threshold
      predictedLetterText.textContent = "-";
      predictedConfidenceText.textContent = "0%";
      confidenceBar.style.width = "0%";
      activeCharPreview.textContent = "-";
      lastPredictedChar = "";
      charStreakCount = 0;
    }
  } else {
    predictedLetterText.textContent = "-";
    predictedConfidenceText.textContent = "0%";
    confidenceBar.style.width = "0%";
    activeCharPreview.textContent = "-";
    lastPredictedChar = "";
    charStreakCount = 0;
    latencyCounter.textContent = "0 ms";
  }


}

// -------------------------------------------------------------
// 6. Sentence Builder Operations
// -------------------------------------------------------------
function appendCharacter(char) {
  // If custom developer model is active, treat output label as a completed word
  if (modelSelector.value === 'custom') {
    if (char === 'space') {
      if (currentSentence.length > 0 && !currentSentence.endsWith(" ")) {
        currentSentence += " ";
      }
      outputSentenceDiv.textContent = currentSentence;
      speakText("space");
      return;
    }
    if (char === 'del') {
      if (currentSentence.length > 0) {
        const words = currentSentence.trimEnd().split(' ');
        words.pop();
        currentSentence = words.join(' ');
        if (currentSentence.length > 0) {
          currentSentence += " ";
        }
        outputSentenceDiv.textContent = currentSentence;
      }
      speakText("delete");
      return;
    }
    currentSentence += char + " ";
    outputSentenceDiv.textContent = currentSentence;
    speakText(char);
    return;
  }

  // If it's a numeric digit, treat as immediate input, else letters build words

  // Handle special characters from pre-trained model
  if (char === 'space') {
    if (currentWord.length > 0) {
      currentSentence += currentWord + " ";
      currentWord = "";
    } else if (currentSentence.length > 0 && !currentSentence.endsWith(" ")) {
      currentSentence += " ";
    }
    outputSentenceDiv.textContent = currentSentence;
    speakText("space");
    return;
  }

  if (char === 'del' || char === 'delete' || char === 'Backspace') {
    if (currentWord.length > 0) {
      currentWord = currentWord.slice(0, -1);
      activeCharPreview.textContent = currentWord.slice(-1) || "-";
      outputSentenceDiv.textContent = currentSentence + currentWord;
    } else if (currentSentence.length > 0) {
      currentSentence = currentSentence.trimEnd().slice(0, -1);
      outputSentenceDiv.textContent = currentSentence;
    }
    speakText("delete");
    return;
  }

  // Handle auto-spacing for alphabetic characters
  if (currentWord.length > 0 && currentWord.endsWith(char)) {
    // Avoid spamming double letters instantly unless held
    return;
  }
  
  currentWord += char;
  outputSentenceDiv.textContent = currentSentence + currentWord;
  
  // Play subtle feedback click sound or TTS of the letter
  speakText(char.toLowerCase());
}

function handleNoHandsTimeout() {
  // When hands leave screen for 1.5 seconds, we finalize the active word
  if (currentWord.length > 0) {
    currentSentence += currentWord;
    
    if (autospaceToggle.checked) {
      currentSentence += " ";
    }
    
    outputSentenceDiv.textContent = currentSentence;
    
    // Auto speak the completed word
    if (ttsToggle.checked) {
      speakText(currentWord);
    }
    
    currentWord = "";
  }
  noHandFramesCount = 0;
}

// -------------------------------------------------------------
// 7. Auxiliary Utilities (Coordinate Inspector & Setup)
// -------------------------------------------------------------
function updateCoordinateInspector(landmarks) {
  if (!landmarks || landmarks.length === 0) return;
  coordInspector.innerHTML = '';
  // Show first 6 joint details for size
  const joints = ['Wrist', 'Thumb C.', 'Thumb T.', 'Index K.', 'Index T.', 'Middle T.'];
  const indices = [0, 2, 4, 5, 8, 12];
  
  indices.forEach((index, i) => {
    const pt = landmarks[index];
    const item = document.createElement('div');
    item.style.marginBottom = '6px';
    item.innerHTML = `
      <strong>${joints[i]}:</strong>
      <span class="coord-item">X: ${pt.x.toFixed(3)}</span>
      <span class="coord-item">Y: ${pt.y.toFixed(3)}</span>
      <span class="coord-item">Z: ${pt.z.toFixed(3)}</span>
    `;
    coordInspector.appendChild(item);
  });
}

// -------------------------------------------------------------
// 8. Main Initialization
// -------------------------------------------------------------
window.addEventListener('load', () => {
  // Load virtual mic & word recognition configurations
  const virtualMicEnabled = localStorage.getItem('virtualMicEnabled') === 'true';
  virtualMicToggle.checked = virtualMicEnabled;
  
  const wordRecEnabled = localStorage.getItem('wordRecEnabled') !== 'false'; // default true
  wordRecToggle.checked = wordRecEnabled;
  
  // Try to load custom model on startup, fallback to auto-training from dataset if not deployed
  loadDeployedCustomModel().then(async (loaded) => {
    if (!loaded) {
      await autoTrainFromDataset();
      loaded = tfModel !== null;
    }
    if (wordRecEnabled) {
      modelSelector.value = loaded ? 'custom' : 'pretrained';
    } else {
      modelSelector.value = 'similarity';
    }
  });
  
  // Start checking connection status
  checkVirtualMicConnection();
  setInterval(checkVirtualMicConnection, 5000);

  // Set up canvas default resolution
  canvasElement.width = 640;
  canvasElement.height = 480;

  // Initialize MediaPipe Hands
  handsEngine = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  handsEngine.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.65,
    minTrackingConfidence: 0.65
  });

  handsEngine.onResults(onResults);

  // 1. Independent Rendering Loop (runs at 60 FPS)
  function renderLoop() {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);
    
    // Draw live webcam frame
    canvasCtx.drawImage(webcamElement, 0, 0, canvasElement.width, canvasElement.height);
        // 2. Overlay landmarks if they exist (Minimal B&W style)
      if (lastHandLandmarks) {
        for (let i = 0; i < lastHandLandmarks.length; i++) {
          const landmarks = lastHandLandmarks[i];
          drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#ffffff', lineWidth: 3});
          drawLandmarks(canvasCtx, landmarks, {color: '#888888', lineWidth: 1, radius: 4});
        }
      }
    
    canvasCtx.restore();
    requestAnimationFrame(renderLoop);
  }

  const THROTTLE_MS = 50; // Limit processing to max 20 FPS for ML

  // Initialize Camera
  cameraHelper = new Camera(webcamElement, {
    onFrame: async () => {
      const t0 = performance.now();
      inferenceStartTime = t0; // Mark start time of ML pass
      
      try {
        await handsEngine.send({image: webcamElement});
      } catch (err) {
        console.error("MediaPipe Processing Error:", err);
      }
      
      // Calculate dynamic delay to throttle to target 20 FPS
      const elapsed = performance.now() - t0;
      const delay = Math.max(0, THROTTLE_MS - elapsed);
      if (delay > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    },
    width: 640,
    height: 480
  });

  cameraHelper.start()
    .then(() => {
      console.log("Webcam feed started successfully.");
      requestAnimationFrame(renderLoop); // Start the independent drawing loop
    })
    .catch(err => {
      console.error("Failed to acquire webcam feed: ", err);
      engineStatus.textContent = "Camera Error";
      alert("Error: Camera acquisition blocked or not found. Please grant permissions and reload.");
    });
});
