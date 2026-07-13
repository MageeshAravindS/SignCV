// Global developer state
let handsEngine = null;
let cameraHelper = null;
let lastHandLandmarks = null;
let isRecording = false;
let tempRecordFrames = [];

// Datasets
let customLabels = [];
let dataset = {}; // label -> array of 63-dim arrays
let activeLabel = null;

// UI Elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');
const coordInspector = document.getElementById('coord-inspector');
const fpsCounter = document.getElementById('fps-counter');

// Trainer Elements
const newGestureNameInput = document.getElementById('new-gesture-name');
const addGestureBtn = document.getElementById('add-gesture-btn');
const gestureLabelsUl = document.getElementById('gesture-labels-ul');
const recorderPanel = document.getElementById('recorder-panel');
const recordingActiveLabel = document.getElementById('recording-active-label');
const recordedSamplesCount = document.getElementById('recorded-samples-count');
const recordBtn = document.getElementById('record-btn');
const clearSamplesBtn = document.getElementById('clear-samples-btn');
const trainModelBtn = document.getElementById('train-model-btn');
const terminalLog = document.getElementById('terminal-log');

// FPS calculations
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;

// -------------------------------------------------------------
// Terminal Logging Helper
// -------------------------------------------------------------
function appendTerminalLog(text) {
  const time = new Date().toTimeString().split(' ')[0];
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `<span class="log-time">[${time}]</span><span class="log-text">&gt; ${text}</span>`;
  terminalLog.appendChild(line);
  terminalLog.scrollTop = terminalLog.scrollHeight;
}

// -------------------------------------------------------------
// Category manager
// -------------------------------------------------------------
addGestureBtn.addEventListener('click', () => {
  const name = newGestureNameInput.value.trim().toUpperCase();
  if (!name) return;
  
  if (customLabels.includes(name)) {
    appendTerminalLog(`Error: Label "${name}" already exists!`);
    return;
  }
  
  customLabels.push(name);
  dataset[name] = [];
  newGestureNameInput.value = "";
  appendTerminalLog(`Added gesture category label: "${name}"`);
  updateLabelsList();
});

function updateLabelsList() {
  if (customLabels.length === 0) {
    gestureLabelsUl.innerHTML = '<li class="empty-note">No training labels active. Enter label tag above.</li>';
    recorderPanel.style.display = 'none';
    trainModelBtn.disabled = true;
    return;
  }
  
  gestureLabelsUl.innerHTML = '';
  customLabels.forEach(label => {
    const li = document.createElement('li');
    li.textContent = label;
    if (label === activeLabel) {
      li.className = 'selected';
    }
    
    const countSpan = document.createElement('span');
    countSpan.className = 'label-count';
    countSpan.textContent = `[${dataset[label].length} samples]`;
    
    li.appendChild(countSpan);
    
    li.addEventListener('click', () => {
      activeLabel = label;
      updateLabelsList();
      showRecorder(label);
    });
    
    gestureLabelsUl.appendChild(li);
  });
  
  // Enable training if we have at least 2 categories and each has at least 5 samples
  const canTrain = customLabels.length >= 2 && customLabels.every(l => dataset[l].length >= 5);
  trainModelBtn.disabled = !canTrain;
}

function showRecorder(label) {
  recorderPanel.style.display = 'flex';
  recordingActiveLabel.textContent = label;
  recordedSamplesCount.textContent = dataset[label].length;
  appendTerminalLog(`Selected active target label: "${label}"`);
}

// -------------------------------------------------------------
// Keypoint Dataset Recorder
// -------------------------------------------------------------
let recordInterval = null;

function startRecording(e) {
  if (e) e.preventDefault();
  if (isRecording || !activeLabel) return;
  
  if (!lastHandLandmarks || lastHandLandmarks.length === 0) {
    appendTerminalLog(`Warning: No hand detected in camera frame. Cannot start recording.`);
    return;
  }
  
  isRecording = true;
  tempRecordFrames = [];
  recordBtn.textContent = "RECORDING...";
  recordBtn.style.background = "var(--danger)";
  recordBtn.style.color = "#fff";
  recordBtn.style.borderColor = "var(--danger)";
  appendTerminalLog(`Continuous 20-frame sequence recording started for label: "${activeLabel}"...`);
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  
  recordBtn.textContent = "Record Sample (Hold)";
  recordBtn.style.background = "";
  recordBtn.style.color = "";
  recordBtn.style.borderColor = "";
  appendTerminalLog(`Recording hold released. Total dataset size for "${activeLabel}": ${dataset[activeLabel].length}`);
  
  pushDatasetBtn.disabled = false;
  updateLabelsList();
}

// Attach listeners for Hold action (Mouse & Touch)
recordBtn.addEventListener('mousedown', startRecording);
recordBtn.addEventListener('mouseup', stopRecording);
recordBtn.addEventListener('mouseleave', stopRecording);
recordBtn.addEventListener('touchstart', startRecording);
recordBtn.addEventListener('touchend', stopRecording);

// Dataset Export, Import & Git Push Operations
const exportDatasetBtn = document.getElementById('export-dataset-btn');
const importDatasetTriggerBtn = document.getElementById('import-dataset-trigger-btn');
const importDatasetFile = document.getElementById('import-dataset-file');
const pushDatasetBtn = document.getElementById('push-dataset-btn');

exportDatasetBtn.addEventListener('click', () => {
  if (customLabels.length === 0) {
    appendTerminalLog("Export aborted: Dataset is empty.");
    return;
  }
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(dataset, null, 2));
  const downloadAnchor = document.createElement('a');
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", "asl_custom_dataset.json");
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
  appendTerminalLog("Exported raw dataset JSON file to downloads.");
});

importDatasetTriggerBtn.addEventListener('click', () => {
  importDatasetFile.click();
});

importDatasetFile.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = (event) => {
    try {
      const importedData = JSON.parse(event.target.result);
      
      // Validation check
      let valid = true;
      for (const k in importedData) {
        if (!Array.isArray(importedData[k])) {
          valid = false;
        }
      }
      if (!valid) throw new Error("File must be a JSON object containing coordinate arrays.");
      
      // Merge / append instead of overwrite!
      let newCount = 0;
      for (const k in importedData) {
        if (!dataset[k]) {
          dataset[k] = [];
          customLabels.push(k);
        }
        const initialLen = dataset[k].length;
        importedData[k].forEach(sample => {
          dataset[k].push(sample);
        });
        newCount += (dataset[k].length - initialLen);
      }
      
      activeLabel = customLabels[0] || null;
      updateLabelsList();
      if (activeLabel) {
        showRecorder(activeLabel);
      }
      appendTerminalLog(`Successfully merged dataset! Appended ${newCount} samples across categories [${Object.keys(importedData).join(', ')}]`);
      pushDatasetBtn.disabled = false;
    } catch (err) {
      appendTerminalLog("Error parsing dataset JSON: " + err.message);
    }
  };
  reader.readAsText(file);
});

pushDatasetBtn.addEventListener('click', async () => {
  pushDatasetBtn.disabled = true;
  appendTerminalLog("Uploading current dataset JSON to local server...");
  
  try {
    const res = await fetch('/api/push-dataset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ dataset })
    });
    const result = await res.json();
    if (result.success) {
      appendTerminalLog("DATASET COMMITTED AND PUSHED TO GITHUB SUCCESSFULLY!");
      appendTerminalLog(`Git Output: ${result.git_output || 'Pushed successfully.'}`);
    } else {
      appendTerminalLog("Dataset Git deployment failed: " + result.error);
    }
  } catch (err) {
    appendTerminalLog("Network error pushing dataset: " + err.message);
  }
  pushDatasetBtn.disabled = false;
});

clearSamplesBtn.addEventListener('click', () => {
  if (!activeLabel) return;
  dataset[activeLabel] = [];
  recordedSamplesCount.textContent = 0;
  appendTerminalLog(`Cleared all recorded samples for category: "${activeLabel}"`);
  updateLabelsList();
});

// -------------------------------------------------------------
// Coordinate Normalization Engine
// -------------------------------------------------------------
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const shifted = landmarks.map(pt => ({
    x: pt.x - wrist.x,
    y: pt.y - wrist.y,
    z: pt.z - wrist.z
  }));
  
  const scaleRef = shifted[5];
  const handSize = Math.sqrt(scaleRef.x**2 + scaleRef.y**2 + scaleRef.z**2) || 1.0;
  
  const flattened = [];
  shifted.forEach(pt => {
    flattened.push(pt.x / handSize, pt.y / handSize, pt.z / handSize);
  });
  
  return flattened;
}

function extractTwoHandFeatures(multiHandLandmarks, multiHandedness) {
  let leftHandFeatures = Array(63).fill(0);
  let rightHandFeatures = Array(63).fill(0);
  
  if (multiHandLandmarks && multiHandedness) {
    for (let i = 0; i < multiHandLandmarks.length; i++) {
      const handLabel = multiHandedness[i].label;
      const landmarks = multiHandLandmarks[i];
      const flattened = normalizeLandmarks(landmarks);
      
      if (handLabel === 'Left') {
        leftHandFeatures = flattened;
      } else if (handLabel === 'Right') {
        rightHandFeatures = flattened;
      }
    }
  }
  return leftHandFeatures.concat(rightHandFeatures);
}

// Update coordinate stream viewer
function updateCoordinateStream(multiHandLandmarks, multiHandedness) {
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
    coordInspector.innerHTML = '<span class="empty-note">&gt; No hand coordinates intercepted. Stream offline.</span>';
    return;
  }
  
  coordInspector.innerHTML = '';
  
  for (let i = 0; i < multiHandLandmarks.length; i++) {
    const label = multiHandedness[i].label;
    const landmarks = multiHandLandmarks[i];
    const ptList = normalizeLandmarks(landmarks);
    
    const title = document.createElement('div');
    title.style.color = 'var(--primary)';
    title.style.marginTop = i > 0 ? '10px' : '0';
    title.style.marginBottom = '6px';
    title.textContent = `${label.toUpperCase()} HAND - 63-FLOAT ARRAY:`;
    coordInspector.appendChild(title);
    
    for (let j = 0; j < 21; j++) {
      const idx = j * 3;
      const item = document.createElement('div');
      item.innerHTML = `[Pt ${String(j).padStart(2, '0')}] X: ${ptList[idx].toFixed(4)} | Y: ${ptList[idx+1].toFixed(4)} | Z: ${ptList[idx+2].toFixed(4)}`;
      coordInspector.appendChild(item);
    }
  }
}

function dist3D(pt1, pt2) {
  return Math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 + (pt1.z - pt2.z)**2);
}

// -------------------------------------------------------------
// MLP Trainer and Serializer
// -------------------------------------------------------------
trainModelBtn.addEventListener('click', async () => {
  trainModelBtn.disabled = true;
  appendTerminalLog(`Initiating TensorFlow.js custom MLP network compiler...`);
  
  // 1. Prepare tensors
  const inputData = [];
  const outputData = [];
  
  customLabels.forEach((label, index) => {
    dataset[label].forEach(features => {
      inputData.push(features);
      // One-hot encoding
      const oneHot = Array(customLabels.length).fill(0);
      oneHot[index] = 1;
      outputData.push(oneHot);
    });
  });
  
  const xs = tf.tensor2d(inputData);
  const ys = tf.tensor2d(outputData);
  
  appendTerminalLog(`Compiled tensors successfully! Features shape: [${xs.shape}], Labels shape: [${ys.shape}]`);
  
  // 2. Define Model
  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 128,
    activation: 'relu',
    inputShape: [2520] // 20 frames * 126 features
  }));
  model.add(tf.layers.dense({
    units: 64,
    activation: 'relu'
  }));
  model.add(tf.layers.dense({
    units: customLabels.length,
    activation: 'softmax'
  }));
  
  model.compile({
    optimizer: tf.train.adam(0.005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  appendTerminalLog(`Starting neural network training. Total Epochs: 50...`);
  
  // 3. Train Model
  await model.fit(xs, ys, {
    epochs: 50,
    batchSize: 8,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 10 === 0 || epoch === 0 || epoch === 49) {
          appendTerminalLog(`Epoch ${String(epoch + 1).padStart(2, '0')}/50 - loss: ${logs.loss.toFixed(4)} - acc: ${logs.acc.toFixed(4)}`);
        }
      }
    }
  });
  
  appendTerminalLog(`Training completed with success. Beginning Base64 serialization...`);
  
  // 4. Serialize weights and POST to API
  try {
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
      const base64Weights = arrayBufferToBase64(artifacts.weightData);
      
      const payload = {
        modelTopology: artifacts.modelTopology,
        weightSpecs: artifacts.weightSpecs,
        weightsBase64: base64Weights,
        labels: customLabels
      };
      
      appendTerminalLog(`Model serialized. Payload size: ${Math.round(payload.weightsBase64.length / 1024)} KB. Uploading to local server...`);
      
      const res = await fetch('/api/deploy-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const result = await res.json();
      if (result.success) {
        appendTerminalLog(`DEPLOYMENT SUCCESSFUL!`);
        appendTerminalLog(`Git Push Output: ${result.git_output || 'Model pushed successfully to remote GitHub repository.'}`);
      } else {
        appendTerminalLog(`Deployment Failed: ${result.error}`);
      }
      
      return { modelArtifactsInfo: { dateSaved: new Date().toISOString() } };
    }));
  } catch (err) {
    appendTerminalLog(`Serialization Error: ${err}`);
  }
  
  // Cleanup Tensors
  xs.dispose();
  ys.dispose();
  model.dispose();
  updateLabelsList();
});

// Convert ArrayBuffer to Base64
function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

// -------------------------------------------------------------
// MediaPipe Listener & Render Loop
// -------------------------------------------------------------
function onResults(results) {
  // Update FPS count
  frameCount++;
  const time = performance.now();
  if (time >= lastFrameTime + 1000) {
    fps = Math.round((frameCount * 1000) / (time - lastFrameTime));
    fpsCounter.textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastFrameTime = time;
  }

  lastHandLandmarks = results.multiHandLandmarks || [];
  
  // Render webcam frames and skeletal maps
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.translate(canvasElement.width, 0);
  canvasCtx.scale(-1, 1);
  
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
  
  if (lastHandLandmarks.length > 0) {
    for (const landmarks of lastHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00ff66', lineWidth: 2});
      drawLandmarks(canvasCtx, landmarks, {color: '#00cc52', lineWidth: 1, radius: 3});
    }
  }

  // 1. Extract 126-dim two-hand features
  const twoHandFeatures = extractTwoHandFeatures(results.multiHandLandmarks, results.multiHandedness);
  
  // 2. Stream coordinates
  updateCoordinateStream(results.multiHandLandmarks, results.multiHandedness);
  
  // 3. Dataset hold-to-record collection
  if (isRecording) {
    tempRecordFrames.push(twoHandFeatures);
    if (tempRecordFrames.length === 20) {
      const flattenedSequence = [];
      tempRecordFrames.forEach(frame => {
        flattenedSequence.push(...frame);
      });
      
      dataset[activeLabel].push(flattenedSequence);
      recordedSamplesCount.textContent = dataset[activeLabel].length;
      appendTerminalLog(`Saved 20-frame motion sequence sample #${dataset[activeLabel].length} for "${activeLabel}"`);
      
      // Implement sliding window to collect smooth overlays
      tempRecordFrames = tempRecordFrames.slice(10); 
      updateLabelsList();
    }
  }
  canvasCtx.restore();
}

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
window.addEventListener('load', () => {
  canvasElement.width = 640;
  canvasElement.height = 480;
  
  // Setup MediaPipe Hands Engine
  handsEngine = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });
  
  handsEngine.setOptions({
    maxNumHands: 2, // Support two hands tracking simultaneously
    modelComplexity: 1,
    minDetectionConfidence: 0.70,
    minTrackingConfidence: 0.70
  });
  
  handsEngine.onResults(onResults);
  
  // Initialize webcam
  cameraHelper = new Camera(webcamElement, {
    onFrame: async () => {
      await handsEngine.send({image: webcamElement});
    },
    width: 640,
    height: 480
  });
  
  cameraHelper.start()
    .then(() => {
      appendTerminalLog(`Camera stream acquired. MediaPipe pipeline initialized.`);
    })
    .catch(err => {
      appendTerminalLog(`CRITICAL: Camera acquisition failed! ${err}`);
    });

  // Automatically pull the existing custom dataset on startup if it exists
  fetch('asl_custom_dataset.json')
    .then(res => {
      if (!res.ok) throw new Error("Dataset file not found on server");
      return res.json();
    })
    .then(data => {
      dataset = data;
      customLabels = Object.keys(dataset);
      activeLabel = customLabels[0] || null;
      
      updateLabelsList();
      if (activeLabel) {
        showRecorder(activeLabel);
      }
      appendTerminalLog(`Successfully synced dataset from repository. Loaded categories: [${customLabels.join(', ')}]`);
    })
    .catch(err => {
      console.log("No existing dataset found on repository startup:", err.message);
      appendTerminalLog("No existing custom dataset found in repository. Ready to record new tags.");
    });
});
