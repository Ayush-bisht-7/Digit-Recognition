// DOM Elements
const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const clearBtn = document.getElementById('clear-btn');
const predictedDigitEl = document.getElementById('predicted-digit');
const confidencePctEl = document.getElementById('confidence-percentage');
const digitBarsListEl = document.getElementById('digit-bars-list');

// App variables
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let modelWeights = null;

// Initialize Canvas
function initCanvas() {
    // Fill canvas with black background (matching MNIST format)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing styles
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Add event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    
    // Touch support for mobile devices
    canvas.addEventListener('touchstart', (e) => {
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        lastX = touch.clientX - rect.left;
        lastY = touch.clientY - rect.top;
        isDrawing = true;
        e.preventDefault();
    });
    
    canvas.addEventListener('touchmove', (e) => {
        if (!isDrawing) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        lastX = x;
        lastY = y;
        runInference();
        e.preventDefault();
    });
    
    canvas.addEventListener('touchend', () => {
        isDrawing = false;
    });

    clearBtn.addEventListener('click', clearCanvas);
}

// Drawing Logic
function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    lastX = e.clientX - rect.left;
    lastY = e.clientY - rect.top;
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    lastX = x;
    lastY = y;
    runInference();
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictedDigitEl.textContent = '-';
    predictedDigitEl.parentElement.style.textShadow = 'none';
    confidencePctEl.textContent = '0.0%';
    updateBars(new Array(10).fill(0));
}

// Initialize distribution bars
function initBars() {
    digitBarsListEl.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'digit-row';
        row.id = `digit-row-${i}`;
        
        row.innerHTML = `
            <div class="digit-label">${i}</div>
            <div class="digit-bar-bg">
                <div class="digit-bar-fill" id="bar-${i}" style="width: 0%"></div>
            </div>
            <div class="digit-percent" id="percent-${i}">0.0%</div>
        `;
        digitBarsListEl.appendChild(row);
    }
}

// Update the distribution bars and predicted digit styles
function updateBars(probs, predictedDigitIndex = -1) {
    for (let i = 0; i < 10; i++) {
        const fill = document.getElementById(`bar-${i}`);
        const text = document.getElementById(`percent-${i}`);
        const row = document.getElementById(`digit-row-${i}`);
        
        const pct = (probs[i] * 100).toFixed(1) + '%';
        fill.style.width = pct;
        text.textContent = pct;
        
        if (i === predictedDigitIndex) {
            row.classList.add('predicted');
        } else {
            row.classList.remove('predicted');
        }
    }
}

// Load Model Weights
async function loadModel() {
    try {
        console.log("Fetching model parameters...");
        const response = await fetch('mnist_weights.json');
        if (!response.ok) {
            throw new Error(`Failed to load weights: ${response.statusText}`);
        }
        const data = await response.json();
        
        // Flatten the bias arrays (shape [10, 1] in JSON) into 1D vectors of size 10
        modelWeights = {
            W1: data.W1,
            b1: data.b1.map(x => x[0]),
            W2: data.W2,
            b2: data.b2.map(x => x[0])
        };
        console.log("Model parameters loaded successfully!", modelWeights);
    } catch (error) {
        console.error("Error loading model parameters:", error);
        predictedDigitEl.textContent = 'Err';
        confidencePctEl.textContent = 'Weights not loaded';
    }
}

// Get 28x28 grayscale vector from canvas matching MNIST preprocessing
function getGrayscalePixels() {
    // 1. Get raw pixel data from main canvas (280x280)
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    
    // 2. Find the bounding box of the drawing (non-black pixels)
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = 0;
    let maxY = 0;
    let hasDrawing = false;
    
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const idx = (y * canvas.width + x) * 4;
            const r = data[idx]; // Since drawing is white on black, R channel represents stroke
            
            if (r > 10) { // Threshold for drawn pixel
                hasDrawing = true;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    
    if (!hasDrawing) {
        return new Float32Array(784);
    }
    
    // Bounding box dimensions
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;
    
    // Create an offscreen canvas to hold the cropped digit
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = width;
    cropCanvas.height = height;
    const cropCtx = cropCanvas.getContext('2d');
    
    // Copy only the bounding box region from main canvas
    cropCtx.drawImage(canvas, minX, minY, width, height, 0, 0, width, height);
    
    // 3. MNIST standard: Scale digit to fit in a 20x20 box
    const targetSize = 20;
    const scale = Math.min(targetSize / width, targetSize / height);
    const scaledWidth = width * scale;
    const scaledHeight = height * scale;
    
    // 4. Create final 28x28 canvas and center the scaled digit
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = 28;
    finalCanvas.height = 28;
    const finalCtx = finalCanvas.getContext('2d');
    
    // Black background
    finalCtx.fillStyle = '#000000';
    finalCtx.fillRect(0, 0, 28, 28);
    
    // Center it in 28x28 canvas (leaving a margin)
    const dx = (28 - scaledWidth) / 2;
    const dy = (28 - scaledHeight) / 2;
    finalCtx.drawImage(cropCanvas, 0, 0, width, height, dx, dy, scaledWidth, scaledHeight);
    
    // 5. Get normalized grayscale pixels
    const finalImgData = finalCtx.getImageData(0, 0, 28, 28);
    const finalData = finalImgData.data;
    const pixels = new Float32Array(784);
    
    for (let i = 0; i < 784; i++) {
        // Red channel value / 255.0 to normalize to [0, 1]
        pixels[i] = finalData[i * 4] / 255.0;
    }
    
    return pixels;
}

// Vector & Matrix Operations
function dotProduct(matrix, vector) {
    const result = new Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        const row = matrix[i];
        for (let j = 0; j < row.length; j++) {
            sum += row[j] * vector[j];
        }
        result[i] = sum;
    }
    return result;
}

function vectorAdd(v1, v2) {
    const result = new Array(v1.length);
    for (let i = 0; i < v1.length; i++) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

function relu(vector) {
    return vector.map(x => Math.max(0, x));
}

function softmax(vector) {
    // Numerically stable softmax
    const max = Math.max(...vector);
    const exps = vector.map(x => Math.exp(x - max));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
}

// Inference Pipeline
function runInference() {
    if (!modelWeights) return;
    
    // 1. Get input vector (784)
    const X = getGrayscalePixels();
    
    // Check if canvas is entirely black (no drawing)
    const isCanvasEmpty = X.every(val => val === 0);
    if (isCanvasEmpty) {
        clearCanvas();
        return;
    }
    
    // 2. Hidden Layer: Z1 = W1 * X + b1
    const Z1 = vectorAdd(dotProduct(modelWeights.W1, X), modelWeights.b1);
    const A1 = relu(Z1);
    
    // 3. Output Layer: Z2 = W2 * A1 + b2
    const Z2 = vectorAdd(dotProduct(modelWeights.W2, A1), modelWeights.b2);
    const A2 = softmax(Z2);
    
    // 4. Get Prediction
    let maxProb = -1;
    let predictedDigit = -1;
    for (let i = 0; i < 10; i++) {
        if (A2[i] > maxProb) {
            maxProb = A2[i];
            predictedDigit = i;
        }
    }
    
    // 5. Update UI
    predictedDigitEl.textContent = predictedDigit;
    predictedDigitEl.parentElement.style.textShadow = '0 0 25px rgba(99, 102, 241, 0.8)';
    confidencePctEl.textContent = (maxProb * 100).toFixed(1) + '%';
    updateBars(A2, predictedDigit);
}

// Init App
async function init() {
    initCanvas();
    initBars();
    await loadModel();
}

window.onload = init;
