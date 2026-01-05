const API_URL = "http://localhost:8001";

// --- Auth Handling ---
async function handleLogin(event) {
    event.preventDefault();
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;
    const errorDiv = document.getElementById('loginError');

    const formData = new FormData();
    formData.append('username', email); // OAuth2 expects 'username'
    formData.append('password', password);

    try {
        const response = await fetch(`${API_URL}/auth/token`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Login failed');
        }

        const data = await response.json();
        localStorage.setItem('token', data.access_token);
        localStorage.setItem('userEmail', email);
        window.location.href = 'app.html';
    } catch (err) {
        errorDiv.textContent = err.message;
        errorDiv.style.display = 'block';
    }
}

async function handleSignup(event) {
    event.preventDefault();
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;
    const errorDiv = document.getElementById('signupError');

    try {
        const response = await fetch(`${API_URL}/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || 'Signup failed');
        }

        // Auto login after signup
        // Reuse handleLogin logic or alert
        alert('Account created! Please sign in.');
        toggleAuth(); // Switch to login view
    } catch (err) {
        errorDiv.textContent = err.message;
        errorDiv.style.display = 'block';
    }
}

function toggleAuth() {
    const loginForm = document.getElementById('loginForm');
    const signupForm = document.getElementById('signupForm');

    if (loginForm.classList.contains('hidden')) {
        loginForm.classList.remove('hidden');
        signupForm.classList.add('hidden');
    } else {
        loginForm.classList.add('hidden');
        signupForm.classList.remove('hidden');
    }
}

function logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('userEmail');
    window.location.href = 'index.html';
}

// --- App Logic ---

// Drag & Drop
const uploadArea = document.getElementById('uploadArea');
if (uploadArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(event) {
    const files = event.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG/PNG).');
            return;
        }

        // Show loading
        document.getElementById('uploadArea').parentElement.classList.add('hidden');
        document.getElementById('loading').classList.remove('hidden');

        // Preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
        }
        reader.readAsDataURL(file);

        uploadFile(file);
    }
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const token = localStorage.getItem('token');

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                // 'Authorization': `Bearer ${token}` // Prediction endpoint is unprotected in code, but good practice
            },
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        showResult(data);
    } catch (err) {
        alert(err.message);
        resetUpload();
    }
}

function showResult(data) {
    document.getElementById('loading').classList.add('hidden');
    const resultContainer = document.getElementById('resultContainer');
    resultContainer.style.display = 'block';

    document.getElementById('predictionLabel').textContent = data.prediction;

    // Confidence bar
    const confidencePercent = Math.round(data.confidence * 100);
    const fill = document.getElementById('confidenceFill');

    // Reset width first for animation
    fill.style.width = '0%';
    setTimeout(() => {
        fill.style.width = `${confidencePercent}%`;
    }, 100);

    document.getElementById('confidenceText').textContent = `${confidencePercent}% Confidence`;
    document.getElementById('disclaimerText').textContent = data.medical_disclaimer;
}

function resetUpload() {
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('uploadArea').parentElement.classList.remove('hidden');
    document.getElementById('fileInput').value = '';
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    const emailDisplay = document.getElementById('userEmail');
    if (emailDisplay) {
        emailDisplay.textContent = localStorage.getItem('userEmail');
    }
});
