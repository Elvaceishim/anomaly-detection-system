// API Base URL - Configure for production
// When deployed, this should point to your Render backend URL
const API_BASE = window.ANOMALY_API_URL || 'https://anomaly-detection-api.onrender.com';

// DOM Elements
const form = document.getElementById('transactionForm');
const submitBtn = document.getElementById('submitBtn');
const resultsCard = document.getElementById('resultsCard');
const apiStatus = document.getElementById('apiStatus');
const transactionsList = document.getElementById('transactionsList');
const clearHistoryBtn = document.getElementById('clearHistory');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const modelInfo = document.getElementById('modelInfo');

// Stats elements
const statTotal = document.getElementById('statTotal');
const statFlagged = document.getElementById('statFlagged');
const statSafe = document.getElementById('statSafe');
const statFlagRate = document.getElementById('statFlagRate');

// State
let transactionHistory = JSON.parse(localStorage.getItem('txnHistory') || '[]');
let currentThreshold = 0.25;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setDefaultTimestamp();
    checkApiStatus();
    fetchModelInfo();
    renderHistory();
    updateStats();
});

// Set default timestamp to now
function setDefaultTimestamp() {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    document.getElementById('timestamp').value = now.toISOString().slice(0, 16);
}

// Check API health
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();

        if (data.model_loaded) {
            apiStatus.className = 'status-badge online';
            apiStatus.innerHTML = '<span class="status-dot"></span><span>Model Ready</span>';
        } else {
            apiStatus.className = 'status-badge offline';
            apiStatus.innerHTML = '<span class="status-dot"></span><span>Model Not Loaded</span>';
        }
    } catch (error) {
        apiStatus.className = 'status-badge offline';
        apiStatus.innerHTML = '<span class="status-dot"></span><span>API Offline</span>';
    }
}

// Fetch model info
async function fetchModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model/info`);

        if (!response.ok) {
            modelInfo.innerHTML = `
                <div class="model-stat">
                    <span class="model-stat-label">Status</span>
                    <span class="model-stat-value">Model not loaded</span>
                </div>
            `;
            return;
        }

        const data = await response.json();
        currentThreshold = data.threshold;

        // Update threshold slider
        thresholdSlider.value = Math.round(data.threshold * 100);
        thresholdValue.textContent = (data.threshold * 100).toFixed(0) + '%';

        // Render model info
        modelInfo.innerHTML = `
            <div class="model-stat">
                <span class="model-stat-label">Model Type</span>
                <span class="model-stat-value highlight">${data.model_type || 'Unknown'}</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">PR-AUC</span>
                <span class="model-stat-value">${data.metrics?.pr_auc ? (data.metrics.pr_auc * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">ROC-AUC</span>
                <span class="model-stat-value">${data.metrics?.roc_auc ? (data.metrics.roc_auc * 100).toFixed(1) + '%' : 'N/A'}</span>
            </div>
            <div class="model-stat">
                <span class="model-stat-label">Features</span>
                <span class="model-stat-value">${data.feature_count || 0}</span>
            </div>
        `;
    } catch (error) {
        modelInfo.innerHTML = `
            <div class="model-stat">
                <span class="model-stat-label">Status</span>
                <span class="model-stat-value">Failed to load info</span>
            </div>
        `;
    }
}

// Threshold slider change
let thresholdDebounce = null;
thresholdSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    thresholdValue.textContent = value + '%';

    // Debounce API call
    clearTimeout(thresholdDebounce);
    thresholdDebounce = setTimeout(() => {
        updateThreshold(value / 100);
    }, 300);
});

// Update threshold via API
async function updateThreshold(newThreshold) {
    try {
        const response = await fetch(`${API_BASE}/threshold/update?new_threshold=${newThreshold}`, {
            method: 'POST'
        });

        if (response.ok) {
            currentThreshold = newThreshold;
            // Re-render history with new threshold consideration
            renderHistory();
        }
    } catch (error) {
        console.error('Failed to update threshold:', error);
    }
}

// Update statistics
function updateStats() {
    const total = transactionHistory.length;
    const flagged = transactionHistory.filter(t => t.is_flagged).length;
    const safe = total - flagged;
    const rate = total > 0 ? ((flagged / total) * 100).toFixed(1) : 0;

    animateCounter(statTotal, parseInt(statTotal.textContent), total);
    animateCounter(statFlagged, parseInt(statFlagged.textContent), flagged);
    animateCounter(statSafe, parseInt(statSafe.textContent), safe);
    statFlagRate.textContent = rate + '%';
}

// Animate counter
function animateCounter(element, start, end) {
    if (start === end) return;

    const duration = 500;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Disable button and show spinner
    submitBtn.disabled = true;
    submitBtn.querySelector('span:first-child').textContent = 'Analyzing...';
    submitBtn.querySelector('.spinner').style.display = 'block';

    try {
        const transaction = {
            transaction_id: document.getElementById('transaction_id').value,
            user_id: document.getElementById('user_id').value,
            amount: parseFloat(document.getElementById('amount').value),
            timestamp: new Date(document.getElementById('timestamp').value).toISOString(),
            transaction_type: document.getElementById('transaction_type').value,
            merchant_category: document.getElementById('merchant_category').value,
            location: document.getElementById('location').value,
            is_failed: document.getElementById('is_failed').checked
        };

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(transaction)
        });

        const result = await response.json();

        if (response.ok) {
            displayResult(result);
            addToHistory(transaction, result);
            incrementTransactionId();
            updateStats();
        } else {
            alert(`Error: ${result.detail || 'Unknown error'}`);
        }
    } catch (error) {
        alert(`Network error: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector('span:first-child').textContent = 'Analyze Transaction';
        submitBtn.querySelector('.spinner').style.display = 'none';
    }
});

// Display result with animation
function displayResult(result) {
    resultsCard.style.display = 'block';

    const score = result.risk_score;
    const percentage = Math.round(score * 100);

    // Determine risk level
    let riskClass = 'safe';
    if (score >= result.threshold) riskClass = 'danger';
    else if (score >= result.threshold * 0.7) riskClass = 'warning';

    // Update score ring
    const progress = document.getElementById('scoreProgress');
    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (score * circumference);

    progress.className = `score-progress ${riskClass}`;
    progress.style.strokeDashoffset = offset;

    // Update score value with animation
    const scoreValue = document.getElementById('scoreValue');
    animateValue(scoreValue, 0, percentage, 1000);

    // Update details
    document.getElementById('resultTxnId').textContent = result.transaction_id;

    const flaggedEl = document.getElementById('resultFlagged');
    flaggedEl.textContent = result.is_flagged ? 'YES' : 'NO';
    flaggedEl.className = `detail-value ${result.is_flagged ? 'flagged' : 'safe'}`;

    document.getElementById('resultThreshold').textContent =
        (result.threshold * 100).toFixed(1) + '%';

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Animate number
function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current + '%';

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Add to history
function addToHistory(transaction, result) {
    const entry = {
        ...transaction,
        risk_score: result.risk_score,
        is_flagged: result.is_flagged,
        timestamp: new Date().toISOString()
    };

    transactionHistory.unshift(entry);
    if (transactionHistory.length > 20) transactionHistory.pop();

    localStorage.setItem('txnHistory', JSON.stringify(transactionHistory));
    renderHistory();
}

// Render history
function renderHistory() {
    if (transactionHistory.length === 0) {
        transactionsList.innerHTML = '<div class="empty-state">No transactions scored yet</div>';
        return;
    }

    transactionsList.innerHTML = transactionHistory.map(txn => {
        const score = txn.risk_score;
        let scoreClass = 'safe';
        if (score >= currentThreshold) scoreClass = 'danger';
        else if (score >= currentThreshold * 0.7) scoreClass = 'warning';

        return `
            <div class="transaction-item ${txn.is_flagged ? 'flagged' : 'safe'}">
                <div class="txn-info">
                    <span class="txn-id">${txn.transaction_id}</span>
                    <span class="txn-meta">$${txn.amount.toLocaleString()} • ${txn.location}</span>
                </div>
                <div class="txn-score">
                    <div class="txn-score-value ${scoreClass}">${(score * 100).toFixed(1)}%</div>
                    <div class="txn-score-label">${txn.is_flagged ? 'Flagged' : 'Safe'}</div>
                </div>
            </div>
        `;
    }).join('');
}

// Clear history
clearHistoryBtn.addEventListener('click', () => {
    transactionHistory = [];
    localStorage.removeItem('txnHistory');
    renderHistory();
    updateStats();
});

// Auto-increment transaction ID
function incrementTransactionId() {
    const input = document.getElementById('transaction_id');
    const match = input.value.match(/^(.+_)(\d+)$/);
    if (match) {
        const prefix = match[1];
        const num = parseInt(match[2]) + 1;
        input.value = prefix + num.toString().padStart(3, '0');
    }
}
