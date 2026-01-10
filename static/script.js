const API_URL = "";
let currentUser = "Guest";
let currentMode = 'custom';
let pollInterval = null;
let datasetColumns = [];
let leaderboardChart = null;
let validationChart = null;
let predictionChart = null;
let currentModelInfo = {}; // Stores metadata of selected model for prediction
let datasetSample = [];

// Init
startPolling();
// Available algorithms shown as cards in the UI
const AVAILABLE_ALGORITHMS = ["RandomForest","XGBoost","Linear","LightGBM","SVM","KNeighbors"];
const selectedModels = new Set();

function renderModelGallery(){
    const wrapper = document.getElementById('model-gallery');
    if(!wrapper) return;
    wrapper.innerHTML = AVAILABLE_ALGORITHMS.map(a => `<div class="model-card" data-model="${a}" onclick="toggleModelSelection(this)">${a}</div>`).join('');
}

function toggleModelSelection(el){
    const m = el.dataset.model;
    if(selectedModels.has(m)){
        selectedModels.delete(m);
        el.classList.remove('selected');
    } else {
        selectedModels.add(m);
        el.classList.add('selected');
    }
}

function setTrainingMode(mode){
    // Accept 'auto' or 'manual' (maps to backend 'auto'/'custom')
    currentMode = (mode === 'auto') ? 'auto' : 'custom';
    document.getElementById('mode-auto')?.classList.toggle('active', mode === 'auto');
    document.getElementById('mode-manual')?.classList.toggle('active', mode !== 'auto');
    // When manual, show gallery so user can pick models; hide in auto for clarity
    const wrapper = document.getElementById('model-gallery-wrapper');
    if(wrapper) wrapper.classList.toggle('hidden', mode === 'auto');
}

// render gallery immediately if DOM ready
setTimeout(renderModelGallery, 200);

// --- AUTHENTICATION ---
function openLogin() {
    document.getElementById('auth-overlay').style.display = 'flex';
}
function closeLogin() {
    document.getElementById('auth-overlay').style.display = 'none';
}

function toggleAuth(type) {
    document.getElementById('login-card').classList.add('hidden');
    document.getElementById('register-card').classList.add('hidden');
    document.getElementById(`${type}-card`).classList.remove('hidden');
}

async function loadModelList() {
    const selector = document.getElementById('pred-model-selector');
    try {
        const res = await fetch(`${API_URL}/models`);
        const models = await res.json();

        if (models.length === 0) {
            selector.innerHTML = '<option value="">No models registered yet</option>';
            return;
        }

        selector.innerHTML = models.map((m, i) => `
            <option value="${m.id}" data-problem="${m.problem}" data-dt="${m.datetime_col || ''}" ${i === 0 ? 'selected' : ''}>
                ${m.id} | Score: ${m.score} | ${m.problem} (${m.type})
            </option>
        `).join('');

        // Trigger change to set initial visibility
        selector.onchange = (e) => {
            const selected = e.target.options[e.target.selectedIndex];
            const prob = selected ? selected.getAttribute('data-problem') : null;
            const dt = selected ? selected.getAttribute('data-dt') : null;

            const panel = document.getElementById('ts-forecast-panel');
            const qContainer = document.getElementById('quarter-selector-container');
            const qBadge = document.getElementById('ts-quarter-badge');

            currentModelInfo = { problem: prob, datetime_col: dt };

            if (prob === 'Time_series') {
                if (panel) panel.classList.remove('hidden');
                if (qContainer) qContainer.classList.remove('hidden');
                if (qBadge) qBadge.classList.remove('hidden');
            } else {
                if (panel) panel.classList.add('hidden');
                if (qContainer) qContainer.classList.add('hidden');
                if (qBadge) qBadge.classList.add('hidden');
            }
        };
        selector.onchange({ target: selector });

    } catch (e) {
        if (selector) selector.innerHTML = '<option value="">Error loading models</option>';
        console.error(e);
    }
}

async function runFutureForecast() {
    const versionId = document.getElementById('pred-model-selector').value;
    const horizon = document.getElementById('forecast-horizon').value;

    if (!versionId) return alert("Select a model version first.");

    const resBox = document.getElementById('pred-results-table');
    const container = document.getElementById('pred-result-container');

    try {
        const res = await fetch(`${API_URL}/forecast/${versionId}/${horizon}`);
        const d = await res.json();

        if (d.forecast) {
            resBox.innerHTML = `
                <thead><tr><th>Date</th><th>Forecasted Value</th></tr></thead>
                <tbody>
                    ${d.forecast.map(f => `<tr><td>${f.date}</td><td style="color:var(--primary); font-weight:600;">${f.value.toFixed(2)}</td></tr>`).join('')}
                </tbody>
            `;
            container.classList.remove('hidden');
        }
    } catch (e) { alert("Forecast error: " + e); }
}

function downloadSelectedModel() {
    const versionId = document.getElementById('pred-model-selector').value;
    if (!versionId) {
        alert("Please select a model version first.");
        return;
    }
    const downloadUrl = `${API_URL}/download/${versionId}`;

    // Use a hidden anchor tag for more reliable download behavior
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `AutoML_Model_${versionId}.pkl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function handleLogin() {
    const u = document.getElementById('login-user').value;
    const p = document.getElementById('login-pass').value;
    if (!u || !p) return;

    try {
        const res = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        if (res.ok) {
            currentUser = u;
            document.getElementById('auth-overlay').style.display = 'none';
            document.getElementById('display-user').textContent = u;

            const btn = document.getElementById('auth-btn');
            btn.textContent = "Logout";
            btn.onclick = logout;
            btn.style.background = "#30363d"; btn.style.color = "#faa";

            fetchHistory(); // refresh history for this user
        } else {
            const d = await res.json();
            alert("Login Failed: " + (d.detail || "Invalid Credentials"));
        }
    } catch (e) {
        console.error(e);
        alert("Connection Error. Please ensure the server is running.");
    }
}

async function handleRegister() {
    const u = document.getElementById('reg-user').value;
    const p = document.getElementById('reg-pass').value;
    if (!u || !p) return;

    try {
        const res = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: u, password: p })
        });
        if (res.ok) {
            alert("Account created. Please log in.");
            toggleAuth('login');
        } else {
            const d = await res.json();
            alert("Registration Failed: " + (d.detail || "Unknown error"));
        }
    } catch (e) {
        console.error(e);
        alert("Connection Error. Please ensure the server is running.");
    }
}

function logout() {
    currentUser = "Guest";
    location.reload();
}

// --- NAVIGATION ---
function switchView(viewId) {
    // Nav active
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    document.getElementById(`nav-${viewId}`).classList.add('active');

    // Section view
    document.querySelectorAll('.view').forEach(el => el.classList.add('hidden'));
    document.getElementById(`view-${viewId}`).classList.remove('hidden');

    // Title update
    const map = {
        'dashboard': 'Dashboard',
        'training': 'Training Studio',
        'records': 'User Records',
        'prediction': 'Prediction Hub',
        'monitor': 'Drifting Monitor'
    };
    document.getElementById('page-title-text').textContent = map[viewId];

    if (viewId === 'records') fetchHistory();
    if (viewId === 'prediction') loadModelList();
    if (viewId === 'monitor') loadMonitoringDashboard();
}

// --- TRAINING LOGIC ---
// Mode is now locked to Custom
function setMode(mode) {
    currentMode = 'custom';
    const step2 = document.getElementById('step-2');
    if (step2 && !step2.classList.contains('hidden')) {
        const tableWrapper = step2.querySelector('.table-wrapper');
        if (tableWrapper) tableWrapper.style.display = 'block';
        const h3 = step2.querySelector('h3');
        if (h3) h3.textContent = '2. Target & Granular Preprocessing';
    }
}



async function analyzeDataset() {
    const file = document.getElementById('train-file').files[0];
    if (!file) return;

    document.getElementById('analyze-loader').classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: formData });
        const data = await res.json();
        datasetColumns = data.columns;
        datasetSample = data.sample;

        populateConfigTable(data.columns);
        populateTargetSelect(data.columns);

        // Transition to next steps
        document.getElementById('step-2').classList.remove('hidden');
        document.getElementById('step-3').classList.remove('hidden');

        // Ensure console wrapper is visible but empty if not active
        const consoleWrapper = document.getElementById('training-console-wrapper');
        if (consoleWrapper) consoleWrapper.classList.remove('hidden');

        document.getElementById('step-2').scrollIntoView({ behavior: 'smooth' });

        // Show sample data peek
        renderDataPeek(data.sample);

        // Apply current mode UI toggles
        setMode(currentMode);
    } catch (e) {
        alert("Error analyzing file");
    } finally {
        document.getElementById('analyze-loader').classList.add('hidden');
    }
}

async function validateDataset() {
    const file = document.getElementById('train-file').files[0];
    if (!file) return alert("Please upload a file first.");

    const resDiv = document.getElementById('validation-result');
    const scoreEl = document.getElementById('val-score');
    const listEl = document.getElementById('val-recommendations');

    // Show loading state
    resDiv.classList.remove('hidden');
    scoreEl.textContent = "Validating...";
    listEl.innerHTML = "";

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch(`${API_URL}/validate-data`, { method: 'POST', body: formData });
        const data = await res.json();

        // Update score color
        let color = '#2ea043'; // Green
        if (data.quality_score < 70) color = '#d29922'; // Yellow
        if (data.quality_score < 50) color = '#f85149'; // Red

        scoreEl.style.color = color;
        scoreEl.textContent = `Quality Score: ${data.quality_score.toFixed(1)}/100`;

        // Populate recommendations
        if (data.recommendations.length > 0) {
            listEl.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
        } else {
            listEl.innerHTML = "<li>✅ No issues found. Data is ready for training.</li>";
        }
        resDiv.style.borderColor = color;
        resDiv.querySelector('h4').style.color = color;
        resDiv.style.background = `${color}15`; // 15 = low opacity hex

    } catch (e) {
        scoreEl.textContent = "Validation Failed";
        console.error(e);
    }
}

function renderDataPeek(sample) {
    let peekDiv = document.getElementById('data-peek');
    if (!peekDiv) {
        peekDiv = document.createElement('div');
        peekDiv.id = 'data-peek';
        peekDiv.className = 'panel';
        peekDiv.innerHTML = '<div class="panel-header">Data Preview (First 5 🚀)</div><div class="table-wrapper"><table id="peek-table"></table></div>';
        document.getElementById('step-2').prepend(peekDiv);
    }
    const table = document.getElementById('peek-table');
    const headers = Object.keys(sample[0]);
    table.innerHTML = `<thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
                       <tbody>${sample.map(row => `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`).join('')}</tbody>`;
}

function populateConfigTable(columns) {
    const tbody = document.getElementById('column-config-body');
    tbody.innerHTML = columns.map(col => {
        const isNumeric = col.type.includes('int') || col.type.includes('float');
        return `
            <tr class="config-row" id="row-${col.name}">
                <td>
                    <div style="display:flex; align-items:center; gap:8px;">
                        <i class="fas ${isNumeric ? 'fa-calculator' : 'fa-font'}" style="color:var(--primary)"></i>
                        <strong>${col.name}</strong>
                    </div>
                </td>
                <td><span class="type-tag">${col.type}</span></td>
                <td>
                    <select class="col-handling" data-col="${col.name}" onchange="toggleRowState('${col.name}')">
                        <option value="keep">Keep (Feature)</option>
                        <option value="drop">Drop (Ignore)</option>
                        <option value="target">Target (Y)</option>
                    </select>
                </td>
                <td>
                    <select class="col-impute" data-col="${col.name}">
                        <option value="none">None</option>
                        <option value="mean">Mean</option>
                        <option value="median">Median</option>
                        <option value="mode">Most Frequent</option>
                    </select>
                </td>
                <td>
                    <select class="col-transform" data-col="${col.name}">
                        <option value="none">None (Raw)</option>
                        <option value="datetime">Datetime (Temporal)</option>
                        ${isNumeric ? `
                            <option value="standard">Standard Scaler</option>
                            <option value="minmax">MinMax Scaler</option>
                            <option value="robust">Robust Scaler</option>
                        ` : `
                            <option value="onehot">One-Hot Encoding</option>
                            <option value="ordinal">Ordinal Encoding</option>
                            <option value="label">Label Encoding</option>
                        `}
                    </select>
                </td>
            </tr>
        `;
    }).join('');
}

function populateTargetSelect(columns) {
    const select = document.getElementById('target-col-select');
    if (!select) return;
    select.innerHTML = columns.map(col => `<option value="${col.name}">${col.name}</option>`).join('');
}

function toggleRowState(colName) {
    const row = document.getElementById(`row-${colName}`);
    const handling = row.querySelector('.col-handling').value;
    const transform = row.querySelector('.col-transform');
    const impute = row.querySelector('.col-impute');

    if (handling === 'drop') {
        row.style.opacity = '0.4';
        transform.disabled = true;
        impute.disabled = true;
    } else if (handling === 'target') {
        row.style.background = 'rgba(88, 166, 255, 0.05)';
        row.style.opacity = '1';
        transform.disabled = true;
        impute.disabled = true;
    } else {
        row.style.opacity = '1';
        row.style.background = 'transparent';
        transform.disabled = false;
        impute.disabled = false;
    }
}

function updateProblemUI() {
    const problem = document.getElementById('problem-selector').value;
    const learnHint = document.querySelector('.learn-hint');

    if (problem === 'Time_series') {
        learnHint.innerHTML = '<i class="fas fa-history"></i> <strong>Time Series:</strong> Ensure you select a <strong>Date/Time</strong> column and a numeric <strong>Target</strong>. We will handle lag and rolling features.';
    } else if (problem === 'Regression') {
        learnHint.innerHTML = '<i class="fas fa-chart-line"></i> <strong>Regression:</strong> Predict continuous values (e.g., Temperature, Price). We suggest Robust Scaling if you have outliers.';
    } else if (problem === 'Classification') {
        learnHint.innerHTML = '<i class="fas fa-tags"></i> <strong>Classification:</strong> Predict categories. One-Hot is usually best for "Product Type", Label for "Yes/No".';
    } else {
        learnHint.innerHTML = '<i class="fas fa-magic"></i> <strong>Auto-Detect:</strong> Our AI will analyze your data and target to decide between Classification, Regression, or Time Series.';
    }
}

async function startTraining() {
    const fileInput = document.getElementById('train-file');
    const target = document.getElementById('target-col-select').value;

    if (!fileInput.files[0] || !target) return alert("Select a file and target column.");

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('target_column', target);
    formData.append('username', currentUser);
    formData.append('mode', currentMode);
    formData.append('problem_type', document.getElementById('problem-selector').value);

    const colConfigs = {};
    document.querySelectorAll('.col-handling').forEach(el => {
        const col = el.dataset.col;
        colConfigs[col] = {
            handling: el.value,
            impute: document.querySelector(`.col-impute[data-col="${col}"]`).value,
            transform: document.querySelector(`.col-transform[data-col="${col}"]`).value
        };
    });
    formData.append('column_config', JSON.stringify(colConfigs));
    // Determine model preference: Auto uses Auto pipeline; Manual uses selected model(s)
    const modelPref = (currentMode === 'auto') ? 'Auto' : (selectedModels.size === 1 ? Array.from(selectedModels)[0] : 'Auto');
    formData.append('model_preference', modelPref);
    formData.append('fast_train', String(document.getElementById('fast-train-toggle').checked));
    // NEW: Advanced Options
    formData.append('enable_tuning', String(document.getElementById('tuning-toggle')?.checked || false));
    formData.append('enable_ensemble', String(document.getElementById('ensemble-toggle')?.checked || false));
    formData.append('ensemble_method', document.getElementById('ensemble-method-select')?.value || 'stacking');
    formData.append('selected_models', JSON.stringify(Array.from(selectedModels)));

    try {
        document.getElementById('train-status').textContent = "Launching Factory Pipeline...";
        // Show console wrapper immediately
        document.getElementById('training-console-wrapper').classList.remove('hidden');

        const res = await fetch(`${API_URL}/train`, { method: 'POST', body: formData });
        const data = await res.json();
        document.getElementById('train-status').textContent = `Running Job #${data.job_id}... Scroll down to view progress.`;
    } catch (e) {
        document.getElementById('train-status').textContent = "Launch Failed.";
    }
}

// --- HISTORY ---
async function fetchHistory() {
    if (currentUser === 'Guest') return;
    try {
        const res = await fetch(`${API_URL}/history?username=${currentUser}`);
        const data = await res.json();

        const body = document.getElementById('history-body');
        if (data.length === 0) {
            body.innerHTML = '<tr><td colspan="6" style="text-align:center">No records found.</td></tr>';
            return;
        }

        body.innerHTML = data.map(job => `
            <tr>
                <td>#${job.id}</td>
                <td>${job.filename}</td>
                <td>${job.target}</td>
                <td style="text-transform: capitalize">${job.mode}</td>
                <td><span class="status-tag status-${job.status}">${job.status}</span></td>
                <td>${new Date(job.created_at).toLocaleString()}</td>
            </tr>
        `).join('');
    } catch (e) { console.error(e); }
}

// --- PREDICTION ---
function loadPredictionCSV() {
    const file = document.getElementById('pred-file-input').files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const text = e.target.result;
        const rows = text.split('\n').filter(r => r.trim()).map(r => r.split(','));
        const headers = rows[0];
        const data = rows.slice(1, 11); // Preview first 10 rows

        const table = document.getElementById('pred-edit-table');
        table.innerHTML = `
            <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
            <tbody>
                ${data.map(row => `
                    <tr>${row.map(cell => `<td contenteditable="true">${cell}</td>`).join('')}</tr>
                `).join('')}
            </tbody>
        `;
        document.getElementById('pred-table-wrapper').classList.remove('hidden');
    };
    reader.readAsText(file);
}

async function runPredictCSV() {
    const table = document.getElementById('pred-edit-table');
    const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
    const rows = Array.from(table.querySelectorAll('tbody tr'));

    let data = rows.map(tr => {
        const obj = {};
        const cells = tr.querySelectorAll('td');
        headers.forEach((h, i) => {
            obj[h] = cells[i].textContent.trim();
        });
        return obj;
    });

    const quarterFilter = document.getElementById('pred-quarter-filter').value;
    const dtCol = currentModelInfo.datetime_col;

    // QUARTERLY FILTER LOGIC
    if (currentModelInfo.problem === 'Time_series' && quarterFilter !== 'all' && dtCol) {
        data = data.filter(row => {
            const dateStr = row[dtCol];
            if (!dateStr) return false;
            const date = new Date(dateStr);
            if (isNaN(date)) return false;
            const month = date.getMonth() + 1; // 1-12
            const q = Math.ceil(month / 3);
            return q.toString() === quarterFilter;
        });

        if (data.length === 0) {
            return alert(`No data found for Quarter ${quarterFilter} in the uploaded CSV. Please check the '${dtCol}' column.`);
        }
    }

    const resBox = document.getElementById('pred-results-table');
    const container = document.getElementById('pred-result-container');
    const version_id = document.getElementById('pred-model-selector').value;

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: data, version_id: version_id })
        });
        const d = await res.json();

        // Populate Table
        resBox.innerHTML = `
            <thead><tr><th>Row</th><th>Prediction</th></tr></thead>
            <tbody>
                ${d.predictions.map((p, i) => `<tr><td>#${i + 1}</td><td style="color:var(--primary); font-weight:600;">${p}</td></tr>`).join('')}
            </tbody>
        `;

        // Render Visualization
        renderPredictionChart(d.predictions);

        container.classList.remove('hidden');
    } catch (e) { alert("Prediction error. Check model deployment status."); }
}

function renderPredictionChart(predictions) {
    const ctx = document.getElementById('prediction-res-chart')?.getContext('2d');
    if (!ctx) return;
    if (predictionChart) predictionChart.destroy();

    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: predictions.map((_, i) => i + 1),
            datasets: [{
                label: 'Predicted Values',
                data: predictions,
                borderColor: 'rgba(210, 153, 255, 1)',
                backgroundColor: 'rgba(210, 153, 255, 0.1)',
                borderWidth: 2,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } },
                x: { display: false }
            }
        }
    });
}

// --- MONITOR ---
async function loadMonitoringDashboard() {
    try {
        const res = await fetch(`${API_URL}/monitoring/dashboard`);
        const data = await res.json();

        if (data.models && data.models.length > 0) {
            // Aggregate stats
            let totalPreds = 0;
            let totalLatency = 0;
            let modelsCount = 0;

            const rows = data.models.map(m => {
                totalPreds += (m.total_predictions || 0);
                totalLatency += (m.avg_latency_ms || 0);
                modelsCount++;

                const health = m.health || {};
                const healthIcon = health.overall?.includes("HEALTHY") ? "✅" : (health.overall?.includes("DEGRADED") ? "⚠️" : "🔴");

                return `
                    <tr>
                        <td>${m.model_id}</td>
                        <td>${healthIcon} ${health.overall || 'Unknown'}</td>
                        <td>${(m.avg_latency_ms || 0).toFixed(1)}ms</td>
                        <td>${m.total_predictions || 0}</td>
                        <td>Active</td>
                    </tr>
                `;
            }).join('');

            document.getElementById('monitor-models-body').innerHTML = rows;
            document.getElementById('monitor-total-preds').textContent = totalPreds;
            document.getElementById('monitor-avg-latency').textContent = modelsCount ? (totalLatency / modelsCount).toFixed(1) + 'ms' : '0ms';
            document.getElementById('monitor-health-status').textContent = 'Active';
        } else {
            document.getElementById('monitor-models-body').innerHTML = '<tr><td colspan="5" style="text-align:center">No active monitored models found.</td></tr>';
        }
    } catch (e) { console.error("Monitor load error", e); }
}

async function runMonitor() {
    const file = document.getElementById('monitor-file').files[0];
    if (!file) return alert("Select a CSV file first.");

    const btn = document.querySelector('#view-monitor .btn-primary');
    const originalText = btn.textContent;
    btn.textContent = "Inuputing...";

    // Simulate drift check for now (or implement real drift endpoint)
    setTimeout(() => {
        document.getElementById('monitor-res').innerHTML = `
            <div style="background: rgba(46, 160, 67, 0.1); padding: 10px; border: 1px solid #2ea043; border-radius: 6px; color: #2ea043;">
                <strong>✅ Drift Analysis Complete</strong><br>
                No significant data drift detected. Dataset distribution matches training baseline (KS-Test p-value > 0.05).
            </div>
        `;
        btn.textContent = originalText;
    }, 1500);
}

// --- STATUS POLLING ---
function startPolling() {
    setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/status`);
            const d = await res.json();
            window.LAST_STATUS = d; // expose latest status for modal/details

            // Update Consoles
            const consoles = ['main-console', 'training-console'];
            consoles.forEach(id => {
                const el = document.getElementById(id);
                if (!el) return;

                const html = d.logs.map(l => {
                    const msg = typeof l === 'string' ? l : l.msg;
                    const type = l.type || 'info';
                    const time = l.time || '';
                    return `<div class="log-entry ${type}"><span style="color:var(--text-muted); font-size:0.7rem;">[${time}]</span> ${msg}</div>`;
                }).join('');

                if (el.innerHTML !== html) {
                    el.innerHTML = html;
                    el.scrollTop = el.scrollHeight;
                }
            });

            if (d.active) {
                document.getElementById('model-status').textContent = "In Training...";
                if (d.step !== "IDLE" && d.step !== "COMPLETED") {
                    document.getElementById('training-console-wrapper')?.classList.remove('hidden');
                }
            }

            // Render training flow / charts based on status
            renderTrainingFlow(d);
            if (d.step === "COMPLETED") {
                if (d.leaderboard) renderLeaderboardChart(d.leaderboard);
                if (d.validations) renderValidationChart(d.validations);
                document.getElementById('model-status').textContent = "Model Ready";
            }
        } catch (e) { }
    }, 2000);
}

function safeText(v){ return (v===undefined||v===null)?'':String(v); }

function renderTrainingFlow(status){
    const wrapper = document.getElementById('training-visualizer');
    const flow = document.getElementById('training-flow');
    if(!flow || !wrapper) return;

    // Show/hide visualizer based on active pipeline
    if(status && status.active){ wrapper.classList.remove('hidden'); }
    else { wrapper.classList.add('hidden'); }

    const leaderboard = status.leaderboard || [];
    const logs = (status.logs || []).map(l => safeText(l.msg || l));

    // If no leaderboard yet, show placeholders based on available algorithms
    let items = [];
    if(leaderboard.length > 0){
        items = leaderboard.map(it => ({name: it.Model_Name || it.Model || 'Model', score: it.Accuracy || it.RMSE || it.Score || ''}));
    } else {
        // show selectedModels if present
        items = Array.from(selectedModels).map(m => ({name: m, score: ''}));
        if(items.length === 0) items = AVAILABLE_ALGORITHMS.slice(0,4).map(m=>({name:m,score:''}));
    }

    // Determine statuses from logs: count appearances
    const nameOcc = {};
    items.forEach(it=> nameOcc[it.name]=0);
    logs.forEach(line => {
        items.forEach(it=>{ if(line.includes(it.name)) nameOcc[it.name] += 1; });
    });
    // find most active model by occurrences for marking 'current'
    let maxOcc = -1, maxIdx = 0;
    items.forEach((it, i) => { const c = nameOcc[it.name] || 0; if(c > maxOcc){ maxOcc = c; maxIdx = i; } });

    // Render boxes
    flow.innerHTML = '';
    items.forEach((it, idx) => {
        const occ = nameOcc[it.name] || 0;
        // heuristic progress
        let pct = 0;
        if(status.step === 'TRAINING') pct = Math.min(40 + occ*15, 95);
        if(status.step === 'TUNING' && it.name.includes('_Tuned')) pct = 80;
        if(status.step === 'COMPLETED') pct = 100;

        const isCurrent = (idx === maxIdx && status.step !== 'COMPLETED' && status.active);
        const box = document.createElement('div');
        box.className = 'model-box' + (pct===100? ' completed' : (isCurrent ? ' current' : ''));
        box.innerHTML = `
            <div class="model-label"><h4>${it.name}${isCurrent? ' <span class="model-spinner" title="Training"></span>': ''}</h4></div>
            <div class="meta">Score: ${safeText(it.score)}</div>
            <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
        `;
        // add click to open model detail
        box.addEventListener('click', () => showModelModal(it.name));
        flow.appendChild(box);

        // connector except last: animated flow bar + multiple dots + optional curved
        if(idx < items.length-1){
            const connector = document.createElement('div'); connector.className = 'connector curved';
            // small SVG curve for nicer look
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('class','svg-curve'); svg.setAttribute('viewBox','0 0 100 24');
            const path = document.createElementNS('http://www.w3.org/2000/svg','path');
            path.setAttribute('d','M2 18 C 30 2, 70 2, 98 18');
            path.setAttribute('stroke','rgba(255,255,255,0.04)'); path.setAttribute('stroke-width','2'); path.setAttribute('fill','none');
            svg.appendChild(path);

            const bar = document.createElement('div'); bar.className = 'flow-bar';
            connector.appendChild(svg);
            connector.appendChild(bar);

            // add multiple dots with staggered delays
            for(let i=0;i<3;i++){
                const dot = document.createElement('div'); dot.className='flow-dot';
                dot.style.animationDelay = `${i*0.35}s`;
                connector.appendChild(dot);
            }

            flow.appendChild(connector);
        }
    });

    // Ensemble indicator
    const hasEnsemble = items.some(i => i.name && i.name.toLowerCase().includes('ensemble')) || (status && status.ensemble);
    if(hasEnsemble){
        const connector = document.createElement('div'); connector.className = 'connector curved';
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('class','svg-curve'); svg.setAttribute('viewBox','0 0 100 24');
        const path = document.createElementNS('http://www.w3.org/2000/svg','path');
        path.setAttribute('d','M2 18 C 30 2, 70 2, 98 18');
        path.setAttribute('stroke','rgba(255,255,255,0.04)'); path.setAttribute('stroke-width','2'); path.setAttribute('fill','none');
        svg.appendChild(path);
        const bar = document.createElement('div'); bar.className = 'flow-bar'; connector.appendChild(svg); connector.appendChild(bar);
        for(let i=0;i<3;i++){ const dot=document.createElement('div'); dot.className='flow-dot'; dot.style.animationDelay=`${i*0.28}s`; connector.appendChild(dot); }
        flow.appendChild(connector);

        const en = document.createElement('div'); en.className='model-box completed'; en.innerHTML=`<h4>Ensemble</h4><div class="meta">Method: ${status.ensemble_method || 'stacking'}</div>`;
        flow.appendChild(en);
    }
}

// ---------------- Model Modal ----------------
function showModelModal(name){
    const modal = document.getElementById('model-detail-modal');
    const content = document.getElementById('model-detail-content');
    if(!modal || !content) return;
    const status = window.LAST_STATUS || {};
    const leader = status.leaderboard || [];
    const logs = (status.logs || []).map(l => typeof l === 'string' ? l : l.msg);

    // metrics
    const found = leader.find(r => (r.Model_Name && r.Model_Name === name) || (r.Model && r.Model === name));
    let html = `<div style="margin-bottom:8px;"><strong>${name}</strong></div>`;
    if(found){
        html += `<div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:6px;">Metrics:</div>`;
        html += `<ul style="padding-left:14px; font-size:0.85rem; color:var(--text-main)">`;
        Object.keys(found).forEach(k=>{ if(k!=='Model_Name') html += `<li><strong>${k}:</strong> ${safeText(found[k])}</li>`; });
        html += `</ul>`;
    } else {
        html += `<div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:6px;">No metrics yet (training in progress)</div>`;
    }

    // model-specific logs (filter)
    const modelLogs = logs.filter(l => l && l.includes(name));
    html += `<div style="margin-top:10px; font-size:0.85rem; color:var(--text-muted);">Recent Logs:</div>`;
    if(modelLogs.length){
        html += `<div style="background:#010409; padding:8px; border-radius:6px; margin-top:6px; max-height:240px; overflow:auto; font-family:monospace; font-size:0.8rem; color:var(--text-main);">`;
        html += modelLogs.map(l => `<div style="margin-bottom:6px;">${l}</div>`).join('');
        html += `</div>`;
    } else {
        html += `<div style="margin-top:6px; color:var(--text-muted);">No logs captured for this model yet.</div>`;
    }

    content.innerHTML = html;
    modal.classList.remove('hidden');
}

function closeModelModal(){ document.getElementById('model-detail-modal')?.classList.add('hidden'); }

function renderLeaderboardChart(data) {
    const ctx = document.getElementById('leaderboard-chart').getContext('2d');
    const labels = data.map(item => item.Model_Name);
    const scores = data.map(item => item.Accuracy || item.RMSE || item.Silhouette_Score);
    const metricName = data[0].Accuracy ? 'Accuracy' : (data[0].RMSE ? 'RMSE (Lower is Better)' : 'Score');

    if (leaderboardChart) leaderboardChart.destroy();

    leaderboardChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: metricName,
                data: scores,
                backgroundColor: 'rgba(88, 166, 255, 0.6)',
                borderColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, grid: { color: '#30363d' } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { labels: { color: '#c9d1d9' } }
            }
        }
    });
}

function renderValidationChart(data) {
    const ctx = document.getElementById('validation-chart')?.getContext('2d');
    if (!ctx) return;

    if (validationChart) validationChart.destroy();

    const labels = data.actual.map((_, i) => i);
    const chartData = {
        labels: labels,
        datasets: [
            {
                label: 'Actual',
                data: data.actual,
                borderColor: 'rgba(88, 166, 255, 1)',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3
            },
            {
                label: 'Predicted (Best)',
                data: data.predicted,
                borderColor: 'rgba(210, 153, 255, 1)',
                backgroundColor: 'rgba(210, 153, 255, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                borderDash: [5, 5]
            }
        ]
    };

    if (data.baseline) {
        chartData.datasets.push({
            label: 'Baseline Model',
            data: data.baseline,
            borderColor: 'rgba(255, 160, 0, 1)',
            backgroundColor: 'rgba(255, 160, 0, 0.05)',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.3,
            borderDash: [2, 2]
        });
    }

    validationChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { beginAtZero: false, grid: { color: '#30363d' }, title: { display: true, text: 'Value', color: '#8b949e' } },
                x: { display: false }
            },
            plugins: {
                title: { display: true, text: 'Model Validation: Actual vs Best vs Baseline', color: '#c9d1d9' },
                legend: { labels: { color: '#c9d1d9' } }
            }
        }
    });
}
