document.addEventListener('DOMContentLoaded', () => {
    // ═══ DOM References ═══
    const qubitSlider = document.getElementById('qubit-slider');
    const qubitValue = document.getElementById('qubit-value');
    const contextSelect = document.getElementById('context-length');
    const epochSlider = document.getElementById('epoch-slider');
    const epochValue = document.getElementById('epoch-value');
    const loadModelBtn = document.getElementById('load-model-btn');
    const trainModelBtn = document.getElementById('train-model-btn');
    const trainingPanel = document.getElementById('training-panel');
    const progressBar = document.getElementById('progress-bar');
    const trainingStatusText = document.getElementById('training-status-text');
    const circuitVisualizer = document.getElementById('circuit-visualizer');
    const activeModelTitle = document.getElementById('active-model-title');
    const activeModelSubtitle = document.getElementById('active-model-subtitle');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const chatForm = document.getElementById('chat-form');
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container');
    const welcomeScreen = document.getElementById('welcome-screen');
    const tempSlider = document.getElementById('temp-slider');
    const tempValue = document.getElementById('temp-value');

    // Training Mode buttons
    const modeBtns = {
        'new': document.getElementById('mode-new'),
        'continue': document.getElementById('mode-continue'),
        'finetune': document.getElementById('mode-finetune')
    };
    let currentMode = 'new';

    // Architecture info elements
    const infoParams = document.getElementById('info-params');
    const infoHidden = document.getElementById('info-hidden');
    const infoLayers = document.getElementById('info-layers');
    const infoHeads = document.getElementById('info-heads');
    const infoState = document.getElementById('info-state');
    const infoDepth = document.getElementById('info-depth');

    let modelLoaded = false;
    let availableModels = [];

    // ═══ Slider Update ═══
    qubitSlider.addEventListener('input', () => {
        qubitValue.textContent = qubitSlider.value;
        checkAvailability();
    });

    contextSelect.addEventListener('change', checkAvailability);

    tempSlider.addEventListener('input', () => {
        tempValue.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    epochSlider.addEventListener('input', () => {
        epochValue.textContent = epochSlider.value;
    });

    Object.keys(modeBtns).forEach(mode => {
        modeBtns[mode].addEventListener('click', () => {
            currentMode = mode;
            // Update UI
            Object.values(modeBtns).forEach(btn => btn.classList.remove('active'));
            modeBtns[mode].classList.add('active');
            
            // Update Training Button text
            if (mode === 'new') trainModelBtn.textContent = 'Train New';
            else if (mode === 'continue') trainModelBtn.textContent = 'Continue Training';
            else if (mode === 'finetune') trainModelBtn.textContent = 'Fine-tune Model';
        });
    });

    async function fetchmodels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            availableModels = data.models || [];
            checkAvailability();
        } catch (e) { console.error('Failed to fetch models list', e); }
    }

    function checkAvailability() {
        const q = parseInt(qubitSlider.value);
        const c = parseInt(contextSelect.value);

        // Find if this exact config exists
        const exactMatch = availableModels.find(m => m.num_qubits === q && m.max_context_length === c);

        // Find if ANY context exists for this qubit count
        const anyCtx = availableModels.filter(m => m.num_qubits === q);

        if (exactMatch) {
            loadModelBtn.classList.add('pulse-cyan');
            loadModelBtn.classList.remove('secondary-btn');
            loadModelBtn.classList.add('primary-btn');
            loadModelBtn.textContent = 'Load Existing Model';
            statusText.textContent = `Pretrained model found (${q}q / ${c}ctx)`;
            statusDot.className = 'status-indicator trained';
        } else if (anyCtx.length > 0) {
            loadModelBtn.classList.remove('pulse-cyan');
            loadModelBtn.textContent = 'Model Found (Diff Ctx)';
            const availableCtx = anyCtx.map(m => m.max_context_length).join(', ');
            statusText.textContent = `${q}q model exists for ctx: ${availableCtx}`;
            statusDot.className = 'status-indicator offline';

            // Suggest switching to the first available context
            if (confirm(`A pretrained ${q}-qubit model exists for ${anyCtx[0].max_context_length} context length. Switch and load now?`)) {
                contextSelect.value = anyCtx[0].max_context_length;
                checkAvailability();
                loadModelBtn.click();
            }
        } else {
            loadModelBtn.classList.remove('pulse-cyan');
            loadModelBtn.textContent = 'Model Not Found';
            statusText.textContent = `No model for ${q}q trained yet`;
            statusDot.className = 'status-indicator offline';
        }
    }

    // ═══ Circuit Visualizer ═══
    const GATE_TYPES = ['H', 'X', 'Z', 'Ry', 'Rz', 'CX'];
    const GATE_COLORS = {
        'H': '#00f0ff', 'X': '#ff6b6b', 'Z': '#ffd93d',
        'Ry': '#6bcb77', 'Rz': '#4d96ff', 'CX': '#c084fc'
    };

    function renderCircuit(circuitData) {
        circuitVisualizer.innerHTML = '';
        if (!circuitData || !circuitData.wires) return;

        const numQubits = circuitData.qubits;
        const depth = circuitData.circuit_depth || 3;
        
        // Use the actual number of layers from the backend
        const layers = circuitData.num_layers || 4;

        for (let i = 0; i < Math.min(numQubits, 16); i++) {
            const wireDiv = document.createElement('div');
            wireDiv.className = 'circuit-wire';

            const label = document.createElement('div');
            label.className = 'wire-label';
            label.textContent = `|q_${i}⟩`;

            const line = document.createElement('div');
            line.className = 'wire-line';

            // Each transformer layer has its own VQC
            // We show a simplified view of the stacked gates
            const layerGates = ['H', 'Ry', 'CX', 'Rz', 'X', 'Z'];
            for (let l = 0; l < layers; l++) {
                const gateType = layerGates[(i + l) % layerGates.length];
                const gate = document.createElement('div');
                gate.className = 'quantum-gate';
                gate.textContent = gateType;
                gate.style.borderColor = GATE_COLORS[gateType];
                gate.style.color = GATE_COLORS[gateType];
                gate.style.left = `${10 + (l * (80 / layers))}%`;
                line.appendChild(gate);
            }

            wireDiv.appendChild(label);
            wireDiv.appendChild(line);
            circuitVisualizer.appendChild(wireDiv);
        }
    }

    // ═══ Update Architecture Info ═══
    function updateModelInfo(data) {
        infoParams.textContent = data.total_parameters ? data.total_parameters.toLocaleString() : '—';
        infoHidden.textContent = data.hidden_dim || '—';
        infoLayers.textContent = data.num_layers || '—';
        infoHeads.textContent = data.num_heads || '—';
        infoState.textContent = data.state_space_size ? `2^${data.num_qubits} = ${data.state_space_size.toLocaleString()}` : '—';
        infoDepth.textContent = data.circuit_depth || '—';
    }

    function setModelActive(data) {
        modelLoaded = true;
        sendBtn.disabled = false;
        statusDot.className = 'status-indicator online';
        statusText.textContent = `${data.num_qubits}-Qubit Model Active`;
        activeModelTitle.textContent = `Quantum Transformer (${data.num_qubits} Qubits)`;
        activeModelSubtitle.textContent = `${data.total_parameters?.toLocaleString() || '?'} parameters · ${data.num_layers} layers · ${data.max_context_length} ctx`;
        updateModelInfo(data);
        if (data.circuit) renderCircuit(data.circuit);
    }

    function setModelInactive(message) {
        modelLoaded = false;
        sendBtn.disabled = true;
        statusDot.className = 'status-indicator offline';
        statusText.textContent = message || 'No model loaded';
    }

    // ═══ Load Model ═══
    loadModelBtn.addEventListener('click', async () => {
        const numQubits = parseInt(qubitSlider.value);
        const ctxLen = parseInt(contextSelect.value);

        loadModelBtn.textContent = 'Loading...';
        loadModelBtn.disabled = true;

        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_qubits: numQubits, max_context_length: ctxLen })
            });
            const data = await res.json();

            if (data.loaded) {
                setModelActive(data);
                welcomeScreen?.remove();
                addMessage(`Model loaded: ${numQubits}-Qubit Quantum Transformer with ${data.total_parameters?.toLocaleString()} parameters.`, 'system');
            } else {
                setModelInactive('Model not found');
                addMessage(data.message || 'No pretrained model found for this configuration. Click "Train New" to create one.', 'system');
            }
        } catch (e) {
            console.error(e);
            addMessage('Failed to connect to backend.', 'system');
        } finally {
            loadModelBtn.textContent = 'Load Model';
            loadModelBtn.disabled = false;
        }
    });

    // ═══ Train Model ═══
    trainModelBtn.addEventListener('click', async () => {
        const numQubits = parseInt(qubitSlider.value);
        const ctxLen = parseInt(contextSelect.value);

        trainModelBtn.disabled = true;
        trainingPanel.classList.remove('hidden');
        trainingStatusText.textContent = 'Initializing training...';
        progressBar.style.width = '5%';

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    num_qubits: numQubits,
                    max_context_length: ctxLen,
                    epochs: parseInt(epochSlider.value),
                    batch_size: 64,
                    learning_rate: currentMode === 'finetune' ? 1e-4 : 6e-4,
                    training_mode: currentMode
                })
            });
            const data = await res.json();

            if (res.ok) {
                welcomeScreen?.remove();
                addMessage(`Training started: ${numQubits}-qubit model. This may take a few minutes...`, 'system');
                pollTrainingStatus();
            } else {
                trainingStatusText.textContent = data.detail || 'Error';
                trainModelBtn.disabled = false;
            }
        } catch (e) {
            console.error(e);
            trainingStatusText.textContent = 'Failed to start training.';
            trainModelBtn.disabled = false;
        }
    });

    function pollTrainingStatus() {
        const interval = setInterval(async () => {
            try {
                const res = await fetch('/api/train/status');
                const data = await res.json();

                if (data.active) {
                    const ppl = data.perplexity ? ` | PPL: ${data.perplexity.toFixed(0)}` : '';
                    trainingStatusText.textContent = (data.progress || 'Training...') + ppl;
                    const currentWidth = parseFloat(progressBar.style.width) || 5;
                    if (currentWidth < 90) {
                        progressBar.style.width = (currentWidth + 1.5) + '%';
                    }
                } else {
                    clearInterval(interval);
                    progressBar.style.width = '100%';
                    trainingStatusText.textContent = data.progress || 'Complete!';
                    trainModelBtn.disabled = false;

                    // Auto-load freshly trained model
                    setTimeout(async () => {
                        trainingPanel.classList.add('hidden');
                        await fetchmodels(); // Refresh lists
                        const configRes = await fetch('/api/config');
                        const configData = await configRes.json();
                        if (configData.loaded) {
                            setModelActive(configData);
                            addMessage(`Training complete! ${configData.num_qubits}-Qubit model ready. Start chatting!`, 'system');
                        }
                    }, 1500);
                }
            } catch (e) {
                console.error(e);
            }
        }, 2000);
    }

    // ═══ Chat ═══
    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = sender === 'user' ? '👤' : '⚛️';

        const content = document.createElement('div');
        content.className = 'message-content';

        if (sender === 'system') {
            // Render with monospace for model output
            const pre = document.createElement('pre');
            pre.className = 'model-output';
            pre.textContent = text;
            content.appendChild(pre);
        } else {
            content.textContent = text;
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addLoading() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system-message loading-msg';

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = '⚛️';

        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = '<div class="loading"><span></span><span></span><span></span></div>';

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgDiv;
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!modelLoaded) return;

        const prompt = promptInput.value.trim();
        if (!prompt) return;

        addMessage(prompt, 'user');
        promptInput.value = '';
        sendBtn.disabled = true;

        const loader = addLoading();

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    prompt, 
                    max_tokens: 256, 
                    temperature: parseFloat(tempSlider.value),
                    top_p: 0.9,
                    top_k: 80
                })
            });
            const data = await res.json();
            loader.remove();

            if (res.ok) {
                addMessage(data.response, 'system');
                
                // If metadata exists, show it in a subtle way
                if (data.metadata) {
                    const statsDiv = document.createElement('div');
                    statsDiv.className = 'message-stats';
                    statsDiv.innerHTML = `
                        <span>Confidence: <b>${data.metadata.confidence}</b></span>
                        <span>Time: <b>${data.metadata.generation_time}</b></span>
                        <span>Model: <b>${data.metadata.model}</b></span>
                    `;
                    chatContainer.appendChild(statsDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            } else {
                addMessage('Error: ' + (data.detail || 'Generation failed'), 'system');
            }
        } catch (e) {
            loader.remove();
            addMessage('Error: Could not reach the backend.', 'system');
        } finally {
            sendBtn.disabled = !modelLoaded;
        }
    });

    // ═══ Initial Load ═══
    fetchmodels(); // Fetch available models first
    fetch('/api/config')
        .then(r => r.json())
        .then(data => {
            if (data.loaded) {
                setModelActive(data);
                welcomeScreen?.remove();
                addMessage(`Model auto-loaded: ${data.num_qubits}-Qubit Quantum Transformer (${data.total_parameters?.toLocaleString()} params)`, 'system');
            }
        })
        .catch(err => console.error('Initial config fetch failed:', err));
});
