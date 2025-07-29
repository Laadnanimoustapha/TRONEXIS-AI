// app.js
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.12.0';

let textGenerator = null;
let modelLoaded = false;

window.addEventListener('DOMContentLoaded', async () => {
    const elements = {
        generateBtn: document.getElementById('generate-btn'),
        inputText: document.getElementById('input-text'),
        outputText: document.getElementById('output-text'),
        loading: document.getElementById('loading'),
        messageBox: document.getElementById('message-box'),
        functionSelect: document.getElementById('function-select'),
        copyBtn: document.getElementById('copy-btn'),
        downloadBtn: document.getElementById('download-btn'),
        modelNotice: document.getElementById('model-notice')
    };

    // Show message function
    function showMessage(msg, type = 'info') {
        elements.messageBox.textContent = msg;
        elements.messageBox.className = `message-box ${type}`;
        elements.messageBox.classList.add('show');
        setTimeout(() => elements.messageBox.classList.remove('show'), 4000);
    }

    // Create progress UI
    function createProgressUI() {
        elements.loading.innerHTML = `
            <div class="progress-container">
                <div class="progress-bar"></div>
            </div>
            <p>Downloading AI model... <span id="progress-percent">0%</span></p>
            <p class="model-info">Model: LaMini-Flan-T5-77M (40MB)</p>
            <p class="connection-info">First-time setup - will be cached for future use</p>
        `;
        elements.loading.style.display = 'block';
        return elements.loading.querySelector('#progress-percent');
    }

    // Initialize the model
    async function initializeModel() {
        try {
            // Show progress UI
            const progressElement = createProgressUI();
            
            // Load the model with progress tracking
            textGenerator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-77M', {
                quantized: true,
                progress_callback: (progress) => {
                    const percent = Math.round(progress * 100);
                    progressElement.textContent = `${percent}%`;
                    const progressBar = elements.loading.querySelector('.progress-bar');
                    progressBar.style.width = `${percent}%`;
                }
            });
            
            modelLoaded = true;
            elements.loading.style.display = 'none';
            elements.modelNotice.style.display = 'none';
            showMessage("✅ AI Model Ready!", 'success');
        } catch (error) {
            console.error("Model loading error:", error);
            handleModelError(error);
        }
    }

    // Handle model loading errors
    function handleModelError(error) {
        elements.loading.innerHTML = `
            <div class="error-icon">⚠️</div>
            <p>Failed to load AI model</p>
            <p class="error-detail">${error.message || 'Network issue'}</p>
            <div class="troubleshooting">
                <p>Possible solutions:</p>
                <ul>
                    <li>Refresh the page</li>
                    <li>Check your internet connection</li>
                    <li>Try a different browser</li>
                </ul>
            </div>
            <button id="retry-btn" class="action-btn">🔄 Retry</button>
        `;
        elements.loading.style.display = 'block';
        
        // Add retry functionality
        document.getElementById('retry-btn').addEventListener('click', initializeModel);
    }

    // Function prompts
    const functionPrompts = {
        summarize: (text) => `Summarize this in 2-3 sentences:\n\n${text}`,
        rewrite: (text) => `Rewrite this to improve clarity:\n\n${text}`,
        expand: (text) => `Expand this by adding details:\n\n${text}`,
        simplify: (text) => `Simplify this for easier understanding:\n\n${text}`,
        professional: (text) => `Rewrite this professionally:\n\n${text}`,
        casual: (text) => `Rewrite this casually:\n\n${text}`
    };

    // Initialize the model when page loads
    showMessage("🧠 Preparing AI...");
    initializeModel();

    // Generate button handler
    elements.generateBtn.addEventListener('click', async () => {
        if (!modelLoaded) {
            showMessage("⚠️ Model still loading. Please wait...", "warning");
            return;
        }

        const text = elements.inputText.value.trim();
        const selectedFunction = elements.functionSelect.value;

        if (!text) {
            showMessage("❗ Please enter text", "error");
            return;
        }

        elements.loading.innerHTML = '<div class="spinner"></div><p>Processing text...</p>';
        elements.loading.style.display = 'block';
        elements.outputText.value = '';

        try {
            const prompt = functionPrompts[selectedFunction](text);
            const output = await textGenerator(prompt, {
                max_new_tokens: 500,
                temperature: 0.7,
                repetition_penalty: 1.3
            });

            // Extract just the generated text
            const generatedText = output[0].generated_text.replace(prompt, '').trim();
            elements.outputText.value = generatedText;
            showMessage("✨ Done!", "success");
        } catch (err) {
            console.error("Generation error:", err);
            showMessage("⚠️ Error processing text", "error");
            elements.outputText.value = "An error occurred. Please try again.";
        } finally {
            elements.loading.style.display = 'none';
        }
    });

    // Copy to clipboard
    elements.copyBtn.addEventListener('click', () => {
        elements.outputText.select();
        try {
            document.execCommand('copy');
            showMessage("📋 Copied to clipboard!", "success");
        } catch (err) {
            showMessage("❗ Copy failed. Please copy manually", "error");
        }
    });

    // Download text
    elements.downloadBtn.addEventListener('click', () => {
        const text = elements.outputText.value;
        if (!text) {
            showMessage("❗ No text to download", "error");
            return;
        }
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'smartwriter_output.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showMessage("💾 Text downloaded!", "success");
    });
});