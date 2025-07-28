// This file is located at smartwriter/public/app.js
window.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('input-text');
    const outputText = document.getElementById('output-text');
    const generateBtn = document.getElementById('generate-btn');
    const functionSelect = document.getElementById('function-select');
    const loadingIndicator = document.getElementById('loading');
    const messageBox = document.getElementById('message-box');

    // IMPORTANT: This is the public URL of your deployed backend server on Vercel
    const BACKEND_URL = 'https://nexus-ai-amber.vercel.app';

    function showMessage(msg, type = 'info') {
        messageBox.textContent = msg;
        messageBox.className = `message-box ${type}`;
        messageBox.style.display = 'block';
        setTimeout(() => {
            messageBox.style.display = 'none';
        }, 3000);
    }

    generateBtn.addEventListener('click', async () => {
        const text = inputText.value.trim();
        const mode = functionSelect.value;

        if (!text) {
            showMessage("❗ Please enter some text", "error");
            return;
        }

        loadingIndicator.style.display = 'block';
        outputText.value = '';

        try {
            // Use the defined BACKEND_URL for the fetch request
            const response = await fetch(`${BACKEND_URL}/summarize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, mode })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(`Server error: ${response.statusText} - ${errorData.error || 'Unknown error'}`);
            }

            const data = await response.json();
            outputText.value = data.result;
            showMessage("✅ AI magic done!", "success");

        } catch (err) {
            console.error("❌ Frontend error:", err);
            showMessage(`⚠️ Failed to connect to backend: ${err.message}`, "error");
        } finally {
            loadingIndicator.style.display = 'none';
        }
    });

    // Copy to clipboard functionality
    document.getElementById('copy-btn').addEventListener('click', () => {
        outputText.select();
        document.execCommand('copy');
        showMessage("📋 Copied to clipboard!", "info");
    });

    // Download functionality
    document.getElementById('download-btn').addEventListener('click', () => {
        const content = outputText.value;
        if (content) {
            const blob = new Blob([content], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'smartwriter_output.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            showMessage("📥 Downloaded!", "success");
        } else {
            showMessage("❗ Nothing to download", "info");
        }
    });
});
