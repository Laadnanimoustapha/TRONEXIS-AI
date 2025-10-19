document.addEventListener('DOMContentLoaded', () => {
            const modal = document.getElementById('test-warning-modal');
            const closeBtn = document.getElementById('close-modal-btn');
            closeBtn.onclick = () => {
                modal.style.display = 'none';
            };
        });