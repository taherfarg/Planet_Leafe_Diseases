document.addEventListener('DOMContentLoaded', function() {
    const resultDiv = document.getElementById('result');

    if (resultDiv) {
        // Fade in animation for result
        resultDiv.style.opacity = '0';
        setTimeout(() => {
            resultDiv.style.transition = 'opacity 1s ease';
            resultDiv.style.opacity = '1';
        }, 500);
    }
});
