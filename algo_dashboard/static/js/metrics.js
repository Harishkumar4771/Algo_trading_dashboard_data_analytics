// Function to update metrics when input values change
function updateMetrics() {
    const form = document.getElementById('metrics-form');
    const formData = new FormData(form);
    const initialCapitalInput = document.getElementById('initial_capital');
    let initialCapital = parseFloat(formData.get('initial_capital'));
    const timePeriod = formData.get('time_period');

    // Validate initial capital
    if (isNaN(initialCapital) || initialCapital <= 0) {
        alert('Please enter a valid positive number for initial capital');
        initialCapitalInput.value = '100000';
        initialCapital = 100000;
    }
    if (initialCapital > 1000000000) {
        alert('Initial capital is too large. Please enter a smaller value.');
        initialCapitalInput.value = '100000';
        initialCapital = 100000;
    }

    fetch('/update_metrics', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            initial_capital: initialCapital,
            time_period: timePeriod
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            alert('Error updating metrics: ' + data.error);
            return;
        }

        // Update metrics values
        document.getElementById('sharpe-ratio').textContent = data.metrics.Sharpe_Ratio.toFixed(2);
        document.getElementById('sortino-ratio').textContent = data.metrics.Sortino_Ratio.toFixed(2);
        document.getElementById('max-drawdown').textContent = data.metrics.Max_Drawdown.toFixed(2) + '%';
        document.getElementById('cagr').textContent = data.metrics.CAGR.toFixed(2) + '%';

        // Update chart with animation
        Plotly.newPlot('metrics-chart', data.chart_data, data.layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while updating the metrics');
    });
}

// Debounce function to limit the rate of API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add event listeners to form inputs for real-time updates
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('metrics-form');
    if (form) {
        const debouncedUpdate = debounce(updateMetrics, 300); // Wait 300ms after last input

        const initialCapitalInput = document.getElementById('initial_capital');
        const timePeriodSelect = document.getElementById('time_period');

        if (initialCapitalInput) {
            // Listen for real-time input changes
            initialCapitalInput.addEventListener('input', debouncedUpdate);
            // Also listen for 'change' event for non-text input methods
            initialCapitalInput.addEventListener('change', debouncedUpdate);
        }

        if (timePeriodSelect) {
            timePeriodSelect.addEventListener('change', updateMetrics);
        }

        // Run initial update
        updateMetrics();
    }
});