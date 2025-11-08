// Animated metrics

// Animated metrics
function animateMetrics() {
    const metrics = document.querySelectorAll('.metric-value');
    metrics.forEach((metric, index) => {
        setTimeout(() => {
            metric.style.opacity = '1';
            metric.style.transform = 'translateY(0)';
        }, index * 100);
    });
}

// Progress bar animations
function animateProgressBars() {
    const bars = document.querySelectorAll('.progress-bar');
    bars.forEach((bar, index) => {
        const target = bar.getAttribute('aria-valuenow');
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = target + '%';
        }, index * 100);
    });
}

// Chart date range selector
function initializeDateRangeSelector() {
    const ranges = document.querySelectorAll('.date-range-btn');
    ranges.forEach(btn => {
        btn.addEventListener('click', () => {
            const range = btn.getAttribute('data-range');
            const now = new Date();
            let start;

            switch(range) {
                case '1d':
                    start = new Date(now - 24*60*60*1000);
                    break;
                case '1w':
                    start = new Date(now - 7*24*60*60*1000);
                    break;
                case '1m':
                    start = new Date(now - 30*24*60*60*1000);
                    break;
                case '3m':
                    start = new Date(now - 90*24*60*60*1000);
                    break;
                case '1y':
                    start = new Date(now - 365*24*60*60*1000);
                    break;
                default:
                    return;
            }

            // Update chart range
            const update = {
                'xaxis.range': [start, now]
            };
            Plotly.relayout('main-chart', update);
        });
    });
}

// Enhanced tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-tooltip]'));
    tooltipTriggerList.forEach(el => {
        el.addEventListener('mouseenter', e => {
            const tooltip = document.createElement('div');
            tooltip.className = 'custom-tooltip-popup';
            tooltip.textContent = el.getAttribute('data-tooltip');
            
            const rect = el.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width/2) + 'px';
            tooltip.style.top = (rect.top - 10) + 'px';
            
            document.body.appendChild(tooltip);
            
            setTimeout(() => tooltip.classList.add('visible'), 10);
        });
        
        el.addEventListener('mouseleave', () => {
            const tooltip = document.querySelector('.custom-tooltip-popup');
            if (tooltip) {
                tooltip.classList.remove('visible');
                setTimeout(() => tooltip.remove(), 200);
            }
        });
    });
}

// Initialize all features when document is ready
document.addEventListener('DOMContentLoaded', () => {
    initializeChartControls();
    animateMetrics();
    animateProgressBars();
    initializeDateRangeSelector();
    initializeTooltips();
});