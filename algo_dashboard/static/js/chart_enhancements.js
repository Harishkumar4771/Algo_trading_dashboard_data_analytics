// Enhanced chart configuration
function enhanceChartConfig(figure) {
    // Set custom theme
    const customTheme = {
        'paper_bgcolor': 'rgba(248, 250, 252, 0.95)',
        'plot_bgcolor': 'rgba(248, 250, 252, 0.5)',
        'font': {
            'family': 'Inter, -apple-system, system-ui, sans-serif',
            'color': '#1e293b'
        },
        'grid': {
            'xaxis': {
                'color': '#e2e8f0',
                'zerolinecolor': '#cbd5e1'
            },
            'yaxis': {
                'color': '#e2e8f0',
                'zerolinecolor': '#cbd5e1'
            }
        },
        'xaxis': {
            'gridcolor': '#e2e8f0',
            'linecolor': '#cbd5e1',
            'tickfont': { 'size': 11 },
            'title': { 'font': { 'size': 13 } }
        },
        'yaxis': {
            'gridcolor': '#e2e8f0',
            'linecolor': '#cbd5e1',
            'tickfont': { 'size': 11 },
            'title': { 'font': { 'size': 13 } }
        },
        'legend': {
            'font': { 'size': 11 },
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#e2e8f0',
            'borderwidth': 1,
            'x': 0,
            'y': 1.1,
            'orientation': 'h'
        },
        'hoverlabel': {
            'font': { 'size': 12 },
            'bgcolor': '#1e293b',
            'bordercolor': '#1e293b',
            'font': { 'color': '#ffffff' }
        },
        'annotations': {
            'font': { 'size': 12 }
        }
    };

    // Merge custom theme with existing layout
    figure.layout = { ...figure.layout, ...customTheme };

    // Enhanced modebar configuration
    figure.config = {
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: [
            'drawopenpath',
            'drawclosedpath',
            'eraseshape'
        ],
        modeBarButtonsToRemove: [
            'lasso2d',
            'select2d'
        ],
        toImageButtonOptions: {
            format: 'png',
            filename: 'chart_export',
            height: 800,
            width: 1200,
            scale: 2
        }
    };

    // Add responsive behavior
    figure.layout.autosize = true;
    figure.layout.responsive = true;

    // Enhanced hover interactions
    figure.layout.hovermode = 'x unified';
    figure.layout.hoverdistance = 100;
    figure.layout.spikedistance = 1000;

    // Add crosshair and spikes
    figure.layout.xaxis.showspikes = true;
    figure.layout.yaxis.showspikes = true;
    figure.layout.xaxis.spikemode = 'across';
    figure.layout.yaxis.spikemode = 'across';
    figure.layout.xaxis.spikethickness = 1;
    figure.layout.yaxis.spikethickness = 1;
    figure.layout.xaxis.spikecolor = '#94a3b8';
    figure.layout.yaxis.spikecolor = '#94a3b8';

    return figure;
}

// Enhanced tooltips with more information
function createEnhancedTooltip(data) {
    const formatNumber = (num) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 2
        }).format(num);
    };

    const formatDate = (date) => {
        return new Date(date).toLocaleDateString('en-IN', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    return `
        <div style="padding: 10px; background: rgba(30, 41, 59, 0.95); border-radius: 4px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <div style="color: #94a3b8; margin-bottom: 5px;">${formatDate(data.x)}</div>
            <div style="color: #f8fafc; font-weight: 600;">${formatNumber(data.y)}</div>
            ${data.additional ? `
                <div style="color: #94a3b8; margin-top: 5px; font-size: 0.9em;">
                    ${Object.entries(data.additional).map(([key, value]) => 
                        `<div>${key}: ${typeof value === 'number' ? formatNumber(value) : value}</div>`
                    ).join('')}
                </div>
            ` : ''}
        </div>
    `;
}

// Animate chart transitions
function animateChart(figure) {
    const frames = [];
    const animationSteps = 30;
    
    figure.data.forEach(trace => {
        if (trace.visible !== false) {
            const yValues = trace.y;
            for (let i = 0; i <= animationSteps; i++) {
                const frame = {
                    data: [{
                        y: yValues.map(y => y * (i / animationSteps))
                    }],
                    traces: [figure.data.indexOf(trace)],
                    name: `frame${i}`
                };
                frames.push(frame);
            }
        }
    });

    figure.frames = frames;
    figure.layout.updatemenus = [{
        type: 'buttons',
        showactive: false,
        y: 0,
        x: 0,
        xanchor: 'left',
        yanchor: 'top',
        pad: {t: 0, r: 10},
        buttons: [{
            method: 'animate',
            args: [null, {
                mode: 'immediate',
                frame: {
                    duration: 500,
                    redraw: false
                },
                transition: {
                    duration: 500,
                    easing: 'cubic-in-out'
                }
            }],
            label: 'Play'
        }]
    }];

    return figure;
}

// Initialize chart interactivity
document.addEventListener('DOMContentLoaded', function() {
    // Add zoom synchronization between charts
    const charts = document.querySelectorAll('.js-plotly-plot');
    let syncingZoom = false;

    charts.forEach(chart => {
        chart.on('plotly_relayout', function(eventdata) {
            if (!syncingZoom && (eventdata['xaxis.range[0]'] || eventdata['xaxis.range[1]'])) {
                syncingZoom = true;
                const range = [eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]']];
                
                charts.forEach(otherChart => {
                    if (otherChart !== chart) {
                        Plotly.relayout(otherChart, {
                            'xaxis.range': range
                        });
                    }
                });
                
                setTimeout(() => { syncingZoom = false; }, 100);
            }
        });
    });

    // Add chart state persistence
    charts.forEach(chart => {
        const chartId = chart.id;
        const savedState = localStorage.getItem(`chart_state_${chartId}`);
        
        if (savedState) {
            try {
                const state = JSON.parse(savedState);
                Plotly.relayout(chart, state);
            } catch (e) {
                console.warn('Failed to restore chart state:', e);
            }
        }

        chart.on('plotly_relayout', function(eventdata) {
            localStorage.setItem(`chart_state_${chartId}`, JSON.stringify(eventdata));
        });
    });
});