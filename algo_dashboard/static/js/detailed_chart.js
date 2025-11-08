// Enhanced chart configuration with detailed tracking
function initializeDetailedCharts() {
    // Configure detailed tooltips
    const customHoverTemplate = `
        <div style="background: rgba(30, 41, 59, 0.95); padding: 10px; border-radius: 4px;">
            <div style="color: #f8fafc; font-size: 14px; font-weight: 600; margin-bottom: 5px;">
                %{fullData.name}
            </div>
            <div style="color: #e2e8f0; font-size: 12px;">
                <div>Date: %{x}</div>
                <div>Value: ₹%{y:,.2f}</div>
                %{customData}
            </div>
        </div>
    `;

    // Enhanced cursor tracking
    const cursorConfig = {
        showSpikes: true,
        spikeColor: '#94a3b8',
        spikeDash: 'solid',
        spikeMode: 'across+marker',
        spikeThickness: 1,
        spikesnap: 'cursor'
    };

    // Configure detailed chart layout
    const detailedLayout = {
        grid: {rows: 4, columns: 1, pattern: 'independent'},
        showlegend: true,
        legend: {
            orientation: 'h',
            yanchor: 'bottom',
            y: 1.02,
            xanchor: 'right',
            x: 1,
            bgcolor: 'rgba(255, 255, 255, 0.9)',
            bordercolor: '#e2e8f0',
            borderwidth: 1
        },
        annotations: [],
        shapes: [],
        xaxis: {
            ...cursorConfig,
            rangeslider: {visible: true},
            rangeselector: {
                buttons: [
                    {count: 1, label: '1h', step: 'hour', stepmode: 'backward'},
                    {count: 6, label: '6h', step: 'hour', stepmode: 'backward'},
                    {count: 1, label: '1d', step: 'day', stepmode: 'backward'},
                    {count: 7, label: '1w', step: 'day', stepmode: 'backward'},
                    {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
                    {step: 'all'}
                ]
            }
        },
        yaxis: {
            ...cursorConfig,
            tickprefix: '₹ ',
            tickformat: ',.2f',
            gridcolor: '#e2e8f0',
            zerolinecolor: '#cbd5e1'
        },
        margin: {t: 50, r: 50, b: 50, l: 50}
    };

    // Add detailed price breakdown annotations
    function addPriceBreakdown(chart, data) {
        const annotations = [];
        const lastPrice = data[data.length - 1];
        
        annotations.push({
            x: lastPrice.x,
            y: lastPrice.y,
            text: `Current: ₹${lastPrice.y.toFixed(2)}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            ax: 40,
            ay: -40,
            font: {size: 12, color: '#1e293b'}
        });

        // Add key level annotations
        const max = Math.max(...data.map(d => d.y));
        const min = Math.min(...data.map(d => d.y));
        
        annotations.push({
            x: data.find(d => d.y === max).x,
            y: max,
            text: `High: ₹${max.toFixed(2)}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            ax: 40,
            ay: 40,
            font: {size: 12, color: '#059669'}
        });

        annotations.push({
            x: data.find(d => d.y === min).x,
            y: min,
            text: `Low: ₹${min.toFixed(2)}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            ax: -40,
            ay: -40,
            font: {size: 12, color: '#dc2626'}
        });

        chart.layout.annotations = annotations;
    }

    // Add live cursor tracking
    function addLiveCursor(chart) {
        let cursorDiv = document.createElement('div');
        cursorDiv.className = 'chart-cursor-info';
        cursorDiv.style.cssText = `
            position: absolute;
            background: rgba(30, 41, 59, 0.95);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            z-index: 1000;
        `;
        chart.parentElement.appendChild(cursorDiv);

        chart.on('plotly_hover', function(data) {
            const pt = data.points[0];
            const x = pt.xaxis.l2p(pt.x) + pt.xaxis._offset;
            const y = pt.yaxis.l2p(pt.y) + pt.yaxis._offset;
            
            cursorDiv.innerHTML = `
                <div style="font-weight: 600">Date: ${pt.x}</div>
                <div>Price: ₹${pt.y.toFixed(2)}</div>
                ${pt.customdata ? `<div>${pt.customdata}</div>` : ''}
            `;
            
            cursorDiv.style.display = 'block';
            cursorDiv.style.left = `${x + 10}px`;
            cursorDiv.style.top = `${y - 10}px`;
        });

        chart.on('plotly_unhover', function() {
            cursorDiv.style.display = 'none';
        });
    }

    // Initialize all charts with detailed features
    document.querySelectorAll('.js-plotly-plot').forEach(chart => {
        // Add cursor tracking
        addLiveCursor(chart);
        
        // Update layout with detailed configuration
        Plotly.update(chart, {}, detailedLayout);
        
        // Add synchronization between charts
        chart.on('plotly_relayout', function(eventdata) {
            if (eventdata['xaxis.range[0]']) {
                const charts = document.querySelectorAll('.js-plotly-plot');
                charts.forEach(otherChart => {
                    if (otherChart !== chart) {
                        Plotly.relayout(otherChart, {
                            'xaxis.range': [
                                eventdata['xaxis.range[0]'],
                                eventdata['xaxis.range[1]']
                            ]
                        });
                    }
                });
            }
        });
    });
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', initializeDetailedCharts);