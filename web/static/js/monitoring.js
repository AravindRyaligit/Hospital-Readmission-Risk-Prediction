
const API_BASE_URL = 'http://localhost:8000';


async function updateSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        const apiStatus = document.getElementById('apiStatus');
        const modelsLoaded = document.getElementById('modelsLoaded');

        if (data.status === 'healthy') {
            apiStatus.textContent = '✓ Online';
            apiStatus.style.color = 'var(--success-color)';
            modelsLoaded.textContent = data.models_loaded ? '✓ Yes' : '✗ No';
            modelsLoaded.style.color = data.models_loaded ? 'var(--success-color)' : 'var(--danger-color)';
        } else {
            apiStatus.textContent = '✗ Offline';
            apiStatus.style.color = 'var(--danger-color)';
            modelsLoaded.textContent = '✗ No';
            modelsLoaded.style.color = 'var(--danger-color)';
        }


        const now = new Date();
        document.getElementById('lastUpdated').textContent = now.toLocaleTimeString();

    } catch (error) {
        console.error('Error checking API health:', error);
        document.getElementById('apiStatus').textContent = '✗ Unreachable';
        document.getElementById('apiStatus').style.color = 'var(--danger-color)';
    }
}


function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart')?.getContext('2d');
    if (!ctx) return;


    const dates = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Current'];
    const xgboostScores = [0.72, 0.71, 0.73, 0.72, 0.72];
    const lightgbmScores = [0.73, 0.74, 0.73, 0.74, 0.73];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'XGBoost AUC-ROC',
                    data: xgboostScores,
                    borderColor: 'rgba(37, 99, 235, 1)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'LightGBM AUC-ROC',
                    data: lightgbmScores,
                    borderColor: 'rgba(124, 58, 237, 1)',
                    backgroundColor: 'rgba(124, 58, 237, 0.1)',
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.65,
                    max: 0.80,
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f1f5f9'
                    }
                }
            }
        }
    });
}


function createPredictionDistChart() {
    const ctx = document.getElementById('predictionDistChart')?.getContext('2d');
    if (!ctx) return;


    const bins = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'];
    const counts = [150, 200, 250, 300, 280, 220, 180, 120, 80, 20];

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: bins,
            datasets: [{
                label: 'Number of Predictions',
                data: counts,
                backgroundColor: 'rgba(96, 165, 250, 0.7)',
                borderColor: 'rgba(96, 165, 250, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(51, 65, 85, 0.5)'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f1f5f9'
                    }
                }
            }
        }
    });
}


function initMonitoring() {
    updateSystemStatus();
    createPerformanceChart();
    createPredictionDistChart();


    setInterval(updateSystemStatus, 30000);
}


if (window.location.pathname.includes('monitoring.html')) {
    initMonitoring();
}
