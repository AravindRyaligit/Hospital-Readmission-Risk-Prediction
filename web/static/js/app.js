
const API_BASE_URL = 'http://localhost:8000';


document.getElementById('predictionForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const patientData = {};

    for (let [key, value] of formData.entries()) {
        if (['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
            'n_outpatient', 'n_inpatient', 'n_emergency'].includes(key)) {
            patientData[key] = parseInt(value);
        } else {
            patientData[key] = value;
        }
    }

    try {
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.innerHTML = '<span class="loading"></span> Predicting...';
        submitBtn.disabled = true;

        const response = await fetch(`${API_BASE_URL}/predict?model=lightgbm`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(patientData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();

        displayResults(result);

        submitBtn.textContent = originalText;
        submitBtn.disabled = false;

    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction. Please ensure the API server is running.');

        const submitBtn = e.target.querySelector('button[type="submit"]');
        submitBtn.textContent = 'Predict Risk';
        submitBtn.disabled = false;
    }
});

function displayResults(result) {
    const resultsCard = document.getElementById('resultsCard');
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });

    const riskLevelElement = document.getElementById('riskLevel');
    riskLevelElement.textContent = result.risk_level;
    riskLevelElement.className = `value risk-${result.risk_level.toLowerCase()}`;

    const riskProbability = (result.readmission_risk * 100).toFixed(1);
    document.getElementById('riskProbability').textContent = `${riskProbability}%`;

    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence').textContent = `${confidence}%`;

    document.getElementById('modelUsed').textContent = result.model_used.toUpperCase();

    createRiskGauge(result.readmission_risk);
    createRiskGauge(result.readmission_risk);
}

function createRiskGauge(riskValue) {
    const canvas = document.getElementById('riskGauge');
    const ctx = canvas.getContext('2d');

    if (window.riskChart) {
        window.riskChart.destroy();
        if (window.riskChart) {
            window.riskChart.destroy();
        }

        let color;
        if (riskValue < 0.3) {
            color = '#10b981';
        } else if (riskValue < 0.6) {
            color = '#f59e0b';
        } else {
            color = '#ef4444';
        }

        window.riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [riskValue * 100, (1 - riskValue) * 100],
                    backgroundColor: [color, 'rgba(51, 65, 85, 0.3)'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '75%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            },
            plugins: [{
                id: 'centerText',
                beforeDraw: function (chart) {
                    const width = chart.width;
                    const height = chart.height;
                    const ctx = chart.ctx;

                    ctx.restore();
                    const fontSize = (height / 114).toFixed(2);
                    ctx.font = `bold ${fontSize}em sans-serif`;
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = color;

                    const text = `${(riskValue * 100).toFixed(1)}%`;
                    const textX = Math.round((width - ctx.measureText(text).width) / 2);
                    const textY = height / 2;

                    ctx.fillText(text, textX, textY);
                    ctx.save();
                }
            }]
        });
    }

    async function loadModelPerformance() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/performance`);
            const data = await response.json();

            document.getElementById('xgboost-auc-roc').textContent = data.xgboost.auc_roc.toFixed(4);
            document.getElementById('xgboost-auc-pr').textContent = data.xgboost.auc_pr.toFixed(4);

            document.getElementById('lightgbm-auc-roc').textContent = data.lightgbm.auc_roc.toFixed(4);
            document.getElementById('lightgbm-auc-pr').textContent = data.lightgbm.auc_pr.toFixed(4);

            createComparisonChart(data);

        } catch (error) {
            console.error('Error loading model performance:', error);
        }
    }

    function createComparisonChart(data) {
        const ctx = document.getElementById('comparisonChart')?.getContext('2d');
        if (!ctx) return;

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['AUC-ROC', 'AUC-PR'],
                datasets: [
                    {
                        label: 'XGBoost',
                        data: [data.xgboost.auc_roc, data.xgboost.auc_pr],
                        backgroundColor: 'rgba(37, 99, 235, 0.7)',
                        borderColor: 'rgba(37, 99, 235, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'LightGBM',
                        data: [data.lightgbm.auc_roc, data.lightgbm.auc_pr],
                        backgroundColor: 'rgba(124, 58, 237, 0.7)',
                        borderColor: 'rgba(124, 58, 237, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
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

    if (window.location.pathname.includes('dashboard.html')) {
        loadModelPerformance();
    }
    async function checkAPIHealth() {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            const data = await response.json();

            if (data.status === 'healthy') {
                console.log('API is healthy');
            } else {
                console.warn('API is unhealthy');
            }
        } catch (error) {
            console.error('API is not reachable:', error);
        }
    }

    checkAPIHealth();
}
