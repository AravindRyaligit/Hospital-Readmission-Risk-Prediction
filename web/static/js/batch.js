
const API_BASE_URL = 'http://localhost:8000';
let batchResultsData = [];



document.getElementById('batchUploadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('csvFile');
    const modelSelect = document.getElementById('batchModel');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a CSV file');
        return;
    }

    try {

        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.innerHTML = '<span class="loading"></span> Processing...';
        submitBtn.disabled = true;


        const formData = new FormData();
        formData.append('file', file);


        const response = await fetch(`${API_BASE_URL}/batch_predict?model=${modelSelect.value}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Batch prediction failed');
        }

        const result = await response.json();
        batchResultsData = result.predictions;


        displayBatchResults(result);


        submitBtn.textContent = originalText;
        submitBtn.disabled = false;

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing batch predictions. Please ensure the CSV format is correct and the API is running.');


        const submitBtn = e.target.querySelector('button[type="submit"]');
        submitBtn.textContent = 'Process Batch';
        submitBtn.disabled = false;
    }
});


function displayBatchResults(result) {
    const resultsCard = document.getElementById('batchResults');
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth' });


    document.getElementById('totalPatients').textContent = result.total_patients;
    document.getElementById('highRiskCount').textContent = result.high_risk_count;


    let mediumCount = 0;
    let lowCount = 0;
    result.predictions.forEach(pred => {
        if (pred.risk_level === 'Medium') mediumCount++;
        if (pred.risk_level === 'Low') lowCount++;
    });

    document.getElementById('mediumRiskCount').textContent = mediumCount;
    document.getElementById('lowRiskCount').textContent = lowCount;


    createRiskDistributionChart(result.high_risk_count, mediumCount, lowCount);


    createResultsTable(result.predictions);
}


function createRiskDistributionChart(highCount, mediumCount, lowCount) {
    const ctx = document.getElementById('riskDistributionChart').getContext('2d');


    if (window.riskDistChart) {
        window.riskDistChart.destroy();
    }

    window.riskDistChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [highCount, mediumCount, lowCount],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderColor: [
                    '#ef4444',
                    '#f59e0b',
                    '#10b981'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
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


function createResultsTable(predictions) {
    const tableContainer = document.getElementById('resultsTable');

    let tableHTML = `
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: rgba(15, 23, 42, 0.8); border-bottom: 2px solid var(--border-color);">
                    <th style="padding: 12px; text-align: left;">#</th>
                    <th style="padding: 12px; text-align: left;">Risk Level</th>
                    <th style="padding: 12px; text-align: left;">Risk Probability</th>
                    <th style="padding: 12px; text-align: left;">Confidence</th>
                    <th style="padding: 12px; text-align: left;">Model</th>
                </tr>
            </thead>
            <tbody>
    `;

    predictions.forEach((pred, index) => {
        const riskClass = `risk-${pred.risk_level.toLowerCase()}`;
        tableHTML += `
            <tr style="border-bottom: 1px solid rgba(51, 65, 85, 0.5);">
                <td style="padding: 12px;">${index + 1}</td>
                <td style="padding: 12px;"><span class="${riskClass}">${pred.risk_level}</span></td>
                <td style="padding: 12px;">${(pred.readmission_risk * 100).toFixed(1)}%</td>
                <td style="padding: 12px;">${(pred.confidence * 100).toFixed(1)}%</td>
                <td style="padding: 12px;">${pred.model_used.toUpperCase()}</td>
            </tr>
        `;
    });

    tableHTML += `
            </tbody>
        </table>
    `;

    tableContainer.innerHTML = tableHTML;
}


function downloadResults() {
    if (batchResultsData.length === 0) {
        alert('No results to download');
        return;
    }


    let csv = 'Patient #,Risk Level,Risk Probability,Confidence,Model Used\n';
    batchResultsData.forEach((pred, index) => {
        csv += `${index + 1},${pred.risk_level},${pred.readmission_risk},${pred.confidence},${pred.model_used}\n`;
    });


    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_predictions_results.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}


function downloadSampleCSV() {
    const sampleCSV = `age,time_in_hospital,n_lab_procedures,n_procedures,n_medications,n_outpatient,n_inpatient,n_emergency,medical_specialty,diag_1,diag_2,diag_3,glucose_test,A1Ctest,change,diabetes_med
[70-80),5,45,2,15,0,0,0,InternalMedicine,Circulatory,Diabetes,Other,normal,high,yes,yes
[60-70),3,30,1,12,1,0,0,Family/GeneralPractice,Respiratory,Circulatory,Other,no,normal,no,yes
[50-60),7,60,3,20,0,1,0,Cardiology,Circulatory,Circulatory,Diabetes,high,high,yes,yes`;

    const blob = new Blob([sampleCSV], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sample_batch_input.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
