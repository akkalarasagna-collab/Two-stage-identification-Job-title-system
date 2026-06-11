document.addEventListener("DOMContentLoaded", () => {
    const predictionForm = document.getElementById("predictionForm");
    if (predictionForm) {
        predictionForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const message = document.getElementById("predictionMessage");
            const title = document.getElementById("predictedTitle");
            const confidence = document.getElementById("confidenceText");
            const button = predictionForm.querySelector("button");

            message.className = "message";
            message.textContent = "Analyzing description...";
            button.disabled = true;

            try {
                const formData = new FormData(predictionForm);
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData,
                });
                const data = await response.json();
                if (!response.ok || !data.success) {
                    throw new Error(data.message || "Prediction failed");
                }

                title.textContent = data.predicted_job_title;
                confidence.textContent = data.confidence_score === null
                    ? "Confidence score is unavailable for this model."
                    : `Estimated confidence: ${data.confidence_score}%`;
                message.className = "message success";
                message.textContent = "Prediction saved to your history.";
            } catch (error) {
                message.className = "message error";
                message.textContent = error.message;
            } finally {
                button.disabled = false;
            }
        });
    }

    renderPredictionChart();
    renderActivityChart();
});

function readChartData(id) {
    const canvas = document.getElementById(id);
    if (!canvas || !window.Chart) {
        return null;
    }
    return {
        canvas,
        data: JSON.parse(canvas.dataset.chart || "[]"),
    };
}

function renderPredictionChart() {
    const chart = readChartData("predictionChart");
    if (!chart) {
        return;
    }
    new Chart(chart.canvas, {
        type: "bar",
        data: {
            labels: chart.data.map((item) => item.predicted_job_title),
            datasets: [{
                label: "Predictions",
                data: chart.data.map((item) => item.count),
                backgroundColor: "#246bfe",
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
        },
    });
}

function renderActivityChart() {
    const chart = readChartData("activityChart");
    if (!chart) {
        return;
    }
    new Chart(chart.canvas, {
        type: "doughnut",
        data: {
            labels: chart.data.map((item) => item.username),
            datasets: [{
                data: chart.data.map((item) => item.count),
                backgroundColor: ["#246bfe", "#00a79d", "#f59f00", "#845ef7", "#e03131", "#12b886", "#228be6", "#495057"],
            }],
        },
        options: {
            responsive: true,
            plugins: { legend: { position: "bottom" } },
        },
    });
}
