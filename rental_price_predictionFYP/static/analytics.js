let charts = {}; // central chart registry

/* ---------------- LOAD FILTER OPTIONS ---------------- */
async function loadFilters() {
    const res = await fetch("/api/options");
    const data = await res.json();

    populateSelect("regionFilter", data.regions);
    populateSelect("propertyTypeFilter", data.property_types);

    document.getElementById("regionFilter").addEventListener("change", () => {
        updateCities(data.cities_by_region);
        loadAnalytics();
    });

    document.getElementById("cityFilter").addEventListener("change", loadAnalytics);
    document.getElementById("propertyTypeFilter").addEventListener("change", loadAnalytics);
}

function populateSelect(id, values) {
    const select = document.getElementById(id);
    values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
    });
}

function updateCities(citiesByRegion) {
    const region = document.getElementById("regionFilter").value;
    const citySelect = document.getElementById("cityFilter");

    citySelect.innerHTML = `<option value="">All Cities</option>`;

    if (region && citiesByRegion[region]) {
        citiesByRegion[region].forEach(c => {
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = c;
            citySelect.appendChild(opt);
        });
    }
}

/* ---------------- LOAD ANALYTICS ---------------- */
async function loadAnalytics() {
    const params = new URLSearchParams({
        region: document.getElementById("regionFilter").value,
        city: document.getElementById("cityFilter").value,
        property_type: document.getElementById("propertyTypeFilter").value
    });

    const res = await fetch(`/api/analytics?${params}`);
    const data = await res.json();

    renderKPIs(data.kpis);
    renderCharts(data);
}

/* ---------------- KPI RENDER ---------------- */
function renderKPIs(kpis) {
    const row = document.getElementById("kpiRow");

    const items = [
        ["Avg Rent", `RM ${kpis.avg_rent}`],
        ["Highest Rent", `RM ${kpis.max_rent}`],
        ["Median Rent", `RM ${kpis.median_rent}`],
        ["Avg Size (sqft)", kpis.avg_size],
        ["Total Listings", kpis.total_listings],
        ["Avg RM / sqft", kpis.avg_psf]
    ];

    row.innerHTML = items.map(i => `
        <div class="col-md-2">
            <div class="kpi-card">
                <h6>${i[0]}</h6>
                <h4>${i[1]}</h4>
            </div>
        </div>
    `).join("");
}

/* ---------------- CHART RENDER ---------------- */
function renderCharts(data) {
    createBarChart(
        "rentByRegionChart",
        data.avg_rent_by_region,
        "Average Rent (RM)",
        "#4fb3ad"
    );

    createBarChart(
        "rentByPropertyChart",
        data.avg_rent_by_property_type,
        "Average Rent (RM)",
        "#6c8cff"
    );

    createBarChart(
        "topCitiesChart",
        data.top_cities,
        "Average Rent (RM)",
        "#ffa94d",
        true
    );
}

function createBarChart(canvasId, data, label, color, horizontal=false) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Destroy existing chart
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(data),
            datasets: [{
                label,
                data: Object.values(data),
                backgroundColor: color,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: horizontal ? "y" : "x",
            responsive: true,
            animation: {
                duration: 900,
                easing: "easeOutQuart"
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

/* ---------------- INIT ---------------- */
document.addEventListener("DOMContentLoaded", () => {
    loadFilters();
    loadAnalytics();
});
