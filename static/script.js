// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const refreshBtn = document.getElementById('refresh-btn');
    const activeGrid = document.getElementById('active-grid');
    const eliminatedGrid = document.getElementById('eliminated-grid');
    const activeCount = document.getElementById('active-count');
    const eliminatedCount = document.getElementById('eliminated-count');
    const leaderDriver = document.getElementById('leader-driver');
    const yearTitle = document.getElementById('year-title');
    const sortSelect = document.getElementById('sort-select');

    let currentData = null;

    // Load initial data
    loadData();

    refreshBtn.addEventListener('click', loadData);

    sortSelect.addEventListener('change', function() {
        if (currentData) {
            renderData(currentData);
        }
    });

    async function loadData() {
        try {
            refreshBtn.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Loading...';
            refreshBtn.disabled = true;

            const response = await fetch('/api/predict');
            const data = await response.json();
            
            currentData = data;
            renderData(data);
            
        } catch (error) {
            console.error('Error loading data:', error);
            activeGrid.innerHTML = '<div class="loading"><i class="fas fa-exclamation-triangle"></i> Error loading data</div>';
        } finally {
            refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Predictions';
            refreshBtn.disabled = false;
        }
    }

    function renderData(data) {
        yearTitle.textContent = `F1 Champion Predictor ${data.current_year}`;
        activeCount.textContent = data.total_active;
        eliminatedCount.textContent = data.total_eliminated;
        leaderDriver.textContent = `${data.leader.driver} (${data.leader.points} pts)`;

        // Sort active drivers based on selection
        let sortedActive = [...data.active_drivers];
        switch(sortSelect.value) {
            case 'points':
                sortedActive.sort((a, b) => b.points - a.points);
                break;
            case 'wins':
                sortedActive.sort((a, b) => b.wins - a.wins);
                break;
            default:
                sortedActive.sort((a, b) => b.probability - a.probability);
        }

        // Render active drivers
        activeGrid.innerHTML = sortedActive.map(driver => createDriverCard(driver, false)).join('');

        // Render eliminated drivers (sorted by points descending)
        eliminatedGrid.innerHTML = data.eliminated_drivers.map(driver => createDriverCard(driver, true)).join('');
    }

    function createDriverCard(driver, isEliminated) {
        const probabilityPercent = driver.probability;
        const cardClass = isEliminated ? 'driver-card eliminated' : 'driver-card';
        
        return `
            <div class="${cardClass}">
                <div class="driver-info">
                    <div class="driver-name">${driver.driver}</div>
                    <div class="driver-probability">${probabilityPercent}%</div>
                </div>
                
                <div class="driver-stats">
                    <div class="stat-item">
                        <span class="stat-label">Points</span>
                        <span class="stat-value">${driver.points}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Pos</span>
                        <span class="stat-value">${driver.avg_pos}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Wins</span>
                        <span class="stat-value">${driver.wins}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Podiums</span>
                        <span class="stat-value">${driver.podiums}</span>
                    </div>
                </div>
                
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probabilityPercent}%"></div>
                </div>
            </div>
        `;
    }
});