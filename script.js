document.addEventListener('DOMContentLoaded', function() {
    const predictBtn = document.getElementById('predict-btn');
    const activeGrid = document.getElementById('active-grid');
    const eliminatedGrid = document.getElementById('eliminated-grid');
    const activeCount = document.getElementById('active-count');
    const eliminatedCount = document.getElementById('eliminated-count');
    const leaderDriver = document.getElementById('leader-driver');
    const sortSelect = document.getElementById('sort-select');
    const yearTitle = document.getElementById('year-title');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const outputContent = document.getElementById('output-content');

    let currentData = null;
    let progressInterval = null;

    predictBtn.addEventListener('click', startSimulation);

    sortSelect.addEventListener('change', function() {
        if (currentData) {
            renderData(currentData);
        }
    });

    async function startSimulation() {
        try {
            // Reset UI
            outputContent.textContent = '';
            progressFill.style.width = '0%';
            progressText.textContent = '0%';
            activeGrid.innerHTML = '';
            eliminatedGrid.innerHTML = '';
            
            // Start the simulation
            const response = await fetch('/api/start_simulation');
            const data = await response.json();
            
            // Start polling for progress
            if (progressInterval) {
                clearInterval(progressInterval);
            }
            
            progressInterval = setInterval(async () => {
                try {
                    // Get progress
                    const progressResponse = await fetch('/api/simulation_progress');
                    const progressData = await progressResponse.json();
                    
                    progressFill.style.width = `${progressData.progress}%`;
                    progressText.textContent = `${Math.round(progressData.progress)}%`;
                    
                    // Get output
                    const outputResponse = await fetch('/api/simulation_output');
                    const outputData = await outputResponse.json();
                    
                    // Update output display
                    if (outputData.output && outputData.output.length > 0) {
                        outputContent.textContent = outputData.output.join('\n');
                        outputContent.scrollTop = outputContent.scrollHeight;
                    }
                    
                    // Check if simulation is complete
                    if (!progressData.running) {
                        clearInterval(progressInterval);
                        
                        // Get final result
                        const resultResponse = await fetch('/api/simulation_result');
                        const resultData = await resultResponse.json();
                        
                        if (resultData.status !== "waiting") {
                            currentData = resultData;
                            renderData(resultData);
                        }
                    }
                } catch (error) {
                    console.error('Error getting progress:', error);
                }
            }, 500);
            
        } catch (error) {
            console.error('Error starting simulation:', error);
            outputContent.textContent = `Error: ${error.message}`;
        }
    }

    function renderData(data) {
        yearTitle.textContent = data.current_year;
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
        const sortedEliminated = [...data.eliminated_drivers].sort((a, b) => b.points - a.points);
        eliminatedGrid.innerHTML = sortedEliminated.map(driver => createDriverCard(driver, true)).join('');
    }

    function createDriverCard(driver, isEliminated) {
        const probabilityPercent = driver.probability;
        const cardClass = isEliminated ? 'driver-card eliminated' : 'driver-card';
        
        // Driver images based on driver code
        const driverImages = {
            'VER': '🏎️',
            'PER': '🏎️',
            'HAM': '🏎️',
            'RUS': '🏎️',
            'LEC': '🏎️',
            'SAI': '🏎️',
            'NOR': '🏎️',
            'PIA': '🏎️',
            'ALO': '🏎️',
            'STR': '🏎️',
            'OCO': '🏎️',
            'GAS': '🏎️',
            'BOT': '🏎️',
            'ZHO': '🏎️',
            'MAG': '🏎️',
            'HUL': '🏎️',
            'TSU': '🏎️',
            'DEV': '🏎️',
            'ALB': '🏎️',
            'SAR': '🏎️'
        };

        return `
            <div class="${cardClass}">
                <div class="driver-image">${driverImages[driver.driver] || '🏎️'}</div>
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
                        <span class="stat-value">${driver.avg_pos || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Wins</span>
                        <span class="stat-value">${driver.wins || 0}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Podiums</span>
                        <span class="stat-value">${driver.podiums || 0}</span>
                    </div>
                </div>
                
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${probabilityPercent}%"></div>
                </div>
            </div>
        `;
    }
});
