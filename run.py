<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deriv API Market Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/moment@2.29.1/moment.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-moment"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --secondary: #0f172a;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #1e293b;
            --light: #f8fafc;
            --gray: #94a3b8;
            --card-bg: rgba(15, 23, 42, 0.7);
            --border: rgba(255, 255, 255, 0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: var(--light);
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .logo i {
            font-size: 28px;
            color: var(--primary);
        }
        
        .logo h1 {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(to right, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .currency-selector {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 15px;
            color: var(--light);
            font-weight: 500;
            min-width: 150px;
        }
        
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
        }
        
        .btn-outline {
            background: transparent;
            border: 1px solid var(--primary);
            color: var(--primary);
        }
        
        .btn-outline:hover {
            background: rgba(37, 99, 235, 0.1);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .main-content {
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 20px;
        }
        
        .chart-container {
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 20px;
            height: 500px;
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .chart-title {
            font-size: 18px;
            font-weight: 600;
        }
        
        .timeframe-selector {
            display: flex;
            gap: 10px;
        }
        
        .time-btn {
            background: transparent;
            color: var(--gray);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .time-btn.active {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .indicators-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .indicator-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.3s ease;
        }
        
        .indicator-card:hover {
            transform: translateY(-5px);
            border-color: rgba(59, 130, 246, 0.5);
        }
        
        .indicator-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .indicator-title {
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .indicator-value {
            font-size: 24px;
            font-weight: 700;
        }
        
        .trend-up {
            color: var(--success);
        }
        
        .trend-down {
            color: var(--danger);
        }
        
        .indicator-chart {
            height: 100px;
            margin-top: 15px;
        }
        
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .market-overview {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        
        .overview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .price-display {
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0;
            background: linear-gradient(to right, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        
        .price-change {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            font-weight: 600;
        }
        
        .change-up {
            color: var(--success);
        }
        
        .change-down {
            color: var(--danger);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 20px;
            font-weight: 700;
            margin: 5px 0;
        }
        
        .stat-label {
            color: var(--gray);
            font-size: 14px;
        }
        
        .prediction-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        
        .prediction-header {
            margin-bottom: 15px;
        }
        
        .prediction-content {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .prediction-item:last-child {
            border-bottom: none;
        }
        
        .prediction-strength {
            display: flex;
            gap: 5px;
        }
        
        .strength-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--gray);
        }
        
        .strength-dot.active {
            background: var(--success);
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: var(--gray);
            font-size: 14px;
            border-top: 1px solid var(--border);
            margin-top: 20px;
        }
        
        .indicator-icon {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(37, 99, 235, 0.2);
            color: var(--primary);
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 15px;
                text-align: center;
            }
            
            .controls {
                width: 100%;
                justify-content: center;
            }
            
            .indicators-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <i class="fas fa-chart-line"></i>
                <h1>Deriv Market Analytics</h1>
            </div>
            <div class="controls">
                <select class="currency-selector" id="currencySelector">
                    <option value="EUR/USD">EUR/USD</option>
                    <option value="GBP/USD">GBP/USD</option>
                    <option value="USD/JPY" selected>USD/JPY</option>
                    <option value="AUD/USD">AUD/USD</option>
                    <option value="USD/CAD">USD/CAD</option>
                    <option value="BTC/USD">BTC/USD</option>
                    <option value="ETH/USD">ETH/USD</option>
                </select>
                <button class="btn" id="connectBtn">
                    <i class="fas fa-plug"></i> Connect to API
                </button>
                <button class="btn btn-outline">
                    <i class="fas fa-cog"></i> Settings
                </button>
            </div>
        </header>
        
        <div class="dashboard">
            <div class="main-content">
                <div class="chart-container">
                    <div class="chart-header">
                        <div class="chart-title">Market Price - USD/JPY</div>
                        <div class="timeframe-selector">
                            <button class="time-btn active">1H</button>
                            <button class="time-btn">4H</button>
                            <button class="time-btn">1D</button>
                            <button class="time-btn">1W</button>
                            <button class="time-btn">1M</button>
                        </div>
                    </div>
                    <canvas id="priceChart"></canvas>
                </div>
                
                <div class="indicators-grid">
                    <div class="indicator-card">
                        <div class="indicator-header">
                            <div class="indicator-title">
                                <div class="indicator-icon">
                                    <i class="fas fa-wave-square"></i>
                                </div>
                                <span>RSI (14)</span>
                            </div>
                            <div class="indicator-value trend-down">42.36</div>
                        </div>
                        <div class="indicator-chart">
                            <canvas id="rsiChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="indicator-card">
                        <div class="indicator-header">
                            <div class="indicator-title">
                                <div class="indicator-icon">
                                    <i class="fas fa-compress-alt"></i>
                                </div>
                                <span>MACD (12,26,9)</span>
                            </div>
                            <div class="indicator-value trend-up">1.25</div>
                        </div>
                        <div class="indicator-chart">
                            <canvas id="macdChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="indicator-card">
                        <div class="indicator-header">
                            <div class="indicator-title">
                                <div class="indicator-icon">
                                    <i class="fas fa-chart-area"></i>
                                </div>
                                <span>Bollinger Bands (20,2)</span>
                            </div>
                            <div class="indicator-value">147.82</div>
                        </div>
                        <div class="indicator-chart">
                            <canvas id="bollingerChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="indicator-card">
                        <div class="indicator-header">
                            <div class="indicator-title">
                                <div class="indicator-icon">
                                    <i class="fas fa-bolt"></i>
                                </div>
                                <span>Stochastic (14,3,3)</span>
                            </div>
                            <div class="indicator-value trend-up">68.74</div>
                        </div>
                        <div class="indicator-chart">
                            <canvas id="stochasticChart"></canvas>
                        </div>
                    </div>
                    
                    <div class="indicator-card">
                        <div class="indicator-header">
                            <div class="indicator-title">
                                <div class="indicator-icon">
                                    <i class="fas fa-weight-hanging"></i>
                                </div>
                                <span>Volume</span>
                            </div>
                            <div class="indicator-value">1.24M</div>
                        </div>
                        <div class="indicator-chart">
                            <canvas id="volumeChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="market-overview">
                    <div class="overview-header">
                        <h2>Market Overview</h2>
                        <div class="price-change change-up">
                            <i class="fas fa-arrow-up"></i>
                            <span>+0.32%</span>
                        </div>
                    </div>
                    <div class="price-display">148.26</div>
                    <div>USD/JPY • Forex</div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">24h High</div>
                            <div class="stat-value">148.92</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">24h Low</div>
                            <div class="stat-value">147.85</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">24h Volume</div>
                            <div class="stat-value">$1.24B</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Market Sentiment</div>
                            <div class="stat-value trend-up">Bullish</div>
                        </div>
                    </div>
                </div>
                
                <div class="prediction-card">
                    <div class="prediction-header">
                        <h2>Indicator Predictions</h2>
                    </div>
                    <div class="prediction-content">
                        <div class="prediction-item">
                            <div>RSI (14)</div>
                            <div class="trend-down">Oversold</div>
                            <div class="prediction-strength">
                                <div class="strength-dot active"></div>
                                <div class="strength-dot active"></div>
                                <div class="strength-dot"></div>
                            </div>
                        </div>
                        <div class="prediction-item">
                            <div>MACD</div>
                            <div class="trend-up">Bullish</div>
                            <div class="prediction-strength">
                                <div class="strength-dot active"></div>
                                <div class="strength-dot active"></div>
                                <div class="strength-dot active"></div>
                            </div>
                        </div>
                        <div class="prediction-item">
                            <div>Bollinger Bands</div>
                            <div>Neutral</div>
                            <div class="prediction-strength">
                                <div class="strength-dot active"></div>
                                <div class="strength-dot"></div>
                                <div class="strength-dot"></div>
                            </div>
                        </div>
                        <div class="prediction-item">
                            <div>Stochastic</div>
                            <div class="trend-up">Bullish</div>
                            <div class="prediction-strength">
                                <div class="strength-dot active"></div>
                                <div class="strength-dot active"></div>
                                <div class="strength-dot"></div>
                            </div>
                        </div>
                        <div class="prediction-item">
                            <div>Volume</div>
                            <div class="trend-down">Decreasing</div>
                            <div class="prediction-strength">
                                <div class="strength-dot active"></div>
                                <div class="strength-dot"></div>
                                <div class="strength-dot"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Deriv Market Analytics Dashboard • Data provided by Deriv API • Real-time market analysis</p>
            <p>This is a demo interface. Connect to Deriv API for live data.</p>
        </footer>

        <div id="settingsModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.7); z-index:1000; align-items:center; justify-content:center;">
            <div style="background:var(--card-bg); border-radius:16px; padding:32px 24px; min-width:320px; max-width:90vw; box-shadow:0 8px 32px rgba(0,0,0,0.3); border:1px solid var(--border); position:relative;">
                <button id="closeSettings" style="position:absolute; top:12px; right:12px; background:none; border:none; color:var(--gray); font-size:20px; cursor:pointer;"><i class="fas fa-times"></i></button>
                <h2 style="margin-bottom:18px;">Settings</h2>
                <label for="derivApiToken" style="font-weight:500;">Deriv API Token</label>
                <input id="derivApiToken" type="text" placeholder="Paste your Deriv API token here" style="width:100%; margin:10px 0 18px 0; padding:10px; border-radius:8px; border:1px solid var(--border); background:var(--dark); color:var(--light);">
                <button id="saveSettingsBtn" class="btn" style="width:100%;">Save</button>
                <div id="settingsSavedMsg" style="color:var(--success); margin-top:10px; display:none;">Saved!</div>
            </div>
        </div>
    </div>

    <script>
        // Fetch data from Flask API and update charts
        async function fetchMarketData() {
            const response = await fetch('http://127.0.0.1:5000/api/market_data');
            return await response.json();
        }

        function updateCharts(data) {
            // Update all charts with real data from API
            // Price Chart
            const priceCtx = document.getElementById('priceChart').getContext('2d');
            if (window.priceChartInstance) window.priceChartInstance.destroy();
            window.priceChartInstance = new Chart(priceCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps,
                    datasets: [{
                        label: 'USD/JPY',
                        data: data.prices,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { type: 'time', time: { unit: 'minute' }, grid: { display: false } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // RSI Chart
            const rsiCtx = document.getElementById('rsiChart').getContext('2d');
            if (window.rsiChartInstance) window.rsiChartInstance.destroy();
            window.rsiChartInstance = new Chart(rsiCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps,
                    datasets: [{
                        data: data.rsi,
                        borderColor: '#22c55e',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: { display: true, min: 0, max: 100, grid: { display: false }, ticks: { display: false } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // MACD Chart
            const macdCtx = document.getElementById('macdChart').getContext('2d');
            if (window.macdChartInstance) window.macdChartInstance.destroy();
            window.macdChartInstance = new Chart(macdCtx, {
                type: 'bar',
                data: {
                    labels: data.timestamps,
                    datasets: [{
                        data: data.macd,
                        backgroundColor: data.macd.map(val => val >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)'),
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { x: { display: false }, y: { display: false } },
                    plugins: { legend: { display: false } }
                }
            });

            // Bollinger Bands Chart
            const bollingerCtx = document.getElementById('bollingerChart').getContext('2d');
            if (window.bollingerChartInstance) window.bollingerChartInstance.destroy();
            window.bollingerChartInstance = new Chart(bollingerCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps,
                    datasets: [
                        { data: data.bollinger_upper, borderColor: 'rgba(239, 68, 68, 0.7)', borderWidth: 1, pointRadius: 0 },
                        { data: data.bollinger_middle, borderColor: '#94a3b8', borderWidth: 1, pointRadius: 0 },
                        { data: data.bollinger_lower, borderColor: 'rgba(34, 197, 94, 0.7)', borderWidth: 1, pointRadius: 0 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { x: { display: false }, y: { display: false } },
                    plugins: { legend: { display: false } }
                }
            });

            // Volume Chart
            const volumeCtx = document.getElementById('volumeChart').getContext('2d');
            if (window.volumeChartInstance) window.volumeChartInstance.destroy();
            window.volumeChartInstance = new Chart(volumeCtx, {
                type: 'bar',
                data: {
                    labels: data.timestamps,
                    datasets: [{
                        data: data.volumes,
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { x: { display: false }, y: { display: false } },
                    plugins: { legend: { display: false } }
                }
            });

            // Stochastic Chart
            const stochasticCtx = document.getElementById('stochasticChart').getContext('2d');
            if (window.stochasticChartInstance) window.stochasticChartInstance.destroy();
            window.stochasticChartInstance = new Chart(stochasticCtx, {
                type: 'line',
                data: {
                    labels: data.timestamps,
                    datasets: [{
                        data: data.stochastic,
                        borderColor: '#f59e0b',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: { x: { display: false }, y: { min: 0, max: 100, display: false } },
                    plugins: { legend: { display: false } }
                }
            });
        }

        document.addEventListener('DOMContentLoaded', async function() {
            // Add connection status indicator
            let connectionStatus = document.createElement('span');
            connectionStatus.id = 'connectionStatus';
            connectionStatus.style.marginLeft = '12px';
            connectionStatus.style.fontWeight = 'bold';
            connectionStatus.style.fontSize = '15px';
            connectionStatus.style.verticalAlign = 'middle';
            document.getElementById('connectBtn').parentNode.appendChild(connectionStatus);

            function setStatus(text, color) {
                connectionStatus.textContent = text;
                connectionStatus.style.color = color;
            }

            // Poll the Python backend for real market data every 2 seconds
            let polling = true;
            async function pollMarketData() {
                if (!polling) return;
                try {
                    const data = await fetchMarketData();
                    updateCharts(data);
                    document.querySelector('.price-display').textContent = data.prices.length ? data.prices[data.prices.length-1].toFixed(2) : '--';
                } catch (e) {
                    setStatus('No Data', '#ef4444');
                }
                setTimeout(pollMarketData, 2000);
            }
            pollMarketData();

            // Connect button functionality
            document.getElementById('connectBtn').addEventListener('click', function() {
                derivApiTokenInput.value = localStorage.getItem('derivApiToken') || '';
                settingsModal.style.display = 'flex';
                settingsSavedMsg.style.display = 'none';
                saveSettingsBtn.onclick = async function() {
                    const token = derivApiTokenInput.value.trim();
                    localStorage.setItem('derivApiToken', token);
                    settingsSavedMsg.style.display = 'block';
                    setStatus('Connecting...', '#f59e0b');
                    // POST token to backend to start Deriv connection
                    try {
                        const resp = await fetch('http://127.0.0.1:5000/api/connect', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ token })
                        });
                        const result = await resp.json();
                        if (result.status === 'started') {
                            setStatus('Connected', '#22c55e');
                            document.getElementById('connectBtn').innerHTML = '<i class="fas fa-check"></i> Connected';
                            document.getElementById('connectBtn').style.background = 'var(--success)';
                            setTimeout(()=>{ settingsModal.style.display = 'none'; }, 1000);
                        } else {
                            setStatus('Failed', '#ef4444');
                        }
                    } catch (e) {
                        setStatus('Failed', '#ef4444');
                        alert('Failed to connect to backend.');
                    }
                };
            });
        });
        
        // Settings modal logic
        const settingsModal = document.getElementById('settingsModal');
        const settingsBtn = document.querySelector('.btn-outline');
        const closeSettings = document.getElementById('closeSettings');
        const saveSettingsBtn = document.getElementById('saveSettingsBtn');
        const derivApiTokenInput = document.getElementById('derivApiToken');
        const settingsSavedMsg = document.getElementById('settingsSavedMsg');

        // Open modal
        settingsBtn.addEventListener('click', function() {
            derivApiTokenInput.value = localStorage.getItem('derivApiToken') || '';
            settingsModal.style.display = 'flex';
            settingsSavedMsg.style.display = 'none';
        });
        // Close modal
        closeSettings.addEventListener('click', function() {
            settingsModal.style.display = 'none';
        });
        // Save token
        saveSettingsBtn.addEventListener('click', function() {
            localStorage.setItem('derivApiToken', derivApiTokenInput.value.trim());
            settingsSavedMsg.style.display = 'block';
            setTimeout(()=>{ settingsModal.style.display = 'none'; }, 1000);
        });
        // Utility to get token
        function getDerivApiToken() {
            return localStorage.getItem('derivApiToken') || '';
        }
    </script>
</body>
</html>
