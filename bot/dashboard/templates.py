DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>V4 Bot Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --border: rgba(255,255,255,0.07);
      --text: #e2e8f0;
      --muted: #8b949e;
      --cyan: #22d3ee;
      --green: #34d399;
      --red: #f87171;
      --yellow: #facc15;
      --orange: #fb923c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      min-height: 100vh;
    }

    /* ── Header ─────────────────────────────────────────────── */
    #header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 24px;
      flex-wrap: wrap;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
    .brand em { color: var(--cyan); font-style: normal; }
    .stat { display: flex; flex-direction: column; gap: 2px; }
    .stat .lbl { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px; }
    .stat .val { font-size: 15px; font-weight: 600; }
    .badges { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .badge {
      font-size: 10px; font-weight: 600;
      padding: 3px 10px; border-radius: 20px;
      border: 1px solid transparent;
    }
    .badge-paper    { background: rgba(34,211,238,0.10);  color: var(--cyan);   border-color: rgba(34,211,238,0.20); }
    .badge-live     { background: rgba(250,204,21,0.10);   color: var(--yellow); border-color: rgba(250,204,21,0.20); }
    .badge-up       { background: rgba(52,211,153,0.10);   color: var(--green);  border-color: rgba(52,211,153,0.20); }
    .badge-down     { background: rgba(248,113,113,0.10);  color: var(--red);    border-color: rgba(248,113,113,0.20); }
    .badge-active   { background: rgba(52,211,153,0.10);   color: var(--green);  border-color: rgba(52,211,153,0.20); }
    .badge-halted   { background: rgba(248,113,113,0.10);  color: var(--red);    border-color: rgba(248,113,113,0.20); }
    .badge-short-on { background: rgba(251,146,60,0.12);   color: var(--orange); border-color: rgba(251,146,60,0.25); }
    .badge-short-ok { background: rgba(52,211,153,0.10);   color: var(--green);  border-color: rgba(52,211,153,0.20); }
    .badge-short-blocked { background: rgba(250,204,21,0.10); color: var(--yellow); border-color: rgba(250,204,21,0.20); }
    .badge-short-off{ background: rgba(255,255,255,0.05);  color: var(--muted);  border-color: rgba(255,255,255,0.10); }
    .badge-strategy { background: rgba(34,211,238,0.07);   color: var(--cyan);   border-color: rgba(34,211,238,0.15); }

    /* ── Body layout ────────────────────────────────────────── */
    #body {
      display: grid;
      grid-template-columns: 1fr 240px;
      gap: 16px;
      padding: 20px 24px;
      max-width: 1400px;
    }
    .main-col, .side-col { display: flex; flex-direction: column; gap: 16px; }

    /* ── Panels ─────────────────────────────────────────────── */
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .panel-title {
      color: var(--muted);
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 12px;
    }
    .panel-subtitle {
      color: var(--muted);
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      margin: 12px 0 8px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    /* ── Tables ─────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; }
    th {
      color: var(--muted); font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px;
      padding: 4px 10px; text-align: left;
      border-bottom: 1px solid var(--border);
    }
    td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 12px; }
    tr:last-child td { border-bottom: none; }
    .g { color: var(--green); font-weight: 600; }
    .r { color: var(--red);   font-weight: 600; }
    .o { color: var(--orange); font-weight: 600; }
    .m { color: var(--muted); }
    .empty-msg { color: var(--muted); font-size: 11px; text-align: center; padding: 20px 0; }

    /* Direction pill in position table */
    .dir-long  { background: rgba(52,211,153,0.12); color: var(--green);  padding: 1px 7px; border-radius: 10px; font-size: 10px; font-weight: 700; }
    .dir-short { background: rgba(251,146,60,0.12);  color: var(--orange); padding: 1px 7px; border-radius: 10px; font-size: 10px; font-weight: 700; }

    /* ── Sidebar KV pairs ───────────────────────────────────── */
    .kv { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .kv:last-child { margin-bottom: 0; }
    .k { color: var(--muted); font-size: 11px; }
    .v { font-size: 12px; font-weight: 600; }

    /* ── Heat progress bars ─────────────────────────────────── */
    .progress-wrap { background: rgba(255,255,255,0.06); border-radius: 4px; height: 4px; margin: -2px 0 10px; }
    .progress-fill { border-radius: 4px; height: 4px; background: linear-gradient(90deg, var(--cyan), var(--green)); transition: width 0.5s; }
    .progress-fill-short { border-radius: 4px; height: 4px; background: linear-gradient(90deg, var(--orange), var(--yellow)); transition: width 0.5s; }

    /* ── Chart ──────────────────────────────────────────────── */
    #pnl-chart-wrap { height: 130px; position: relative; }
  </style>
</head>
<body>

<div id="header">
  <span class="brand">V4 <em>Bot</em></span>
  <div class="stat"><div class="lbl">Equity</div><div class="val" id="h-equity">—</div></div>
  <div class="stat"><div class="lbl">Cash</div><div class="val" id="h-cash">—</div></div>
  <div class="stat"><div class="lbl">Day P&amp;L</div><div class="val" id="h-pnl">—</div></div>
  <div class="badges">
    <span class="badge badge-paper"    id="b-env">PAPER</span>
    <span class="badge badge-strategy" id="b-strategy">GAP-HOLD</span>
    <span class="badge badge-up"       id="b-regime">&#8593; UPTREND</span>
    <span class="badge badge-short-off" id="b-shorts">SHORTS OFF</span>
    <span class="badge badge-active"   id="b-kill">&#10003; ACTIVE</span>
  </div>
</div>

<div id="body">
  <div class="main-col">

    <div class="panel">
      <div class="panel-title">Intraday P&amp;L</div>
      <div id="pnl-chart-wrap"><canvas id="pnl-chart"></canvas></div>
    </div>

    <div class="panel">
      <div class="panel-title">Open Positions (<span id="pos-count">0</span>)</div>
      <table>
        <thead>
          <tr>
            <th>Ticker</th><th>Dir</th><th>Shares</th><th>Entry</th><th>Stop</th>
            <th>Target</th><th>Last</th><th>Unreal. P&amp;L</th><th>Risk</th><th>Time In</th>
          </tr>
        </thead>
        <tbody id="positions-body">
          <tr><td colspan="10" class="empty-msg">No open positions</td></tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-title">Closed Trades Today (<span id="trades-count">0</span>)</div>
      <table>
        <thead>
          <tr>
            <th>Time</th><th>Ticker</th><th>Dir</th><th>Entry</th>
            <th>Exit</th><th>Shares</th><th>P&amp;L</th><th>Reason</th>
          </tr>
        </thead>
        <tbody id="trades-body">
          <tr><td colspan="8" class="empty-msg">No closed trades</td></tr>
        </tbody>
      </table>
    </div>

  </div>
  <div class="side-col">

    <div class="panel">
      <div class="panel-title">Portfolio Risk</div>

      <div class="kv" style="margin-bottom:4px">
        <span class="k">Long Heat</span><span class="v" id="s-heat">—</span>
      </div>
      <div class="progress-wrap"><div class="progress-fill" id="heat-bar" style="width:0%"></div></div>

      <div id="short-heat-row" style="display:none">
        <div class="kv" style="margin-bottom:4px">
          <span class="k">Short Heat</span><span class="v" id="s-short-heat">—</span>
        </div>
        <div class="progress-wrap"><div class="progress-fill-short" id="short-heat-bar" style="width:0%"></div></div>
      </div>

      <div class="kv"><span class="k">Positions</span><span class="v" id="s-pos">—</span></div>
      <div class="kv"><span class="k">Consec. Losses</span><span class="v" id="s-losses">—</span></div>
      <div class="kv"><span class="k">Cooldown</span><span class="v m" id="s-cooldown">—</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Config</div>
      <div id="config-body"><span class="m" style="font-size:11px">Loading...</span></div>
    </div>

  </div>
</div>

<script>
// ── Chart setup ──────────────────────────────────────────────
const ctx = document.getElementById('pnl-chart').getContext('2d');
const pnlChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Cumulative P&L',
      data: [],
      borderColor: '#22d3ee',
      backgroundColor: 'rgba(34,211,238,0.06)',
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: '#22d3ee',
      fill: true,
      tension: 0.3,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: function(c) { return '$' + c.parsed.y.toFixed(2); } }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { size: 10 }, callback: function(v) { return '$' + v.toFixed(0); } }
      }
    }
  }
});

// ── Helpers ──────────────────────────────────────────────────
function fmt2(n) {
  if (n == null) return '—';
  return Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
}
function timeFmt(iso) { return iso ? iso.substring(11, 16) : '—'; }
function timeIn(iso) {
  var mins = Math.floor((Date.now() - new Date(iso)) / 60000);
  return mins < 60 ? mins + 'm' : Math.floor(mins / 60) + 'h ' + (mins % 60) + 'm';
}
function signOf(n) { return n >= 0 ? '+' : '-'; }
function dirPill(d) {
  return d === 'short'
    ? '<span class="dir-short">SHORT</span>'
    : '<span class="dir-long">LONG</span>';
}

// ── Updaters ─────────────────────────────────────────────────
function updateHeader(d) {
  // equity tracks realized PnL; add only unrealized to get current mark-to-market
  var currentEquity = d.equity + (d.unrealized_pnl || 0);
  document.getElementById('h-equity').textContent = '$' + fmt2(currentEquity);
  document.getElementById('h-cash').textContent = '$' + fmt2(d.cash);

  var pnlEl = document.getElementById('h-pnl');
  pnlEl.textContent = signOf(d.day_pnl) + '$' + fmt2(d.day_pnl) + ' (' + signOf(d.day_pnl_pct) + (Math.abs(d.day_pnl_pct) * 100).toFixed(2) + '%)';
  pnlEl.className = 'val ' + (d.day_pnl >= 0 ? 'g' : 'r');

  var bEnv = document.getElementById('b-env');
  bEnv.textContent = d.is_paper ? 'PAPER' : 'LIVE';
  bEnv.className = 'badge ' + (d.is_paper ? 'badge-paper' : 'badge-live');

  var bStrat = document.getElementById('b-strategy');
  var stratName = d.long_strategy_name === 'gap_hold' ? 'GAP-HOLD' : 'STANDARD';
  bStrat.textContent = stratName;

  var bReg = document.getElementById('b-regime');
  bReg.innerHTML = d.regime_uptrend ? '&#8593; UPTREND' : '&#8595; LONGS BLOCKED';
  bReg.className = 'badge ' + (d.regime_uptrend ? 'badge-up' : 'badge-down');

  var bShorts = document.getElementById('b-shorts');
  if (!d.short_enabled) {
    bShorts.textContent = 'SHORTS OFF';
    bShorts.className = 'badge badge-short-off';
  } else if (d.short_allowed) {
    bShorts.innerHTML = '&#8595; SHORTS OK';
    bShorts.className = 'badge badge-short-ok';
  } else {
    bShorts.innerHTML = '&#8856; SHORTS BLOCKED';
    bShorts.className = 'badge badge-short-blocked';
  }

  var bKill = document.getElementById('b-kill');
  bKill.innerHTML = d.kill_switch_active ? '&#10007; HALTED' : '&#10003; ACTIVE';
  bKill.className = 'badge ' + (d.kill_switch_active ? 'badge-halted' : 'badge-active');
}

function updatePositions(d) {
  document.getElementById('pos-count').textContent = d.open_positions_count;
  var tbody = document.getElementById('positions-body');
  if (!d.positions.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty-msg">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = d.positions.map(function(p) {
    var isShort = p.direction === 'short';
    var pnlCls = isShort
      ? (p.unrealized_pnl >= 0 ? 'g' : 'r')
      : (p.unrealized_pnl >= 0 ? 'g' : 'r');
    return '<tr>' +
      '<td><strong>' + p.ticker + '</strong></td>' +
      '<td>' + dirPill(p.direction) + '</td>' +
      '<td>' + p.shares + '</td>' +
      '<td>$' + fmt2(p.entry_price) + '</td>' +
      '<td>$' + fmt2(p.stop_price) + '</td>' +
      '<td>$' + fmt2(p.target_price) + '</td>' +
      '<td>$' + fmt2(p.last_price) + '</td>' +
      '<td class="' + pnlCls + '">' + signOf(p.unrealized_pnl) + '$' + fmt2(p.unrealized_pnl) + '</td>' +
      '<td class="m">$' + fmt2(p.open_risk) + '</td>' +
      '<td class="m">' + timeIn(p.entry_time) + '</td>' +
      '</tr>';
  }).join('');
}

function updateTrades(d) {
  document.getElementById('trades-count').textContent = d.closed_trades.length;
  var tbody = document.getElementById('trades-body');
  if (!d.closed_trades.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">No closed trades</td></tr>';
    return;
  }
  tbody.innerHTML = d.closed_trades.map(function(t) {
    var cls = (t.pnl || 0) >= 0 ? 'g' : 'r';
    return '<tr>' +
      '<td class="m">' + timeFmt(t.exit_time) + '</td>' +
      '<td><strong>' + t.ticker + '</strong></td>' +
      '<td>' + dirPill(t.direction) + '</td>' +
      '<td>$' + fmt2(t.entry_price) + '</td>' +
      '<td>$' + fmt2(t.exit_price) + '</td>' +
      '<td>' + t.shares + '</td>' +
      '<td class="' + cls + '">' + signOf(t.pnl || 0) + '$' + fmt2(t.pnl) + '</td>' +
      '<td class="m">' + (t.exit_reason || '—') + '</td>' +
      '</tr>';
  }).join('');
}

function updateChart(d) {
  var sorted = d.closed_trades
    .filter(function(t) { return t.exit_time && t.pnl != null; })
    .slice()
    .sort(function(a, b) { return a.exit_time.localeCompare(b.exit_time); });
  var cum = 0;
  var labels = [], values = [];
  for (var i = 0; i < sorted.length; i++) {
    cum += sorted[i].pnl;
    labels.push(timeFmt(sorted[i].exit_time));
    values.push(parseFloat(cum.toFixed(2)));
  }
  pnlChart.data.labels = labels;
  pnlChart.data.datasets[0].data = values;
  var color = cum >= 0 ? '#22d3ee' : '#f87171';
  pnlChart.data.datasets[0].borderColor = color;
  pnlChart.data.datasets[0].backgroundColor = cum >= 0 ? 'rgba(34,211,238,0.06)' : 'rgba(248,113,113,0.06)';
  pnlChart.update('none');
}

function updateRisk(d) {
  var hPct = (d.portfolio_heat_pct * 100).toFixed(1);
  var maxPct = (d.max_portfolio_heat * 100).toFixed(0);
  document.getElementById('s-heat').textContent = hPct + '% / ' + maxPct + '%';
  var fill = Math.min((d.portfolio_heat_pct / (d.max_portfolio_heat || 1)) * 100, 100);
  document.getElementById('heat-bar').style.width = fill + '%';

  var shortRow = document.getElementById('short-heat-row');
  if (d.short_enabled) {
    shortRow.style.display = 'block';
    var shPct = (d.short_heat_pct * 100).toFixed(1);
    var shMax = (d.short_max_heat * 100).toFixed(0);
    document.getElementById('s-short-heat').textContent = shPct + '% / ' + shMax + '%';
    var sFill = Math.min((d.short_heat_pct / (d.short_max_heat || 1)) * 100, 100);
    document.getElementById('short-heat-bar').style.width = sFill + '%';
  } else {
    shortRow.style.display = 'none';
  }

  document.getElementById('s-pos').textContent = d.open_positions_count + ' / ' + d.max_open_positions;
  document.getElementById('s-losses').textContent = d.consecutive_losses;

  var cdEl = document.getElementById('s-cooldown');
  if (d.cooldown_until) {
    var secs = Math.max(0, Math.floor((new Date(d.cooldown_until) - Date.now()) / 1000));
    cdEl.textContent = Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
    cdEl.className = 'v r';
  } else {
    cdEl.textContent = '—';
    cdEl.className = 'v m';
  }
}

function updateConfig(d) {
  var c = d.config;
  if (!c || !Object.keys(c).length) return;
  var stratLabel = d.long_strategy_name === 'gap_hold' ? 'Gap-Hold' : 'Standard';
  var rows = [
    ['Strategy', stratLabel],
    ['Risk/Trade', (c.risk_per_trade * 100).toFixed(1) + '%'],
    ['Max Heat', (c.max_portfolio_heat * 100).toFixed(0) + '%'],
    ['Min RelVol', c.stage2_min_relative_volume + '×'],
    ['Min Δ Price', (c.stage1_min_price_change_pct * 100).toFixed(0) + '%'],
    ['Buy Pressure', '≥ ' + (c.stage2_buying_pressure_min * 100).toFixed(0) + '%'],
    ['EOD Exit', c.eod_evaluation],
    ['Conf. Tiers', c.confidence_tiers],
  ];
  var html = rows.map(function(row) {
    return '<div class="kv"><span class="k">' + row[0] + '</span><span class="v">' + row[1] + '</span></div>';
  }).join('');

  if (d.short_enabled && d.short_config && Object.keys(d.short_config).length) {
    var sc = d.short_config;
    var shortRows = [
      ['Strategy', sc.strategy || 'HOD Rejection'],
      ['Min Run', sc.min_run_pct || '—'],
      ['Rejection', sc.rejection_bars || '—'],
      ['Stop/Target', sc.stop_target || '—'],
      ['Regime', sc.regime_filter || '—'],
      ['Risk/Trade', sc.risk_per_trade != null ? (sc.risk_per_trade * 100).toFixed(1) + '%' : '—'],
      ['Max Heat', sc.max_portfolio_heat != null ? (sc.max_portfolio_heat * 100).toFixed(0) + '%' : '—'],
      ['Min $Vol', sc.min_dollar_vol != null ? '$' + (sc.min_dollar_vol / 1e6).toFixed(0) + 'M' : '—'],
    ];
    html += '<div class="panel-subtitle">Short Strategy</div>';
    html += shortRows.map(function(row) {
      return '<div class="kv"><span class="k">' + row[0] + '</span><span class="v">' + row[1] + '</span></div>';
    }).join('');
  }

  document.getElementById('config-body').innerHTML = html;
}

// ── Poll loop ────────────────────────────────────────────────
function poll() {
  fetch('/api/state')
    .then(function(r) { return r.ok ? r.json() : null; })
    .then(function(d) {
      if (!d) return;
      updateHeader(d);
      updatePositions(d);
      updateTrades(d);
      updateChart(d);
      updateRisk(d);
      updateConfig(d);
    })
    .catch(function(e) { console.error('Dashboard poll error:', e); });
}

poll();
setInterval(poll, 3000);
</script>
</body>
</html>"""
