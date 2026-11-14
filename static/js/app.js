/**
 * app.js — Drowning Detection System Frontend
 * Drag-and-drop upload, progress tracking, alert sounds,
 * real-time alert toasts, and stream dashboard logic.
 */

'use strict';

// ============================================================
// DRAG-AND-DROP UPLOAD
// ============================================================

(function initUpload() {
  const zone      = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const banner    = document.getElementById('file-selected-banner');
  const bannerName= document.getElementById('selected-file-name');
  const form      = document.getElementById('upload-form');
  const submitBtn = document.getElementById('submit-btn');
  const progress  = document.getElementById('progress-container');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const progressPct  = document.getElementById('progress-pct');

  if (!zone) return;

  // ---- Drag events ----
  ['dragenter', 'dragover'].forEach(evt =>
    zone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('drag-over');
    })
  );

  ['dragleave', 'drop'].forEach(evt =>
    zone.addEventListener(evt, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drag-over');
    })
  );

  zone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      showFileBanner(files[0].name, files[0].size);
    }
  });

  // ---- File input change ----
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      const f = fileInput.files[0];
      showFileBanner(f.name, f.size);
    }
  });

  function showFileBanner(name, size) {
    if (banner) {
      banner.classList.add('visible');
      bannerName.textContent = `${name} (${formatBytes(size)})`;
    }
    if (submitBtn) submitBtn.disabled = false;
  }

  // ---- Form submit with progress simulation ----
  if (form) {
    form.addEventListener('submit', () => {
      if (!fileInput.files.length) return;
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner"></span> Processing…';
      if (progress) {
        progress.classList.add('visible');
        simulateProgress();
      }
    });
  }

  // Simulate progress (actual backend processing)
  function simulateProgress() {
    let pct = 0;
    const stages = [
      { target: 15, label: 'Uploading file…',          duration: 600 },
      { target: 30, label: 'Pre-processing footage…',  duration: 800 },
      { target: 55, label: 'Running D-FINE detection…',duration: 1200 },
      { target: 75, label: 'Analysing poses…',         duration: 900 },
      { target: 90, label: 'Classifying drowning risk…',duration: 700 },
      { target: 98, label: 'Encoding output video…',   duration: 500 },
    ];

    function runStage(idx) {
      if (idx >= stages.length) return;
      const { target, label, duration } = stages[idx];
      if (progressText) progressText.textContent = label;
      const start  = pct;
      const diff   = target - start;
      const tStart = performance.now();

      function step(now) {
        const elapsed = now - tStart;
        const t       = Math.min(elapsed / duration, 1);
        pct = start + diff * easeOut(t);
        setProgress(pct);
        if (t < 1) requestAnimationFrame(step);
        else runStage(idx + 1);
      }

      requestAnimationFrame(step);
    }

    runStage(0);
  }

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function setProgress(pct) {
    if (progressFill) progressFill.style.width = pct.toFixed(1) + '%';
    if (progressPct)  progressPct.textContent   = Math.round(pct) + '%';
  }

  function formatBytes(bytes) {
    if (bytes < 1024)        return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
})();


// ============================================================
// ALERT TOASTS
// ============================================================

function showToast(title, description, type = 'drowning') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  const icons = { drowning: '🚨', distress: '⚠️', geofence: '🚧' };
  toast.innerHTML = `
    <div class="toast-icon">${icons[type] || '⚠️'}</div>
    <div class="toast-body">
      <div class="toast-title">${escHtml(title)}</div>
      <div class="toast-desc">${escHtml(description)}</div>
    </div>
  `;

  container.appendChild(toast);

  // Auto-dismiss after 6s
  setTimeout(() => {
    toast.style.animation = 'toast-out 0.4s ease forwards';
    toast.addEventListener('animationend', () => toast.remove());
  }, 6000);

  // Beep sound
  playAlertBeep(type);
}

function playAlertBeep(type) {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.type      = 'sine';
    osc.frequency.setValueAtTime(
      type === 'drowning' ? 880 : (type === 'geofence' ? 760 : 660),
      ctx.currentTime
    );
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);

    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.5);

    if (type === 'drowning') {
      // Double beep for drowning
      const osc2 = ctx.createOscillator();
      const g2   = ctx.createGain();
      osc2.connect(g2);
      g2.connect(ctx.destination);
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(1100, ctx.currentTime + 0.6);
      g2.gain.setValueAtTime(0.3, ctx.currentTime + 0.6);
      g2.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.0);
      osc2.start(ctx.currentTime + 0.6);
      osc2.stop(ctx.currentTime + 1.0);
    }
  } catch (_) {
    // Audio unavailable — silent fail
  }
}

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}


// ============================================================
// AUTO-SHOW TOASTS FOR DETECTED ALERTS (results page)
// ============================================================

(function initResultAlerts() {
  const alertRows = document.querySelectorAll('[data-alert-row]');
  if (!alertRows.length) return;

  // Show toasts with staggered delay
  alertRows.forEach((row, idx) => {
    const risk   = row.dataset.risk   || 'UNKNOWN';
    const tid    = row.dataset.tid    || '?';
    const ts     = row.dataset.ts     || '0';
    const reason = row.dataset.reason || '';

    if (risk === 'DROWNING' || risk === 'DISTRESS') {
      setTimeout(() => {
        showToast(
          `${risk} — Track #${tid} at ${ts}s`,
          reason || 'Risk detected',
          risk.toLowerCase()
        );
      }, idx * 800 + 400);
    }
  });
})();


// ============================================================
// LIVE STREAM CONTROLS
// ============================================================

(function initStream() {
  const startBtn  = document.getElementById('btn-start-stream');
  const stopBtn   = document.getElementById('btn-stop-stream');
  const feedImg   = document.getElementById('stream-feed-img');
  const statusDot = document.getElementById('stream-status-text');

  if (!startBtn) return;

  let streamInterval = null;
  let lastAlertCount = 0;

  startBtn.addEventListener('click', () => {
    if (feedImg) {
      // Start MJPEG stream
      const ts = Date.now();
      feedImg.src = `/video_feed?t=${ts}`;
    }
    if (statusDot) statusDot.textContent = 'LIVE';
    startBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    lastAlertCount = 0;
    const tbody = document.getElementById('stream-alert-tbody');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--clr-text-muted)">Waiting for events…</td></tr>';
    }

    // Poll alert endpoint every 1.5 seconds
    if (streamInterval) clearInterval(streamInterval);
    streamInterval = setInterval(pollStreamAlerts, 1500);
  });

  stopBtn.addEventListener('click', () => {
    if (feedImg) feedImg.src = '';
    if (statusDot) statusDot.textContent = 'STOPPED';
    startBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    if (streamInterval) clearInterval(streamInterval);
    fetch('/stream_control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'stop'})
    }).catch(() => {});
  });

  async function pollStreamAlerts() {
    try {
      const res  = await fetch('/stream_alerts', {cache: 'no-store'});
      const data = await res.json();
      const alerts = data.alerts || [];
      if (statusDot && data.status) statusDot.textContent = data.status;

      if (alerts.length > lastAlertCount) {
        const newAlerts = alerts.slice(lastAlertCount);
        newAlerts.forEach(alert => {
          if (alert.risk_level === 'DROWNING' || alert.risk_level === 'DISTRESS'
              || alert.risk_level === 'GEOFENCE') {
            const type = alert.risk_level === 'DROWNING'
              ? 'drowning'
              : (alert.risk_level === 'GEOFENCE' ? 'geofence' : 'distress');
            showToast(
              `${alert.risk_level} — Track #${alert.track_id}`,
              alert.reasons || 'Risk detected during live stream',
              type
            );
          }
        });
        lastAlertCount = alerts.length;
        updateStreamAlertTable(alerts);
      }
    } catch (_) {
      // Network error — silent fail
    }
  }

  function updateStreamAlertTable(alerts) {
    const tbody = document.getElementById('stream-alert-tbody');
    const counter = document.getElementById('live-alert-count');
    if (counter) counter.textContent = alerts.length;
    if (!tbody) return;

    if (!alerts.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--clr-text-muted)">No alerts yet.</td></tr>';
      return;
    }

    tbody.innerHTML = alerts.slice().reverse().slice(0, 20).map(a => `
      <tr>
        <td>${a.timestamp.toFixed ? a.timestamp.toFixed(1) : a.timestamp}s</td>
        <td><span class="risk-badge ${a.risk_level.toLowerCase()}">${a.risk_level}</span></td>
        <td>#${a.track_id}</td>
        <td>${escHtml(a.reasons || '')}</td>
      </tr>
    `).join('');
  }
})();


// ============================================================
// RISK COLOR LEGEND TOOLTIP
// ============================================================

(function initLegend() {
  document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.setAttribute('title', el.dataset.tooltip);
  });
})();


// ============================================================
// GEOFENCE DRAWING (Live Stream page)
// ============================================================

(function initGeofenceEditor() {
  const canvas   = document.getElementById('geofence-canvas');
  const img      = document.getElementById('stream-feed-img');
  const stage    = document.getElementById('geofence-stage');
  const btnSafe  = document.getElementById('btn-draw-safe');
  const btnDanger= document.getElementById('btn-draw-danger');
  const btnUndo  = document.getElementById('btn-undo-point');
  const btnClear = document.getElementById('btn-clear-geofence');
  const btnSave  = document.getElementById('btn-save-geofence');
  const enabled  = document.getElementById('geofence-enabled');
  const statusEl = document.getElementById('geofence-status');

  if (!canvas || !img || !stage) return;

  const ctx = canvas.getContext('2d');
  const COLORS = { safe: '#22c55e', danger: '#ef4444' };

  let zones = [];          // saved zones: [{name, mode, points:[[x,y],...]}]
  let draft = null;        // zone being drawn: {mode, points:[]}
  let drawingMode = null;  // 'safe' | 'danger' | null

  function setStatus(text, isError) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.style.color = isError ? '#ef4444' : 'var(--clr-text-muted)';
  }

  function syncCanvas() {
    const w = Math.max(1, img.clientWidth || stage.clientWidth || 640);
    const h = Math.max(1, img.clientHeight || 280);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    draw();
  }

  function drawPolygon(points, mode, lineWidth, fillAlpha, closed) {
    if (!points.length) return;
    const color = COLORS[mode] || COLORS.safe;
    const pts = points.map(([x, y]) => [x * canvas.width, y * canvas.height]);

    if (pts.length >= 3 && closed) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath();
      ctx.fillStyle = hexToRgba(color, fillAlpha);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (closed && pts.length >= 3) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();

    // Vertices
    pts.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  function hexToRgba(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    zones.forEach(z => drawPolygon(z.points, z.mode, 2, 0.07, true));
    if (draft) drawPolygon(draft.points, draft.mode, 3, 0.12, false);
  }

  function pointFromEvent(ev) {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / Math.max(1, rect.width);
    const y = (ev.clientY - rect.top) / Math.max(1, rect.height);
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
  }

  canvas.addEventListener('click', ev => {
    if (!drawingMode) {
      setStatus('Choose "Draw safe zone" or "Draw danger zone" first.');
      return;
    }
    if (!draft || draft.mode !== drawingMode) {
      draft = { mode: drawingMode, points: [] };
    }
    draft.points.push(pointFromEvent(ev));
    draw();
    setStatus(`${draft.mode} zone: ${draft.points.length} point(s). ` +
              'Click more points, then Save.');
  });

  function startDrawing(mode) {
    drawingMode = mode;
    draft = { mode, points: [] };
    canvas.classList.add('drawing');
    draw();
    setStatus(`Drawing ${mode} zone — click points on the feed.`);
  }

  if (btnSafe)   btnSafe.addEventListener('click', () => startDrawing('safe'));
  if (btnDanger) btnDanger.addEventListener('click', () => startDrawing('danger'));

  if (btnUndo) {
    btnUndo.addEventListener('click', () => {
      if (draft && draft.points.length) draft.points.pop();
      draw();
    });
  }

  if (btnClear) {
    btnClear.addEventListener('click', () => {
      zones = [];
      draft = null;
      drawingMode = null;
      canvas.classList.remove('drawing');
      draw();
      setStatus('All zones cleared locally. Press Save to apply.');
    });
  }

  if (btnSave) {
    btnSave.addEventListener('click', async () => {
      let next = zones.map(z => ({
        name: z.name, mode: z.mode,
        points: z.points.map(p => [p[0], p[1]]),
      }));

      if (draft && draft.points.length >= 3) {
        // Replace the previous zone of the same mode with the new drawing.
        next = next.filter(z => z.mode !== draft.mode);
        next.push({
          name: draft.mode === 'safe' ? 'drawn-safe-zone' : 'drawn-danger-zone',
          mode: draft.mode,
          points: draft.points.map(p => [p[0], p[1]]),
        });
      }

      if (enabled && enabled.checked && !next.length) {
        setStatus('Draw at least one zone (3+ points) before enabling.', true);
        return;
      }

      setStatus('Saving geofence…');
      try {
        const res = await fetch('/geofence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: !!(enabled && enabled.checked),
            zones: next,
          }),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          setStatus(data.error || 'Could not save geofence.', true);
          return;
        }
        zones = (data.geofence && data.geofence.zones) || [];
        draft = null;
        drawingMode = null;
        canvas.classList.remove('drawing');
        if (enabled) enabled.checked = !!data.geofence.enabled;
        draw();
        setStatus(`Saved: ${zones.length} zone(s), ` +
                  (data.geofence.enabled ? 'geofence ON' : 'geofence OFF') + '.');
      } catch (err) {
        setStatus('Network error while saving geofence.', true);
      }
    });
  }

  async function load() {
    try {
      const res = await fetch('/geofence', { cache: 'no-store' });
      const data = await res.json();
      zones = data.zones || [];
      if (enabled) enabled.checked = !!data.enabled;
      draw();
      setStatus(`${zones.length} zone(s) configured, ` +
                (data.enabled ? 'enabled' : 'disabled') + '.');
    } catch (err) {
      setStatus('Could not load geofence configuration.', true);
    }
  }

  if (window.ResizeObserver) {
    new ResizeObserver(syncCanvas).observe(img);
  }
  window.addEventListener('resize', syncCanvas);
  img.addEventListener('load', syncCanvas);

  syncCanvas();
  load();
})();
