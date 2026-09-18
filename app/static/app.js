/* Exoplanet Habitability Explorer — front end.
 *
 * State flow: slider/toggle input -> debounce(175ms) -> POST /api/predict
 * -> animate dial, SHAP rows and the plot marker. The probability surface is
 * only re-requested when one of the three held-constant features changes.
 */

const DEBOUNCE_MS = 175;
const PLOT_PAD = { top: 16, right: 16, bottom: 44, left: 58 };

const EARTH = {
  Planet_Mass: 1.0,
  Planet_Radius: 1.0,
  Planet_Surface_Temperature: 288.0,
  Planet_Period: 365.25,
  Planet_Type: 'Terran',
};

/* Sequential blue ramp (reference palette), low -> high on a dark surface. */
const RAMP = [
  [0.00, [26, 26, 25]],
  [0.06, [13, 54, 107]],
  [0.22, [24, 79, 149]],
  [0.38, [37, 106, 191]],
  [0.55, [57, 135, 229]],
  [0.70, [109, 167, 236]],
  [0.85, [158, 197, 244]],
  [1.00, [205, 226, 251]],
];

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const dur = (ms) => (reduceMotion ? 0 : ms);

const state = {
  meta: null,
  values: {},
  featureByName: {},
  axes: null,
  surface: null,
  surfaceKey: null,
  lastResult: null,
  seq: 0,
  marker: { px: 0, py: 0, glow: 0, settled: false },
  hover: null,
};

const el = (id) => document.getElementById(id);

/* ───────────────────────── scales ───────────────────────── */

const lerp = (a, b, t) => a + (b - a) * t;

function toPosition(feature, value) {
  if (feature.scale === 'log') {
    const lo = Math.log10(feature.min), hi = Math.log10(feature.max);
    return (Math.log10(value) - lo) / (hi - lo);
  }
  return (value - feature.min) / (feature.max - feature.min);
}

function fromPosition(feature, t) {
  if (feature.scale === 'log') {
    const lo = Math.log10(feature.min), hi = Math.log10(feature.max);
    return Math.pow(10, lerp(lo, hi, t));
  }
  return lerp(feature.min, feature.max, t);
}

function fmt(value, feature) {
  const abs = Math.abs(value);
  if (feature && feature.name === 'Planet_Surface_Temperature') return value.toFixed(0);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function rampColor(p) {
  const t = Math.max(0, Math.min(1, p));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i];
      const k = (t - t0) / (t1 - t0);
      return [
        Math.round(lerp(c0[0], c1[0], k)),
        Math.round(lerp(c0[1], c1[1], k)),
        Math.round(lerp(c0[2], c1[2], k)),
      ];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

/* ───────────────────────── bootstrap ───────────────────────── */

async function init() {
  const res = await fetch('/api/meta');
  if (!res.ok) throw new Error('Could not load model metadata');
  const meta = await res.json();
  state.meta = meta;
  meta.features.forEach((f) => { state.featureByName[f.name] = f; });

  meta.features.forEach((f) => { state.values[f.name] = clampToFeature(f, EARTH[f.name]); });
  state.values.Planet_Type = meta.planet_types.includes(EARTH.Planet_Type)
    ? EARTH.Planet_Type : meta.default_planet_type;

  renderModelStats(meta);
  buildSliders(meta);
  buildTypeToggle(meta);
  setupPlot(meta);
  setupPlanet();
  el('earthBtn').addEventListener('click', () => applyPreset(EARTH));
  el('liveSurface').addEventListener('change', () => { state.surfaceKey = null; request(); });

  el('footNote').textContent =
    `HistGradientBoostingClassifier over ${meta.metrics.n_samples.toLocaleString()} NASA archive planets · ` +
    `${meta.metrics.n_habitable} labelled habitable (${(meta.metrics.positive_rate * 100).toFixed(2)}%) · ` +
    `SHAP values are exact (2^5 coalitions).`;

  request(true);
}

function clampToFeature(f, v) {
  if (v == null || !isFinite(v)) return f.median;
  return Math.min(f.max, Math.max(f.min, v));
}

function renderModelStats(meta) {
  const m = meta.metrics;
  const stats = [
    ['ROC-AUC', m.roc_auc.toFixed(3)],
    ['PR-AUC', m.pr_auc.toFixed(3)],
    ['Recall', m.recall_habitable.toFixed(2)],
    ['Precision', m.precision_habitable.toFixed(2)],
  ];
  el('modelStats').innerHTML = stats
    .map(([k, v]) => `<div><b>${v}</b>${k}</div>`)
    .join('');
  el('modelStats').title = `${m.cv_folds}-fold stratified cross-validation on the full labelled set`;
}

/* ───────────────────────── controls ───────────────────────── */

function buildSliders(meta) {
  const host = el('sliders');
  host.innerHTML = '';

  meta.features.forEach((f) => {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    wrap.style.marginTop = '0';
    wrap.innerHTML = `
      <div class="field-head">
        <label for="s_${f.name}">${f.label}</label>
        <span class="field-value" id="v_${f.name}"></span>
      </div>
      <input type="range" id="s_${f.name}" min="0" max="1000" step="1"
             aria-label="${f.label} in ${f.unit || 'units'}">
      <div class="field-scale">
        <span>${fmt(f.min, f)}</span><span>${fmt(f.max, f)} ${f.unit}</span>
      </div>`;
    host.appendChild(wrap);

    const input = wrap.querySelector('input');
    input.value = Math.round(toPosition(f, state.values[f.name]) * 1000);
    paintSlider(input);
    updateValueLabel(f);

    input.addEventListener('input', () => {
      state.values[f.name] = fromPosition(f, input.value / 1000);
      paintSlider(input);
      updateValueLabel(f);
      updatePlanet();
      request();
    });
  });
}

function paintSlider(input) {
  input.style.setProperty('--fill', `${(input.value / 1000) * 100}%`);
}

function updateValueLabel(f) {
  const node = el(`v_${f.name}`);
  if (node) node.innerHTML = `${fmt(state.values[f.name], f)}<span>${f.unit}</span>`;
}

function buildTypeToggle(meta) {
  const host = el('planetTypes');
  host.innerHTML = '';
  meta.planet_types.forEach((type) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'type-btn';
    btn.role = 'radio';
    btn.textContent = type;
    btn.setAttribute('aria-checked', String(type === state.values.Planet_Type));
    const habitable = meta.habitable_by_type[type] || 0;
    const total = meta.planet_type_counts[type] || 0;
    btn.title = `${total} in training data · ${habitable} labelled habitable`;
    btn.addEventListener('click', () => {
      state.values.Planet_Type = type;
      [...host.children].forEach((c) =>
        c.setAttribute('aria-checked', String(c.textContent === type)));
      updateTypeNote();
      updatePlanet();
      request();
    });
    host.appendChild(btn);
  });
  updateTypeNote();
}

function updateTypeNote() {
  const meta = state.meta;
  const type = state.values.Planet_Type;
  const habitable = meta.habitable_by_type[type] || 0;
  const total = meta.planet_type_counts[type] || 0;
  el('typeNote').textContent =
    `${total.toLocaleString()} ${type} planets in training data, ${habitable} labelled habitable.`;
}

function applyPreset(preset) {
  state.meta.features.forEach((f) => {
    state.values[f.name] = clampToFeature(f, preset[f.name]);
    const input = el(`s_${f.name}`);
    input.value = Math.round(toPosition(f, state.values[f.name]) * 1000);
    paintSlider(input);
    updateValueLabel(f);
  });
  if (state.meta.planet_types.includes(preset.Planet_Type)) {
    state.values.Planet_Type = preset.Planet_Type;
    [...el('planetTypes').children].forEach((c) =>
      c.setAttribute('aria-checked', String(c.textContent === preset.Planet_Type)));
    updateTypeNote();
  }
  updatePlanet();
  request();
}

/* ───────────────────────── planet viewport ───────────────────────── */

let planet = null;
let orbit = null;

function setupPlanet() {
  const hint = el('dragHint');
  planet = PlanetView.create(el('planetCanvas'), {
    onFirstDrag: () => hint.classList.add('gone'),
  });
  planet.start();

  orbit = OrbitView.create(el('orbitCanvas'), {
    onChange: (period, aAU, lap) => {
      el('orbitMeta').textContent =
        `${fmt(period, state.featureByName.Planet_Period)} d · a ≈ ${formatAU(aAU)} AU · lap ${lap.toFixed(1)}s`;
    },
  });
  orbit.start();

  updatePlanet(true);
}

function formatAU(a) {
  if (a >= 100) return a.toFixed(0);
  if (a >= 10) return a.toFixed(1);
  if (a >= 1) return a.toFixed(2);
  return a.toFixed(3);
}

/* Driven straight off the inputs, so the planet reacts on the drag itself
 * rather than waiting for the debounced prediction to come back. */
function updatePlanet(immediate) {
  if (!planet) return;
  planet.setParams({
    mass: state.values.Planet_Mass,
    radius: state.values.Planet_Radius,
    surface_temperature: state.values.Planet_Surface_Temperature,
    period: state.values.Planet_Period,
    planet_type: state.values.Planet_Type,
  }, immediate);
  if (orbit) orbit.setPeriod(state.values.Planet_Period);
  updatePlanetLegend();
}

function updatePlanetLegend() {
  const v = state.values;
  const f = state.featureByName;
  const text = {
    radius: `${fmt(v.Planet_Radius, f.Planet_Radius)} R⊕`,
    temp: `${fmt(v.Planet_Surface_Temperature, f.Planet_Surface_Temperature)} K · ${surfaceWord()}`,
    mass: `${fmt(v.Planet_Mass, f.Planet_Mass)} M⊕`,
    period: `${fmt(v.Planet_Period, f.Planet_Period)} d · a ≈ ${formatAU(Math.pow(v.Planet_Period / 365.25, 2 / 3))} AU`,
  };
  el('planetLegend').querySelectorAll('dd').forEach((dd) => {
    dd.textContent = text[dd.dataset.k];
  });
}

function surfaceWord() {
  const t = state.values.Planet_Surface_Temperature;
  const type = state.values.Planet_Type;
  if (type === 'Jovian' || type === 'Neptunian') return t > 700 ? 'glowing' : 'banded';
  if (t < 245) return 'frozen';
  if (t < 340) return 'temperate';
  if (t < 800) return 'arid';
  return 'molten';
}

/* ───────────────────────── requests ───────────────────────── */

let debounceTimer = null;

function request(immediate = false) {
  clearTimeout(debounceTimer);
  setStatus('busy', 'predicting');
  if (immediate) { send(); return; }
  debounceTimer = setTimeout(send, DEBOUNCE_MS);
}

/* The surface only depends on the three features held constant across the
 * grid, so skip recomputing it while the user drags period or temperature. */
function surfaceKey() {
  if (!el('liveSurface').checked) return 'median';
  return [state.values.Planet_Mass.toFixed(4),
          state.values.Planet_Radius.toFixed(4),
          state.values.Planet_Type].join('|');
}

async function send() {
  const seq = ++state.seq;
  const key = surfaceKey();
  const needSurface = key !== state.surfaceKey;
  const live = el('liveSurface').checked;

  // The prediction always describes the user's own planet; only the surface's
  // held-constant features change with the toggle.
  const body = {
    mass: state.values.Planet_Mass,
    radius: state.values.Planet_Radius,
    surface_temperature: state.values.Planet_Surface_Temperature,
    period: state.values.Planet_Period,
    planet_type: state.values.Planet_Type,
    include_surface: needSurface,
  };

  if (needSurface) {
    body.surface_x = { ...state.axes.x };
    body.surface_y = { ...state.axes.y };
    if (!live) {
      body.surface_hold = {
        mass: state.meta.medians.Planet_Mass,
        radius: state.meta.medians.Planet_Radius,
        planet_type: state.meta.default_planet_type,
      };
    }
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const result = await res.json();

    if (seq !== state.seq) return;   // a newer request already landed

    if (result.surface) {
      state.surface = result.surface;
      state.surfaceKey = key;
      paintSurface();
    }

    applyResult(result);
    setStatus('', 'live');
  } catch (err) {
    if (seq !== state.seq) return;
    console.error(err);
    setStatus('error', 'error');
  }
}

function setStatus(cls, text) {
  const chip = el('statusChip');
  chip.className = `status-chip ${cls}`.trim();
  chip.textContent = text;
}

/* ───────────────────────── result rendering ───────────────────────── */

function applyResult(result) {
  const prev = state.lastResult;
  state.lastResult = result;

  animateProbability(prev ? prev.probability : 0, result.probability);
  renderShap(result);
  moveMarker();
  if (planet) planet.setProbability(result.probability);
}

/* Persistent so a new prediction cancels the previous count-up rather than
 * letting two animations fight over the same text node. */
const probCounter = { v: 0 };

function animateProbability(from, to) {
  const circumference = 2 * Math.PI * 52;

  anime.remove(probCounter);
  probCounter.v = from;
  anime({
    targets: probCounter,
    v: to,
    duration: dur(520),
    easing: 'easeOutCubic',
    update: () => { el('probNumber').textContent = (probCounter.v * 100).toFixed(1); },
  });

  anime({
    targets: '#dialValue',
    strokeDashoffset: circumference * (1 - to),
    stroke: rgb(rampColor(Math.max(0.18, to))),
    duration: dur(620),
    easing: 'easeOutCubic',
  });

  const label = el('verdictLabel');
  const habitable = to >= 0.5;
  label.textContent = habitable ? 'Likely habitable' : 'Unlikely to be habitable';
  label.className = `verdict-label ${habitable ? 'yes' : 'no'}`;

  el('verdictNote').innerHTML = habitable
    ? 'Above the 0.5 decision threshold used by <code>predict</code>.'
    : 'Below the 0.5 decision threshold used by <code>predict</code>.';
}

const rgb = (c) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

function renderShap(result) {
  const list = el('shapList');
  const top = result.shap.filter((s) => Math.abs(s.contribution) > 1e-4).slice(0, 3);

  el('shapBase').textContent = `baseline ${(result.base_value * 100).toFixed(1)}%`;

  if (!top.length) {
    list.innerHTML = '<li class="shap-empty">No feature moves this prediction measurably.</li>';
    return;
  }

  const scale = Math.max(...top.map((s) => Math.abs(s.contribution)));
  const feature = (name) => state.featureByName[name];

  list.innerHTML = top.map((s) => {
    const f = feature(s.feature);
    const shown = f ? `${fmt(state.values[s.feature], f)} ${f.unit}` : state.values.Planet_Type;
    const width = scale > 0 ? (Math.abs(s.contribution) / scale) * 50 : 0;
    const sign = s.contribution >= 0 ? '+' : '−';
    return `
      <li class="shap-item">
        <div class="shap-item-top">
          <span class="shap-name">${s.label} <b>${shown}</b></span>
          <span class="shap-amount">${sign}${(Math.abs(s.contribution) * 100).toFixed(1)} pts</span>
        </div>
        <div class="shap-track">
          <span class="shap-zero"></span>
          <span class="shap-fill ${s.direction}" data-width="${width}"></span>
        </div>
      </li>`;
  }).join('');

  anime({
    targets: '#shapList .shap-item',
    opacity: [0, 1],
    translateY: [6, 0],
    delay: anime.stagger(dur(60)),
    duration: dur(340),
    easing: 'easeOutQuad',
  });

  anime({
    targets: '#shapList .shap-fill',
    width: (elem) => `${elem.dataset.width}%`,
    delay: anime.stagger(dur(60), { start: dur(70) }),
    duration: dur(480),
    easing: 'easeOutCubic',
  });
}

/* ───────────────────────── plot ───────────────────────── */

let canvas, ctx, surfaceCanvas, scatterCanvas, plotRect, dpr = 1;

function setupPlot(meta) {
  canvas = el('plot');
  ctx = canvas.getContext('2d');

  const period = state.featureByName.Planet_Period;
  const temp = state.featureByName.Planet_Surface_Temperature;
  // Axes mirror the sliders' own scales, so the marker and the slider travel
  // stay in step.
  state.axes = {
    x: { min: period.min, max: period.max, scale: period.scale },
    y: { min: temp.min, max: temp.max, scale: temp.scale },
  };

  surfaceCanvas = document.createElement('canvas');
  surfaceCanvas.width = meta.grid.width;
  surfaceCanvas.height = meta.grid.height;

  scatterCanvas = document.createElement('canvas');

  el('plotSub').textContent =
    `${meta.scatter.length.toLocaleString()} archive planets have both a period and a ` +
    `surface temperature. Shading is the model's probability across the plane.`;

  new ResizeObserver(() => resize()).observe(canvas.parentElement);
  canvas.addEventListener('mousemove', onHover);
  canvas.addEventListener('mouseleave', () => { state.hover = null; el('tooltip').hidden = true; draw(); });
  resize();
}

function resize() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);

  plotRect = {
    x: PLOT_PAD.left,
    y: PLOT_PAD.top,
    w: Math.max(10, rect.width - PLOT_PAD.left - PLOT_PAD.right),
    h: Math.max(10, rect.height - PLOT_PAD.top - PLOT_PAD.bottom),
    cssW: rect.width,
    cssH: rect.height,
  };

  scatterCanvas.width = canvas.width;
  scatterCanvas.height = canvas.height;
  paintScatter();
  snapMarker();
  draw();
}

function axisFraction(axis, value) {
  if (axis.scale === 'log') {
    const lo = Math.log10(axis.min), hi = Math.log10(axis.max);
    return (Math.log10(value) - lo) / (hi - lo);
  }
  return (value - axis.min) / (axis.max - axis.min);
}

function xPix(period) {
  return plotRect.x + axisFraction(state.axes.x, period) * plotRect.w;
}

function yPix(temp) {
  return plotRect.y + plotRect.h - axisFraction(state.axes.y, temp) * plotRect.h;
}

/* Readable ticks for either scale: 1-2-5 decades on log, round steps on linear. */
function axisTicks(axis, targetCount) {
  const out = [];
  if (axis.scale === 'log') {
    for (let e = Math.floor(Math.log10(axis.min)); e <= Math.ceil(Math.log10(axis.max)); e++) {
      for (const m of [1, 2, 5]) {
        const v = m * Math.pow(10, e);
        if (v >= axis.min && v <= axis.max) out.push(v);
      }
    }
    return out;
  }
  const raw = (axis.max - axis.min) / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  for (let v = Math.ceil(axis.min / step) * step; v <= axis.max; v += step) out.push(v);
  return out;
}

function tickLabel(v) {
  if (v >= 1000) return `${+(v / 1000).toFixed(1)}k`;
  if (v >= 1) return String(Math.round(v));
  return String(+v.toFixed(2));
}

function paintSurface() {
  const s = state.surface;
  if (!s) return;
  const sctx = surfaceCanvas.getContext('2d');
  const img = sctx.createImageData(s.width, s.height);
  for (let row = 0; row < s.height; row++) {
    // Grid row 0 is the lowest temperature; canvas row 0 is the top.
    const src = (s.height - 1 - row) * s.width;
    for (let col = 0; col < s.width; col++) {
      const c = rampColor(s.values[src + col]);
      const o = (row * s.width + col) * 4;
      img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
  }
  sctx.putImageData(img, 0, 0);
  draw();
}

function paintScatter() {
  if (!plotRect) return;
  const sctx = scatterCanvas.getContext('2d');
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sctx.clearRect(0, 0, plotRect.cssW, plotRect.cssH);

  const pts = state.meta.scatter;
  const inRange = (p) =>
    p.period >= state.axes.x.min && p.period <= state.axes.x.max &&
    p.temp >= state.axes.y.min && p.temp <= state.axes.y.max;

  // Background planets: a recessive density cloud, deliberately achromatic.
  sctx.fillStyle = 'rgba(154, 152, 144, 0.42)';
  pts.forEach((p) => {
    if (p.habitable || !inRange(p)) return;
    sctx.beginPath();
    sctx.arc(xPix(p.period), yPix(p.temp), 1.9, 0, Math.PI * 2);
    sctx.fill();
  });

  // Known-habitable planets: few enough to carry a 2px surface ring.
  pts.forEach((p) => {
    if (!p.habitable || !inRange(p)) return;
    const x = xPix(p.period), y = yPix(p.temp);
    sctx.beginPath();
    sctx.arc(x, y, 4.5, 0, Math.PI * 2);
    sctx.fillStyle = '#d95926';
    sctx.fill();
    sctx.lineWidth = 2;
    sctx.strokeStyle = 'rgba(18,18,17,0.9)';
    sctx.stroke();
  });
}

function draw() {
  if (!plotRect) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, plotRect.cssW, plotRect.cssH);

  ctx.fillStyle = '#121211';
  ctx.fillRect(0, 0, plotRect.cssW, plotRect.cssH);

  // Probability surface, smoothed up from the coarse model grid.
  if (state.surface) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(plotRect.x, plotRect.y, plotRect.w, plotRect.h);
    ctx.clip();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(surfaceCanvas, plotRect.x, plotRect.y, plotRect.w, plotRect.h);
    ctx.restore();
  }

  drawAxes();

  ctx.drawImage(scatterCanvas, 0, 0, plotRect.cssW, plotRect.cssH);

  drawMarker();
  drawHover();
}

function drawAxes() {
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#8c8b80';
  ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';

  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  axisTicks(state.axes.x, 6).forEach((v) => {
    const x = Math.round(xPix(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, plotRect.y);
    ctx.lineTo(x, plotRect.y + plotRect.h);
    ctx.stroke();
    ctx.fillText(tickLabel(v), x, plotRect.y + plotRect.h + 7);
  });

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  axisTicks(state.axes.y, 6).forEach((v) => {
    const y = Math.round(yPix(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(plotRect.x, y);
    ctx.lineTo(plotRect.x + plotRect.w, y);
    ctx.stroke();
    ctx.fillText(tickLabel(v), plotRect.x - 8, y);
  });

  ctx.fillStyle = '#c3c2b7';
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('Orbital period (days, log scale)',
               plotRect.x + plotRect.w / 2, plotRect.cssH - 4);

  ctx.save();
  ctx.translate(13, plotRect.y + plotRect.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText('Surface temperature (K, log scale)', 0, 0);
  ctx.restore();
}

function markerTarget() {
  return {
    px: xPix(state.values.Planet_Period),
    py: yPix(state.values.Planet_Surface_Temperature),
  };
}

function snapMarker() {
  const t = markerTarget();
  state.marker.px = t.px;
  state.marker.py = t.py;
  state.marker.settled = true;
}

function moveMarker() {
  const t = markerTarget();
  if (!state.marker.settled) { snapMarker(); draw(); return; }

  anime.remove(state.marker);
  anime({
    targets: state.marker,
    px: t.px,
    py: t.py,
    duration: dur(520),
    easing: 'easeOutElastic(1, 0.75)',
    update: draw,
  });

  anime({
    targets: state.marker,
    glow: [1, 0],
    duration: dur(760),
    easing: 'easeOutQuad',
    update: draw,
  });
}

function drawMarker() {
  const { px, py, glow } = state.marker;
  if (!isFinite(px) || !isFinite(py)) return;

  const inside = px >= plotRect.x - 2 && px <= plotRect.x + plotRect.w + 2 &&
                 py >= plotRect.y - 2 && py <= plotRect.y + plotRect.h + 2;
  if (!inside) return;

  ctx.save();

  if (glow > 0.01) {
    ctx.beginPath();
    ctx.arc(px, py, 9 + glow * 16, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,${0.18 * glow})`;
    ctx.fill();
  }

  // Crosshair tying the marker back to both axes.
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = 'rgba(255,255,255,0.32)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plotRect.x, py); ctx.lineTo(px, py);
  ctx.moveTo(px, plotRect.y + plotRect.h); ctx.lineTo(px, py);
  ctx.stroke();
  ctx.setLineDash([]);

  // 2px surface ring keeps the mark legible over the scatter cloud.
  ctx.beginPath();
  ctx.arc(px, py, 7, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#121211';
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(px, py, 3, 0, Math.PI * 2);
  ctx.fillStyle = state.lastResult
    ? rgb(rampColor(Math.max(0.2, state.lastResult.probability))) : '#3987e5';
  ctx.fill();

  ctx.restore();
}

/* ───────────────────────── hover ───────────────────────── */

function onHover(event) {
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;

  let best = null, bestDist = Infinity;
  for (const p of state.meta.scatter) {
    const dx = xPix(p.period) - mx, dy = yPix(p.temp) - my;
    const d = dx * dx + dy * dy;
    // Known-habitable points are the interesting ones, so they get a larger
    // hit target than the background cloud.
    const reach = p.habitable ? 16 : 11;
    if (d <= reach * reach && d < bestDist) { best = p; bestDist = d; }
  }

  if (best !== state.hover) { state.hover = best; draw(); }

  const tip = el('tooltip');
  if (!best) { tip.hidden = true; return; }

  tip.hidden = false;
  tip.style.left = `${xPix(best.period)}px`;
  tip.style.top = `${yPix(best.temp)}px`;
  tip.innerHTML =
    `<b>${escapeHtml(best.name)}</b>` +
    `${best.type || 'unknown type'}<br>` +
    `${best.period.toLocaleString(undefined, { maximumFractionDigits: 2 })} d · ` +
    `${best.temp.toFixed(0)} K` +
    (best.mass != null ? `<br>${best.mass} M⊕ · ${best.radius} R⊕` : '') +
    (best.habitable ? '<br><span class="tt-hab">labelled habitable</span>' : '');
}

function drawHover() {
  const p = state.hover;
  if (!p) return;
  ctx.beginPath();
  ctx.arc(xPix(p.period), yPix(p.temp), 6, 0, Math.PI * 2);
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#ffffff';
  ctx.stroke();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

init().catch((err) => {
  console.error(err);
  setStatus('error', 'failed to load');
  el('verdictLabel').textContent = 'Could not reach the model';
});
