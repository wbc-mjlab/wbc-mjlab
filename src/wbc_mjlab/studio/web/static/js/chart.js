// Small canvas line charts in the brand palette: several series, limit bands, a zero line and a
// playhead; redrawn on theme changes. lineChart(title, xs, series, opts) → element with .setPlayhead(x).
import { THEME_EVENT, h, token } from "./ui.js";

// Categorical order: accent (cyan) first, then amber, success, a violet and danger.
const PALETTE = ["--color-accent", "--color-amber", "--color-success", "#9b8cf0", "--color-danger", "#7fb3ff"];
const colorOf = (i) => (PALETTE[i % PALETTE.length].startsWith("--") ? token(PALETTE[i % PALETTE.length]) : PALETTE[i % PALETTE.length]);

export function fmtNum(v) {
  const a = Math.abs(v);
  return a >= 1000 ? v.toExponential(1) : a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}

/**
 * @param title  label in the head
 * @param xs     shared x values
 * @param series [{label, values}] (values same length as xs)
 * @param opts   {unit, limits: [lo, hi], zero: bool, height, onSeek(x), last: bool}
 */
export function lineChart(title, xs, series, opts = {}) {
  const canvas = h("canvas");
  if (opts.height) canvas.style.height = `${opts.height}px`;
  const lastText = () => {
    const s = series[0];
    if (!s || !s.values.length) return "no data yet";
    const v = s.values[s.values.length - 1];
    return `${fmtNum(v)}${opts.unit ? ` ${opts.unit}` : ""}${opts.last === false ? "" : ` @ ${fmtNum(xs[xs.length - 1])}`}`;
  };
  const legend = series.length > 1 ? h("div.legend", series.map((s, i) => h("span", h("i", { style: { background: colorOf(i) } }), s.label))) : null;
  const el = h("div.chart", h("div.chart-head", title, h("span", opts.headRight ?? lastText())), canvas, legend);
  // The chart is drawn once into ``base``; the playhead only blits it back and adds one line.
  let geo = null;
  let base = null;
  let amber = "";
  const draw = () => {
    geo = drawChart(canvas, xs, series, opts);
    base = document.createElement("canvas");
    base.width = canvas.width;
    base.height = canvas.height;
    base.getContext("2d").drawImage(canvas, 0, 0);
    amber = token("--color-amber");
  };
  el.setPlayhead = (x) => {
    if (!geo || !base || base.width !== canvas.width || !canvas.isConnected) return;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(base, 0, 0);
    const dpr = window.devicePixelRatio || 1;
    const px = geo.X(Math.max(geo.x0, Math.min(geo.x1, x))) * dpr;
    ctx.strokeStyle = amber;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(px, geo.top * dpr);
    ctx.lineTo(px, geo.bottom * dpr);
    ctx.stroke();
  };
  if (opts.onSeek) {
    canvas.style.cursor = "pointer";
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const pad = 34;
      const f = (e.clientX - r.left - pad) / Math.max(1, r.width - pad - 6);
      opts.onSeek(xs[0] + Math.max(0, Math.min(1, f)) * (xs[xs.length - 1] - xs[0]));
    });
  }
  requestAnimationFrame(draw);
  const onTheme = () => (el.isConnected ? draw() : window.removeEventListener(THEME_EVENT, onTheme));
  window.addEventListener(THEME_EVENT, onTheme);
  return el;
}

function drawChart(canvas, xs, series, opts) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const hgt = canvas.clientHeight || 92;
  canvas.width = w * dpr;
  canvas.height = hgt * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const muted = token("--color-text-muted");
  const grid = token("--color-border");
  ctx.font = "10px ui-monospace, Menlo, monospace";
  ctx.fillStyle = muted;
  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  if (xs.length < 2 || !all.length) {
    ctx.fillText(opts.empty || "waiting for data…", 6, hgt / 2);
    return;
  }
  const pad = { l: 34, r: 6, t: 6, b: 14 };
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (opts.limits) {
    lo = Math.min(lo, opts.limits[0]);
    hi = Math.max(hi, opts.limits[1]);
  }
  if (opts.zero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const span = hi - lo;
  lo -= span * 0.04;
  hi += span * 0.04;
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const X = (x) => pad.l + ((x - x0) / Math.max(1e-9, x1 - x0)) * (w - pad.l - pad.r);
  const Y = (y) => pad.t + (1 - (y - lo) / (hi - lo)) * (hgt - pad.t - pad.b);
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  for (const y of [lo + span * 0.04, (lo + hi) / 2, hi - span * 0.04]) {
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(y));
    ctx.lineTo(w - pad.r, Y(y));
    ctx.stroke();
    ctx.fillText(fmtNum(y), 2, Y(y) + 3);
  }
  ctx.fillText(fmtNum(x0), pad.l, hgt - 2);
  const xl = fmtNum(x1);
  ctx.fillText(xl, w - pad.r - ctx.measureText(xl).width, hgt - 2);
  if (opts.limits) {
    const danger = token("--color-danger");
    ctx.fillStyle = `${danger}1f`;
    ctx.fillRect(pad.l, pad.t, w - pad.l - pad.r, Math.max(0, Y(opts.limits[1]) - pad.t));
    ctx.fillRect(pad.l, Y(opts.limits[0]), w - pad.l - pad.r, Math.max(0, hgt - pad.b - Y(opts.limits[0])));
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = `${danger}aa`;
    for (const y of opts.limits) {
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(y));
      ctx.lineTo(w - pad.r, Y(y));
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  if (opts.zero && lo < 0 && hi > 0) {
    ctx.strokeStyle = muted;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(0));
    ctx.lineTo(w - pad.r, Y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  series.forEach((s, i) => {
    const color = colorOf(i);
    ctx.beginPath();
    let started = false;
    s.values.forEach((v, k) => {
      if (!Number.isFinite(v)) return;
      if (!started) { ctx.moveTo(X(xs[k]), Y(v)); started = true; }
      else ctx.lineTo(X(xs[k]), Y(v));
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    if (series.length === 1 && !opts.noFill) {
      ctx.lineTo(X(xs[xs.length - 1]), hgt - pad.b);
      ctx.lineTo(X(xs[0]), hgt - pad.b);
      ctx.closePath();
      ctx.fillStyle = `${color}22`;
      ctx.fill();
    }
  });
  return { X, x0, x1, top: pad.t, bottom: hgt - pad.b };
}
