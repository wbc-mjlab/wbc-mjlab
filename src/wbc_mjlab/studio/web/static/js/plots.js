// Plots: a floating, draggable, resizable window over the stage (server: studio/plots.py).
//  - Data step: the open clip's root, joints and feet over time; a converted clip's raw source dashed.
//    The playhead follows the transport, a click seeks the clip.
//  - Play step: a rolling record of the focused env — the robot solid, the reference it tracks dashed,
//    or their difference (Error); PD targets, torques, actions, keybody tracking errors and every reward
//    term. Live: the view follows the newest steps; a click pins a cursor. The server records every env
//    of a small session, so switching env shows that env's record at once.
// Cheap per frame: new rows are pulled (binary) with each stage frame, draws are coalesced to one per
// animation frame, the canvas is resized only when its box changes, and nothing reads layout or styles
// while drawing (box from a ResizeObserver, theme colours cached).
// Up to eight channels in the validated categorical palette, each keeping its colour while selected; one
// unit on the one y-axis; a crosshair reads every plotted channel; CSV download is the table view.
import { THEME_EVENT, getTheme, h, post, store, token } from "./ui.js";

const MAX = 8;
// Categorical slots in fixed order (dataviz reference palette; validated on the Studio panel surface,
// light #f8fafc and dark #101b26: all checks pass; light needs labels — legend, direct labels, tooltip).
const PALETTE = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
const PAD = { left: 56, right: 12, top: 12, bottom: 26 };
const LABEL_W = 116; // direct labels at the line ends (<= 4 series)
const KINEMATIC = new Set(["rad", "m"]);
const QUANTITY = { value: "", vel: "/s", acc: "/s²" };
const WHAT = { rad: "angle", m: "position", "m/s": "speed", Nm: "torque", reward: "reward (weighted)", raw: "reward (raw)", "": "action" };
const WINDOWS = [5, 10, 30, 60];
const PULL_MS = 30; // at most one Play pull in flight, at most this often (stage frames drive it)
const IDLE_MS = 500; // and at least this often (no stage frames: paused, nothing changed)
const STATS_MS = 250; // legend RMSE refresh while live
const HELP = "drag: scrub · ctrl+drag: zoom box · wheel: zoom · shift+drag: pan · double-click: fit";

const ICON = {
  chart: "M4 19h16M5 15l4-5 4 3 6-8",
  zoom: "M11 5a6 6 0 1 0 0 12 6 6 0 0 0 0-12zM20 20l-4.3-4.3M8.5 11h5M11 8.5v5",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  max: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  min: "M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5",
  close: "M6 6l12 12M18 6L6 18",
};

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", ICON[name]);
  svg.appendChild(p);
  return svg;
}

/** A binary Play pull: [u32 header length][header JSON][pad] then float32 data, ref, meta rows. */
async function fetchRows(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  const buf = await res.arrayBuffer();
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  const rows = head.end - head.start;
  let off = 4 + n;
  const [data, ref, meta] = head.cols.map((cols) => {
    const a = new Float32Array(buf, off, rows * cols);
    off += rows * cols * 4;
    return a;
  });
  return { head, data, ref, meta };
}

function decode(b64) {
  if (!b64) return new Float32Array(0);
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Round tick steps (1, 2, 2.5, 5 × 10^k) covering [lo, hi] with about `count` ticks. */
function niceTicks(lo, hi, count) {
  const span = Math.max(hi - lo, 1e-9);
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, step };
}

function fmt(v, step) {
  const digits = Math.max(0, Math.min(3, -Math.floor(Math.log10(Math.max(step, 1e-9))) + (step < 1 ? 1 : 0)));
  return Number(v).toFixed(digits);
}

const num = (v) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const short = (name) => name.replace(/_joint$/, "").replace(/_link$/, "");

export class Plots {
  constructor(app) {
    this.app = app;
    const saved = store.get("plots", {});
    this.quantity = saved.quantity || "value";
    this.units = saved.units || "deg";
    this.showLimits = saved.limits ?? true;
    this.showRef = saved.ref ?? true;
    this.errorMode = false;
    this.windowS = saved.windowS || 10;
    this.geometry = saved.geometry || null;
    this.picks = saved.picks || {}; // "<source>:<robot>" -> [channel ids]
    this.collapsed = new Set(saved.collapsed || []);
    this.open = false;
    this.source = null; // "clip" | "play" | null
    this.key = null; // what is loaded (clip key or play epoch)
    this.meta = null; // { channels, fps, title, ref_label, clips }
    this.C = 0;
    this.rows = 0;
    this.start = 0; // absolute index of row 0 (play: steps since the record began)
    this.data = null;
    this.ref = null;
    this.hasRef = [];
    this.cuts = null; // Uint8Array: row starts a new episode / clip (play)
    this.markers = [];
    this.selected = [];
    this.slots = new Map();
    this.view = null;
    this.yRange = null;
    this.box = null;
    this.zoomMode = false;
    this.hover = null;
    this.cursor = null; // play: a pinned row
    this.live = true; // play: the view follows the newest steps
    this.listHover = -1;
    this.query = "";
    this.tokens = new Map(); // theme colours, read once per theme
    this.plotBox = null; // the plot's size (ResizeObserver), never measured while drawing
    this.canvasSize = "";
    this.build();
    this.bind();
  }

  // ---------------------------------------------------------------- DOM
  build() {
    const seg = (id, items, value) => h(`div.pl-seg#${id}`, { role: "radiogroup" },
      items.map(([v, label, title]) => h("button", { type: "button", "data-v": v, role: "radio", "aria-checked": String(v === value), title }, label)));
    const toggle = (id, label, title, on) => h(`button.pl-toggle#${id}`, { type: "button", "aria-pressed": String(on), title }, label);
    const iconBtn = (id, name, title) => h(`button.icon-btn.pl-icon#${id}`, { type: "button", title, "aria-label": title }, icon(name));
    this.el = h("section.pl-panel#pl-panel", { role: "dialog", "aria-label": "Plots", hidden: true },
      h("header.pl-head#pl-head",
        icon("chart"),
        h("div.pl-title", h("strong", "Plots"), h("small#pl-subtitle", "")),
        seg("pl-quantity", [["value", "Pos", "Values"], ["vel", "Vel", "Rate of change (finite differences)"], ["acc", "Acc", "Second derivative"]], this.quantity),
        seg("pl-units", [["deg", "deg", "Angles in degrees"], ["rad", "rad", "Angles in radians"]], this.units),
        toggle("pl-ref", "Reference", "Draw the reference dashed next to each channel", this.showRef),
        toggle("pl-error", "Error", "Plot actual − reference (channels that have a reference)", false),
        toggle("pl-limits", "Limits", "Joint limits (angle plots)", this.showLimits),
        h("button.pl-toggle#pl-zoom", { type: "button", "aria-pressed": "false", title: "Drag a box in the plot to zoom (also ctrl+drag) · double-click fits" }, icon("zoom"), "Zoom"),
        h("span.pl-live-box#pl-live-box", { hidden: true },
          h("select#pl-window", { title: "Window shown while following the newest steps" },
            WINDOWS.map((s) => h("option", { value: String(s), selected: s === this.windowS }, `${s} s`))),
          toggle("pl-live", "Live", "Follow the newest steps", true)),
        iconBtn("pl-csv", "download", "Download the plotted channels as CSV (table view)"),
        iconBtn("pl-max", "max", "Full screen (F)"),
        iconBtn("pl-close", "close", "Close (Esc)")),
      h("div.pl-body",
        h("aside.pl-side",
          h("input#pl-search", { type: "search", placeholder: "Search · knee, reward, error…", autocomplete: "off", spellcheck: "false", "aria-label": "Search channels" }),
          h("div.pl-list#pl-list", { role: "listbox", "aria-multiselectable": "true", tabindex: "0", "aria-label": "Channels" }),
          h("small.pl-note#pl-note", `Pick up to ${MAX} channels.`)),
        h("div.pl-main",
          h("div.pl-legend#pl-legend"),
          h("div.pl-plot#pl-plot",
            h("canvas#pl-canvas", { role: "img", "aria-label": "Plot" }),
            h("div.pl-tip#pl-tip", { role: "status", hidden: true }),
            h("div.pl-empty#pl-empty", "")),
          h("div.pl-status",
            h("span#pl-status", ""),
            h("span.pl-range", "from ", h("input#pl-from", { type: "number", step: "0.1", title: "Start of the view [s]" }), " to ",
              h("input#pl-to", { type: "number", step: "0.1", title: "End of the view [s]" }), " s"),
            h("span.pl-help#pl-help", HELP)))),
      h("div.pl-resize#pl-resize", { "aria-hidden": "true" }));
    document.body.appendChild(this.el);
    this.$ = (id) => this.el.querySelector(`#${id}`);
  }

  bind() {
    const $ = this.$;
    $("pl-max").addEventListener("click", () => this.setMaximized(!this.maximized));
    $("pl-head").addEventListener("dblclick", (e) => {
      if (!e.target.closest("button, input, select")) this.setMaximized(!this.maximized);
    });
    $("pl-close").addEventListener("click", () => this.close());
    $("pl-csv").addEventListener("click", () => this.downloadCsv());
    $("pl-zoom").addEventListener("click", () => this.setZoomMode(!this.zoomMode));
    const flip = (id, key, after) => $(id).addEventListener("click", () => {
      this[key] = !this[key];
      $(id).setAttribute("aria-pressed", String(this[key]));
      this.yRange = null;
      after?.();
      this.save();
      this.renderLegend();
      this.draw();
    });
    flip("pl-ref", "showRef");
    flip("pl-error", "errorMode");
    flip("pl-limits", "showLimits");
    $("pl-live").addEventListener("click", () => this.setLive(!this.live));
    $("pl-window").addEventListener("change", (e) => {
      this.windowS = Number(e.target.value);
      this.save();
      this.setLive(true);
    });
    this.radio("pl-quantity", (v) => { this.quantity = v; });
    this.radio("pl-units", (v) => { this.units = v; });
    $("pl-search").addEventListener("input", () => {
      this.query = $("pl-search").value.trim().toLowerCase();
      this.renderList();
    });
    $("pl-search").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = this.visibleRows()[0];
        if (first !== undefined) this.togglePick(first);
      }
    });
    $("pl-list").addEventListener("click", (e) => {
      const all = e.target.closest("[data-plot-group]");
      if (all) return this.pickGroup(all.dataset.plotGroup);
      const head = e.target.closest("[data-group]");
      if (head) return this.toggleGroup(head.dataset.group);
      const row = e.target.closest("[data-ch]");
      if (row) this.togglePick(Number(row.dataset.ch));
    });
    $("pl-list").addEventListener("mouseover", (e) => {
      const row = e.target.closest("[data-ch]");
      this.setListHover(row ? Number(row.dataset.ch) : -1);
    });
    $("pl-list").addEventListener("mouseleave", () => this.setListHover(-1));
    $("pl-legend").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-ch]");
      if (b) this.togglePick(Number(b.dataset.ch));
      if (e.target.closest("[data-clear]")) this.clearPicks();
    });
    for (const id of ["pl-from", "pl-to"]) {
      $(id).addEventListener("change", () => {
        if (!this.meta || this.rows < 2) return;
        const a = Number($("pl-from").value) * this.fps - this.start;
        const b = Number($("pl-to").value) * this.fps - this.start;
        if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return;
        this.live = false;
        this.renderLive();
        this.setView(a, b);
        // Data: the clip goes to the start of the range (paused), so the stage shows it too
        const f = this.app.hud.frame ?? 0;
        if (this.source === "clip" && (f < this.view[0] || f > this.view[1])) {
          post("/api/data/playing", { playing: false })
            .then(() => post("/api/data/frame", { frame: Math.round(this.view[0]) })).catch(() => {});
        }
        this.draw();
      });
    }
    this.bindPanel();
    this.bindPlot();
    window.addEventListener(THEME_EVENT, () => {
      this.tokens.clear();
      this.renderList();
      this.renderLegend();
      this.draw();
    });
    document.addEventListener("keydown", (e) => {
      const t = e.target;
      const typing = t instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName);
      const inside = this.open && this.el.contains(document.activeElement);
      if (e.key === "Escape" && inside && !this.box) {
        e.preventDefault();
        if (this.maximized) this.setMaximized(false);
        else this.close();
      } else if (inside && !typing && !e.ctrlKey && !e.metaKey && !e.altKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        e.stopPropagation();
        this.setMaximized(!this.maximized);
      }
    }, true);
    new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this.plotBox = { width, height };
      if (this.open) this.draw();
    }).observe($("pl-plot"));
  }

  radio(id, onPick) {
    const root = this.$(id);
    root.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      for (const x of root.querySelectorAll("button")) x.setAttribute("aria-checked", String(x === b));
      onPick(b.dataset.v);
      this.yRange = null;
      this.save();
      this.renderLegend();
      this.draw();
    });
  }

  /** Drag by the header, resize from the corner; kept inside the window and remembered. */
  bindPanel() {
    const panel = this.el;
    const clamp = () => {
      const r = panel.getBoundingClientRect();
      panel.style.left = `${Math.min(Math.max(0, r.left), Math.max(0, window.innerWidth - 120))}px`;
      panel.style.top = `${Math.min(Math.max(0, r.top), Math.max(0, window.innerHeight - 48))}px`;
    };
    const drag = (event, onMove) => {
      event.preventDefault();
      const start = { x: event.clientX, y: event.clientY, rect: panel.getBoundingClientRect() };
      const move = (e) => onMove(e.clientX - start.x, e.clientY - start.y, start.rect);
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        clamp();
        this.rememberGeometry();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    };
    this.$("pl-head").addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || this.maximized || e.target.closest("button, input, select")) return;
      drag(e, (dx, dy, rect) => {
        panel.style.left = `${rect.left + dx}px`;
        panel.style.top = `${rect.top + dy}px`;
      });
    });
    this.$("pl-resize").addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || this.maximized) return;
      drag(e, (dx, dy, rect) => {
        panel.style.width = `${Math.max(560, Math.min(window.innerWidth - rect.left, rect.width + dx))}px`;
        panel.style.height = `${Math.max(320, Math.min(window.innerHeight - rect.top, rect.height + dy))}px`;
      });
    });
    window.addEventListener("resize", () => this.open && !this.maximized && clamp());
  }

  setMaximized(on) {
    this.maximized = Boolean(on);
    this.el.classList.toggle("is-max", this.maximized);
    const b = this.$("pl-max");
    b.replaceChildren(icon(this.maximized ? "min" : "max"));
    b.title = this.maximized ? "Leave full screen (F / Esc)" : "Full screen (F)";
    if (!this.maximized) this.placePanel();
    this.draw();
  }

  rememberGeometry() {
    const r = this.el.getBoundingClientRect();
    this.geometry = { left: r.left, top: r.top, width: r.width, height: r.height };
    this.save();
  }

  placePanel() {
    const g = this.geometry;
    const width = Math.min(g?.width || 940, window.innerWidth - 16);
    const height = Math.min(g?.height || 440, window.innerHeight - 16);
    const left = g ? Math.min(Math.max(8, g.left), window.innerWidth - width - 8) : 24;
    const top = g ? Math.min(Math.max(8, g.top), window.innerHeight - height - 8) : Math.max(8, window.innerHeight - height - 24);
    Object.assign(this.el.style, { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` });
  }

  save() {
    store.set("plots", {
      quantity: this.quantity, units: this.units, limits: this.showLimits, ref: this.showRef,
      windowS: this.windowS, geometry: this.geometry, picks: this.picks, collapsed: [...this.collapsed],
    });
  }

  // ---------------------------------------------------------------- open / source
  toggle() {
    if (this.open) this.close();
    else this.show();
  }

  show() {
    this.open = true;
    this.el.hidden = false;
    this.placePanel();
    this.app.onPlotsToggle?.(true);
    this.key = null; // reload what the stage shows
    this.sync(this.app.hud, this.app.state.owner);
    // no focus grab: the Studio's single-key shortcuts keep working while the window is open
  }

  close() {
    this.open = false;
    this.el.hidden = true;
    clearTimeout(this.idleTimer);
    this.idleTimer = null;
    if (this.maximized) this.setMaximized(false);
    this.app.onPlotsToggle?.(false);
  }

  /** Open on given channels of the Data clip (a QA issue): `{channels: [ids], frame, quantity}`
   *  plots them with their limits, in that quantity, and seeks the clip to `frame`. */
  focusOn(req) {
    this.pending = req;
    if (!this.open) this.show();
    else if (this.source === "clip" && this.meta && !this.loading) this.applyPending();
  }

  applyPending() {
    const req = this.pending;
    this.pending = null;
    if (!req || !this.meta) return;
    const index = new Map(this.meta.channels.map((c, i) => [c.id, i]));
    const picks = (req.channels || []).map((id) => index.get(id)).filter((i) => i !== undefined);
    if (!picks.length) return;
    for (const i of picks) this.collapsed.delete(this.meta.channels[i].group);
    this.query = "";
    this.$("pl-search").value = "";
    this.selected = [];
    this.slots = new Map();
    for (const i of picks) this.addPick(i, { quiet: true });
    const set = (id, key, on) => {
      this[key] = on;
      this.$(id).setAttribute("aria-pressed", String(on));
    };
    set("pl-limits", "showLimits", true);
    set("pl-error", "errorMode", false);
    const quantity = req.quantity || "value";
    if (this.kinematic()) {
      this.quantity = quantity;
      for (const b of this.$("pl-quantity").querySelectorAll("button")) b.setAttribute("aria-checked", String(b.dataset.v === quantity));
    }
    this.note("");
    const f = req.frame;
    if (Number.isInteger(f) && this.view && (f < this.view[0] || f > this.view[1])) {
      const span = this.view[1] - this.view[0];
      this.setView(f - span / 2, f + span / 2);
    }
    this.afterPick();
    this.$("pl-list").querySelector(`[data-ch="${picks[0]}"]`)?.scrollIntoView({ block: "nearest" });
    if (Number.isInteger(f)) {
      post("/api/data/playing", { playing: false })
        .then(() => post("/api/data/frame", { frame: f })).catch(() => {});
    }
  }

  /** Every HUD frame: follow the stage (the Data clip or the Play session) and its playhead. */
  sync(hud, owner) {
    if (!this.open) return;
    const source = hud.owner === "data" && !hud.grid ? "clip" : hud.owner === "play" ? "play" : null;
    if (source !== this.source) {
      this.source = source;
      this.key = null;
      this.reset();
      this.renderSource();
    }
    if (source === "clip") {
      const key = `${hud.title}|${hud.kind}|${hud.frames}`;
      if (key !== this.key && !this.loading) this.loadClip(key);
      else if (this.data) {
        this.followPlayhead();
        this.requestDraw();
      }
    } else if (source === "play") {
      // the stage frame that brought this HUD also means new steps or another focused env
      if (this.meta && (hud.focus ?? 0) !== this.env) this.key = null;
      this.pull();
    } else {
      this.renderEmpty(owner === "data" && hud.grid ? "Open one clip (not a grid) to plot it." : "Open a clip in Data, or start a Play session.");
    }
  }

  renderSource() {
    const play = this.source === "play";
    this.$("pl-live-box").hidden = !play;
    this.$("pl-help").textContent = play ? HELP.replace("scrub", "cursor") : HELP;
    this.renderLive();
  }

  reset() {
    this.meta = this.data = this.ref = this.cuts = this.refCols = null;
    this.env = null;
    this.subtitle = "";
    this.rows = this.C = this.start = 0;
    this.markers = [];
    this.selected = [];
    this.slots = new Map();
    this.cursor = null;
    this.hover = null;
    this.live = true;
    this.renderList();
    this.renderLegend();
    this.draw();
  }

  renderEmpty(text) {
    if (this.emptyText === text && !this.$("pl-empty").hidden) return;
    this.emptyText = text;
    this.$("pl-subtitle").textContent = "";
    this.$("pl-empty").textContent = text;
    this.$("pl-empty").hidden = false;
    this.renderList();
    this.draw();
  }

  async loadClip(key) {
    this.loading = true;
    this.$("pl-subtitle").textContent = "loading…";
    try {
      const res = await fetch("/api/plots/clip");
      const info = await res.json();
      if (!res.ok) throw new Error(info.detail || res.statusText);
      if (this.source !== "clip") return;
      this.key = key;
      this.setChannels(info);
      this.fps = info.fps;
      this.start = 0;
      this.data = decode(info.data);
      this.ref = info.ref ? decode(info.ref) : null;
      this.rows = info.frames;
      this.cuts = null;
      this.markers = [];
      this.scanRef(0);
      this.view = [0, Math.max(1, this.rows - 1)];
      this.yRange = null;
      this.restorePicks();
      this.$("pl-subtitle").textContent = `${info.title} · ${info.frames} frames @ ${info.fps.toFixed(0)} Hz${this.ref ? ` · dashed: ${info.ref_label}` : ""}`;
    } catch (err) {
      this.key = key;
      this.pending = null;
      this.renderEmpty(`Can't plot this clip: ${err.message}`);
      return;
    } finally {
      this.loading = false;
    }
    this.renderList();
    this.renderLegend();
    this.draw();
    if (this.pending) this.applyPending();
  }

  setChannels(info) {
    this.meta = info;
    this.C = info.channels.length;
    this.hasRef = new Array(this.C).fill(false);
    this.wrapCols = info.channels.map((c, i) => (c.wrap ? i : -1)).filter((i) => i >= 0);
  }

  /** Whether each channel has any reference value (rows from `from` on). */
  scanRef(from) {
    if (!this.ref) return;
    const C = this.C;
    const cols = (this.source === "play" && this.refCols) || [...Array(C).keys()];
    for (const i of cols) {
      if (this.hasRef[i]) continue;
      for (let f = from; f < this.rows; f += 1) {
        if (Number.isFinite(this.ref[f * C + i])) {
          this.hasRef[i] = true;
          break;
        }
      }
    }
  }

  // ---------------------------------------------------------------- Play: pulling
  /** New rows of the focused env: driven by stage frames (each brings steps or a focus change), at most
   *  one request in flight and one per PULL_MS, plus an idle timer when no frames come. */
  async pull() {
    if (!this.open || this.source !== "play") return;
    const now = performance.now();
    if (this.pulling || now - (this.pulledAt || 0) < PULL_MS) {
      if (!this.pulling && !this.pullTimer) {
        this.pullTimer = setTimeout(() => { this.pullTimer = null; this.pull(); }, PULL_MS);
      }
      return;
    }
    this.pulling = true;
    this.pulledAt = now;
    clearTimeout(this.idleTimer);
    try {
      const focus = this.app.hud.focus ?? 0;
      const fresh = !this.meta || this.key === null;
      const since = fresh ? 0 : this.start + this.rows;
      const epoch = fresh ? "" : `&epoch=${this.key}`;
      const got = await fetchRows(`/api/plots/play?env=${focus}&since=${since}${epoch}`);
      if (this.source === "play" && this.open) this.onPlayRows(got);
    } catch (err) {
      if (this.source === "play" && !this.meta) this.renderEmpty(String(err.message || err));
    } finally {
      this.pulling = false;
      if (this.open && this.source === "play") this.idleTimer = setTimeout(() => this.pull(), IDLE_MS);
    }
  }

  onPlayRows({ head: info, data, ref, meta }) {
    const n = info.end - info.start;
    if (info.fresh) {
      this.key = info.epoch;
      this.env = info.env;
      this.setChannels(info);
      this.refCols = info.ref_cols;
      this.fps = info.fps;
      this.cap = Math.round(info.history_s * info.fps);
      // room for a quarter more than the server keeps: old rows are dropped in batches, not every pull
      this.room = this.cap + Math.max(64, Math.ceil(this.cap / 4));
      this.data = new Float32Array(this.room * this.C);
      this.ref = new Float32Array(this.room * this.C).fill(NaN);
      this.metaRows = new Float32Array(this.room * 3);
      this.cuts = new Uint8Array(this.room);
      this.rows = 0;
      this.start = info.start;
      this.markers = [];
      const C = this.C;
      this.unwrap = { data: new Float64Array(C), ref: new Float64Array(C), lastData: new Float64Array(C).fill(NaN), lastRef: new Float64Array(C).fill(NaN) };
      this.cursor = null;
      this.hover = null;
      this.view = null;
      this.yRange = null;
      this.restorePicks(); // the remembered picks: the same channels for every env
    }
    if (n > 0) this.append(data, ref, meta, n);
    this.emptyText = "";
    const now = performance.now();
    if (info.fresh) {
      this.statsAt = now;
      this.renderList();
      this.renderLegend();
    } else if (n > 0 && now - (this.statsAt || 0) > STATS_MS) {
      // channels gain their "ref" tag with the first rows; the RMSE follows the live view
      this.statsAt = now;
      if (this.refTagged !== this.hasRef.filter(Boolean).length) this.renderList();
      this.updateStats();
    }
    const t = (this.start + this.rows) / (this.fps || 50);
    const sub = `env ${info.env} · ${this.clipAt(this.rows - 1) || "—"} · ${t.toFixed(1)} s recorded (last ${Math.round(this.cap / this.fps)} s kept)`;
    if (sub !== this.subtitle) this.$("pl-subtitle").textContent = this.subtitle = sub;
    if (n > 0 || info.fresh) {
      if (this.live || !this.view) this.followLive();
      this.draw();
    }
  }

  append(d, r, m, n) {
    const C = this.C;
    if (this.rows + n > this.room) {
      // the oldest rows fall out (down to the server's horizon, so this happens once in a while)
      const drop = Math.min(this.rows, this.rows + n - this.cap);
      this.data.copyWithin(0, drop * C, this.rows * C);
      this.ref.copyWithin(0, drop * C, this.rows * C);
      this.metaRows.copyWithin(0, drop * 3, this.rows * 3);
      this.cuts.copyWithin(0, drop, this.rows);
      this.rows -= drop;
      this.start += drop;
      this.markers = this.markers.map((k) => ({ ...k, f: k.f - drop })).filter((k) => k.f >= 0);
      if (this.view) this.view = [Math.max(0, this.view[0] - drop), Math.max(1, this.view[1] - drop)];
      if (this.cursor !== null) this.cursor = this.cursor - drop >= 0 ? this.cursor - drop : null;
      if (this.hover !== null) this.hover = this.hover - drop >= 0 ? this.hover - drop : null;
    }
    const keep = Math.min(n, this.room);
    const off = n - keep;
    const R = this.refCols.length;
    const from = this.rows;
    this.data.set(d.subarray(off * C, n * C), from * C);
    this.metaRows.set(m.subarray(off * 3, n * 3), from * 3);
    const u = this.unwrap;
    const clips = this.meta.clips || [];
    for (let k = 0; k < keep; k += 1) {
      const f = from + k;
      const src = (off + k) * R;
      for (let j = 0; j < R; j += 1) this.ref[f * C + this.refCols[j]] = r[src + j];
      // a new episode (episode step went back) or another clip: a cut + a marker
      const clip = this.metaRows[f * 3];
      const cut = f === 0 || clip !== this.metaRows[(f - 1) * 3] || this.metaRows[f * 3 + 2] < this.metaRows[(f - 1) * 3 + 2];
      this.cuts[f] = cut ? 1 : 0;
      if (cut) this.markers.push({ f, label: short(clips[clip] || `clip ${clip}`) });
      // unwrap roll / pitch / yaw within a segment (reset at cuts)
      for (const i of this.wrapCols) {
        this.unwrapAt(this.data, u.data, u.lastData, f * C + i, i, cut);
        this.unwrapAt(this.ref, u.ref, u.lastRef, f * C + i, i, cut);
      }
    }
    this.rows += keep;
    this.scanRef(from);
    if (!this.selected.length) this.restorePicks();
  }

  unwrapAt(arr, turns, last, at, i, cut) {
    const v = arr[at];
    if (!Number.isFinite(v)) return;
    if (cut || !Number.isFinite(last[i])) turns[i] = 0;
    else {
      const dv = v - last[i];
      if (dv > Math.PI) turns[i] -= 2 * Math.PI;
      else if (dv < -Math.PI) turns[i] += 2 * Math.PI;
    }
    last[i] = v;
    arr[at] = v + turns[i];
  }

  clipAt(f) {
    if (!this.metaRows || f < 0 || f >= this.rows) return "";
    const clips = this.meta.clips || [];
    return clips[this.metaRows[f * 3]] || "";
  }

  setLive(on) {
    this.live = Boolean(on);
    if (this.live) {
      this.cursor = null;
      this.yRange = null;
      this.followLive();
    }
    this.renderLive();
    this.draw();
  }

  renderLive() {
    const b = this.$("pl-live");
    b.setAttribute("aria-pressed", String(this.live));
    b.title = this.live ? "Following the newest steps (click to hold the view)" : "Follow the newest steps again";
  }

  followLive() {
    if (this.rows < 2) {
      this.view = [0, 1];
      return;
    }
    const span = Math.min(this.rows - 1, Math.max(8, this.windowS * this.fps));
    this.view = [this.rows - 1 - span, this.rows - 1];
  }

  // ---------------------------------------------------------------- selection
  storeKey() {
    return `${this.source}:${this.app.state.robot}`;
  }

  restorePicks() {
    if (!this.meta) return;
    const index = new Map(this.meta.channels.map((c, i) => [c.id, i]));
    let picks = (this.picks[this.storeKey()] || []).map((id) => index.get(id)).filter((i) => i !== undefined);
    if (!picks.length) {
      // a first look: the knees (with their reference in Play), else the first two joints
      const joints = this.meta.channels.map((c, i) => (/(Left|Right) leg/.test(c.group) ? i : -1)).filter((i) => i >= 0);
      picks = joints.filter((i) => /knee/i.test(this.meta.channels[i].name)).slice(0, 2);
      if (!picks.length) picks = joints.slice(0, 2);
    }
    this.selected = [];
    this.slots = new Map();
    for (const i of picks) this.addPick(i, { quiet: true });
  }

  unitOf(i) {
    return this.meta.channels[i].unit;
  }

  freeSlot() {
    const used = new Set(this.slots.values());
    for (let s = 0; s < MAX; s += 1) if (!used.has(s)) return s;
    return -1;
  }

  addPick(i, { quiet = false } = {}) {
    if (this.selected.includes(i)) return;
    if (this.selected.length && this.unitOf(this.selected[0]) !== this.unitOf(i)) {
      // one unit on the one axis: another unit replaces the plot, never a second y-axis
      if (!quiet) this.note(`${short(this.meta.channels[i].name)} is in ${WHAT[this.unitOf(i)] || this.unitOf(i)}: the ${WHAT[this.unitOf(this.selected[0])] || "previous"} channels were cleared (one unit per plot).`);
      this.selected = [];
      this.slots = new Map();
    }
    if (this.selected.length >= MAX) {
      if (!quiet) this.note(`${MAX} channels at most: remove one first.`, true);
      return;
    }
    this.selected.push(i);
    this.slots.set(i, this.freeSlot());
  }

  togglePick(i) {
    if (!this.meta) return;
    if (this.selected.includes(i)) {
      this.selected = this.selected.filter((x) => x !== i);
      this.slots.delete(i);
      this.note("");
    } else this.addPick(i);
    this.afterPick();
  }

  pickGroup(group) {
    if (!this.meta) return;
    let members = this.meta.channels.map((c, i) => (c.group === group ? i : -1)).filter((i) => i >= 0);
    if (members.length > MAX && this.rows && /^(reward|raw)$/.test(this.meta.channels[members[0]].unit)) {
      // rewards: the largest contributors over the record (mean |value|), not the definition order
      const size = (i) => {
        let s = 0;
        for (let f = 0; f < this.rows; f += 1) s += Math.abs(this.data[f * this.C + i]) || 0;
        return s;
      };
      members = members.map((i) => [i, size(i)]).sort((x, y) => y[1] - x[1]).map(([i]) => i);
    }
    this.selected = [];
    this.slots = new Map();
    for (const i of members.slice(0, MAX)) this.addPick(i, { quiet: true });
    const rewards = /^(reward|raw)$/.test(this.meta.channels[members[0]]?.unit);
    this.note(members.length > MAX ? `${group}: ${rewards ? "the largest" : "the first"} ${MAX} of ${members.length}.` : "");
    this.afterPick();
  }

  clearPicks() {
    this.selected = [];
    this.slots = new Map();
    this.note("");
    this.afterPick();
  }

  afterPick() {
    this.yRange = null;
    if (this.meta) {
      this.picks[this.storeKey()] = this.selected.map((i) => this.meta.channels[i].id);
      this.save();
    }
    this.renderList();
    this.renderLegend();
    this.draw();
  }

  toggleGroup(group) {
    if (this.collapsed.has(group)) this.collapsed.delete(group);
    else this.collapsed.add(group);
    this.save();
    this.renderList();
  }

  note(text, warn = false) {
    const el = this.$("pl-note");
    el.textContent = text || `Pick up to ${MAX} channels.`;
    el.classList.toggle("is-warn", Boolean(text) && warn);
  }

  color(i) {
    return (PALETTE[getTheme()] || PALETTE.dark)[this.slots.get(i) ?? 0];
  }

  // ---------------------------------------------------------------- list + legend
  visibleRows() {
    if (!this.meta) return [];
    const terms = this.query.split(/\s+/).filter(Boolean);
    return this.meta.channels
      .map((c, i) => ({ c, i }))
      .filter(({ c }) => terms.every((t) => `${c.name} ${c.group}`.toLowerCase().replace(/_/g, " ").includes(t.replace(/_/g, " "))))
      .map(({ i }) => i);
  }

  renderList() {
    const list = this.$("pl-list");
    list.replaceChildren();
    if (!this.meta) return;
    this.refTagged = this.hasRef.filter(Boolean).length;
    const rows = this.visibleRows();
    if (!rows.length) {
      list.append(h("div.pl-list-empty", "No channel matches the search."));
      return;
    }
    const groups = new Map();
    for (const i of rows) {
      const g = this.meta.channels[i].group;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(i);
    }
    for (const [group, members] of groups) {
      const closed = this.collapsed.has(group) && !this.query;
      list.append(h("div.pl-group", { "data-group": group, title: closed ? "Show" : "Hide" },
        h("span.pl-caret", closed ? "▸" : "▾"), h("span.pl-group-name", group), h("small", String(members.length)),
        h("button.pl-all", { type: "button", "data-plot-group": group, title: `Plot ${group} (first ${MAX})` }, "plot")));
      if (closed) continue;
      for (const i of members) {
        const c = this.meta.channels[i];
        const on = this.selected.includes(i);
        const swatch = h("span.pl-swatch");
        if (on) swatch.style.background = this.color(i);
        list.append(h(`div.pl-row${on ? ".is-on" : ""}`, { "data-ch": String(i), role: "option", "aria-selected": String(on), title: this.limitsOf(i) ? `${c.name} · limits ${this.limitText(i)}` : c.name },
          swatch, h("span.pl-name", short(c.name)), this.hasRef[i] ? h("small.pl-has-ref", { title: "has a reference" }, "ref") : null));
      }
    }
  }

  setListHover(i) {
    if (this.listHover === i) return;
    this.listHover = i;
    this.draw();
  }

  renderLegend() {
    const legend = this.$("pl-legend");
    legend.replaceChildren();
    const kin = this.kinematic();
    for (const b of this.$("pl-quantity").querySelectorAll("button")) b.disabled = !kin;
    for (const b of this.$("pl-units").querySelectorAll("button")) b.disabled = !this.selected.length || this.unitOf(this.selected[0]) !== "rad";
    const anyRef = this.selected.some((i) => this.hasRef[i]);
    this.$("pl-ref").disabled = !anyRef || this.errorMode;
    this.$("pl-error").disabled = !anyRef;
    this.$("pl-limits").disabled = !(this.quantity === "value" || !kin) || this.errorMode || !this.anyLimits();
    if (!this.meta) return;
    const stats = this.rmse();
    for (const i of this.selected) {
      const err = stats.get(i);
      legend.append(h("button.pl-chip", { type: "button", "data-ch": String(i), title: `Remove ${this.meta.channels[i].name}` },
        h("span.pl-key", { style: { background: this.color(i) } }),
        h("span", short(this.meta.channels[i].name)),
        this.hasRef[i] ? h("em", { "data-rmse": String(i), title: "RMSE actual vs reference over the view" }, err !== undefined ? `RMSE ${num(err)} ${this.unitLabel()}` : "") : null,
        h("span.pl-x", { "aria-hidden": "true" }, "×")));
    }
    if (this.selected.length) {
      if (anyRef && this.showRef && !this.errorMode) {
        legend.append(h("span.pl-compare", h("i.solid"), this.source === "play" ? "robot" : "clip", h("i.dashed"), this.meta.ref_label || "reference"));
      }
      if (this.errorMode) legend.append(h("span.pl-compare", `actual − ${this.meta.ref_label || "reference"}`));
      legend.append(h("button.pl-clear", { type: "button", "data-clear": "1" }, "clear"));
    }
  }

  /** The legend's RMSE figures, in place (no rebuild while live). */
  updateStats() {
    const ems = this.$("pl-legend").querySelectorAll("em[data-rmse]");
    if (ems.length !== this.selected.filter((i) => this.hasRef[i]).length) {
      this.renderLegend();
      return;
    }
    const stats = this.rmse();
    const unit = this.unitLabel();
    for (const em of ems) {
      const err = stats.get(Number(em.dataset.rmse));
      const text = err !== undefined ? `RMSE ${num(err)} ${unit}` : "";
      if (em.textContent !== text) em.textContent = text;
    }
  }

  // ---------------------------------------------------------------- values
  kinematic() {
    return this.selected.length > 0 && KINEMATIC.has(this.unitOf(this.selected[0]));
  }

  scale(i) {
    return this.unitOf(i) === "rad" && this.units === "deg" ? 180 / Math.PI : 1;
  }

  unitLabel() {
    if (!this.selected.length) return "";
    const u = this.unitOf(this.selected[0]);
    const base = u === "rad" ? (this.units === "deg" ? "°" : "rad") : u === "reward" || u === "raw" ? "" : u;
    return `${base}${this.kinematic() ? QUANTITY[this.quantity] : ""}`;
  }

  /** Neighbour row of f in direction dir inside f's segment (no differences across episode cuts). */
  nb(f, dir) {
    const j = f + dir;
    if (j < 0 || j >= this.rows) return f;
    if (this.cuts && ((dir > 0 && this.cuts[j]) || (dir < 0 && this.cuts[f]))) return f;
    return j;
  }

  /** Channel i at row f of `arr` in the chosen quantity and unit. */
  raw(arr, i, f) {
    const C = this.C;
    const at = (k) => arr[k * C + i];
    const q = this.kinematic() ? this.quantity : "value";
    let v;
    if (q === "value") v = at(f);
    else {
      const a = this.nb(f, -1);
      const b = this.nb(f, 1);
      if (q === "vel") v = b === a ? NaN : ((at(b) - at(a)) / (b - a)) * this.fps;
      else v = b - f === 1 && f - a === 1 ? (at(b) - 2 * at(f) + at(a)) * this.fps * this.fps : NaN;
    }
    return v * this.scale(i);
  }

  /** What is drawn for channel i at row f: the series (`ref` for its reference), or actual − reference. */
  value(i, f, which = "data") {
    if (this.errorMode) {
      if (which === "ref" || !this.hasRef[i]) return NaN;
      return this.raw(this.data, i, f) - this.raw(this.ref, i, f);
    }
    if (which === "ref") return this.ref && this.hasRef[i] ? this.raw(this.ref, i, f) : NaN;
    return this.raw(this.data, i, f);
  }

  /** RMSE of actual vs reference over the visible rows, per plotted channel that has a reference. */
  rmse() {
    const out = new Map();
    if (!this.view || !this.ref || this.rows < 2) return out;
    const a = Math.max(0, Math.floor(this.view[0]));
    const b = Math.min(this.rows - 1, Math.ceil(this.view[1]));
    for (const i of this.selected) {
      if (!this.hasRef[i]) continue;
      let s = 0;
      let n = 0;
      for (let f = a; f <= b; f += 1) {
        const d = this.raw(this.data, i, f) - this.raw(this.ref, i, f);
        if (Number.isFinite(d)) { s += d * d; n += 1; }
      }
      if (n) out.set(i, Math.sqrt(s / n));
    }
    return out;
  }

  limitsOf(i) {
    const c = this.meta.channels[i];
    if (c.lower === null || c.lower === undefined || c.upper === null || c.upper === undefined) return null;
    const k = this.scale(i);
    return [c.lower * k, c.upper * k];
  }

  /** Limits apply to values: joint angles (and root / feet positions with limits) in Pos, torques
   *  (effort limits from the model or the actuators) always. */
  limitsOn() {
    return this.showLimits && !this.errorMode && (!this.kinematic() || this.quantity === "value") && this.anyLimits();
  }

  anyLimits() {
    return Boolean(this.meta) && this.selected.some((i) => this.limitsOf(i));
  }

  /** "±88 Nm, actuator" for a channel's list tooltip and the shaded label. */
  limitText(i) {
    const c = this.meta.channels[i];
    const lim = this.limitsOf(i);
    if (!lim) return "";
    const unit = this.unitOf(i) === "rad" ? (this.units === "deg" ? "°" : " rad") : ` ${this.unitOf(i)}`;
    const range = Math.abs(lim[0] + lim[1]) < 1e-6 * Math.abs(lim[1]) ? `±${num(lim[1])}${unit}` : `${num(lim[0])} … ${num(lim[1])}${unit}`;
    return c.limit_note ? `${range}, ${c.limit_note}` : range;
  }

  // ---------------------------------------------------------------- plot interaction
  bindPlot() {
    const canvas = this.$("pl-canvas");
    let press = null;
    const pointAt = (x) => {
      const f = Math.round(this.xToFrame(x, this.layout()));
      if (this.source === "clip") {
        if (this.app.hud.playing) post("/api/data/playing", { playing: false }).catch(() => {});
        const now = performance.now();
        if (now - (this._seekAt || 0) > 40) {
          this._seekAt = now;
          post("/api/data/frame", { frame: f }).catch(() => {});
        }
      } else {
        this.cursor = f;
        if (this.live) {
          this.live = false;
          this.renderLive();
        }
      }
    };
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.ready() || e.button !== 0) return;
      canvas.setPointerCapture(e.pointerId);
      const mode = e.shiftKey ? "pan" : e.ctrlKey || e.metaKey || this.zoomMode ? "box" : "scrub";
      press = { mode, x: e.clientX, view: [...this.view] };
      if (mode === "scrub") pointAt(e.offsetX);
      else if (mode === "box") this.box = { x0: e.offsetX, y0: e.offsetY, x1: e.offsetX, y1: e.offsetY };
      if (mode !== "scrub") this.hold();
      canvas.classList.toggle("is-panning", mode === "pan");
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.ready()) return;
      const L = this.layout();
      if (press?.mode === "pan") {
        const span = press.view[1] - press.view[0];
        const shift = (-(e.clientX - press.x) / Math.max(1, L.w)) * span;
        this.setView(press.view[0] + shift, press.view[1] + shift);
      } else if (press?.mode === "scrub") pointAt(e.offsetX);
      else if (press?.mode === "box" && this.box) Object.assign(this.box, { x1: e.offsetX, y1: e.offsetY });
      this.hover = press?.mode === "box" ? null : Math.round(this.xToFrame(e.offsetX, L));
      this.requestDraw();
    });
    const release = (e) => {
      if (press?.mode === "box" && this.box && e?.type === "pointerup") this.applyBox();
      this.box = null;
      press = null;
      canvas.classList.remove("is-panning");
      this.renderLegend();
      this.requestDraw();
    };
    canvas.addEventListener("pointerup", release);
    canvas.addEventListener("pointercancel", release);
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.requestDraw();
    });
    canvas.addEventListener("wheel", (e) => {
      if (!this.ready()) return;
      e.preventDefault();
      this.hold();
      const L = this.layout();
      const [a, b] = this.view;
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
        const shift = ((e.shiftKey ? e.deltaY : e.deltaX) / Math.max(1, L.w)) * (b - a);
        this.setView(a + shift, b + shift);
      } else {
        const at = this.xToFrame(e.offsetX, L);
        const k = Math.exp(e.deltaY * 0.0015);
        this.setView(at - (at - a) * k, at + (b - at) * k);
      }
      this.renderLegend();
      this.requestDraw();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => {
      if (!this.ready()) return;
      this.hold();
      this.view = [0, Math.max(1, this.rows - 1)];
      this.yRange = null;
      this.renderLegend();
      this.requestDraw();
    });
  }

  /** Play: stop following the newest steps (the user is looking at a part of the record). */
  hold() {
    if (this.source === "play" && this.live) {
      this.live = false;
      this.renderLive();
    }
  }

  setZoomMode(on) {
    this.zoomMode = Boolean(on);
    this.$("pl-zoom").setAttribute("aria-pressed", String(this.zoomMode));
    this.$("pl-canvas").classList.toggle("is-zoom", this.zoomMode);
    this.$("pl-help").textContent = this.zoomMode ? "drag a box to zoom · double-click: fit · Zoom again to scrub" : HELP;
  }

  applyBox() {
    const L = this.layout();
    const box = this.box;
    const x0 = Math.max(PAD.left, Math.min(box.x0, box.x1));
    const x1 = Math.min(PAD.left + L.w, Math.max(box.x0, box.x1));
    const y0 = Math.max(PAD.top, Math.min(box.y0, box.y1));
    const y1 = Math.min(PAD.top + L.h, Math.max(box.y0, box.y1));
    if (x1 - x0 < 6) return;
    this.setView(this.xToFrame(x0, L), this.xToFrame(x1, L));
    if (y1 - y0 >= 24 && this.lastY) {
      const { lo, hi } = this.lastY;
      const value = (y) => lo + (1 - (y - PAD.top) / L.h) * (hi - lo);
      this.yRange = [value(y1), value(y0)];
    }
  }

  /** Data: while zoomed in and playing, keep the playhead in view (the window pages along). */
  followPlayhead() {
    if (this.source !== "clip" || !this.view || !this.app.hud.playing) return;
    const f = this.app.hud.frame ?? 0;
    const [a, b] = this.view;
    const span = b - a;
    if (span >= this.rows - 1.5) return;
    if (f < a || f > b) this.setView(f - span * 0.1, f - span * 0.1 + span);
  }

  setView(a, b) {
    const last = Math.max(1, this.rows - 1);
    let span = Math.max(Math.min(8, last), Math.min(last, b - a));
    const lo = Math.max(0, Math.min(a, last - span));
    span = Math.min(span, last - lo);
    this.view = [lo, lo + span];
  }

  layout() {
    const canvas = this.$("pl-canvas");
    if (!this.plotBox || !this.plotBox.width) {
      const r = this.$("pl-plot").getBoundingClientRect(); // before the first ResizeObserver report
      this.plotBox = { width: r.width, height: r.height };
    }
    const box = this.plotBox;
    const labels = this.selected.length && this.selected.length <= 4 ? LABEL_W : 0;
    const w = Math.max(10, box.width - PAD.left - PAD.right - labels);
    const hgt = Math.max(10, box.height - PAD.top - PAD.bottom);
    return { canvas, width: box.width, height: box.height, w, h: hgt, labels, dpr: window.devicePixelRatio || 1 };
  }

  frameToX(f, L) {
    const [a, b] = this.view;
    return PAD.left + ((f - a) / Math.max(1e-9, b - a)) * L.w;
  }

  xToFrame(x, L) {
    const [a, b] = this.view;
    return Math.max(0, Math.min(this.rows - 1, a + ((x - PAD.left) / Math.max(1, L.w)) * (b - a)));
  }

  ready() {
    return Boolean(this.meta && this.data && this.rows >= 2 && this.view);
  }

  /** One paint per animation frame, however many pulls, HUD frames and pointer moves asked for it. */
  draw() {
    if (!this.open || this.pending) return;
    this.pending = requestAnimationFrame(() => {
      this.pending = null;
      this.paint();
    });
  }

  requestDraw() {
    this.draw();
  }

  /** A theme colour (custom property), read once per theme. */
  tok(name) {
    let v = this.tokens.get(name);
    if (v === undefined) {
      v = token(name);
      this.tokens.set(name, v);
    }
    return v;
  }

  /** The row the playhead marks: the clip frame (Data), the pinned cursor (Play), else none. */
  playhead() {
    if (this.source === "clip") return this.app.hud.frame ?? null;
    return this.cursor;
  }

  // ---------------------------------------------------------------- draw
  paint() {
    if (!this.open) return;
    const L = this.layout();
    const { canvas, dpr } = L;
    const size = `${Math.round(L.width * dpr)}x${Math.round(L.height * dpr)}`;
    if (size !== this.canvasSize) {
      // resizing reallocates the backing store: only when the box (or the screen's dpr) changed
      this.canvasSize = size;
      canvas.width = Math.round(L.width * dpr);
      canvas.height = Math.round(L.height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, L.width, L.height);
    const ready = this.ready() && this.selected.length;
    const empty = this.$("pl-empty");
    if (!ready) {
      if (this.meta) {
        empty.textContent = this.rows < 2 ? "Waiting for steps — press play." : "Pick channels on the left to plot them.";
      } else if (!empty.textContent || this.source) {
        empty.textContent = this.source ? "Loading…" : "Open a clip in Data, or start a Play session.";
      }
      empty.hidden = false;
      this.$("pl-tip").hidden = true;
      this.$("pl-status").textContent = "";
      return;
    }
    empty.hidden = true;
    const ink3 = this.tok("--color-text-muted");
    const faint = this.tok("--color-text-faint");
    const grid = this.tok("--color-border");
    const [f0, f1] = this.view;
    const a = Math.max(0, Math.floor(f0));
    const b = Math.min(this.rows - 1, Math.ceil(f1));
    const refOn = this.showRef && !this.errorMode && this.ref;
    // y range over the visible rows of every plotted series (and its reference), plus limits
    let lo = Infinity;
    let hi = -Infinity;
    const stride = Math.max(1, Math.floor((b - a) / (L.w * 2)));
    for (const i of this.selected) {
      for (const which of refOn ? ["data", "ref"] : ["data"]) {
        for (let f = a; f <= b; f += stride) {
          const v = this.value(i, f, which);
          if (!Number.isFinite(v)) continue;
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    const limitsOn = this.limitsOn();
    const focus = limitsOn ? this.limitChannel() : null;
    if (limitsOn) {
      for (const i of focus === null || this.selected.includes(focus) ? this.selected : [...this.selected, focus]) {
        const lim = this.limitsOf(i);
        if (!lim) continue;
        lo = Math.min(lo, lim[0]);
        hi = Math.max(hi, lim[1]);
      }
    }
    if (this.errorMode) {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    if (!Number.isFinite(lo)) [lo, hi] = [-1, 1];
    if (this.yRange) [lo, hi] = this.yRange;
    if (hi - lo < 1e-6) [lo, hi] = [lo - 0.5, hi + 0.5];
    const padY = this.yRange ? 0 : (hi - lo) * 0.06;
    lo -= padY;
    hi += padY;
    const yOf = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * L.h;
    this.lastY = { lo, hi };
    // grid + y axis (recessive hairlines)
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    const yt = niceTicks(lo, hi, Math.max(3, Math.round(L.h / 48)));
    ctx.lineWidth = 1;
    for (const v of yt.ticks) {
      const y = Math.round(yOf(v)) + 0.5;
      ctx.strokeStyle = grid;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(PAD.left + L.w, y);
      ctx.stroke();
      ctx.fillStyle = ink3;
      ctx.fillText(fmt(v, yt.step), PAD.left - 6, y);
    }
    ctx.save();
    ctx.translate(11, PAD.top + L.h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    const u = this.unitOf(this.selected[0]);
    const what = this.kinematic() ? { value: WHAT[u], vel: "velocity", acc: "acceleration" }[this.quantity] : WHAT[u] ?? u;
    const unit = this.unitLabel();
    ctx.fillText(`${this.errorMode ? "error: " : ""}${what}${unit ? ` (${unit})` : ""}`, 0, 0);
    ctx.restore();
    // x axis in seconds (Play: since the record began)
    const fps = this.fps;
    const s0 = (this.start + f0) / fps;
    const s1 = (this.start + f1) / fps;
    const xt = niceTicks(s0, s1, Math.max(3, Math.round(L.w / 90)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const s of xt.ticks) {
      const x = this.frameToX(s * fps - this.start, L);
      if (x < PAD.left - 1 || x > PAD.left + L.w + 1) continue;
      ctx.fillStyle = ink3;
      ctx.fillText(`${fmt(s, xt.step)} s`, x, PAD.top + L.h + 7);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.left, PAD.top, L.w, L.h);
    ctx.clip();
    // episode / clip changes (Play): a faint dashed line, labelled when there is room
    if (this.markers.length) {
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      let lastLabel = -Infinity;
      for (const m of this.markers) {
        if (m.f < a - 1 || m.f > b + 1) continue;
        const x = Math.round(this.frameToX(m.f, L)) + 0.5;
        ctx.strokeStyle = faint;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, PAD.top + L.h);
        ctx.stroke();
        if (x - lastLabel > 90) {
          ctx.fillStyle = faint;
          let label = m.label;
          while (label.length > 4 && ctx.measureText(label).width > 84) label = `${label.slice(0, -2)}…`;
          ctx.fillText(label, x + 3, PAD.top + 2);
          lastLabel = x;
        }
      }
      ctx.setLineDash([]);
    }
    if (this.errorMode) {
      ctx.strokeStyle = ink3;
      ctx.lineWidth = 1;
      const y0 = Math.round(yOf(0)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y0);
      ctx.lineTo(PAD.left + L.w, y0);
      ctx.stroke();
    }
    if (limitsOn) {
      const focusLim = focus === null ? null : this.limitsOf(focus);
      if (focusLim) {
        const danger = this.tok("--color-danger");
        const yUp = yOf(focusLim[1]);
        const yLo = yOf(focusLim[0]);
        ctx.fillStyle = `${danger}1f`;
        ctx.fillRect(PAD.left, PAD.top, L.w, Math.max(0, yUp - PAD.top));
        ctx.fillRect(PAD.left, yLo, L.w, Math.max(0, PAD.top + L.h - yLo));
        ctx.fillStyle = danger;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText(`${short(this.meta.channels[focus].name)} limits (${this.limitText(focus)})`, PAD.left + 4, Math.max(PAD.top + 12, yUp - 2));
      }
      ctx.lineWidth = 1;
      for (const i of this.selected) {
        const lim = this.limitsOf(i);
        if (!lim) continue;
        ctx.globalAlpha = 0.75;
        ctx.strokeStyle = this.color(i);
        for (const v of lim) {
          const y = Math.round(yOf(v)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(PAD.left, y);
          ctx.lineTo(PAD.left + L.w, y);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }
    // series (gaps where a value is missing): the reference dashed under the actual one
    const step = Math.max(1, Math.floor((b - a) / (L.w * dpr)));
    const path = (i, which) => {
      ctx.beginPath();
      let pen = false;
      const visit = (f) => {
        const v = this.value(i, f, which);
        if (!Number.isFinite(v) || (this.cuts && this.cuts[f] && pen && f > a)) {
          pen = false;
          if (!Number.isFinite(v)) return;
        }
        const x = this.frameToX(f, L);
        const y = yOf(v);
        if (pen) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        pen = true;
      };
      for (let f = a; f <= b; f += step) visit(f);
      if ((b - a) % step) visit(b);
      ctx.stroke();
    };
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const emphasis = this.selected.includes(this.listHover) ? this.listHover : null;
    for (const i of this.selected) {
      ctx.globalAlpha = emphasis === null || emphasis === i ? 1 : 0.28;
      ctx.strokeStyle = this.color(i);
      if (refOn && this.hasRef[i]) {
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.5;
        path(i, "ref");
        ctx.setLineDash([]);
      }
      ctx.lineWidth = emphasis === i ? 2.5 : 2;
      path(i, "data");
    }
    ctx.globalAlpha = 1;
    // playhead (Data: the clip frame) / cursor (Play)
    const play = this.playhead();
    if (play !== null && play !== undefined) {
      const px = this.frameToX(play, L);
      if (px >= PAD.left && px <= PAD.left + L.w) {
        ctx.strokeStyle = this.tok("--color-accent");
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, PAD.top);
        ctx.lineTo(px, PAD.top + L.h);
        ctx.stroke();
      }
    }
    // crosshair
    if (this.hover !== null) {
      const hx = Math.round(this.frameToX(this.hover, L)) + 0.5;
      ctx.strokeStyle = this.tok("--color-text-muted");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, PAD.top);
      ctx.lineTo(hx, PAD.top + L.h);
      ctx.stroke();
      for (const i of this.selected) {
        const v = this.value(i, this.hover);
        if (!Number.isFinite(v)) continue;
        ctx.fillStyle = this.color(i);
        ctx.strokeStyle = this.tok("--color-panel");
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(hx, yOf(v), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
    ctx.restore();
    // direct labels at the line ends (<= 4 series): a line key in the series colour, the name in ink
    if (L.labels) {
      const lastFinite = (i) => {
        for (let f = b; f >= a; f -= 1) {
          const v = this.value(i, f);
          if (Number.isFinite(v)) return v;
        }
        return NaN;
      };
      const ends = this.selected.map((i) => ({ i, y: yOf(lastFinite(i)) })).filter((p) => Number.isFinite(p.y)).sort((p, q) => p.y - q.y);
      for (let k = 1; k < ends.length; k += 1) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 14);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      for (const { i, y } of ends) {
        const x = PAD.left + L.w + 8;
        ctx.strokeStyle = this.color(i);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 10, y);
        ctx.stroke();
        ctx.fillStyle = this.tok("--color-text");
        let name = short(this.meta.channels[i].name);
        while (name.length > 4 && ctx.measureText(name).width > L.labels - 20) name = `${name.slice(0, -2)}…`;
        ctx.fillText(name, x + 14, y);
      }
    }
    if (this.box) {
      const bx = Math.min(this.box.x0, this.box.x1);
      const by = Math.min(this.box.y0, this.box.y1);
      const bw = Math.abs(this.box.x1 - this.box.x0);
      const bh = Math.abs(this.box.y1 - this.box.y0);
      const accent = this.tok("--color-accent");
      ctx.fillStyle = `${accent}22`;
      ctx.fillRect(bx, by, bw, bh);
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(bx) + 0.5, Math.round(by) + 0.5, Math.round(bw), Math.round(bh));
    }
    this.renderTip(L);
    this.renderStatus(a, b, play);
  }

  renderStatus(a, b, play) {
    const fps = this.fps;
    const $ = this.$;
    const at = play !== null && play !== undefined ? play : this.hover;
    const bits = [];
    if (at !== null && at !== undefined) {
      bits.push(`${((this.start + at) / fps).toFixed(2)} s`);
      if (this.source === "clip") bits.push(`frame ${at}`);
      else if (this.metaRows) bits.push(`${short(this.clipAt(at))} frame ${this.metaRows[at * 3 + 1]}`);
    }
    bits.push(`showing ${((b - a) / fps).toFixed(1)} of ${((this.rows - 1) / fps).toFixed(1)} s`);
    if (this.limitsOn()) {
      // angles: frames past the joint limits; torques (clipped by the actuator): frames at the limit
      const torque = this.unitOf(this.selected[0]) === "Nm";
      const over = this.selected.map((i) => ({ i, n: this.framesOutside(i, torque) })).filter(({ n }) => n > 0)
        .map(({ i, n }) => `${short(this.meta.channels[i].name)} ${n}`);
      if (torque) bits.push(over.length ? `frames at the effort limit: ${over.join(", ")}` : "below the effort limits");
      else bits.push(over.length ? `frames outside limits: ${over.join(", ")}` : "all within limits");
    }
    const status = bits.join(" · ");
    if ($("pl-status").textContent !== status) $("pl-status").textContent = status;
    for (const [id, f] of [["pl-from", this.view[0]], ["pl-to", this.view[1]]]) {
      if (document.activeElement !== $(id)) $(id).value = ((this.start + f) / fps).toFixed(2);
    }
  }

  /** Rows of the whole record where channel i is outside its limits (`saturated`: at or past them,
   *  within 0.5 % of the range — a clipped torque sits exactly on its limit). */
  framesOutside(i, saturated = false) {
    const c = this.meta.channels[i];
    if (c.lower === null || c.lower === undefined) return 0;
    const tol = saturated ? -0.005 * (c.upper - c.lower) : 1e-4;
    let n = 0;
    for (let f = 0; f < this.rows; f += 1) {
      const v = this.data[f * this.C + i];
      if (v < c.lower - tol || v > c.upper + tol) n += 1;
    }
    return n;
  }

  /** The channel whose limits are shaded: the hovered list row, else the only plotted one. */
  limitChannel() {
    const pick = this.listHover >= 0 ? this.listHover : this.selected.length === 1 ? this.selected[0] : null;
    if (pick === null || pick < 0 || !this.meta) return null;
    if (this.selected.length && this.unitOf(pick) !== this.unitOf(this.selected[0])) return null;
    return this.meta.channels[pick].lower === null || this.meta.channels[pick].lower === undefined ? null : pick;
  }

  renderTip(L) {
    const tip = this.$("pl-tip");
    if (this.hover === null) {
      tip.hidden = true;
      return;
    }
    const f = this.hover;
    const unit = this.unitLabel();
    const refOn = this.ref && !this.errorMode;
    const rows = this.selected
      .map((i) => ({ i, v: this.value(i, f), r: refOn && this.hasRef[i] ? this.raw(this.ref, i, f) : NaN }))
      .sort((p, q) => (Number.isFinite(q.v) ? q.v : -Infinity) - (Number.isFinite(p.v) ? p.v : -Infinity));
    const head = [`${((this.start + f) / this.fps).toFixed(2)} s`];
    if (this.source === "play" && this.metaRows) head.push(short(this.clipAt(f)));
    else head.push(`frame ${f}`);
    tip.replaceChildren(h("div.pl-tip-head", head.join(" · ")));
    for (const { i, v, r } of rows) {
      const row = h("div.pl-tip-row", h("span.pl-key", { style: { background: this.color(i) } }),
        h("strong", `${num(v)} ${unit}`), h("span", short(this.meta.channels[i].name)));
      const lim = this.limitsOn() ? this.limitsOf(i) : null;
      if (lim && this.unitOf(i) === "Nm") {
        const tol = 0.005 * (lim[1] - lim[0]);
        if (v >= lim[1] - tol || v <= lim[0] + tol) row.append(h("em.pl-over", "at the effort limit"));
      } else if (lim && (v < lim[0] - 1e-3 || v > lim[1] + 1e-3)) row.append(h("em.pl-over", v > lim[1] ? "above limit" : "below limit"));
      if (Number.isFinite(r)) row.append(h("em", `${this.meta.ref_label || "ref"} ${num(r)} · Δ ${v - r >= 0 ? "+" : ""}${num(v - r)}`));
      tip.append(row);
    }
    tip.hidden = false;
    const x = this.frameToX(f, L);
    const box = tip.getBoundingClientRect();
    const left = x + 14 + box.width > L.width ? x - 14 - box.width : x + 14;
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${PAD.top + 4}px`;
  }

  // ---------------------------------------------------------------- table view
  downloadCsv() {
    if (!this.ready() || !this.selected.length) return;
    const unit = this.unitLabel();
    const refOn = this.ref && !this.errorMode;
    const cols = [];
    for (const i of this.selected) {
      const name = this.meta.channels[i].name;
      cols.push(`${this.errorMode ? "error " : ""}${name} [${unit}]`);
      if (refOn && this.hasRef[i]) cols.push(`${name} ${this.meta.ref_label || "reference"} [${unit}]`);
    }
    const lines = [["time_s", this.source === "play" ? "step" : "frame", ...(this.source === "play" ? ["clip", "clip_frame"] : []), ...cols].join(",")];
    for (let f = 0; f < this.rows; f += 1) {
      const vals = [];
      for (const i of this.selected) {
        vals.push(this.value(i, f).toFixed(5));
        if (refOn && this.hasRef[i]) vals.push(this.raw(this.ref, i, f).toFixed(5));
      }
      const extra = this.source === "play" ? [`"${this.clipAt(f)}"`, this.metaRows[f * 3 + 1]] : [];
      lines.push([((this.start + f) / this.fps).toFixed(4), this.start + f, ...extra, ...vals].join(","));
    }
    const blob = new Blob([`${lines.join("\n")}\n`], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = h("a", { href: url, download: `${this.source === "play" ? `play_env${this.app.hud.focus ?? 0}` : (this.app.hud.title || "clip")}_${this.quantity}.csv` });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
