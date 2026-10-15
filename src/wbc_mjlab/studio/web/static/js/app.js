// WBC Studio (web): the pipeline stepper, the inspector (one pane per step), the stage
// (viewer, toolbar, clip bar, HUD, transport), websocket (events + scene frames), shortcuts.
import { DataPane } from "./panes/data.js";
import { PlayPane } from "./panes/play.js";
import { PresetPane } from "./panes/preset.js";
import { RobotsPane } from "./panes/robots.js";
import { TrainPane } from "./panes/train.js";
import { Plots } from "./plots.js";
import { api, debounce, h, popover, post, store, svgIcon, toast, wireThemeToggle } from "./ui.js";
import { Viewer } from "./viewer.js";

const STEPS = [
  ["robots", "Robot"],
  ["data", "Data"],
  ["preset", "Preset"],
  ["train", "Train"],
  ["play", "Play"],
];
const OWNER_LABEL = { robots: "Robot", data: "Motion preview", play: "Policy playback", import: "New robot" };
const VIEW_OPTS = { ghost: true, marks: true, labels: true, follow: true };

class App {
  constructor() {
    this.state = { robot: "", robots: [], task: null, owner: "robots", connected: false };
    this.hud = {};
    this.logs = [];
    this.modelKey = null;
    this.loadingModel = false;
    this.viewer = new Viewer(document.getElementById("viewport"), {
      onPick: (i) => post("/api/focus", { index: i }),
      labelsHost: document.getElementById("labels"),
    });
    this.panes = {
      robots: new RobotsPane(this),
      data: new DataPane(this),
      preset: new PresetPane(this),
      train: new TrainPane(this),
      play: new PlayPane(this),
    };
    this._dirty = new Map();
    this._flush = debounce(() => this.flushEvents(), 100);
    this._stepper = debounce(() => this.renderStepper(), 60);
    this.clipNames = [];
    this.clipKey = null;
    this._clipBar = debounce(() => this.refreshClipBar().catch((e) => console.error(e)), 120);
    this.plots = new Plots(this);
    this.bindStage();
    this.bindKeys();
    wireThemeToggle(document.getElementById("theme-toggle"));
    document.getElementById("status-chip").addEventListener("click", (e) => this.showActivity(e.currentTarget));
    const saved = store.get("pane", "robots");
    this.select(this.panes[saved] ? saved : "robots");
    this.init();
  }

  async init() {
    const st = await api("/api/state");
    Object.assign(this.state, st);
    this.logs = st.logs;
    this.renderChrome();
    await Promise.all(Object.values(this.panes).map((p) => p.refresh().catch((e) => console.error(e))));
    this.renderStepper();
    this._clipBar();
    this.connect();
  }

  // ------------------------------------------------------------------ steps
  select(name) {
    if (this.pane && this.pane !== name) this.panes[this.pane]?.onHide?.();
    this.pane = name;
    store.set("pane", name);
    const p = this.panes[name];
    document.getElementById("inspector").replaceChildren(p.frame.el);
    document.getElementById("studio").style.setProperty("--side-w", `${p.width || 420}px`);
    if (this._dirty.has(name)) {
      this._dirty.delete(name);
      p.refresh().catch(() => {});
    }
    p.onShow?.();
    this.renderStepper();
  }

  /** Re-render the stepper (panes call this when their summary changes). */
  touch() {
    this._stepper();
  }

  renderStepper() {
    const host = document.getElementById("stepper");
    const items = [];
    STEPS.forEach(([key, label], i) => {
      const s = this.panes[key].summary?.() || {};
      if (i) items.push(h(`span.step-link${s.done ? ".done" : ""}`));
      const cls = `${key === this.pane ? ".is-active" : ""}${s.done ? ".done" : ""}${s.busy ? ".busy" : ""}`;
      const dotEl = h("span.dot", s.done && key !== this.pane ? svgIcon("check") : String(i + 1));
      const b = h(`button.step${cls}`, { type: "button", title: `${label} (${i + 1})` }, dotEl, h("span.step-text", h("b", label), h("small", s.value || "—")));
      b.addEventListener("click", () => this.select(key));
      items.push(b);
    });
    host.replaceChildren(...items);
  }

  // ------------------------------------------------------------------ chrome
  renderChrome() {
    const s = this.state;
    const robotChip = document.getElementById("robot-chip");
    if (s.robots.length > 1) {
      const sel = h("select", { title: "Switch robot" }, s.robots.map((r) => h("option", { value: r }, r)));
      sel.value = s.robot;
      sel.disabled = !!s.task;
      sel.addEventListener("change", () => post("/api/robot/select", { name: sel.value }).catch(() => (sel.value = s.robot)));
      robotChip.replaceChildren(svgIcon("robot"), sel);
    } else robotChip.replaceChildren(svgIcon("robot"), h("b", s.robot || "—"));
    const led = !s.connected ? "off" : s.task ? "busy" : "";
    document.getElementById("status-chip").replaceChildren(h(`span.led${led ? `.${led}` : ""}`), s.connected ? (s.task || "Ready") : "Offline");
    document.getElementById("brand").dataset.state = !s.connected ? "offline" : s.task ? "busy" : "live";
    const busy = document.getElementById("busy");
    busy.hidden = !s.task;
    if (s.task) {
      document.getElementById("busy-title").textContent = s.task;
      document.getElementById("busy-stage").textContent = s.stage ? `${s.stage} · ${Math.round(100 * (s.fraction || 0))}%` : "working…";
      const bar = document.getElementById("busy-bar");
      bar.parentElement.classList.toggle("indeterminate", s.fraction === undefined);
      bar.style.width = `${100 * (s.fraction || 0)}%`;
    }
  }

  showActivity(anchor) {
    const s = this.state;
    const lines = this.logs.slice(-80).reverse().map((e) =>
      h(`div.log-line.${e.level}`, h("time", new Date(e.t * 1000).toLocaleTimeString([], { hour12: false })), h("span", e.text)));
    popover(anchor, [
      h("h4", "Activity"),
      h("p.help", `${s.robot} · ${s.gpu || s.device}${s.task ? ` · running: ${s.task}` : ""}`),
      h("div", lines.length ? lines : h("p.help", "Nothing yet.")),
    ], { width: 440 });
  }

  // ------------------------------------------------------------------ stage
  bindStage() {
    const v = this.viewer;
    this.view = { ...VIEW_OPTS, ...store.get("view", {}) };
    const apply = () => {
      v.follow = this.view.follow;
      v.showLabels = this.view.labels;
      v.showGhost = this.view.ghost;
      if (!this.view.ghost) v.ghosts.hide();
      v.setOverlayVisible(this.view.marks);
      document.querySelectorAll("#toolbar [data-opt]").forEach((b) => b.classList.toggle("on", !!this.view[b.dataset.opt]));
      store.set("view", this.view);
    };
    this.toggleView = (key) => {
      this.view[key] = !this.view[key];
      apply();
    };
    document.querySelectorAll("#toolbar [data-opt]").forEach((b) => b.addEventListener("click", () => this.toggleView(b.dataset.opt)));
    apply();
    document.getElementById("cam-reset").addEventListener("click", () => v.resetCamera());
    document.getElementById("keys-btn").addEventListener("click", (e) => this.showKeys(e.currentTarget));
    document.getElementById("plots-btn").addEventListener("click", () => this.plots.toggle());

    document.getElementById("t-play").addEventListener("click", () => this.transport("toggle"));
    document.getElementById("t-step").addEventListener("click", () => this.transport("step"));
    document.getElementById("t-restart").addEventListener("click", () => this.transport("restart"));
    const frame = document.getElementById("t-frame");
    frame.addEventListener("input", () => this.transport("seek", Number(frame.value)));
    document.getElementById("t-speed").addEventListener("change", (e) => {
      const speed = Number(e.target.value);
      post("/api/data/options", { speed });
      post("/api/play/options", { speed });
    });

    const env = document.getElementById("c-env");
    const clip = document.getElementById("c-clip");
    env.addEventListener("change", () => post("/api/focus", { index: Math.max(0, Number(env.value) || 0) }));
    clip.addEventListener("focus", () => clip.select());
    clip.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { clip.value = this.hud.title || ""; clip.blur(); }
    });
    clip.addEventListener("change", () => {
      const name = clip.value.trim();
      if (!name || name === this.hud.title) return;
      if (!this.clipNames.includes(name)) {
        toast(`no clip "${name}" in this library`, "warn");
        clip.value = this.hud.title || "";
        return;
      }
      this.pickClip(name);
      clip.blur();
    });
    document.getElementById("c-prev").addEventListener("click", () => this.shiftClip(-1));
    document.getElementById("c-next").addEventListener("click", () => this.shiftClip(1));
    document.getElementById("c-random").addEventListener("click", () => this.pickClip(""));
    document.getElementById("c-push").addEventListener("click", () => post("/api/play/push", { speed: 1.2 }));
  }

  showKeys(anchor) {
    const rows = [
      ["1 – 5", "pipeline steps"],
      ["space", "play / pause"],
      ["← →", "step back / forward"],
      ["[ ]", "previous / next clip"],
      [", .", "previous / next env (grid, play)"],
      ["p", "push the focused robot"],
      ["g · o · l · f", "ghost · markers · labels · follow"],
      ["j", "plots: clip, robot vs reference, rewards"],
      ["r", "reset camera"],
      ["click", "focus a robot (drag orbits, right-drag pans)"],
    ];
    popover(anchor, [h("h4", "Shortcuts"), h("div.keys", rows.flatMap(([k, d]) => [h("kbd", k), h("span", d)]))], { width: 330 });
  }

  bindKeys() {
    window.addEventListener("keydown", (e) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) || e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key;
      if (k === " ") { e.preventDefault(); this.transport("toggle"); }
      else if (k === "ArrowRight") this.transport("step");
      else if (k === "ArrowLeft") this.transport("back");
      else if (["1", "2", "3", "4", "5"].includes(k)) this.select(STEPS[Number(k) - 1][0]);
      else if (k === "g") this.toggleView("ghost");
      else if (k === "o") this.toggleView("marks");
      else if (k === "l") this.toggleView("labels");
      else if (k === "f") this.toggleView("follow");
      else if (k === "j") { e.preventDefault(); this.plots.toggle(); } // not typed into its search
      else if (k === "r") this.viewer.resetCamera();
      else if (k === "?") this.showKeys(document.getElementById("keys-btn"));
      else if (k === "[" || k === "]") this.shiftClip(k === "]" ? 1 : -1);
      else if (k === "," || k === ".") this.shiftFocus(k === "." ? 1 : -1);
      else if (k === "p" && this.hud.owner === "play") post("/api/play/push", { speed: 1.2 });
    });
  }

  shiftFocus(d) {
    const n = this.hud.num_envs || this.hud.grid || 1;
    if (n < 2) return;
    const cur = this.hud.focus ?? 0;
    post("/api/focus", { index: (((cur + d) % n) + n) % n });
  }

  shiftClip(d) {
    if (this.hud.owner === "data") return this.panes.data.stepClip(d);
    const n = this.clipNames.length;
    if (!n || this.hud.owner !== "play") return;
    const i = this.clipNames.indexOf(this.hud.title);
    this.pickClip(this.clipNames[(((i < 0 ? 0 : i + d) % n) + n) % n]);
  }

  onPlotsToggle(open) {
    document.getElementById("plots-btn").classList.toggle("on", open);
  }

  pickClip(name) {
    post("/api/play/clip", { clip: name }).catch(() => {});
  }

  async refreshClipBar() {
    const d = await api("/api/play");
    const bar = document.getElementById("clip-bar");
    bar.hidden = !d.active || this.state.owner !== "play";
    if (!d.active) {
      this.clipKey = null;
      return;
    }
    const key = `${d.run}|${d.motion}|${d.clips_loaded}`;
    if (key !== this.clipKey) {
      this.clipKey = key;
      this.clipNames = await api("/api/play/clips");
      document.getElementById("c-clips").replaceChildren(...this.clipNames.map((n) => h("option", { value: n })));
    }
    const env = document.getElementById("c-env");
    env.max = d.num_envs - 1;
    env.disabled = d.num_envs <= 1;
    if (document.activeElement !== env) env.value = d.focus;
    const clip = document.getElementById("c-clip");
    if (document.activeElement !== clip) clip.value = d.focus_clip;
    clip.placeholder = `${this.clipNames.length} clips · type to search`;
    const st = document.getElementById("c-state");
    st.textContent = d.focus_pinned ? "looping" : "random";
    st.className = `state${d.focus_pinned ? " pinned" : ""}`;
    document.getElementById("c-random").disabled = !d.focus_pinned;
  }

  transport(action, value) {
    const owner = this.hud.owner;
    if (owner === "data") {
      const grid = !!this.hud.grid;
      if (action === "toggle") post("/api/data/playing", { playing: !this.hud.playing });
      if (action === "restart") post("/api/data/time", { t: 0 });
      if (grid) {
        if (action === "step") post("/api/data/time", { t: (this.hud.time || 0) + 0.02 });
        if (action === "back") post("/api/data/time", { t: Math.max(0, (this.hud.time || 0) - 0.02) });
        if (action === "seek") post("/api/data/time", { t: value / 50 });
      } else {
        if (action === "step") post("/api/data/frame", { frame: (this.hud.frame || 0) + 1 });
        if (action === "back") post("/api/data/frame", { frame: Math.max(0, (this.hud.frame || 0) - 1) });
        if (action === "seek") post("/api/data/frame", { frame: value });
      }
    } else if (owner === "play") {
      if (action === "toggle") post("/api/play/playing", { playing: !this.hud.playing });
      if (action === "step") post("/api/play/step");
      if (action === "restart") post("/api/play/restart");
    }
  }

  // ------------------------------------------------------------------ websocket
  connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.state.connected = true;
      this.renderChrome();
      this._ping = setInterval(() => ws.readyState === 1 && ws.send("ping"), 20000);
    };
    ws.onclose = () => {
      this.state.connected = false;
      clearInterval(this._ping);
      this.renderChrome();
      setTimeout(() => this.connect(), 1500);
    };
    ws.onmessage = (msg) => {
      if (typeof msg.data === "string") this.onEvent(JSON.parse(msg.data));
      else this.onFrame(msg.data).catch((e) => console.error(e));
    };
  }

  onEvent(ev) {
    const { topic, payload } = ev;
    if (topic === "log") {
      const e = payload.entry;
      this.logs.push(e);
      if (this.logs.length > 400) this.logs.shift();
      if (e.level === "error" || e.level === "ok" || e.level === "warn") toast(e.text, e.level);
      return;
    }
    if (topic === "busy") {
      Object.assign(this.state, { task: payload.task, stage: payload.stage, fraction: payload.fraction });
      if (!payload.task) Object.assign(this.state, { stage: undefined, fraction: undefined });
      this.renderChrome();
      return;
    }
    if (topic === "robot") {
      this.state.robot = payload.name;
      this.renderChrome();
    }
    if (topic === "play" || topic === "focus" || topic === "owner") this._clipBar();
    for (const [name, pane] of Object.entries(this.panes)) {
      if (!pane.topics.includes(topic)) continue;
      // a robot switch changes every step's summary: refresh hidden panes too
      if (name === this.pane || pane.live?.includes(topic) || topic === "robot") {
        const list = this._dirty.get(`ev:${name}`) || [];
        list.push([topic, payload]);
        this._dirty.set(`ev:${name}`, list);
      } else this._dirty.set(name, true);
    }
    this._flush();
  }

  flushEvents() {
    for (const [key, list] of [...this._dirty]) {
      if (!key.startsWith("ev:")) continue;
      this._dirty.delete(key);
      const pane = this.panes[key.slice(3)];
      const seen = new Set();
      for (const [topic, payload] of list.reverse()) {
        if (seen.has(topic)) continue;
        seen.add(topic);
        Promise.resolve(pane.onEvent(topic, payload)).catch((e) => console.error(e)).finally(() => this.touch());
      }
    }
  }

  async onFrame(buf) {
    const view = new DataView(buf);
    const headLen = view.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
    const blob = new Float32Array(buf, 4 + headLen);
    if (header.terrain_version !== undefined && header.terrain_version !== this.terrainVersion) {
      const v = (this.terrainVersion = header.terrain_version);
      fetch("/api/scene/terrain.bin").then((r) => r.arrayBuffer()).then((b) => {
        if (v === this.terrainVersion) this.viewer.setTerrain(b);
      }).catch(() => {});
    }
    const key = `${this.state.robot}:${header.model_version}`;
    if (key !== this.modelKey && !this.loadingModel) {
      this.loadingModel = true;
      try {
        const [meta, bin] = await Promise.all([api("/api/robot/model"), fetch("/api/robot/model.bin").then((r) => r.arrayBuffer())]);
        this.viewer.setModel(meta, bin);
        this.modelKey = `${meta.robot}:${meta.version}`;
        this.state.robot = meta.robot;
      } finally {
        this.loadingModel = false;
      }
    }
    if (header.n) this.viewer.setPoses(header.n, header.ng, header.nbody, blob, header.focus, header.labels);
    if (header.overlays) this.viewer.setOverlays(header.overlays, blob);
    if (this.state.owner !== header.owner) {
      this.state.owner = header.owner;
      this._clipBar();
    }
    this.onHud(header.hud || {}, header.owner, header);
  }

  onHud(hud, owner, header) {
    const prevOwner = this.hud.owner;
    this.hud = hud.owner === owner ? { ...hud, focus: hud.focus ?? header.focus } : {};
    const title = document.getElementById("stage-title");
    const sub = document.getElementById("stage-sub");
    const pill = document.getElementById("stage-pill");
    const tr = document.getElementById("transport");
    const has = !!this.hud.owner;
    if (tr.hidden === has) tr.hidden = !has;
    if (prevOwner !== this.hud.owner) this.touch();
    if (!has) {
      title.textContent = OWNER_LABEL[owner] || "Scene";
      sub.textContent = owner === "robots" ? `${this.state.robot || "no robot"} · default pose` : owner === "import" ? "preview · not saved yet" : "nothing playing";
      pill.textContent = owner === "robots" ? "pose" : owner === "import" ? "draft" : "idle";
      pill.className = "pill idle";
      document.getElementById("hud").hidden = true;
      this.plots.sync(this.hud, owner);
      return;
    }
    title.textContent = hud.title;
    pill.textContent = hud.playing ? (hud.owner === "play" ? "live" : "playing") : "paused";
    pill.className = `pill${hud.playing ? "" : " paused"}`;
    const clip = document.getElementById("c-clip");
    if (hud.owner === "play" && document.activeElement !== clip && clip.value !== hud.title) clip.value = hud.title;
    const bits = [];
    if (hud.owner === "play") bits.push(`policy · env ${hud.focus} of ${hud.num_envs}`, "ghost = reference");
    if (hud.owner === "data" && hud.grid) bits.push(`grid of ${hud.grid} clips`);
    else if (hud.owner === "data") bits.push(hud.kind === "raw" ? "raw source" : "converted", `${hud.fps} Hz`, hud.ghost ? "ghost = raw source" : null);
    sub.textContent = bits.filter(Boolean).join(" · ");
    document.getElementById("t-play-icon").setAttribute("d", hud.playing ? "M7 4h4v16H7zM13 4h4v16h-4z" : "M7 4l12 8-12 8z");
    const frame = document.getElementById("t-frame");
    frame.max = Math.max(0, hud.frames - 1);
    if (document.activeElement !== frame) frame.value = hud.frame;
    frame.disabled = hud.owner !== "data";
    document.getElementById("t-time").textContent = `${(hud.time ?? 0).toFixed(2)} / ${(hud.duration ?? 0).toFixed(2)} s`;
    document.getElementById("t-frame-label").textContent = `${hud.frame} / ${Math.max(0, hud.frames - 1)}`;
    this.plots.sync(this.hud, owner);
    const now = performance.now();
    if (now - (this._hudAt || 0) < 200) return;
    this._hudAt = now;
    const tiles = [];
    if (hud.owner === "play") {
      const m = hud.metrics || {};
      const tone = (v, good, bad) => (v == null ? "" : v < good ? "good" : v > bad ? "bad" : "warn");
      const num = (v, k = 1, d = 1) => (v == null ? "—" : (k * v).toFixed(d));
      tiles.push(["Return", num(hud.return), ""]);
      if (hud.num_envs > 1) tiles.push(["Mean return", num(hud.mean_return), ""]);
      if ("error_body_pos" in m) tiles.push(["Body error", `${num(m.error_body_pos, 100)} cm`, tone(m.error_body_pos, 0.05, 0.15)]);
      if ("error_joint_pos" in m) tiles.push(["Joint error", `${num(m.error_joint_pos, 1, 2)} rad`, tone(m.error_joint_pos, 0.5, 1.2)]);
      if ("error_anchor_pos" in m) tiles.push(["Anchor drift", `${num(m.error_anchor_pos, 100)} cm`, tone(m.error_anchor_pos, 0.1, 0.5)]);
      tiles.push(["Episode", `${((hud.episode_step || 0) / 50).toFixed(1)} s`, ""]);
      this.panes.play.hud(hud);
    }
    const el = document.getElementById("hud");
    el.hidden = !tiles.length;
    el.replaceChildren(...tiles.map(([k, v, t]) => h(`div.tile.glass${t ? `.${t}` : ""}`, h("small", k), h("b", v))));
  }
}

window.app = new App();
