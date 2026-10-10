// Step 3 · Preset: the MDP a policy trains on. Built-in presets are the paper setups
// (--task Wbc-G1, Wbc-G1-Zest …); a custom preset is one of them + modules (reusable pieces such
// as SE) + its own overrides of observations, rewards, terminations, sampling, randomization
// and PPO. Saved per user (~/.config/wbc-mjlab), trained as task <base>--<name>.
import {
  PaneFrame, api, button, closePopover, del, empty, field, fmtNum, group, h, help, iconButton, input, modal,
  popover, post, put, seg, select, switchEl, tabs, tag, toast,
} from "../ui.js";

const TABS = [
  ["observations", "Observations"],
  ["rewards", "Rewards"],
  ["terminations", "Terminations"],
  ["command", "Sampling"],
  ["events", "Randomization"],
  ["terrain", "Terrain"],
  ["agent", "PPO"],
];
const COMMAND_SECTIONS = [
  ["rsi.", "Reference state init (RSI)", "Where episodes start in the clips and how hard bins are favoured."],
  ["assistive_", "Assistive wrench", "A virtual force that helps on the hardest bins, faded out as they get easier."],
  ["pose_range.", "Reset noise · root pose", "Added to the reference root at reset."],
  ["velocity_range.", "Reset noise · root velocity", null],
  ["", "Episode", null],
];
const AGENT_SECTIONS = [
  ["algorithm.", "Optimisation", null],
  ["actor.", "Policy network", "Changing the network makes older checkpoints incompatible."],
  ["critic.", "Value network", null],
  ["", "Rollouts", null],
];
const TERRAIN_INTRO = "Procedural terrain of the -Terrain-Scan tasks: rows of rising difficulty (the curriculum moves envs up as they track well) and a mix of sub-terrains, one column each.";
const FLAT_TABS = new Set(["command", "terrain", "agent"]);
const OBS_GROUPS = {
  actor: ["Actor · policy input", "What the policy sees, on the robot too. Changing it changes the network input (train from scratch); the export only rebuilds deployable terms."],
  critic: ["Critic · privileged", "Training-only inputs of the value function: ground truth, contacts, the full reference. Free to change."],
};
const KINDS = [
  ["reference", "Reference · what to track"],
  ["state", "Robot state"],
  ["privileged", "Privileged · sim only"],
];
const MODULE = "module:";

const clone = (o) => JSON.parse(JSON.stringify(o || {}));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const pretty = (key) => key.replace(/^(rsi|algorithm|actor|critic|env)\./, "").replace(/_/g, " ");
const prune = (o) => {
  for (const [k, v] of Object.entries(o)) if (v === undefined) delete o[k];
  return o;
};

export class PresetPane {
  constructor(app) {
    this.app = app;
    this.topics = ["presets", "robot", "play"];
    this.width = 640;
    this.deleteBtn = iconButton("trash", () => this.remove(), "Delete");
    this.frame = new PaneFrame("Preset", "The MDP a policy learns: what it sees, how it is rewarded, sampled, randomized and optimised.", { actions: [this.deleteBtn] });
    this.tab = "observations";
    this.tasks = [];
    this.modules = [];
    this.configDir = "";
    this.selected = "";
    this.schema = null;
    this.overrides = {};
    this.saved = {};
    this.using = []; // modules of the edited preset, in order
    this.savedUsing = [];
    this.open = new Set();
    this.playActive = false;
  }

  // ------------------------------------------------------------------ state
  current() {
    return this.tasks.find((t) => t.id === this.selected);
  }

  /** The module being edited (selected as ``module:<name>``), else null. */
  editedModule() {
    return this.selected.startsWith(MODULE) ? this.modules.find((m) => m.name === this.selected.slice(MODULE.length)) || null : null;
  }

  isCustom() {
    return !!this.current()?.custom;
  }

  baseId() {
    if (this.selected.startsWith(MODULE)) return this.tasks.find((t) => !t.custom)?.id || "";
    const t = this.current();
    return t ? (t.custom ? t.base : t.id) : "";
  }

  dirty() {
    return !same(this.overrides, this.saved) || !same(this.using, this.savedUsing);
  }

  /** The task id training should use (what the Train step reads). */
  taskId() {
    return this.selected.startsWith(MODULE) ? this.baseId() : this.selected;
  }

  summary() {
    const m = this.editedModule();
    const t = this.current();
    if (!t && !m) return { value: "" };
    const n = this.changes();
    const name = m ? `module ${m.name}` : t.custom ? t.preset : t.id;
    const mods = this.using.length ? ` + ${this.using.join(" + ")}` : "";
    return { value: `${name}${mods}${n ? ` · ${n} change${n === 1 ? "" : "s"}` : ""}${this.dirty() ? " •" : ""}`, done: true };
  }

  async refresh() {
    const d = await api("/api/presets");
    this.tasks = d.tasks;
    this.modules = d.modules;
    this.configDir = d.config_dir;
    this.playActive = (await api("/api/play")).active;
    if (!this.current() && !this.editedModule()) await this.choose(this.tasks[0]?.id || "", { keepTab: true, force: true });
    else this.render();
  }

  async onEvent(topic, payload) {
    if (topic === "play") {
      const was = this.playActive;
      this.playActive = !!payload.active;
      if (was !== this.playActive) this.renderFooter();
      return;
    }
    if (topic === "robot") {
      this.selected = "";
      this.saved = this.overrides = {};
      this.using = this.savedUsing = [];
    }
    await this.refresh(); // keeps the edited overrides of the current preset
  }

  async choose(id, { keepTab, force } = {}) {
    if (!force && this.dirty() && this.selected && !confirm("Discard unsaved changes?")) {
      this.render();
      return;
    }
    this.selected = id;
    this.schema = null;
    this.overrides = {};
    this.using = [];
    const m = this.editedModule();
    const t = this.current();
    if (m) {
      this.overrides = clone(m.overrides);
    } else if (t?.custom) {
      const preset = await api(`/api/presets/${encodeURIComponent(t.preset)}`);
      this.overrides = clone(preset.overrides);
      this.using = [...(preset.modules || [])];
    }
    this.saved = clone(this.overrides);
    this.savedUsing = [...this.using];
    if (!keepTab) this.open.clear();
    await this.loadSchema();
    this.app.touch();
  }

  async loadSchema() {
    const base = this.baseId();
    if (!base) {
      this.render();
      return;
    }
    const q = `base=${encodeURIComponent(base)}${this.using.length ? `&modules=${encodeURIComponent(this.using.join(","))}` : ""}`;
    this.schema = await api(`/api/presets/schema?${q}`);
    this.render();
  }

  async toggleModule(name, on) {
    this.using = on ? [...this.using.filter((x) => x !== name), name] : this.using.filter((x) => x !== name);
    await this.loadSchema(); // the modules' changes become the defaults the own changes diff against
    this.app.touch();
  }

  // ------------------------------------------------------------------ overrides
  termOv(group, name) {
    return this.overrides[group]?.[name] || {};
  }

  setTerm(group, name, patch) {
    const g = (this.overrides[group] ||= {});
    const o = prune({ ...(g[name] || {}), ...patch });
    if (o.params && !Object.keys(o.params).length) delete o.params;
    if (Object.keys(o).length) g[name] = o;
    else delete g[name];
    if (!Object.keys(g).length) delete this.overrides[group];
    this.changed();
  }

  setParam(group, term, key, value, def) {
    const params = { ...(this.termOv(group, term.name).params || {}) };
    if (same(value, def)) delete params[key];
    else params[key] = value;
    this.setTerm(group, term.name, { params: Object.keys(params).length ? params : undefined });
  }

  setFlat(group, key, value, def) {
    const g = (this.overrides[group] ||= {});
    if (same(value, def)) delete g[key];
    else g[key] = value;
    if (!Object.keys(g).length) delete this.overrides[group];
    this.changed();
  }

  obsOv(g) {
    return this.overrides.observations?.[g] || {};
  }

  obsTermOv(g, name) {
    return this.obsOv(g).terms?.[name] || {};
  }

  /** Patch an observation group's settings, or one of its terms when ``term`` is given. */
  setObs(g, patch, term) {
    const obs = (this.overrides.observations ||= {});
    const grp = (obs[g] ||= {});
    if (term) {
      const terms = (grp.terms ||= {});
      const o = prune({ ...(terms[term] || {}), ...patch });
      if (Object.keys(o).length) terms[term] = o;
      else delete terms[term];
      if (!Object.keys(terms).length) delete grp.terms;
    } else {
      prune(Object.assign(grp, patch));
    }
    if (!Object.keys(grp).length) delete obs[g];
    if (!Object.keys(obs).length) delete this.overrides.observations;
    this.changed();
  }

  changes(group) {
    const ov = this.overrides;
    const count = (g) => {
      if (FLAT_TABS.has(g)) return Object.keys(ov[g] || {}).length;
      if (g === "observations") {
        return Object.values(ov.observations || {}).reduce((n, o) =>
          n + Object.keys(o).filter((k) => k !== "terms").length
            + Object.values(o.terms || {}).reduce((m, t) => m + Object.keys(t).length, 0), 0);
      }
      return Object.values(ov[g] || {}).reduce((n, o) => n + (o.enabled === false ? 1 : 0) + ("weight" in o ? 1 : 0) + Object.keys(o.params || {}).length, 0);
    };
    return group ? count(group) : TABS.reduce((n, [g]) => n + count(g), 0);
  }

  changed() {
    this._softRender();
    this.app.touch();
  }

  _softRender() {
    // keep focus / scroll while typing: only refresh tabs, summaries and footer
    this.renderTabs();
    this.renderFooter();
    if (this.tab === "rewards") this.renderComposition();
    if (this.tab === "observations") this.renderObsSummary();
  }

  // ------------------------------------------------------------------ render
  render() {
    if (this.schema && !this.tabList().some(([g]) => g === this.tab)) this.tab = "observations";
    const m = this.editedModule();
    this.deleteBtn.hidden = !(this.isCustom() || (m && !m.builtin));
    this.deleteBtn.title = m ? "Delete this module" : "Delete this preset";
    if (!this.schema) {
      this.frame.setTabs(null);
      this.frame.setBody(this.pickerGroup(), this.tasks.length ? help("loading…") : empty("layers", "No presets", "This robot has no registered tasks."));
      this.frame.setFooter();
      return;
    }
    this.renderTabs();
    const body = {
      observations: () => this.observationsTab(),
      rewards: () => this.rewardsTab(),
      terminations: () => this.termsTab("terminations", "Episodes end when a term fires. Off = never ends that way."),
      command: () => this.flatTab("command", COMMAND_SECTIONS),
      events: () => this.termsTab("events", "Domain randomization applied at startup, reset or on an interval."),
      terrain: () => h("div", h("p.help", { style: { marginBottom: "12px" } }, TERRAIN_INTRO), this.flatTab("terrain", this.terrainSections())),
      agent: () => this.flatTab("agent", AGENT_SECTIONS),
    }[this.tab]();
    this.frame.setBody(this.pickerGroup(), body);
    this.renderFooter();
  }

  /** Tabs of this task: Terrain only when it has a procedural terrain. */
  tabList() {
    return TABS.filter(([g]) => g !== "terrain" || this.schema?.terrain?.length);
  }

  terrainSections() {
    const subs = [...new Set(this.schema.terrain.map((f) => f.key.split(".")[0]).filter((k) => k !== "curriculum"))];
    return [
      ["curriculum.", "Curriculum", null],
      ...subs.map((k) => [`${k}.`, `Sub-terrain · ${k}`, null]),
    ];
  }

  renderTabs() {
    if (!this.schema) return;
    if (!this.tabList().some(([g]) => g === this.tab)) this.tab = "observations";
    this.frame.setTabs(tabs(this.tabList().map(([g, label]) => {
      const n = this.changes(g);
      const total = g === "observations" ? this.obsRows("actor").rows.length : this.schema[g].length;
      return { value: g, label, count: n || total, mod: !!n };
    }), this.tab, (t) => { this.tab = t; this.render(); }));
    requestAnimationFrame(() => this.frame.el.querySelector(".tabs .is-active")?.scrollIntoView({ block: "nearest", inline: "nearest" }));
  }

  pickerGroup() {
    const builtins = this.tasks.filter((t) => !t.custom).map((t) => ({ value: t.id, label: t.id }));
    const mine = this.tasks.filter((t) => t.custom).map((t) => ({ value: t.id, label: `${t.preset}  (${[t.base, ...(t.modules || [])].join(" + ")})` }));
    const mods = this.modules.map((m) => ({ value: MODULE + m.name, label: `${m.name}${m.builtin ? " (built-in)" : ""} · ${m.changes} change${m.changes === 1 ? "" : "s"}` }));
    const options = [{ group: "Built-in (paper setups)", items: builtins }];
    if (mine.length) options.push({ group: "Your presets", items: mine });
    if (mods.length) options.push({ group: "Modules · reusable pieces", items: mods });
    const sel = select(options, this.selected, (v) => this.choose(v));
    const m = this.editedModule();
    if (m) {
      const text = m.builtin
        ? `Built-in: ${m.description}. Read off ${m.source}. “Save as module…” makes an editable copy.`
        : `${m.description ? `${m.description}. ` : ""}Previewed on ${this.baseId()}; what you change here is the module.`;
      return h("div.group", field("Module", sel, text));
    }
    const t = this.current();
    return h("div.group",
      field(this.isCustom() ? `Custom preset · based on ${t.base}` : "Base preset", sel, t?.description),
      this.modulesRow());
  }

  modulesRow() {
    if (!this.modules.length) return null;
    const chips = this.modules.map((m) => {
      const on = this.using.includes(m.name);
      const order = on && this.using.length > 1 ? this.using.indexOf(m.name) + 1 : 0;
      const b = h(`button.module-chip${on ? ".is-active" : ""}`, {
        type: "button",
        title: `${m.description || m.name}${m.source ? ` (${m.source})` : ""} · ${m.changes} change${m.changes === 1 ? "" : "s"}`,
      }, h("span.box", order ? String(order) : ""), h("b", m.name), h("span.n", `${m.changes}`));
      b.addEventListener("click", () => this.toggleModule(m.name, !on));
      return b;
    });
    return h("div.modules-row",
      h("span.label", "Modules"),
      h("div.module-chips", chips),
      h("span.hint", this.using.length ? "applied in order, then your changes" : "reusable pieces, e.g. SE"));
  }

  // ------------------------------------------------------------------ observations
  /** Rows of a group: the schema's terms (base + modules) + the terms this preset adds. */
  obsRows(g) {
    const obs = this.schema.observations;
    const grp = obs.groups.find((x) => x.name === g);
    if (!grp) return { grp: null, rows: [] };
    const lib = new Map(obs.library.map((x) => [x.name, x]));
    const rows = grp.terms.map((t) => ({ ...t, added: false }));
    for (const [name, o] of Object.entries(this.obsOv(g).terms || {})) {
      if (!o.add || rows.some((r) => r.name === name)) continue;
      const l = lib.get(name) || { name, kind: "state", func: "", deployable: false };
      rows.push({
        name, func: l.func, kind: l.kind, dim: l.dim ?? null, noise: g === "actor" ? l.noise : "", history_length: 0,
        bodies: g === "actor" ? l.bodies || null : null, takes_bodies: !!l.takes_bodies, deployable: g === "actor" ? l.deployable : true, added: true,
      });
    }
    return { grp, rows };
  }

  /** Width of a row with its overrides: 0 when dropped, null when unknown. */
  rowDim(g, r) {
    const o = this.obsTermOv(g, r.name);
    if (o.enabled === false) return 0;
    if (r.dim === null || r.dim === undefined) return null;
    let d = r.dim / Math.max(1, r.history_length || 1);
    if ("bodies" in o && r.takes_bodies) {
      const all = this.schema.observations.bodies;
      d = (d / Math.max(1, (r.bodies || all).length)) * ((o.bodies && o.bodies.length) || all.length);
    }
    return d * Math.max(1, o.history_length ?? r.history_length ?? 1);
  }

  groupStats(g) {
    const { grp, rows } = this.obsRows(g);
    let dims = 0;
    let unknown = 0;
    let terms = 0;
    for (const r of rows) {
      const d = this.rowDim(g, r);
      if (d === 0) continue;
      terms += 1;
      if (d === null) unknown += 1;
      else dims += d;
    }
    const hist = this.obsOv(g).history_length ?? grp?.history_length ?? 0;
    return { terms, dims: dims * Math.max(1, hist || 1), unknown, hist };
  }

  renderObsSummary() {
    for (const [g, el] of Object.entries(this.obsTotals || {})) {
      const s = this.groupStats(g);
      el.textContent = `${s.terms} terms · ${s.unknown ? "≥ " : ""}${s.dims} dims${s.hist > 1 ? ` · ×${s.hist}` : ""}`;
    }
    if (!this.obsWarn) return;
    const bad = this.obsRows("actor").rows.filter((r) => !r.deployable && this.obsTermOv("actor", r.name).enabled !== false);
    const changed = !!this.overrides.observations?.actor || this.using.some((n) => this.modules.find((m) => m.name === n)?.overrides.observations?.actor);
    this.obsWarn.replaceChildren(...[
      bad.length ? h("div.callout.warn", h("b", "Not deployable · "), `${bad.map((r) => r.name).join(", ")}: the export can't rebuild ${bad.length === 1 ? "it" : "them"} on the robot. Fine for experiments in sim.`) : null,
      changed && !this.selected.startsWith(MODULE) ? h("div.callout", `The policy input differs from ${this.baseId()}: train from scratch, other checkpoints won't load.`) : null,
    ].filter(Boolean));
  }

  observationsTab() {
    this.obsTotals = {};
    this.obsWarn = h("div.callouts");
    const out = [this.obsWarn];
    for (const g of ["actor", "critic"]) {
      const { grp, rows } = this.obsRows(g);
      if (!grp) continue;
      const [title, text] = OBS_GROUPS[g];
      const total = h("span.obs-total");
      this.obsTotals[g] = total;
      const add = button("Add term", () => this.addTermPopover(g, add), { cls: "ghost small", icon: "plus" });
      const ov = this.obsOv(g);
      const knobs = h("div.obs-knobs",
        knob({ key: "history", kind: "int", value: grp.history_length }, ov.history_length,
          (v) => this.setObs(g, { history_length: v === grp.history_length ? undefined : v })),
        g === "actor"
          ? knob({ key: "training noise", kind: "bool", value: grp.enable_corruption }, ov.enable_corruption,
            (v) => this.setObs(g, { enable_corruption: v === grp.enable_corruption ? undefined : v }))
          : null,
        add);
      const sections = KINDS.map(([kind, label]) => {
        const mine = rows.filter((r) => r.kind === kind);
        return mine.length ? h("div.obs-section", h("div.obs-section-title", label), h("div.terms", mine.map((r) => this.obsRow(g, r)))) : null;
      });
      out.push(group(title, { collapsible: true, closed: g === "critic", key: `preset-obs-${g}`, aside: total },
        help(text), knobs, sections));
    }
    this.renderObsSummary();
    return h("div", out);
  }

  /** The active module that added term ``name`` to group ``g`` ("" if none). */
  moduleOf(g, name) {
    for (const n of this.using) {
      const m = this.modules.find((x) => x.name === n);
      if (m?.overrides.observations?.[g]?.terms?.[name]?.add) return m.name;
    }
    return "";
  }

  obsRow(g, r) {
    const o = this.obsTermOv(g, r.name);
    const off = o.enabled === false;
    const key = `obs:${g}:${r.name}`;
    const row = h(`div.term.obs-term${off ? ".off" : ""}${this.open.has(key) ? ".open" : ""}`);
    const sw = switchEl("", !off, (on) => {
      if (r.added && !on) {
        this.setObs(g, { add: undefined, noise_scale: undefined, history_length: undefined, bodies: undefined }, r.name);
        this.render();
        return;
      }
      this.setObs(g, { enabled: on ? undefined : false }, r.name);
      row.classList.toggle("off", !on);
    }, off ? "use this term" : r.added ? "remove" : "drop this term");
    const mod = h("button.mod-dot", { type: "button" });
    const dim = h("span.obs-dim", { title: "values in this term" });
    const syncMod = () => {
      const n = Object.keys(this.obsTermOv(g, r.name)).filter((k) => k !== "add").length;
      mod.classList.toggle("on", n > 0);
      mod.title = n ? `${n} change(s) — click to reset` : "";
      const d = this.obsTermOv(g, r.name).enabled === false ? r.dim : this.rowDim(g, r);
      dim.textContent = d === null || d === undefined ? "?" : String(d);
    };
    mod.addEventListener("click", () => {
      this.setObs(g, { enabled: undefined, noise_scale: undefined, history_length: undefined, bodies: undefined }, r.name);
      this.render();
    });
    const from = this.moduleOf(g, r.name);
    const tags = [
      r.added ? tag("added", "accent") : from ? tag(from, "accent", `added by module ${from}`) : null,
      g === "actor" && !r.deployable ? tag("sim only", "amber", "the export can't rebuild this term on the robot") : null,
    ].filter(Boolean);
    let noise = h("span.obs-noise.none", g === "actor" ? "no noise" : "");
    if (r.noise) {
      const w = input(o.noise_scale ?? 1, { type: "number", step: 0.25, min: 0, title: `training noise ${r.noise}, scaled (0 = off)` });
      noise = h(`div.obs-noise${o.noise_scale !== undefined ? ".mod" : ""}`, { title: `uniform noise ${r.noise} × factor` }, h("span", r.noise), h("i", "×"), w);
      w.addEventListener("input", () => {
        const v = Number(w.value);
        if (w.value === "" || !Number.isFinite(v) || v < 0) return;
        this.setObs(g, { noise_scale: v === 1 ? undefined : v }, r.name);
        noise.classList.toggle("mod", v !== 1);
        syncMod();
      });
    }
    const expand = h("button.icon-btn.expand", { type: "button", title: "options" }, svgChev());
    const toggle = () => {
      row.classList.toggle("open");
      row.classList.contains("open") ? this.open.add(key) : this.open.delete(key);
    };
    expand.addEventListener("click", toggle);
    const name = h("div.name", { on: { click: toggle } }, h("b", r.name, ...tags), h("span", r.func));
    const opts = [
      knob({ key: "history", kind: "int", value: r.history_length, help: "frames of this term (0 = the group's)" }, o.history_length,
        (v) => { this.setObs(g, { history_length: v === r.history_length ? undefined : v }, r.name); syncMod(); }),
    ];
    if (r.takes_bodies) opts.push(this.bodyPicker(g, r, syncMod));
    row.append(h("div.term-head", sw, name, dim, noise, mod, expand), h("div.term-params", opts));
    syncMod();
    return row;
  }

  bodyPicker(g, r, syncMod) {
    const all = this.schema.observations.bodies;
    const base = r.bodies || all;
    const ov = this.obsTermOv(g, r.name).bodies;
    let cur = new Set(ov === undefined ? base : ov.length ? ov : all);
    const host = h("div.body-chips");
    const label = h("span");
    const draw = () => {
      label.textContent = `bodies · ${cur.size} of ${all.length}`;
      host.replaceChildren(...all.map((b) => {
      const btn = h(`button${cur.has(b) ? ".is-active" : ""}`, { type: "button" }, b.replace(/_link$/, ""));
      btn.addEventListener("click", () => {
        if (cur.has(b)) cur.delete(b);
        else cur.add(b);
        if (!cur.size) cur = new Set(base);
        const list = all.filter((x) => cur.has(x));
        this.setObs(g, { bodies: same(list, base) ? undefined : list.length === all.length ? [] : list }, r.name);
        draw();
        syncMod();
      });
      return btn;
    }));
    };
    draw();
    return h("div.knob.wide", h("label", label), host,
      h("div.hint", "keybodies this term covers, e.g. hands and feet only"));
  }

  addTermPopover(g, anchor) {
    const present = new Set(this.obsRows(g).rows.filter((r) => this.obsTermOv(g, r.name).enabled !== false).map((r) => r.name));
    const lib = this.schema.observations.library.filter((x) => !present.has(x.name));
    const sections = KINDS.map(([kind, label]) => {
      const mine = lib.filter((x) => x.kind === kind);
      if (!mine.length) return null;
      return h("div.add-section", h("div.obs-section-title", label), mine.map((x) => {
        const b = h("button.add-item", { type: "button", disabled: !!x.requires, title: x.requires ? `needs the ${x.requires} sensor: start from a task that has it (e.g. a -Terrain-Scan task)` : "" },
          h("span.text", h("b", x.name), h("span", x.func)),
          x.requires ? tag(`needs ${x.requires}`) : g === "actor" && !x.deployable ? tag("sim only", "amber") : null,
          h("span.obs-dim", x.dim ? String(x.dim) : "?"));
        if (x.requires) return b;
        b.addEventListener("click", () => {
          closePopover();
          // adding back a term this preset dropped just undoes the drop
          if (this.obsTermOv(g, x.name).enabled === false) this.setObs(g, { enabled: undefined }, x.name);
          else this.setObs(g, { add: true }, x.name);
          this.render();
        });
        return b;
      }));
    });
    popover(anchor, h("div.add-terms",
      h("div.pop-title", `Add to ${g}`),
      lib.length ? h("div.add-list", sections) : help("Every term this robot has is already in.")), { width: 380, align: "left" });
  }

  // ------------------------------------------------------------------ rewards
  rewardsTab() {
    this.compositionHost = h("div");
    this.renderComposition();
    const rows = this.schema.rewards.map((t) => this.termRow("rewards", t, { weight: true }));
    return h("div",
      group("Composition", { aside: "share of |weight|" }, this.compositionHost),
      group("Terms", { aside: "toggle · weight · ▸ parameters" }, h("div.terms", rows)));
  }

  renderComposition() {
    if (!this.compositionHost) return;
    const items = this.schema.rewards
      .map((t) => {
        const o = this.termOv("rewards", t.name);
        return { name: t.name, w: o.enabled === false ? 0 : (o.weight ?? t.weight) };
      })
      .filter((x) => x.w !== 0);
    const color = (x) => (x.w < 0 ? "var(--color-danger)" : x.name.startsWith("motion_") ? "var(--color-accent)" : "var(--color-success)");
    const sum = (f) => items.filter(f).reduce((s, x) => s + Math.abs(x.w), 0);
    const track = sum((x) => x.w > 0 && x.name.startsWith("motion_"));
    const other = sum((x) => x.w > 0 && !x.name.startsWith("motion_"));
    const pen = sum((x) => x.w < 0);
    this.compositionHost?.replaceChildren(
      h("div.composition", items.map((x) => h("div", { title: `${x.name}: ${fmtNum(x.w)}`, style: { flexGrow: String(Math.abs(x.w)), background: color(x), opacity: x.w < 0 ? "0.8" : "1" } }))),
      h("div.composition-legend",
        h("span", h("span", { style: { color: "var(--color-accent)" } }, "■ "), `tracking ${fmtNum(track)}`),
        h("span", h("span", { style: { color: "var(--color-success)" } }, "■ "), `other ${fmtNum(other)}`),
        h("span", h("span", { style: { color: "var(--color-danger)" } }, "■ "), `penalties −${fmtNum(pen)}`)));
  }

  // ------------------------------------------------------------------ term rows
  termRow(group, t, { weight } = {}) {
    const o = this.termOv(group, t.name);
    const off = o.enabled === false;
    const key = `${group}:${t.name}`;
    const row = h(`div.term${off ? ".off" : ""}${this.open.has(key) ? ".open" : ""}`);
    const sw = switchEl("", !off, (on) => {
      this.setTerm(group, t.name, { enabled: on ? undefined : false });
      row.classList.toggle("off", !on);
    }, off ? "enable" : "disable");
    let weightBox = h("span");
    const mod = h("button.mod-dot", { type: "button" });
    const syncMod = () => {
      const cur = this.termOv(group, t.name);
      const n = ("weight" in cur ? 1 : 0) + Object.keys(cur.params || {}).length + (cur.enabled === false ? 1 : 0);
      mod.classList.toggle("on", n > 0);
      mod.title = n ? `${n} change(s) — click to reset` : "";
      weightBox.classList?.toggle("mod", "weight" in cur);
    };
    mod.addEventListener("click", () => {
      this.setTerm(group, t.name, { weight: undefined, params: undefined, enabled: undefined });
      this.render();
    });
    if (weight) {
      const w = input(o.weight ?? t.weight, { type: "number", step: stepFor(t.weight), title: `default ${fmtNum(t.weight)}` });
      w.addEventListener("input", () => {
        const v = Number(w.value);
        if (!Number.isFinite(v)) return;
        this.setTerm(group, t.name, { weight: v === t.weight ? undefined : v });
        syncMod();
      });
      weightBox = h("div.weight", w);
    } else if (t.mode || t.time_out) {
      weightBox = tag(t.time_out ? "time-out" : t.mode);
    }
    const expand = h("button.icon-btn.expand", { type: "button", title: "parameters" }, svgChev());
    const toggle = () => {
      row.classList.toggle("open");
      row.classList.contains("open") ? this.open.add(key) : this.open.delete(key);
    };
    expand.addEventListener("click", toggle);
    const name = h("div.name", { on: { click: toggle } }, h("b", t.name), h("span", `${t.func}${t.params.length ? ` · ${t.params.length} param${t.params.length === 1 ? "" : "s"}` : ""}`));
    const params = h("div.term-params",
      t.params.length
        ? t.params.map((p) => knob(p, o.params?.[p.key], (v) => { this.setParam(group, t, p.key, v, p.value); syncMod(); }))
        : h("div.none", "No tunable parameters."));
    row.append(h("div.term-head", sw, name, weightBox, mod, expand), params);
    syncMod();
    return row;
  }

  termsTab(group, text) {
    return h("div", h("p.help", { style: { marginBottom: "8px" } }, text),
      h("div.terms", this.schema[group].map((t) => this.termRow(group, t))));
  }

  // ------------------------------------------------------------------ flat knobs
  flatTab(groupName, sections) {
    const fields = this.schema[groupName];
    const used = new Set();
    const out = [];
    for (const [prefix, title, text] of sections) {
      const mine = fields.filter((f) => !used.has(f.key) && (prefix ? f.key.startsWith(prefix) : true));
      if (!mine.length) continue;
      mine.forEach((f) => used.add(f.key));
      const n = mine.filter((f) => f.key in (this.overrides[groupName] || {})).length;
      out.push(group(title, { collapsible: true, closed: prefix.endsWith("range."), key: `preset-${groupName}-${title}`, aside: n ? `${n} changed` : "" },
        text ? help(text) : null,
        h("div.knobs", mine.map((f) => knob(f, this.overrides[groupName]?.[f.key], (v) => this.setFlat(groupName, f.key, v, f.value), prefix)))));
    }
    return h("div", out);
  }

  // ------------------------------------------------------------------ footer
  renderFooter() {
    if (!this.schema) return;
    const n = this.changes();
    const m = this.editedModule();
    const t = this.current();
    const dirty = this.dirty();
    const vs = m ? "in this module" : `vs ${[this.baseId(), ...this.using].join(" + ")}`;
    const note = h("div.note", { title: `saved in ${this.configDir}` }, n ? h("b", `${n} change${n === 1 ? "" : "s"}`) : "No changes", ` ${vs}${dirty ? " · unsaved" : ""}`);
    const revert = dirty
      ? button("", () => this.revert(), { cls: "ghost", icon: "undo", title: "Revert unsaved changes" })
      : null;
    if (m) {
      this.frame.setFooter(note, revert,
        !m.builtin && dirty ? button("Save", () => this.saveModule(m.name, m.description), { icon: "save" }) : null,
        button("Save as module…", () => this.saveAs("module"), { cls: m.builtin ? "primary" : "ghost" }));
      return;
    }
    const live = this.playActive
      ? button("Apply live", () => this.applyLive(), { cls: "ghost", icon: "live", title: "Push the reward terms into the running Play session (training is not affected)" })
      : null;
    this.frame.setFooter(
      note,
      revert,
      live,
      t?.custom && dirty ? button("Save", () => this.save(t.preset), { icon: "save" }) : null,
      n || this.using.length || t?.custom ? button("Save as…", () => this.saveAs(), { cls: "ghost" }) : null,
      button("Train", () => this.train(), { cls: "primary", icon: "arrow", title: dirty ? "Saves first" : "" }),
    );
  }

  async revert() {
    const modulesChanged = !same(this.using, this.savedUsing);
    this.overrides = clone(this.saved);
    this.using = [...this.savedUsing];
    if (modulesChanged) await this.loadSchema();
    else this.render();
    this.app.touch();
  }

  async reload(selected) {
    const d = await api("/api/presets");
    this.tasks = d.tasks;
    this.modules = d.modules;
    this.selected = selected;
    this.saved = clone(this.overrides);
    this.savedUsing = [...this.using];
    this.render();
    this.app.touch();
  }

  async save(name, description) {
    const base = this.baseId();
    await put(`/api/presets/${encodeURIComponent(name)}`, {
      base, overrides: this.overrides, modules: this.using,
      description: description ?? (this.current()?.custom ? this.current().description : ""),
    });
    await this.reload(`${base}--${name}`);
  }

  async saveModule(name, description = "") {
    await put(`/api/modules/${encodeURIComponent(name)}`, { overrides: this.overrides, description });
    await this.reload(MODULE + name);
  }

  /** Name + kind dialog: a preset (trainable) or a module (a reusable piece, no base). */
  saveAs(kind) {
    const m = this.editedModule();
    const t = this.current();
    let as = kind || "preset";
    const name = input(m ? `${m.name}_copy` : t?.custom ? `${t.preset}_copy` : `${this.baseId().toLowerCase().replace(/^wbc-/, "")}_custom`);
    const desc = input(m?.description || (t?.custom ? t.description : "") || "", { placeholder: "what it changes, in a few words" });
    const hint = h("p.help");
    const syncHint = () => {
      hint.textContent = as === "module"
        ? `Only the changes (${this.changes()}) are kept, without a base${this.using.length ? ` or the modules ${this.using.join(", ")}` : ""}: switch it on in any preset.`
        : `Trains as task ${this.baseId()}--<name>, logs to <experiment>--<name>. Saved in ${this.configDir}.`;
    };
    syncHint();
    const kinds = m ? null : field("Save as", seg([
      { value: "preset", label: "Preset · trainable" },
      { value: "module", label: "Module · reusable piece" },
    ], as, (v) => { as = v; syncHint(); }, { block: true }));
    return new Promise((resolve) => {
      const finish = async (ok) => {
        dlg.close();
        if (!ok || !name.value.trim()) return resolve(false);
        try {
          if (as === "module") await this.saveModule(name.value.trim(), desc.value.trim());
          else await this.save(name.value.trim(), desc.value.trim());
          resolve(true);
        } catch (e) {
          toast(String(e.message || e), "error");
          resolve(false);
        }
      };
      const dlg = modal(m || kind === "module" ? "Save module" : "Save", [kinds, field("Name", name), field("Description", desc), hint], [
        button("Cancel", () => finish(false), { cls: "ghost" }),
        button("Save", () => finish(true), { cls: "primary" }),
      ], { width: 460, onClose: () => resolve(false) });
      name.addEventListener("keydown", (e) => e.key === "Enter" && finish(true));
      setTimeout(() => { name.focus(); name.select(); }, 30);
    });
  }

  async remove() {
    const m = this.editedModule();
    const t = this.current();
    if (m && !m.builtin) {
      if (!confirm(`Delete module ${m.name}? Presets using it can't be built until you remove it from them; runs keep their snapshot.`)) return;
      await del(`/api/modules/${encodeURIComponent(m.name)}`);
    } else if (t?.custom) {
      if (!confirm(`Delete preset ${t.preset}? Runs trained with it keep their own snapshot.`)) return;
      await del(`/api/presets/${encodeURIComponent(t.preset)}`);
    } else return;
    this.saved = this.overrides = {};
    this.using = this.savedUsing = [];
    const fallback = t?.custom ? t.base : this.tasks.find((x) => !x.custom)?.id || "";
    await this.refresh();
    await this.choose(fallback, { force: true });
  }

  async applyLive() {
    const r = await post("/api/play/rewards", { rewards: this.overrides.rewards || {} });
    toast(`${r.changed} reward term(s) updated in the Play session`, "ok");
  }

  async train() {
    if (this.dirty() || (!this.isCustom() && this.using.length)) {
      if (this.isCustom()) await this.save(this.current().preset);
      else if (!(await this.saveAs("preset"))) return;
    }
    this.app.panes.train.prefill({ task: this.selected });
  }
}

// ---------------------------------------------------------------------- knobs
function stepFor(v) {
  const a = Math.abs(v || 0);
  if (!a) return 0.1;
  return 10 ** Math.floor(Math.log10(a) - 1);
}

function svgChev() {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.innerHTML = '<path d="M9 6l6 6-6 6"/>';
  return s;
}

/** One editable value of the schema ({key, kind, value, choices, help}); ``ov`` = override. */
function knob(f, ov, onChange, prefix = "") {
  const cur = ov === undefined ? f.value : ov;
  const label = prefix && f.key.startsWith(prefix) ? f.key.slice(prefix.length) : f.key;
  const el = h(`div.knob${ov !== undefined ? ".mod" : ""}${f.kind === "bool" ? ".bool" : ""}`, { title: `${f.key} · default ${JSON.stringify(f.value)}` });
  const mark = (v) => el.classList.toggle("mod", JSON.stringify(v) !== JSON.stringify(f.value));
  const emit = (v) => { mark(v); onChange(v); };
  let control;
  if (f.kind === "bool") {
    control = switchEl(pretty(label), cur, (v) => emit(v));
    el.append(control);
  } else {
    if (f.kind === "choice") {
      control = select(f.choices, cur, (v) => emit(v));
    } else if (f.kind === "range") {
      const lo = input(cur[0], { type: "number", step: stepFor(f.value[1] - f.value[0]) });
      const hi = input(cur[1], { type: "number", step: stepFor(f.value[1] - f.value[0]) });
      const read = () => { const a = Number(lo.value); const b = Number(hi.value); if (Number.isFinite(a) && Number.isFinite(b)) emit([a, b]); };
      lo.addEventListener("input", read);
      hi.addEventListener("input", read);
      control = h("div.range", lo, h("i", "to"), hi);
    } else if (f.kind === "ints") {
      control = input(cur.join(", "), {});
      control.addEventListener("change", () => {
        const v = control.value.split(/[\s,]+/).filter(Boolean).map(Number);
        if (v.length && v.every(Number.isInteger)) emit(v);
        else control.value = cur.join(", ");
      });
    } else {
      control = input(cur, { type: "number", step: f.kind === "int" ? 1 : stepFor(f.value), min: f.kind === "int" && f.key.includes("history") ? 0 : undefined });
      control.addEventListener("input", () => {
        const v = Number(control.value);
        if (control.value !== "" && Number.isFinite(v)) emit(f.kind === "int" ? Math.max(0, Math.round(v)) : v);
      });
    }
    el.append(h("label", h("span", pretty(label))), control);
  }
  if (f.help) el.append(h("div.hint", f.help));
  return el;
}
