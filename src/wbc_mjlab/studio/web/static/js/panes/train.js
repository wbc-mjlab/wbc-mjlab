// Step 4 · Train: preset (step 3, what to learn) + dataset (step 2) + recipe (how: the method,
// PPO or the world model, and its settings) → a detached wbc-mjlab-train job; follow jobs
// (progress, curves, log). Runs are browsed in Play.
import { lineChart } from "../chart.js";
import { logBlock } from "../logview.js";
import { clearJobs, openStorage, removeJob } from "../storage.js";
import {
  PaneFrame, api, button, del, dot, empty, field, fmtDuration, group, h, help, iconButton, input, knob, linkButton,
  post, progressBar, prompt, put, seg, select, switchEl, tabs, tag, toast,
} from "../ui.js";

const CURVE_TITLES = {
  "Train/mean_reward": "Mean reward",
  "Train/mean_episode_length": "Episode length",
  "Metrics/motion/error_anchor_pos": "Anchor position error",
  "Metrics/motion/error_body_pos": "Body position error",
  "Metrics/motion/error_joint_pos": "Joint position error",
  "Metrics/motion/error_body_rot": "Body rotation error",
  "Metrics/motion/sampling_entropy": "RSI sampling entropy",
  "Metrics/motion/assist_gain_mean": "Assist gain",
  "Loss/value": "Value loss",
  "Loss/surrogate": "Surrogate loss",
  "Loss/entropy": "Entropy",
  "Loss/critic": "Critic loss",
  "Loss/wm_dynamics": "World model · dynamics loss",
  "Loss/wm_heads": "World model · heads loss",
  "Loss/wm_error_8_step": "World model · 8-step error",
  "Loss/imagined_return": "Imagined return",
  "Loss/imagined_reward": "Imagined reward",
  "Loss/imagined_termination": "Imagined termination",
  "Policy/mean_std": "Action std",
  "Perf/total_fps": "Steps / s",
};
const METHOD_LABEL = { ppo: "PPO", wm: "World model" };
const METHOD_OPTION = { ppo: "PPO", wm: "World model · experimental" };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// settings labelled without their config prefix
const pretty = (key) => key.replace(/^(algorithm|actor|critic)\./, "").replace(/_/g, " ");
const valueText = (v) => (Array.isArray(v) ? v.join(" × ") : String(v));

export class TrainPane {
  constructor(app) {
    this.app = app;
    this.topics = ["jobs", "robot", "data", "presets", "recipes"];
    this.live = ["jobs"];
    this.width = 440;
    this.frame = new PaneFrame("Train", "Train a preset with a recipe: the method (PPO or a world model) and its settings. Jobs keep running when the Studio closes.");
    this.tab = "new";
    this.tasks = [];
    this.datasets = [];
    this.runs = [];
    this.jobs = [];
    this.methods = []; // PPO and the world model, with every setting of each for the chosen preset
    this.recipes = []; // a method + settings, by name (built-in and yours)
    this.architecture = []; // settings a checkpoint fixes
    this.methodsTask = "";
    this.timing = {}; // method → measured seconds per iteration of its runs here
    this.checks = {}; // `${task}|${algo}|${dataset}` → { busy } | { ok, error, details }
    // algo_settings: the recipe's settings, as edited here
    this.form = { task: "", dataset: "", algo: "ppo", recipe: "ppo", algo_settings: {}, num_envs: 4096, max_iterations: 30000, run_name: "", logger: "tensorboard", mirror: false, debias_z: false, resume_run: "", resume_checkpoint: "", extra: "" };
    this.openJob = "";
    this.curveGroup = "train";
  }

  summary() {
    const running = this.jobs.filter((j) => j.status === "running");
    if (running.length) {
      const p = running[0].progress || {};
      return { value: `${running.length} running${p.fraction ? ` · ${Math.round(100 * p.fraction)}%` : ""}`, busy: true, done: true };
    }
    const done = this.jobs.filter((j) => j.status === "done").length;
    return { value: this.jobs.length ? `${done} done · ${this.jobs.length} jobs` : "no jobs yet", done: done > 0 };
  }

  async refresh() {
    const [presets, ds, runs] = await Promise.all([api("/api/presets"), api("/api/data/datasets"), api("/api/runs")]);
    this.tasks = presets.tasks;
    this.datasets = ds.datasets.filter((d) => d.raw > 0 || d.npz > 0);
    this.runs = runs;
    const preset = this.app.panes.preset?.taskId?.();
    if (!this.tasks.some((t) => t.id === this.form.task)) this.form.task = preset || this.tasks[0]?.id || "";
    const picked = this.app.panes.data?.dataset;
    if (!this.datasets.some((d) => d.name === this.form.dataset)) this.form.dataset = (this.datasets.find((d) => d.name === picked) || this.datasets.find((d) => d.name === "samples") || this.datasets[0])?.name || "";
    this.methodsTask = ""; // a preset's defaults may have changed
    this.timing = {}; // runs may have finished
    await Promise.all([this.loadJobs(), this.loadMethods()]);
    this.render();
  }

  async onEvent(topic) {
    if (topic === "jobs") {
      await this.loadJobs();
      if (this.tab === "jobs") this.render();
      else this.renderTabs();
      return;
    }
    if (topic === "presets") this.checks = {};
    if (topic === "recipes") {
      this.methodsTask = "";
      await this.loadMethods();
      this.render();
      return;
    }
    await this.refresh();
  }

  onShow() {
    const preset = this.app.panes.preset?.taskId?.();
    const ds = this.app.panes.data?.dataset;
    let changed = false;
    if (preset && preset !== this.form.task && this.tasks.some((t) => t.id === preset)) {
      this.form.task = preset;
      this.form.resume_run = "";
      changed = true;
    }
    if (ds && ds !== this.form.dataset && this.datasets.some((d) => d.name === ds)) {
      this.form.dataset = ds;
      changed = true;
    }
    if (changed) this.loadMethods().then(() => this.render());
  }

  prefill(values) {
    Object.assign(this.form, values);
    this.tab = "new";
    this.app.select("train");
    this.refresh();
  }

  async loadJobs() {
    this.jobs = (await api("/api/jobs")).filter((j) => ["train", "evaluate"].includes(j.kind));
  }

  /** The methods, their settings for the chosen preset (defaults differ per preset) and the
   * recipes. */
  async loadMethods() {
    const task = this.form.task;
    if (!task || task === this.methodsTask) return;
    const d = await api(`/api/train/methods?task=${encodeURIComponent(task)}`);
    this.methods = d.methods;
    this.recipes = d.recipes;
    this.architecture = d.architecture;
    this.methodsTask = task;
    const f = this.form;
    if (!this.method()) f.algo = "ppo";
    if (!this.recipe()) this.useRecipe(this.method().default_recipe);
    // a preset change can drop settings the other preset had
    const known = new Set(this.method().settings.map((s) => s.key));
    for (const k of Object.keys(f.algo_settings)) if (!known.has(k)) delete f.algo_settings[k];
    this.loadTiming();
  }

  method() {
    return this.methods.find((m) => m.id === this.form.algo);
  }

  /** The recipe the form started from (null if it was deleted). */
  recipe() {
    return this.recipes.find((r) => r.name === this.form.recipe && r.algo === this.form.algo) || null;
  }

  /** The form's settings differ from its recipe's. */
  modified() {
    const r = this.recipe();
    return !r || !same(this.ownSettings(), this.recipeSettings(r));
  }

  /** The form's settings, without those a continued run fixes. */
  ownSettings() {
    const out = { ...this.form.algo_settings };
    if (this.resumed()) for (const k of this.architecture) delete out[k];
    return out;
  }

  recipeSettings(r) {
    const known = new Set(this.method()?.settings.map((s) => s.key) || []);
    return Object.fromEntries(Object.entries(r.settings).filter(([k]) => known.has(k)));
  }

  /** The run being continued, if any. */
  resumed() {
    return this.runs.find((r) => r.label === this.form.resume_run) || null;
  }

  // ------------------------------------------------------------------ method
  setAlgo(id) {
    const f = this.form;
    if (id === f.algo) return;
    // the policy network is the same for both methods: keep it
    const policy = Object.fromEntries(Object.entries(f.algo_settings).filter(([k]) => k.startsWith("actor.")));
    f.algo = id;
    f.resume_run = "";
    f.resume_checkpoint = "";
    this.useRecipe(this.method().default_recipe);
    Object.assign(f.algo_settings, policy);
    const m = this.method();
    if (!m.iterations.includes(f.max_iterations)) f.max_iterations = m.iterations[2] ?? m.iterations[0];
    this.loadTiming();
    this.render();
  }

  useRecipe(name) {
    const f = this.form;
    const r = this.recipes.find((x) => x.name === name);
    if (!r) return;
    f.recipe = name;
    f.algo = r.algo;
    f.algo_settings = { ...this.recipeSettings(r) };
  }

  setSetting(s, v) {
    const f = this.form;
    if (same(v, s.value)) delete f.algo_settings[s.key];
    else f.algo_settings[s.key] = v;
    if (s.key === "actor.architecture") { this.render(); return; }
    this.renderCmd();
    this.renderRecipeRow();
  }

  async saveRecipe() {
    const f = this.form;
    const r = this.recipe();
    const name = await prompt("Save recipe", {
      label: "Name",
      value: r && !r.builtin ? r.name : "",
      placeholder: f.algo === "wm" ? "e.g. wm-short-horizon" : "e.g. ppo-big-net",
      hint: `${METHOD_LABEL[f.algo]} with ${Object.keys(this.ownSettings()).length} setting(s) changed from the preset's defaults. Any preset can train with it.`,
    });
    if (!name) return;
    const saved = await put(`/api/recipes/${encodeURIComponent(name)}`, { algo: f.algo, settings: this.ownSettings() });
    toast(`recipe ${saved.name} saved`, "ok");
    this.methodsTask = "";
    f.recipe = saved.name;
    await this.loadMethods();
    this.render();
  }

  async deleteRecipe() {
    const r = this.recipe();
    if (!r || r.builtin || !confirm(`Delete recipe ${r.name}? Runs trained with it keep their settings.`)) return;
    await del(`/api/recipes/${encodeURIComponent(r.name)}`);
    this.methodsTask = "";
    this.form.recipe = "";
    await this.loadMethods();
    this.render();
  }

  async loadTiming() {
    const algo = this.form.algo;
    if (algo in this.timing) return;
    this.timing[algo] = [];
    try {
      this.timing[algo] = await api(`/api/train/timing?algo=${algo}`);
    } catch {
      return;
    }
    if (this.tab === "new" && this.form.algo === algo) this.renderEstimate();
  }

  checkKey() {
    const f = this.form;
    return `${f.task}|${f.algo}|${f.dataset}`;
  }

  /** Whether the method can train the preset (world model: a 2-env copy is built and checked). */
  check() {
    const f = this.form;
    if (f.algo === "ppo" || !f.task || !f.dataset) return null;
    const key = this.checkKey();
    if (!this.checks[key]) {
      this.checks[key] = { busy: true };
      post("/api/train/check", { task: f.task, algo: f.algo, dataset: f.dataset })
        .then((r) => { this.checks[key] = r; })
        .catch((e) => { this.checks[key] = { ok: false, error: String(e.message || e) }; })
        .finally(() => { if (this.tab === "new" && this.checkKey() === key) this.render(); });
    }
    return this.checks[key];
  }

  /** Method + recipe: how the preset is trained. */
  methodGroup() {
    const f = this.form;
    const m = this.method();
    if (!m) return group("How to train", {}, help("loading…"));
    const c = this.check();
    let status = null;
    if (c?.busy) {
      status = h("div.check", dot("accent"), h("span", "Checking that the world model can train this preset — building a 2-env copy (~10 s)…"));
    } else if (c && !c.ok) {
      status = h("div.check", dot("bad"), h("span", h("b", "The world model can't train this preset. "), c.error));
    } else if (c?.ok) {
      const d = c.details || {};
      const learned = d.learned_rewards?.length ? `learned from the simulator: ${d.learned_rewards.join(", ")}` : "every reward computed exactly";
      status = h("div.check", dot("ok"), h("span", h("b", "Ready. "), `${d.exact_rewards?.length || 0} rewards computed exactly in imagination, ${learned}; randomization given to the model: ${d.context?.join(", ") || "none"}.`));
    }
    const methodSel = select(this.methods.map((x) => ({ value: x.id, label: METHOD_OPTION[x.id] || x.label })), f.algo, (v) => this.setAlgo(v));
    this.recipeRow = h("div.input-row");
    this.renderRecipeRow();
    return group("How to train", {},
      h("div.inputs",
        h("div.input-row", h("span", "Method"), methodSel, h("span")),
        this.recipeRow),
      h("div.stack", { style: { marginTop: "10px" } },
        help(m.summary),
        status,
        m.notes ? help(m.notes) : null));
  }

  /** Recipe picker, its state against the form and the save / delete actions. */
  renderRecipeRow() {
    if (!this.recipeRow) return;
    const f = this.form;
    const r = this.recipe();
    const mine = this.recipes.filter((x) => x.algo === f.algo);
    const options = [
      { group: "Built-in", items: mine.filter((x) => x.builtin).map((x) => ({ value: x.name, label: x.label })) },
      ...(mine.some((x) => !x.builtin) ? [{ group: "Yours", items: mine.filter((x) => !x.builtin).map((x) => ({ value: x.name, label: x.name })) }] : []),
    ];
    if (!r) options.unshift({ value: "", label: "custom settings" });
    const sel = select(options, r ? r.name : "", (v) => { this.useRecipe(v); this.render(); });
    sel.title = r?.description || "";
    const modified = this.modified();
    const save = button("", () => this.saveRecipe(), { cls: modified ? "small" : "ghost small", icon: "save", title: modified ? "Save these settings as a recipe" : "Save as a new recipe" });
    const remove = r && !r.builtin ? iconButton("trash", () => this.deleteRecipe(), "Delete this recipe") : null;
    this.recipeRow.replaceChildren(h("span", "Recipe"), sel, h("div.row", save, remove));
    if (this.settingsAside) {
      const n = Object.keys(this.ownSettings()).length;
      this.settingsAside.textContent = `${n ? `${n} changed` : "preset defaults"}${modified && r ? " · not saved" : ""}`;
    }
  }

  /** Every setting of the method, the main ones first; the preset's values are the defaults. */
  settingsGroup() {
    const f = this.form;
    const m = this.method();
    if (!m) return null;
    const run = this.resumed();
    const locked = new Set(run ? this.architecture : []);
    const row = (s) => {
      if (locked.has(s.key) && run.network?.[s.key] !== undefined) {
        return h("div.knob.locked", { title: `fixed by ${run.name}'s checkpoint` },
          h("label", h("span", pretty(s.key))), h("div.locked-value", valueText(run.network[s.key])), h("div.hint", "from the run you continue"));
      }
      return knob(s, f.algo_settings[s.key], (v) => this.setSetting(s, v), "", { pretty });
    };
    const visible = m.settings.filter((s) => Object.entries(s.visible_if || {}).every(([key, value]) =>
      (run?.network?.[key] ?? f.algo_settings[key] ?? m.settings.find((x) => x.key === key)?.value) === value));
    const main = visible.filter((s) => s.main);
    const rest = visible.filter((s) => !s.main);
    const sections = [...new Set(rest.map((s) => s.section))];
    this.settingsAside = h("span");
    this.renderRecipeRow();
    return group("Settings", { collapsible: true, closed: true, key: `train-settings-${f.algo}`, aside: this.settingsAside },
      help(`Defaults are the preset's. Changes apply to this run; save them as a recipe to reuse with any preset.${run ? " Continuing a run keeps its networks." : ""}`),
      main.length ? h("div.knobs", { style: { marginTop: "10px" } }, main.map(row)) : null,
      sections.map((title) => h("div.obs-section", { style: { marginTop: "14px" } },
        h("div.obs-section-title", title),
        title === "Policy network" ? help("The same network for both methods: the one exported and deployed.") : null,
        h("div.knobs", rest.filter((s) => s.section === title).map(row)))));
  }

  /** Training time of the budget, from this method's runs here. */
  renderEstimate() {
    if (!this.estimateEl) return;
    const f = this.form;
    const all = this.timing[f.algo] || [];
    const own = all.filter((x) => x.task === f.task); // the same preset first
    const runs = own.length ? own : all;
    if (!runs.length) {
      this.estimateEl.textContent = "";
      return;
    }
    // the run whose env count is closest to this one's
    const best = runs.reduce((a, b) => (Math.abs((b.num_envs || 0) - f.num_envs) < Math.abs((a.num_envs || 0) - f.num_envs) ? b : a));
    const total = best.seconds * f.max_iterations;
    const envs = best.num_envs && best.num_envs !== Number(f.num_envs) ? ` with ${best.num_envs} envs` : "";
    const other = best.task !== f.task ? ` (${best.task.split("--").pop()})` : "";
    this.estimateEl.textContent = `≈ ${fmtDuration(total)} · ${best.seconds.toFixed(2)} s per iteration measured on ${best.run.split("/").pop()}${other}${envs}`;
  }

  // ------------------------------------------------------------------ tabs
  renderTabs() {
    const running = this.jobs.filter((j) => j.status === "running").length;
    this.frame.setTabs(tabs([
      { value: "new", label: "New run" },
      { value: "jobs", label: "Jobs", count: running ? `${running} running` : this.jobs.length || null, mod: !!running },
    ], this.tab, (t) => { this.tab = t; this.render(); }));
  }

  render() {
    this.renderTabs();
    if (this.tab === "new") this.renderForm();
    else if (this.openJob) this.renderJob();
    else this.renderJobs();
    this.app.touch();
  }

  // ------------------------------------------------------------------ new run
  renderForm() {
    const f = this.form;
    const set = (k) => (v) => { f[k] = v; this.renderCmd(); };
    const t = this.tasks.find((x) => x.id === f.task);
    const m = this.method();
    const taskSel = select([
      { group: "Built-in", items: this.tasks.filter((x) => !x.custom).map((x) => ({ value: x.id, label: x.id })) },
      ...(this.tasks.some((x) => x.custom) ? [{ group: "Your presets", items: this.tasks.filter((x) => x.custom).map((x) => ({ value: x.id, label: x.preset })) }] : []),
    ], f.task, async (v) => { f.task = v; f.resume_run = ""; await this.loadMethods(); this.render(); });
    const dsSel = select(this.datasets.length ? this.datasets.map((d) => ({ value: d.name, label: `${d.name} · ${d.raw || d.npz} clips` })) : [{ value: "", label: "add a dataset first" }], f.dataset, (v) => { f.dataset = v; this.render(); });
    const ds = this.datasets.find((d) => d.name === f.dataset);
    // a run continues with the method it was trained with
    const sameTask = this.runs.filter((r) => r.task === f.task && !r.demo && (r.algo || "ppo") === f.algo);
    const resume = select([{ value: "", label: "Fresh policy" }, ...sameTask.map((r) => ({ value: r.label, label: `Continue ${r.name}` }))], f.resume_run, (v) => { f.resume_run = v; f.resume_checkpoint = ""; this.render(); });
    const run = sameTask.find((r) => r.label === f.resume_run);
    const iters = input(f.max_iterations, { type: "number", min: 1, step: 1000, onChange: (v) => { f.max_iterations = v; this.render(); } });
    this.estimateEl = h("p.help");
    this.cmdEl = h("div.cmd");
    // jobs share the one GPU: a world model's update alone takes ~10 GB
    const running = this.jobs.filter((x) => x.status === "running");
    const busy = running.length
      ? h("div.callouts", h("div.callout.warn", h("b", `${running.length} training job${running.length === 1 ? " is" : "s are"} running on this GPU · `),
        `${running.map((x) => `${x.meta.task?.split("--").pop()} (${METHOD_LABEL[x.meta.algo || "ppo"]})`).join(", ")}. A second job slows both; two world-model jobs don't fit in 24 GB and the second runs out of memory.`))
      : null;
    this.frame.setBody(
      busy,
      group("Inputs", {},
        h("div.inputs",
          h("div.input-row", h("span", "Preset"), taskSel, button("", () => this.app.select("preset"), { cls: "ghost small", icon: "sliders", title: "Open in Preset" })),
          h("div.input-row", h("span", "Dataset"), dsSel, button("", () => this.app.select("data"), { cls: "ghost small", icon: "database", title: "Open in Data" })),
          h("div.input-row", h("span", "Start"), resume, run ? select([{ value: "", label: "latest" }, ...run.checkpoints.map((c) => ({ value: c, label: c }))], f.resume_checkpoint, set("resume_checkpoint")) : h("span"))),
        t ? h("p.help", { style: { marginTop: "8px" } }, t.description) : null),
      this.methodGroup(),
      this.settingsGroup(),
      group("Budget", {},
        h("div.stack",
          h("div.grid2",
            field("Parallel envs", input(f.num_envs, { type: "number", min: 1, step: 256, onChange: (v) => { f.num_envs = v; this.renderCmd(); this.renderEstimate(); } })),
            field("Iterations", iters)),
          h("div.chips", (m?.iterations || []).map((n) => {
            const b = h(`button${f.max_iterations === n ? ".is-active" : ""}`, { type: "button" }, n >= 1000 ? `${n / 1000}k` : String(n));
            b.addEventListener("click", () => { f.max_iterations = n; this.render(); });
            return b;
          })),
          this.estimateEl)),
      group("Motion", {},
        h("div.stack",
          switchEl("Mirror — add a left/right mirrored copy of every clip", f.mirror, set("mirror")),
          switchEl("Height debias — lowest foot on the ground in each clip", f.debias_z, set("debias_z")),
          h("p.help", ds && !ds.raw
            ? "This dataset has converted clips only: mirror and debias were set when it was converted."
            : "Clips are resampled, differentiated and mirrored on the GPU while training: nothing to convert."))),
      group("Run", {},
        h("div.grid2",
          field("Name", input(f.run_name, { placeholder: f.algo === "ppo" ? "optional suffix" : `optional suffix, e.g. ${f.algo}`, onInput: set("run_name") })),
          field("Logger", seg(["tensorboard", "wandb"], f.logger, set("logger"), { block: true })))),
      group("Advanced", { collapsible: true, closed: true, key: "train-adv" },
        field("Extra arguments", input(f.extra, { placeholder: "--env.commands.motion.rsi.bin-width-s 2.0", onInput: set("extra") })),
        field("Equivalent command", this.cmdEl)),
    );
    this.renderCmd();
    this.renderEstimate();
    const c = this.check();
    const blocked = !!(c && !c.busy && !c.ok);
    const r = this.recipe();
    const how = m ? ` · ${r ? r.label : METHOD_LABEL[m.id]}${this.modified() ? " (edited)" : ""}` : "";
    this.frame.setFooter(
      h("div.note", f.task ? h("b", t?.custom ? t.preset : f.task) : "", f.dataset ? ` on ${f.dataset}` : "", how),
      button("Launch", () => this.launch(), { cls: "primary", icon: "play", disabled: !f.task || !f.dataset || blocked, title: blocked ? "This method can't train the preset (see How to train)" : undefined }),
    );
  }

  renderCmd() {
    if (!this.cmdEl) return;
    const f = this.form;
    const t = this.tasks.find((x) => x.id === f.task);
    const run = this.runs.find((r) => r.label === f.resume_run);
    const parts = t?.custom
      ? ["python -m wbc_mjlab.studio.train_preset", `--preset ${t.preset}`]
      : ["uv run wbc-mjlab-train", `--task ${f.task}`];
    parts.push(`--dataset ${f.dataset}`, `--env.scene.num-envs ${f.num_envs}`, `--agent.max-iterations ${f.max_iterations}`);
    if (f.algo !== "ppo") parts.push(`--algo ${f.algo}`);
    const settings = { ...this.ownSettings(), ...(run?.network || {}) };
    for (const [k, v] of Object.entries(run ? settings : f.algo_settings)) parts.push(`--agent.${k.replace(/_/g, "-")} ${Array.isArray(v) ? `"(${v.join(", ")}${v.length === 1 ? "," : ""})"` : v}`);
    if (f.mirror) parts.push("--mirror");
    if (f.debias_z) parts.push("--debias-z");
    if (f.run_name) parts.push(`--agent.run-name ${f.run_name}`);
    if (f.logger === "wandb") parts.push("--use_wandb");
    if (run) parts.push(`--agent.resume True --agent.load-run ${run.name}`, f.resume_checkpoint ? `--agent.load-checkpoint ${f.resume_checkpoint}` : "");
    if (f.extra) parts.push(f.extra);
    this.cmdEl.textContent = parts.filter(Boolean).join(" ");
  }

  async launch() {
    const f = this.form;
    const r = await post("/api/train/launch", { ...f, algo_settings: this.ownSettings(), recipe: this.recipe() && !this.modified() ? f.recipe : "" });
    toast(`training started`, "ok");
    this.openJob = r.job;
    this.tab = "jobs";
    await this.loadJobs();
    this.render();
  }

  // ------------------------------------------------------------------ jobs
  renderJobs() {
    const cards = this.jobs.slice(0, 20).map((j) => {
      const tone = { running: "accent", done: "ok", failed: "bad", stopped: "amber" }[j.status];
      const pr = j.progress || {};
      const preset = j.meta.task?.includes("--") ? j.meta.task.split("--").pop() : j.meta.task;
      const card = h("div.job",
        h("div.job-head", h("b", `${preset} · ${j.meta.dataset}`), tag(j.status, tone),
          j.status === "running" ? null : iconButton("x", (e) => { e.stopPropagation(); this.forget(j); }, "Remove this job (and, if you choose, its run)")),
        h("div.job-meta",
          `${j.kind === "evaluate" ? "Evaluation" : METHOD_LABEL[j.meta.algo || "ppo"]}${j.meta.recipe ? ` · ${j.meta.recipe}` : ""} · ${j.started_text} · ${fmtDuration(j.elapsed)}`,
          j.kind === "evaluate" && pr.total ? ` · clips ${pr.done}/${pr.total}` : "",
          pr.iteration !== undefined ? ` · it ${pr.iteration + 1}/${pr.total}` : "",
          pr.mean_reward ? ` · reward ${Number(pr.mean_reward).toFixed(2)}` : "",
          j.status === "running" && pr.phase ? ` · ${pr.phase}` : ""),
        j.status === "running" ? progressBar(pr.fraction) : null);
      card.addEventListener("click", () => { this.openJob = j.name; this.render(); });
      return card;
    });
    this.frame.setBody(cards.length ? h("div.jobs", cards) : empty("bolt", "No training jobs", "Start one from New run.", button("New run", () => { this.tab = "new"; this.render(); }, { cls: "small" })));
    this.frame.setFooter(
      ...["train", "evaluate"].map((kind) => {
        const jobs = this.jobs.filter((j) => j.kind === kind);
        const count = jobs.filter((j) => j.status !== "running").length;
        return count ? button(`Clear ${kind === "train" ? "training" : "evaluations"} · ${count}`, () => clearJobs(kind, jobs, { onDone: () => this.reloadJobs() }), { cls: "ghost", icon: "trash" }) : null;
      }),
      button("Runs on disk", () => openStorage({ onChange: () => this.reloadJobs() }), { cls: "ghost", icon: "database", title: "Every run folder with its size: delete runs, prune old checkpoints" }),
      h("span.spacer"),
      linkButton("TensorBoard · all runs", "/tensorboard", { cls: "ghost", icon: "link", title: "Every run under logs/rsl_rl, CLI runs included" }),
    );
  }

  async renderJob() {
    const name = this.openJob;
    const j = this.jobs.find((x) => x.name === name);
    if (!j) {
      this.openJob = "";
      return this.renderJobs();
    }
    const d = await api(`/api/jobs/${name}?tail=40`);
    if (this.openJob !== name || this.tab !== "jobs") return;
    const pr = d.progress || {};
    const evaluation = j.kind === "evaluate";
    const tone = { running: "accent", done: "ok", failed: "bad", stopped: "amber" }[d.status];
    const runLabel = d.run_dir ? d.run_dir.split("/").slice(-2).join("/") : "";
    this.curvesHost ||= h("div.charts");
    this.frame.setBody(
      group("", {},
        h("div.row.between",
          h("button.link", { on: { click: () => { this.openJob = ""; this.render(); } } }, "‹ All jobs"),
          tag(d.status, tone)),
        h("div", { style: { marginTop: "8px", fontWeight: "600", fontSize: "0.95rem" } }, `${j.meta.task} · ${j.meta.dataset}`),
        h("div.job-meta", `${evaluation ? "Evaluation" : METHOD_LABEL[j.meta.algo || "ppo"]}${j.meta.recipe ? ` · recipe ${j.meta.recipe}` : ""}${Object.keys(j.meta.algo_settings || {}).length ? ` · ${Object.keys(j.meta.algo_settings).length} setting(s)` : ""} · ${j.started_text} · ${fmtDuration(j.elapsed)} · ${j.meta.num_envs} envs${j.meta.resume ? ` · from ${j.meta.resume}` : ""}`),
        pr.total ? h("div", { style: { marginTop: "10px" } }, progressBar(pr.fraction), h("div.job-meta", { style: { marginTop: "4px" } }, `${evaluation ? `clips ${pr.done}` : `iteration ${pr.iteration + 1}`} of ${pr.total}${pr.mean_reward ? ` · mean reward ${Number(pr.mean_reward).toFixed(2)}` : ""}${pr.eta ? ` · eta ${pr.eta}` : ""}${d.status === "running" && pr.phase ? ` · ${pr.phase}` : ""}`)) : null),
      evaluation ? null : group("Curves", { action: seg(["train", "errors", "rewards", "sampling", "loss"], this.curveGroup, (g) => { this.curveGroup = g; this.loadCurves(); }) }, this.curvesHost),
      group("Log", { collapsible: true, closed: !evaluation && d.status !== "failed", key: "train-log" }, logBlock(name, d.log, { title: `${j.meta.task} · ${j.meta.dataset} · log` }), h("div.cmd", d.command.replace(/^.*python\S* -m /, "python -m "))),
    );
    if (!evaluation) this.loadCurves();
    this.frame.setFooter(
      d.status === "running" ? button("Stop", () => post(`/api/jobs/${name}/stop`), { cls: "danger", icon: "stop" }) : iconButton("trash", () => this.forget(j), "Remove this job (and, if you choose, its run)"),
      h("span.spacer"),
      evaluation && d.status === "done" ? linkButton("Download JSON", `/api/jobs/${name}/result`, { cls: "primary", icon: "download" }) : null,
      runLabel ? linkButton("TensorBoard", `/tensorboard?run=${encodeURIComponent(runLabel.split("/")[1])}`, { cls: "ghost", icon: "link" }) : null,
      runLabel ? button("Play", () => this.app.panes.play.openRun(runLabel), { cls: "primary", icon: "play" }) : null,
    );
  }

  /** Remove a finished job (record and log), and its run folder if chosen. */
  forget(j) {
    const preset = j.meta.task?.includes("--") ? j.meta.task.split("--").pop() : j.meta.task;
    removeJob(j.name, `${preset} · ${j.meta.dataset}`, {
      onDone: () => {
        if (this.openJob === j.name) this.openJob = "";
        this.reloadJobs();
      },
    });
  }

  async reloadJobs() {
    await this.loadJobs();
    this.render();
  }

  async loadCurves() {
    const host = this.curvesHost;
    const d = await api(`/api/curves?${new URLSearchParams({ job: this.openJob, group: this.curveGroup })}`);
    const charts = Object.entries(d.series).map(([tag_, s]) => lineChart(CURVE_TITLES[tag_] || tag_.replace(/^Episode_Reward\//, ""), s.x, [{ label: tag_, values: s.y }], {}));
    host.className = charts.length > 1 ? "charts two" : "charts";
    host.replaceChildren(...(charts.length ? charts : [help(d.path ? "No scalars yet — they appear after the first logged iterations." : "Waiting for the run folder…")]));
  }
}
