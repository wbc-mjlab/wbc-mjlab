// Step 4 · Train: preset (step 3) + dataset (step 2) → a detached wbc-mjlab-train job;
// follow jobs (progress, curves, log). Runs are browsed in Play (step 5).
import { lineChart } from "../chart.js";
import {
  PaneFrame, api, button, del, empty, field, fmtDuration, group, h, help, iconButton, input, linkButton, post,
  progressBar, seg, select, switchEl, tabs, tag, toast,
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
  "Policy/mean_std": "Action std",
  "Perf/total_fps": "Steps / s",
};
const ITERS = [1000, 5000, 30000, 100000];

export class TrainPane {
  constructor(app) {
    this.app = app;
    this.topics = ["jobs", "robot", "data", "presets"];
    this.live = ["jobs"];
    this.width = 440;
    this.frame = new PaneFrame("Train", "Launch PPO on the GPU; jobs keep running when the Studio closes.");
    this.tab = "new";
    this.tasks = [];
    this.datasets = [];
    this.runs = [];
    this.jobs = [];
    this.form = { task: "", dataset: "", num_envs: 4096, max_iterations: 30000, run_name: "", logger: "tensorboard", mirror: false, debias_z: false, resume_run: "", resume_checkpoint: "", extra: "" };
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
    await this.loadJobs();
    this.render();
  }

  async onEvent(topic) {
    if (topic === "jobs") {
      await this.loadJobs();
      if (this.tab === "jobs") this.render();
      else this.renderTabs();
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
    if (changed) this.render();
  }

  prefill(values) {
    Object.assign(this.form, values);
    this.tab = "new";
    this.app.select("train");
    this.refresh();
  }

  async loadJobs() {
    this.jobs = await api("/api/jobs?kind=train");
  }

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
    const taskSel = select([
      { group: "Built-in", items: this.tasks.filter((x) => !x.custom).map((x) => ({ value: x.id, label: x.id })) },
      ...(this.tasks.some((x) => x.custom) ? [{ group: "Your presets", items: this.tasks.filter((x) => x.custom).map((x) => ({ value: x.id, label: x.preset })) }] : []),
    ], f.task, (v) => { f.task = v; f.resume_run = ""; this.render(); });
    const dsSel = select(this.datasets.length ? this.datasets.map((d) => ({ value: d.name, label: `${d.name} · ${d.raw || d.npz} clips` })) : [{ value: "", label: "add a dataset first" }], f.dataset, set("dataset"));
    const ds = this.datasets.find((d) => d.name === f.dataset);
    const sameTask = this.runs.filter((r) => r.task === f.task && !r.demo);
    const resume = select([{ value: "", label: "Fresh policy" }, ...sameTask.map((r) => ({ value: r.label, label: `Continue ${r.name}` }))], f.resume_run, (v) => { f.resume_run = v; f.resume_checkpoint = ""; this.render(); });
    const run = sameTask.find((r) => r.label === f.resume_run);
    const iters = input(f.max_iterations, { type: "number", min: 1, step: 1000, onChange: (v) => { f.max_iterations = v; this.render(); } });
    this.cmdEl = h("div.cmd");
    this.frame.setBody(
      group("Inputs", {},
        h("div.inputs",
          h("div.input-row", h("span", "Preset"), taskSel, button("", () => this.app.select("preset"), { cls: "ghost small", icon: "sliders", title: "Open in Preset" })),
          h("div.input-row", h("span", "Dataset"), dsSel, button("", () => this.app.select("data"), { cls: "ghost small", icon: "database", title: "Open in Data" })),
          h("div.input-row", h("span", "Start"), resume, run ? select([{ value: "", label: "latest" }, ...run.checkpoints.map((c) => ({ value: c, label: c }))], f.resume_checkpoint, set("resume_checkpoint")) : h("span"))),
        t ? h("p.help", { style: { marginTop: "8px" } }, t.description) : null),
      group("Budget", {},
        h("div.stack",
          h("div.grid2",
            field("Parallel envs", input(f.num_envs, { type: "number", min: 1, step: 256, onChange: set("num_envs") })),
            field("Iterations", iters)),
          h("div.chips", ITERS.map((n) => {
            const b = h(`button${f.max_iterations === n ? ".is-active" : ""}`, { type: "button" }, n >= 1000 ? `${n / 1000}k` : String(n));
            b.addEventListener("click", () => { f.max_iterations = n; this.render(); });
            return b;
          })))),
      group("Motion", {},
        h("div.stack",
          switchEl("Mirror — add a left/right mirrored copy of every clip", f.mirror, set("mirror")),
          switchEl("Height debias — lowest foot on the ground in each clip", f.debias_z, set("debias_z")),
          h("p.help", ds && !ds.raw
            ? "This dataset has converted clips only: mirror and debias were set when it was converted."
            : "Clips are resampled, differentiated and mirrored on the GPU while training: nothing to convert."))),
      group("Run", {},
        h("div.grid2",
          field("Name", input(f.run_name, { placeholder: "optional suffix", onInput: set("run_name") })),
          field("Logger", seg(["tensorboard", "wandb"], f.logger, set("logger"), { block: true })))),
      group("Advanced", { collapsible: true, closed: true, key: "train-adv" },
        field("Extra arguments", input(f.extra, { placeholder: "--env.commands.motion.rsi.bin-width-s 2.0", onInput: set("extra") })),
        field("Equivalent command", this.cmdEl)),
    );
    this.renderCmd();
    this.frame.setFooter(
      h("div.note", f.task ? h("b", t?.custom ? t.preset : f.task) : "", f.dataset ? ` on ${f.dataset}` : ""),
      button("Launch", () => this.launch(), { cls: "primary", icon: "play", disabled: !f.task || !f.dataset }),
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
    if (f.mirror) parts.push("--mirror");
    if (f.debias_z) parts.push("--debias-z");
    if (f.run_name) parts.push(`--agent.run-name ${f.run_name}`);
    if (f.logger === "wandb") parts.push("--use_wandb");
    if (run) parts.push(`--agent.resume True --agent.load-run ${run.name}`, f.resume_checkpoint ? `--agent.load-checkpoint ${f.resume_checkpoint}` : "");
    if (f.extra) parts.push(f.extra);
    this.cmdEl.textContent = parts.filter(Boolean).join(" ");
  }

  async launch() {
    const r = await post("/api/train/launch", this.form);
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
        h("div.job-head", h("b", `${preset} · ${j.meta.dataset}`), tag(j.status, tone)),
        h("div.job-meta",
          `${j.started_text} · ${fmtDuration(j.elapsed)}`,
          pr.iteration !== undefined ? ` · it ${pr.iteration + 1}/${pr.total}` : "",
          pr.mean_reward ? ` · reward ${Number(pr.mean_reward).toFixed(2)}` : ""),
        j.status === "running" ? progressBar(pr.fraction) : null);
      card.addEventListener("click", () => { this.openJob = j.name; this.render(); });
      return card;
    });
    this.frame.setBody(cards.length ? h("div.jobs", cards) : empty("bolt", "No training jobs", "Start one from New run.", button("New run", () => { this.tab = "new"; this.render(); }, { cls: "small" })));
    this.frame.setFooter();
  }

  async renderJob() {
    const name = this.openJob;
    const j = this.jobs.find((x) => x.name === name);
    if (!j) {
      this.openJob = "";
      return this.renderJobs();
    }
    const d = await api(`/api/jobs/${name}?tail=30`);
    if (this.openJob !== name || this.tab !== "jobs") return;
    const pr = d.progress || {};
    const tone = { running: "accent", done: "ok", failed: "bad", stopped: "amber" }[d.status];
    const runLabel = d.run_dir ? d.run_dir.split("/").slice(-2).join("/") : "";
    this.curvesHost ||= h("div.charts");
    this.frame.setBody(
      group("", {},
        h("div.row.between",
          h("button.link", { on: { click: () => { this.openJob = ""; this.render(); } } }, "‹ All jobs"),
          tag(d.status, tone)),
        h("div", { style: { marginTop: "8px", fontWeight: "600", fontSize: "0.95rem" } }, `${j.meta.task} · ${j.meta.dataset}`),
        h("div.job-meta", `${j.started_text} · ${fmtDuration(j.elapsed)} · ${j.meta.num_envs} envs${j.meta.resume ? ` · from ${j.meta.resume}` : ""}`),
        pr.total ? h("div", { style: { marginTop: "10px" } }, progressBar(pr.fraction), h("div.job-meta", { style: { marginTop: "4px" } }, `iteration ${pr.iteration + 1} of ${pr.total}${pr.mean_reward ? ` · mean reward ${Number(pr.mean_reward).toFixed(2)}` : ""}${pr.eta ? ` · eta ${pr.eta}` : ""}`)) : null),
      group("Curves", { action: seg(["train", "errors", "rewards", "sampling", "loss"], this.curveGroup, (g) => { this.curveGroup = g; this.loadCurves(); }) }, this.curvesHost),
      group("Log", { collapsible: true, closed: d.status !== "failed", key: "train-log" }, h("pre.log", d.log || "(no output yet)"), h("div.cmd", d.command.replace(/^.*python\S* -m /, "python -m "))),
    );
    this.loadCurves();
    this.frame.setFooter(
      d.status === "running" ? button("Stop", () => post(`/api/jobs/${name}/stop`), { cls: "danger", icon: "stop" }) : iconButton("trash", async () => { await del(`/api/jobs/${name}`); this.openJob = ""; await this.loadJobs(); this.render(); }, "Forget this job (the run folder stays)"),
      h("span.spacer"),
      runLabel ? linkButton("TensorBoard", `/tensorboard?run=${encodeURIComponent(runLabel.split("/")[1])}`, { cls: "ghost", icon: "link" }) : null,
      runLabel ? button("Play", () => this.app.panes.play.openRun(runLabel), { cls: "primary", icon: "play" }) : null,
    );
  }

  async loadCurves() {
    const host = this.curvesHost;
    const d = await api(`/api/curves?${new URLSearchParams({ job: this.openJob, group: this.curveGroup })}`);
    const charts = Object.entries(d.series).map(([tag_, s]) => lineChart(CURVE_TITLES[tag_] || tag_.replace(/^Episode_Reward\//, ""), s.x, [{ label: tag_, values: s.y }], {}));
    host.className = charts.length > 1 ? "charts two" : "charts";
    host.replaceChildren(...(charts.length ? charts : [help(d.path ? "No scalars yet — they appear after the first logged iterations." : "Waiting for the run folder…")]));
  }
}

