// Step 5 · Play: a run's checkpoint (or the zero action = PD on the reference) on N envs over a
// motion library. Clip / env / push live on the stage; this pane picks the policy and shows the
// focused env's reward terms.
import { openStorage } from "../storage.js";
import { PaneFrame, api, bars, button, empty, field, group, h, help, iconButton, input, linkButton, listRow, post, seg, select, switchEl, tag, toast } from "../ui.js";

const ENVS = ["1", "4", "16", "64"];
const MAX_ENVS = 4096; // play.MAX_ENVS

/** The env count: presets, or any number typed in the box (applied on Enter / leaving it). */
function envPicker(value, onChange) {
  const box = input(value, { type: "number", min: 1, max: MAX_ENVS, step: 1, title: `Any number of envs, 1 to ${MAX_ENVS}` });
  const presets = seg(ENVS.map((n) => ({ value: n, label: n })), String(value), (v) => { box.value = v; onChange(Number(v)); }, { block: true });
  presets.style.flex = "1";
  box.style.width = "84px";
  box.addEventListener("change", () => {
    const n = Math.round(Number(box.value));
    if (!Number.isFinite(n) || n < 1 || n > MAX_ENVS) {
      toast(`Environments: 1 to ${MAX_ENVS}`, "warn");
      box.value = value;
      return;
    }
    box.value = n;
    for (const b of presets.querySelectorAll("button")) b.classList.toggle("is-active", b.textContent === String(n));
    onChange(n);
  });
  return h("div.row", { style: { gap: "8px", alignItems: "stretch" } }, presets, box);
}
const ZERO = "__zero__";

export class PlayPane {
  constructor(app) {
    this.app = app;
    this.topics = ["play", "robot", "jobs", "data", "presets"];
    this.live = ["play"];
    this.width = 420;
    this.frame = new PaneFrame("Play", "Run a policy on many envs — the ghost is the reference it tracks.");
    this.state = null;
    this.runs = [];
    this.libs = [];
    this.choice = { run: "", checkpoint: "", motion: "", num_envs: 16, terminations: false };
    this.picking = false;
    this.pendingClip = null;
  }

  summary() {
    const s = this.state;
    if (s?.active) return { value: `${s.num_envs} envs · ${s.checkpoint === s.zero ? "zero action" : s.checkpoint}`, done: true, busy: false };
    return { value: this.runs.length ? `${this.runs.length} runs` : "no runs", done: false };
  }

  async refresh() {
    [this.state, this.runs] = await Promise.all([api("/api/play"), api("/api/play/runs")]);
    const s = this.state;
    if (s.active && !this.picking) {
      Object.assign(this.choice, { run: s.run, checkpoint: s.checkpoint === s.zero ? ZERO : s.checkpoint, motion: s.motion, num_envs: s.num_envs, terminations: s.terminations });
    }
    if (!this.runs.some((r) => r.label === this.choice.run)) this.choice.run = this.runs.find((r) => !r.demo)?.label || this.runs[0]?.label || "";
    await this.loadLibraries();
    this.render();
  }

  async loadLibraries() {
    this.libs = this.choice.run ? await api(`/api/play/libraries?run=${encodeURIComponent(this.choice.run)}`) : [];
    if (!this.libs.some((l) => l.value === this.choice.motion)) {
      const own = this.libs.find((l) => l.label.startsWith("trained on"));
      const samples = this.libs.find((l) => /\/samples$/.test(l.value));
      this.choice.motion = (own || samples || this.libs[0])?.value || "";
    }
  }

  async onEvent(topic, payload) {
    if (topic === "play") {
      const was = this.state?.active;
      this.state = payload;
      if (this.pendingClip && payload.active && !this.app.state.task) {
        const clip = this.pendingClip;
        this.pendingClip = null;
        post("/api/play/clip", { clip }).catch(() => {});
      }
      if (was !== payload.active) {
        this.picking = false;
        this.render();
      } else this.renderSession();
      return;
    }
    await this.refresh();
  }

  openRun(label) {
    this.choice.run = label;
    this.choice.checkpoint = "";
    this.picking = true;
    this.app.select("play");
    this.loadLibraries().then(() => this.render());
  }

  prefillClip(dataset, clip) {
    this.app.select("play");
    const lib = this.libs.find((l) => l.value.endsWith(`/${dataset}`));
    if (this.state?.active && (!lib || this.state.motion === lib.value)) {
      post("/api/play/clip", { clip }).catch(() => {});
      return;
    }
    if (lib) this.choice.motion = lib.value;
    this.pendingClip = clip;
    this.picking = true;
    this.render();
    toast(`Load a policy: ${clip} will be pinned to env 0`, "info");
  }

  // ------------------------------------------------------------------ render
  render() {
    const s = this.state;
    if (!this.runs.length) {
      this.frame.setBody(empty("play", "No runs yet", "Train a policy, or use the bundled demo checkpoint (demos/wbc_g1)."));
      this.frame.setFooter();
      return;
    }
    if (s?.active && !this.picking) this.renderSession();
    else this.renderPicker();
    this.app.touch();
  }

  renderPicker() {
    const c = this.choice;
    // filter by training method once runs of both exist (the demo is always listed)
    const methods = new Set(this.runs.filter((r) => !r.demo).map((r) => r.algo || "ppo"));
    const filter = methods.size > 1 ? (this.methodFilter || "all") : "all";
    const shown = this.runs.filter((r) => filter === "all" || r.demo || (r.algo || "ppo") === filter || r.label === c.run);
    const rows = shown.map((r) => {
      const preset = r.task.includes("--") ? r.task.split("--").pop() : "";
      return listRow({
        lead: r.demo ? "bolt" : preset ? "sliders" : "layers",
        title: r.demo ? "Bundled demo" : preset || r.task || r.experiment,
        sub: `${preset ? `${r.task.split("--")[0]} + preset · ` : ""}${r.updated_text} · ${r.checkpoints.length} ckpt${r.name.includes("_", 19) ? ` · ${r.name.slice(20)}` : ""}`,
        trail: r.demo ? tag("demo", "accent") : r.algo === "wm" ? tag("world model", "", "Trained with the world model (--algo wm)") : null,
        active: r.label === c.run,
        onClick: async () => { c.run = r.label; c.checkpoint = ""; await this.loadLibraries(); this.render(); },
      });
    });
    const run = this.runs.find((r) => r.label === c.run);
    const ckpts = run?.checkpoints || [];
    if (!c.checkpoint || (c.checkpoint !== ZERO && !ckpts.includes(c.checkpoint))) c.checkpoint = ckpts[0] || ZERO;
    this.frame.setBody(
      group("Run", {
        action: h("div.row", { style: { gap: "12px" } },
          methods.size > 1
            ? seg([{ value: "all", label: "All" }, { value: "ppo", label: "PPO" }, { value: "wm", label: "World model" }], filter, (v) => { this.methodFilter = v; this.render(); })
            : h("span.aside", `${this.runs.length} on disk`),
          h("a.link", { href: "/tensorboard", target: "_blank", rel: "noopener", title: "Every run's training curves" }, "All in TensorBoard"),
          iconButton("database", () => openStorage({ onChange: () => this.refresh() }), "Runs on disk: sizes, delete runs, prune old checkpoints")),
      }, h("div.list", rows)),
      group("Setup", {},
        h("div.stack",
          field("Checkpoint", select([...ckpts.map((x) => ({ value: x, label: x })), { value: ZERO, label: "Zero action · PD on the reference" }], c.checkpoint, (v) => (c.checkpoint = v))),
          field("Motion library", select(this.libs.length ? this.libs : [{ value: "", label: "add a dataset in Data first" }], c.motion, (v) => (c.motion = v))),
          field("Environments", envPicker(c.num_envs, (n) => (c.num_envs = n))),
          switchEl("Terminations — falls end episodes", c.terminations, (v) => (c.terminations = v)))),
    );
    this.frame.setFooter(
      this.state?.active ? button("Cancel", () => { this.picking = false; this.render(); }, { cls: "ghost" }) : h("span.note", run ? run.task : ""),
      h("span.spacer"),
      tensorBoardButton(run),
      button(this.state?.active ? "Apply" : "Load policy", () => this.apply(), { cls: "primary", icon: "play", disabled: !c.motion }),
    );
  }

  async apply() {
    const c = this.choice;
    const r = await post("/api/play/configure", { ...c, checkpoint: c.checkpoint === ZERO ? "" : c.checkpoint });
    if (r.result !== "rebuilding") {
      this.picking = false;
      this.render();
    }
  }

  /** Change the running session's env count or terminations (the rest stays as it is). */
  async reconfigure(patch) {
    const s = this.state;
    const c = Object.assign(this.choice, {
      run: s.run, checkpoint: s.checkpoint === s.zero ? ZERO : s.checkpoint, motion: s.motion,
      num_envs: s.num_envs, terminations: s.terminations, ...patch,
    });
    const r = await post("/api/play/configure", { ...c, checkpoint: c.checkpoint === ZERO ? "" : c.checkpoint });
    if (r.result === "rebuilding") toast(`Rebuilding the session for ${c.num_envs} envs…`, "info");
  }

  renderSession() {
    const s = this.state;
    if (!s?.active || this.picking) return;
    const run = this.runs.find((r) => r.label === s.run);
    const lib = s.motion.split("/").pop();
    this.barsHost ||= h("div", help("No step yet."));
    this.frame.setBody(
      group("Policy", { action: h("button.link", { on: { click: () => { this.picking = true; this.render(); } } }, "Change") },
        h("div.list", listRow({
          lead: run?.demo ? "bolt" : "layers",
          title: run?.demo ? "Bundled demo" : (s.task.includes("--") ? s.task.split("--").pop() : s.task),
          sub: `${run?.algo === "wm" ? "world model · " : ""}${run?.updated_text || ""} · ${s.checkpoint === s.zero ? "zero action" : s.checkpoint} · ${lib} · ${s.num_envs} envs`,
          trail: tag(s.playing ? "live" : "paused", s.playing ? "accent" : "amber"),
          active: true,
        }))),
      group("Session", {},
        h("div.stack",
          field("Environments", envPicker(s.num_envs, (n) => this.reconfigure({ num_envs: n })),
            `up to ${s.capacity}: instant; more rebuilds the session`),
          switchEl("Terminations — falls end episodes", s.terminations, (v) => this.reconfigure({ terminations: v })))),
      group("View", {},
        h("div.stack",
          field("Reference ghost", seg([
            { value: "aligned", label: "Aligned to robot", title: "Reference anchored to the robot's xy + yaw — what the body rewards compare" },
            { value: "world", label: "World", title: "Where the clip is — shows global drift" },
          ], s.ghost_mode, (v) => post("/api/play/options", { ghost_mode: v }), { block: true })),
          switchEl("Keybody targets, coloured by error", s.show_targets, (v) => post("/api/play/options", { show_targets: v })),
          help("Pick the focused env's clip, push it or shuffle from the bar over the stage."))),
      group("Reward terms · focused env", {
        action: h("button.link", { on: { click: () => this.app.select("preset") }, title: "Edit the weights and push them into this session" }, "Tune"),
      }, this.barsHost),
    );
    this.frame.setFooter(
      button("Stop", () => post("/api/play/stop"), { cls: "danger ghost", icon: "stop" }),
      h("span.spacer"),
      tensorBoardButton(run),
      button("Restart all", () => post("/api/play/restart"), { icon: "refresh" }),
    );
  }

  hud(hud) {
    if (this.barsHost && hud.terms) this.barsHost.replaceChildren(bars(hud.terms, 24));
  }
}

/** The run's training curves in TensorBoard (started on demand), for runs that logged there. */
function tensorBoardButton(run) {
  if (!run?.events) return null;
  return linkButton("TensorBoard", `/tensorboard?run=${encodeURIComponent(run.name)}`, { cls: "ghost", icon: "link", title: "This run's training curves" });
}
