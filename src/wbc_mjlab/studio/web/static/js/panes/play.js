// Step 5 · Play: a run's checkpoint (or the zero action = PD on the reference) on N envs over a
// motion library. Clip / env / push live on the stage; this pane picks the policy and shows the
// focused env's reward terms.
import { PaneFrame, api, bars, button, empty, field, group, h, help, listRow, post, seg, select, switchEl, tag, toast } from "../ui.js";

const ENVS = ["1", "4", "16", "64"];
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
    const rows = this.runs.map((r) => {
      const preset = r.task.includes("--") ? r.task.split("--").pop() : "";
      return listRow({
        lead: r.demo ? "bolt" : preset ? "sliders" : "layers",
        title: r.demo ? "Bundled demo" : preset || r.task || r.experiment,
        sub: `${preset ? `${r.task.split("--")[0]} + preset · ` : ""}${r.updated_text} · ${r.checkpoints.length} ckpt${r.name.includes("_", 19) ? ` · ${r.name.slice(20)}` : ""}`,
        trail: r.demo ? tag("demo", "accent") : null,
        active: r.label === c.run,
        onClick: async () => { c.run = r.label; c.checkpoint = ""; await this.loadLibraries(); this.render(); },
      });
    });
    const run = this.runs.find((r) => r.label === c.run);
    const ckpts = run?.checkpoints || [];
    if (!c.checkpoint || (c.checkpoint !== ZERO && !ckpts.includes(c.checkpoint))) c.checkpoint = ckpts[0] || ZERO;
    this.frame.setBody(
      group("Run", { aside: `${this.runs.length} on disk` }, h("div.list", rows)),
      group("Setup", {},
        h("div.stack",
          field("Checkpoint", select([...ckpts.map((x) => ({ value: x, label: x })), { value: ZERO, label: "Zero action · PD on the reference" }], c.checkpoint, (v) => (c.checkpoint = v))),
          field("Motion library", select(this.libs.length ? this.libs : [{ value: "", label: "convert a dataset first" }], c.motion, (v) => (c.motion = v))),
          field("Environments", seg(ENVS.map((n) => ({ value: n, label: n })), String(c.num_envs), (v) => (c.num_envs = Number(v)), { block: true })),
          switchEl("Terminations — falls end episodes", c.terminations, (v) => (c.terminations = v)))),
    );
    this.frame.setFooter(
      this.state?.active ? button("Cancel", () => { this.picking = false; this.render(); }, { cls: "ghost" }) : h("span.note", run ? run.task : ""),
      h("span.spacer"),
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
          sub: `${run?.updated_text || ""} · ${s.checkpoint === s.zero ? "zero action" : s.checkpoint} · ${lib} · ${s.num_envs} envs`,
          trail: tag(s.playing ? "live" : "paused", s.playing ? "accent" : "amber"),
          active: true,
        }))),
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
      button("Restart all", () => post("/api/play/restart"), { icon: "refresh" }),
    );
  }

  hud(hud) {
    if (this.barsHost && hud.terms) this.barsHost.replaceChildren(bars(hud.terms, 24));
  }
}
