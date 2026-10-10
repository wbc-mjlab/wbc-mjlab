// Step 1 · Robot: which embodiment, how the motion command is wired to it, its default pose —
// and adding a robot from its model files (MJCF / URDF + meshes) without writing code.
import {
  PaneFrame, api, button, debounce, del, field, group, h, help, iconButton, input, listRow, post, select, strip,
  switchEl, tag, toast,
} from "../ui.js";

const GROUP_ORDER = ["legs", "torso", "arms", "other"];
const ID_RE = /^[a-z][a-z0-9_]{0,31}$/;
const WIRING = [
  ["keybodies", "Tracked keybodies", "accent", "The bodies whose reference pose the policy tracks (rewards, RSI, terminations). Usually root, torso and 3 per limb."],
  ["ee_bodies", "End-effectors", "blue", "Hands and feet: EE height termination, and the EE task's command."],
  ["feet_bodies", "Feet", "green", "Ground contact, foot slip, air time; height correction in conversion."],
];

export class RobotsPane {
  constructor(app) {
    this.app = app;
    this.topics = ["robot", "robots"];
    this.live = ["robots"];
    this.frame = new PaneFrame("Robot", "The embodiment everything downstream is built for.");
    this.info = null;
    this.catalogue = [];
    this.importing = null; // {id, files, model, info, spec, check} while adding / editing a robot
    this.width = 420;
  }

  async refresh() {
    [this.catalogue, this.info] = await Promise.all([api("/api/robots"), api("/api/robot")]);
    this.render();
  }

  onEvent() {
    return this.importing ? null : this.refresh();
  }

  onShow() {
    const im = this.importing;
    if (im?.info) post("/api/robots/import/inspect", { id: im.id, model: im.model }).then(() => this.preview()).catch(() => {});
    else if (this.app.state.owner !== "robots" && !this.app.hud.playing) post("/api/robot/show");
  }

  onHide() {
    // the draft model only lives in the robot step: other steps get the selected robot back
    if (this.importing?.info) post("/api/robots/import/cancel").catch(() => {});
  }

  summary() {
    if (this.importing) return { value: `adding ${this.importing.id || "a robot"}`, busy: true };
    return { value: this.info?.robot ? `${this.info.robot} · ${this.info.dof} dof` : "", done: !!this.info?.robot };
  }

  setWidth(w) {
    this.width = w;
    if (this.app.pane === "robots") document.getElementById("studio").style.setProperty("--side-w", `${w}px`);
  }

  render() {
    if (this.importing) return this.renderImport();
    this.setWidth(420);
    this.frame.setTitle?.("Robot");
    const info = this.info || {};
    const add = button("Add robot", () => this.startImport(), { cls: "ghost small", icon: "plus" });
    if (!info.robot) {
      this.frame.setBody(group("Robots", { action: add }, help("No robot registered yet: add one from its model files.")));
      this.frame.setFooter();
      return;
    }
    const w = info.wiring || {};
    const robots = this.catalogue.map((r) =>
      listRow({
        lead: "robot",
        title: r.id,
        sub: r.error || `${r.dof} dof · ${r.nbody - 1} bodies · ${r.datasets} datasets · ${r.runs} runs`,
        trail: [
          r.builtin ? tag("built-in") : r.custom ? tag("added", "accent", "added in the Studio (~/.config/wbc-mjlab/robots)") : tag("extension", "accent"),
          r.custom ? iconButton("sliders", (e) => { e.stopPropagation(); this.editRobot(r.id); }, "Edit wiring and gains") : null,
          r.custom ? iconButton("trash", (e) => { e.stopPropagation(); this.deleteRobot(r.id); }, "Delete this robot") : null,
        ],
        active: r.selected,
        onClick: () => (r.selected ? post("/api/robot/show") : post("/api/robot/select", { name: r.id })),
      }),
    );
    const bodies = (w.bodies || []).map((b) => tag(b.replace(/_link$/, ""), b === w.anchor ? "amber" : "", b === w.anchor ? "anchor" : "tracked keybody"));
    this.frame.setBody(
      group("Robots", { action: add }, h("div.list", robots)),
      group("Model", {},
        strip([
          ["DoF", info.dof, "accent"],
          ["Bodies", info.nbody - 1],
          ["Mass", `${info.mass.toFixed(1)} kg`],
          ["Control", w.step_dt ? `${Math.round(1 / w.step_dt)} Hz` : "—"],
        ]),
        h("p.help", { style: { marginTop: "10px" }, title: info.mjcf }, h("code", info.mjcf.split("/").slice(-3).join("/"))),
      ),
      group("Tracking", {
        action: switchEl("Show on model", info.show_wiring, (v) => post("/api/robot/options", { show_wiring: v })),
      },
        h("div.stack",
          h("p.help", { html: `Anchor <b>${w.anchor || "—"}</b> · ${(w.bodies || []).length} keybodies · feet <b>${(w.feet || []).map((f) => f.replace(/_link$/, "")).join(", ") || "—"}</b>` }),
          h("div.tags", bodies),
          help('<span style="color:var(--color-accent)">●</span> keybodies &nbsp; <span style="color:var(--color-amber)">●</span> anchor &nbsp; <span style="color:var(--color-success)">●</span> feet'))),
      this.poseGroup(info),
    );
    this.frame.setFooter(
      h("div.note", h("b", `${info.tasks.length} presets`), " available for this robot"),
      button("Data", () => this.app.select("data"), { cls: "primary", icon: "arrow" }),
    );
  }

  poseGroup(info) {
    const groups = {};
    for (const j of info.joints) (groups[j.group] ||= []).push(j);
    return group("Pose", { collapsible: true, closed: true, key: "robot-pose", aside: `${info.joints.length} joints` },
      h("div.row",
        button("Default", () => post("/api/robot/reset", { zero: false }), { cls: "small" }),
        button("Zero", () => post("/api/robot/reset", { zero: true }), { cls: "small ghost" })),
      ...GROUP_ORDER.filter((g) => groups[g]).map((g) =>
        h("div.field", h("label", g), h("div.joints", groups[g].map((j) => this.slider(j))))),
    );
  }

  slider(j) {
    const out = h("output", j.value.toFixed(2));
    const range = h("input.accent", { type: "range", min: j.min, max: j.max, step: 0.01, title: `${j.name} [${j.min.toFixed(2)}, ${j.max.toFixed(2)}] rad` });
    range.value = j.value;
    let pending = null;
    range.addEventListener("input", () => {
      out.textContent = Number(range.value).toFixed(2);
      if (pending) return;
      pending = setTimeout(() => {
        pending = null;
        post("/api/robot/joint", { name: j.name, value: Number(range.value) });
      }, 30);
    });
    return h("div.joint", h("label", { title: j.name }, j.name.replace(/_joint$/, "")), range, out);
  }

  // ------------------------------------------------------------------ add / edit a robot
  async startImport() {
    const st = await api("/api/robots/import");
    this.importing = { id: "", files: [], model: "", info: null, spec: null, check: null, dir: st.dir, drafts: st.drafts, edit: false };
    this.render();
    this.app.touch();
  }

  async editRobot(id) {
    const st = await api("/api/robots/import");
    const spec = await api(`/api/robots/${encodeURIComponent(id)}/spec`);
    this.importing = { id, files: [], model: spec.model, info: null, spec: null, check: null, dir: st.dir, drafts: [], edit: true };
    await this.loadFiles();
    await this.inspect(spec);
  }

  async deleteRobot(id) {
    if (!confirm(`Delete robot ${id}? Its model folder and spec go; data/${id}/ and its runs stay on disk.`)) return;
    await del(`/api/robots/${encodeURIComponent(id)}`);
    toast(`robot ${id} deleted`, "ok");
    await this.refresh();
  }

  async exitImport() {
    const had = !!this.importing?.info;
    this.importing = null;
    if (had) await post("/api/robots/import/cancel");
    await this.refresh();
    this.app.touch();
  }

  async loadFiles() {
    const im = this.importing;
    im.files = ID_RE.test(im.id) ? await api(`/api/robots/import/files?id=${encodeURIComponent(im.id)}`) : [];
    if (!im.files.some((f) => f.path === im.model)) im.model = im.files[0]?.path || "";
  }

  async upload(fileList) {
    const im = this.importing;
    const files = [...fileList];
    if (!files.length) return;
    const strip0 = files.every((f) => f.webkitRelativePath && f.webkitRelativePath.includes("/"));
    let done = 0;
    this.uploadNote?.replaceChildren(`uploading 0 / ${files.length}…`);
    const queue = files.slice();
    const worker = async () => {
      while (queue.length) {
        const f = queue.shift();
        const rel = strip0 ? f.webkitRelativePath.split("/").slice(1).join("/") : f.name;
        const res = await fetch(`/api/robots/import/upload/${encodeURIComponent(im.id)}/${rel.split("/").map(encodeURIComponent).join("/")}`, { method: "PUT", body: f });
        if (!res.ok) throw new Error(`${rel}: ${res.status}`);
        done += 1;
        this.uploadNote?.replaceChildren(`uploading ${done} / ${files.length}…`);
      }
    };
    try {
      await Promise.all([worker(), worker(), worker(), worker()]);
      toast(`${files.length} file(s) uploaded`, "ok");
    } catch (e) {
      toast(String(e.message || e), "error");
    }
    await this.loadFiles();
    this.render();
  }

  async copyFrom(source) {
    const im = this.importing;
    const r = await post("/api/robots/import/path", { id: im.id, source });
    toast(`${r.files} file(s) copied`, "ok");
    await this.loadFiles();
    this.render();
  }

  async inspect(saved) {
    const im = this.importing;
    const info = await post("/api/robots/import/inspect", { id: im.id, model: im.model });
    const s = info.suggested;
    const prev = saved || info.saved;
    im.info = info;
    im.check = null;
    im.spec = prev ? JSON.parse(JSON.stringify(prev)) : {
      id: im.id,
      model: im.model,
      anchor_body: s.anchor_body,
      keybodies: s.keybodies,
      ee_bodies: s.ee_bodies,
      feet_bodies: s.feet_bodies,
      foot_sites: s.foot_sites,
      foot_geoms: s.foot_geoms,
      foot_sole_z: s.foot_sole_z,
      init_height: info.init_height,
      init_joint_pos: Object.fromEntries(info.joints.map((j) => [j.name, j.default])),
      joints: Object.fromEntries(info.joints.map((j) => [j.name, { effort: round(j.effort), stiffness: round(j.stiffness), damping: round(j.damping), armature: j.armature }])),
      action_scale: 0.25,
      description: "",
      nconmax: s.nconmax,
      njmax: s.njmax,
    };
    this.render();
    this.preview();
    this.app.touch();
  }

  preview = debounce(() => {
    const sp = this.importing?.spec;
    if (sp) post("/api/robots/import/preview", { wiring: sp }).catch(() => {});
  }, 80);

  changed() {
    if (this.importing) this.importing.check = null;
    this.preview();
    this.renderCheck();
  }

  async runCheck() {
    const im = this.importing;
    this.checkHost?.replaceChildren(help("simulating 0.5 s…"));
    im.check = await post("/api/robots/import/check", { spec: im.spec });
    this.renderCheck();
  }

  async save() {
    const im = this.importing;
    const r = await post("/api/robots/import/save", { spec: im.spec });
    toast(`robot ${r.id} saved · train it as ${r.task}`, "ok");
    this.importing = null;
    await this.refresh();
    this.app.touch();
  }

  renderImport() {
    const im = this.importing;
    this.setWidth(im.info ? 600 : 460);
    const body = [this.sourceGroup()];
    if (im.info) body.push(...this.wiringGroups());
    this.frame.setBody(...body);
    this.frame.setFooter(
      button("Cancel", () => this.exitImport(), { cls: "ghost" }),
      h("span.spacer"),
      im.info ? button("Check", () => this.runCheck(), { icon: "check", title: "Drop it on a floor and hold the pose for 0.5 s" }) : null,
      im.info
        ? button(im.edit ? "Save changes" : "Save robot", () => this.save(), { cls: "primary", icon: "save" })
        : button("Inspect", () => this.inspect(), { cls: "primary", icon: "scan", disabled: !im.model }),
    );
  }

  sourceGroup() {
    const im = this.importing;
    const idInput = input(im.id, { placeholder: "e.g. h1, my_humanoid" });
    const idHint = h("div.hint");
    const syncId = () => {
      const ok = ID_RE.test(im.id);
      idHint.textContent = ok ? `registers robot ${im.id} · tasks ${taskId(im.id)}, -SE, -Terrain-Scan · data in data/${im.id}/` : "lowercase letters, digits and _, starting with a letter";
      idHint.classList.toggle("bad", !!im.id && !ok);
    };
    idInput.addEventListener("input", () => { im.id = idInput.value.trim(); syncId(); });
    idInput.addEventListener("change", async () => { await this.loadFiles(); this.render(); });
    idInput.disabled = im.edit;
    syncId();
    const ready = ID_RE.test(im.id);
    const folderIn = h("input", { type: "file", multiple: true, hidden: true });
    folderIn.setAttribute("webkitdirectory", "");
    const filesIn = h("input", { type: "file", multiple: true, hidden: true });
    folderIn.addEventListener("change", () => this.upload(folderIn.files));
    filesIn.addEventListener("change", () => this.upload(filesIn.files));
    this.uploadNote = h("span.hint");
    const path = input("", { placeholder: "/path/to/robot (on the Studio machine)" });
    const files = im.files.map((f) => ({ value: f.path, label: f.path }));
    const fileSel = files.length ? select(files, im.model, (v) => { im.model = v; }) : h("p.help", ready ? "No MJCF / URDF in the folder yet." : "");
    const drafts = !im.edit && im.drafts?.length
      ? h("p.help", "Unfinished: ", ...im.drafts.map((d) => h("button.link", { type: "button", on: { click: async () => { im.id = d.id; await this.loadFiles(); this.render(); } } }, d.id)))
      : null;
    return group(im.edit ? `Edit robot ${im.id}` : "Add a robot", { aside: im.info ? `${im.info.dof} dof` : "" },
      help("A humanoid from its <b>MJCF or URDF</b> with meshes. wbc-mjlab removes the model's own actuators, keyframes and floor, adds a free joint and the IMU sensors when missing, and gives it PD actuators. Saved per user, not in the repo."),
      h("div.stack",
        field("Robot id", idInput),
        idHint,
        drafts,
        im.edit ? null : h("div.row",
          button("Upload folder…", () => folderIn.click(), { cls: "small", icon: "folder", disabled: !ready }),
          button("Upload files…", () => filesIn.click(), { cls: "small ghost", icon: "upload", disabled: !ready }),
          this.uploadNote, folderIn, filesIn),
        im.edit ? null : h("div.row.grow", path, button("Copy", () => this.copyFrom(path.value.trim()), { cls: "small ghost", disabled: !ready })),
        field("Model file", fileSel, im.info ? null : "The robot's main file (not a scene that includes it)."),
        im.info?.notes?.length ? h("div.callout", h("b", "Prepared for wbc-mjlab: "), im.info.notes.join(" · ")) : null),
      im.info ? strip([["DoF", im.info.dof, "accent"], ["Bodies", im.info.bodies.length], ["Mass", `${im.info.mass.toFixed(1)} kg`], ["Height", `${im.info.height.toFixed(2)} m`]]) : null,
    );
  }

  wiringGroups() {
    const im = this.importing;
    const sp = im.spec;
    const info = im.info;
    const bodies = info.bodies.map((b) => b.name);
    const anchor = select(bodies.map((b) => ({ value: b, label: b })), sp.anchor_body, (v) => { sp.anchor_body = v; this.changed(); });
    const groups = [
      group("Wiring", { aside: "markers on the model" },
        help("Suggested from body names and the tree — check it on the model: <span style='color:var(--color-amber)'>●</span> anchor · <span style='color:var(--color-accent)'>●</span> keybodies · <span style='color:#5aa0ff'>●</span> end-effectors · <span style='color:var(--color-success)'>●</span> feet."),
        h("div.stack",
          field("Anchor", anchor, "The body the reference is expressed in (usually the torso)."),
          ...WIRING.map(([key, label, , text]) => this.bodyChips(key, label, text, bodies)),
          info.sites.length ? this.chips("foot_sites", "Foot sites", "Sites under the soles, for the foot-slip reward (none: the reward is off).", info.sites) : null)),
      group("Initial state", { collapsible: true, key: "import-init", aside: `${sp.init_height.toFixed(2)} m` },
        h("div.knobs",
          numberKnob("Root height (m)", sp.init_height, (v) => { sp.init_height = v; this.changed(); }, 0.01, `feet on the floor: ${info.init_height.toFixed(3)}`),
          numberKnob("Sole height (m)", sp.foot_sole_z, (v) => { sp.foot_sole_z = v; }, 0.005, "foot body above the floor when standing — conversion height correction")),
        info.keyframe ? help(`Joint defaults from the model's “${info.keyframe}” keyframe.`) : help("Joint defaults: the model's zero pose (bent knees help a humanoid hold it)."),
        this.jointTable("pose")),
      group("Actuators", { collapsible: true, key: "import-act", aside: `${info.joints.length} joints` },
        help(`PD per joint. Defaults: the model's armature / force limits when it has them, else G1's per joint kind scaled by mass; stiffness = armature·ω², damping = 2ζ·armature·ω (ω = ${info.pd.natural_freq_hz} Hz, ζ = ${info.pd.damping_ratio}, as the H2 extension).`),
        this.jointTable("gains")),
      group("Check", { aside: "0.5 s on a floor, no policy" }, this.checkHost = h("div")),
    ];
    this.renderCheck();
    return groups;
  }

  chips(key, label, text, names) {
    const sp = this.importing.spec;
    const host = h("div.body-chips");
    const count = h("span");
    const draw = () => {
      const cur = new Set(sp[key]);
      count.textContent = `${label} · ${cur.size}`;
      host.replaceChildren(...names.map((n) => {
        const b = h(`button${cur.has(n) ? ".is-active" : ""}`, { type: "button", title: n }, n.replace(/_link$/, ""));
        b.addEventListener("click", () => {
          sp[key] = cur.has(n) ? sp[key].filter((x) => x !== n) : [...sp[key], n];
          draw();
          this.changed();
        });
        return b;
      }));
    };
    draw();
    return h("div.knob.wide", h("label", count), host, h("div.hint", text));
  }

  bodyChips(key, label, text, bodies) {
    return this.chips(key, label, text, bodies);
  }

  jointTable(kind) {
    const im = this.importing;
    const sp = im.spec;
    const rows = im.info.joints.map((j) => {
      if (kind === "pose") {
        const v = input(round(sp.init_joint_pos[j.name] ?? j.default, 3), { type: "number", step: 0.05, min: j.range[0], max: j.range[1] });
        v.addEventListener("input", () => { const x = Number(v.value); if (v.value !== "" && Number.isFinite(x)) { sp.init_joint_pos[j.name] = x; this.changed(); } });
        return h("tr", h("td", { title: j.name }, j.name.replace(/_joint$/, "")), h("td.num", v), h("td.dim", `[${j.range[0].toFixed(2)}, ${j.range[1].toFixed(2)}]`));
      }
      const g = sp.joints[j.name];
      const cell = (field, step, title) => {
        const x = input(round(g[field], 4), { type: "number", step, min: 0, title });
        x.addEventListener("input", () => { const n = Number(x.value); if (x.value !== "" && Number.isFinite(n) && n >= 0) { g[field] = n; this.changed(); } });
        return h("td.num", x);
      };
      return h("tr",
        h("td", { title: j.name }, j.name.replace(/_joint$/, ""), j.effort_from_model ? null : h("span.dim", " *")),
        cell("effort", 1, "peak torque N·m"), cell("stiffness", 1, "N·m/rad"), cell("damping", 0.1, "N·m·s/rad"), cell("armature", 0.001, "kg·m²"));
    });
    const head = kind === "pose"
      ? h("tr", h("th", "joint"), h("th", "default (rad)"), h("th", "range"))
      : h("tr", h("th", "joint"), h("th", "effort"), h("th", "stiffness"), h("th", "damping"), h("th", "armature"));
    const foot = kind === "gains" && im.info.joints.some((j) => !j.effort_from_model) ? help("* effort not in the model: a G1-based guess, check the datasheet.") : null;
    return h("div", h("div.table-scroll", h("table.table.joint-table", h("thead", head), h("tbody", rows))), foot);
  }

  renderCheck() {
    const host = this.checkHost;
    const c = this.importing?.check;
    if (!host) return;
    if (!c) {
      host.replaceChildren(help("<b>Check</b> drops the robot on a floor in its initial pose and holds it with these gains for 0.5 s: stable simulation, start above the floor, lands on its feet, holds the pose, torque within limits. Plain PD doesn't balance for long — the policy does."));
      return;
    }
    host.replaceChildren(h("div.checks", c.checks.map((x) => h(`div.check-row.${x.ok ? "ok" : x.warn_only ? "warn" : "bad"}`,
      h("span.mark", x.ok ? "✓" : x.warn_only ? "!" : "✗"), h("b", x.name), h("span", x.text)))),
    c.ok ? null : help("Fix the ✗ items, then check again — or save anyway and look at it in Play with the zero action."));
  }
}

function round(v, digits = 2) {
  const f = 10 ** digits;
  return Math.round(Number(v) * f) / f;
}

function taskId(id) {
  return `Wbc-${id.split("_").filter(Boolean).map((p) => (p.length <= 3 && /\d/.test(p) ? p.toUpperCase() : p[0].toUpperCase() + p.slice(1))).join("-")}`;
}

function numberKnob(label, value, onChange, step, hint) {
  const x = input(value, { type: "number", step });
  x.addEventListener("input", () => { const n = Number(x.value); if (x.value !== "" && Number.isFinite(n)) onChange(n); });
  return h("div.knob", h("label", h("span", label)), x, hint ? h("div.hint", hint) : null);
}
