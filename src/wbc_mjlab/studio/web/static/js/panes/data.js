// Step 2 · Data: datasets → clips → inspect (QA, raw vs converted; curves in the Plots window).
// Training plays the source clips directly; Export writes npz/<clip>.npz for tools that want them.
import {
  PaneFrame, api, button, chips, debounce, dot, empty, field, fmtDuration, group, h, help, iconButton, input,
  listRow, modal, plural, post, progressBar, qs, ring, searchBox, seg, select, strip, svgIcon, switchEl, tabs, tag, toast,
} from "../ui.js";

const PAGE = 100;

export class DataPane {
  constructor(app) {
    this.app = app;
    this.topics = ["data", "jobs", "robot"];
    this.live = ["jobs"];
    this.width = 440;
    this.frame = new PaneFrame("Data", "Retargeted clips, checked and played straight into training.", {
      actions: [iconButton("plus", () => this.importDialog(), "New dataset / import")],
    });
    this.tab = "library";
    this.datasets = [];
    this.dataset = "";
    this.q = "";
    this.status = "all";
    this.page = 0;
    this.listing = null;
    this.selected = new Set();
    this.preview = null;
    this.jobs = [];
    this.openJob = "";
    this.gridCount = 16;
    this.opts = { which: "missing", batch_size: 8, output_fps: 50, input_fps: 0, mirror: false, debias_z: false };
    this._reloadList = debounce(() => this.loadListing().then(() => this.render()), 180);
  }

  summary() {
    const d = this.ds();
    const n = d ? d.raw || d.npz : 0;
    return { value: d ? `${d.name} · ${plural(n, "clip")}` : "", done: !!n, busy: this.jobs.some((j) => j.status === "running") };
  }

  ds() {
    return this.datasets.find((x) => x.name === this.dataset);
  }

  async refresh() {
    const d = await api("/api/data/datasets");
    this.datasets = d.datasets;
    if (!this.ds()) this.dataset = d.selected || this.datasets[0]?.name || "";
    this.preview = await api("/api/data/preview");
    await Promise.all([this.loadListing(), this.loadJobs()]);
    this.render();
  }

  async onEvent(topic, payload) {
    if (topic === "jobs") {
      await this.loadJobs();
      if (this.tab === "convert") this.render();
      return;
    }
    if (topic === "robot") {
      this.dataset = "";
      this.selected.clear();
      return this.refresh();
    }
    if (payload.preview) {
      const key = this.previewKey();
      this.preview = payload.preview;
      if (this.previewKey() !== key) {
        if (this.preview.active && this.tab !== "clip") this.tab = "clip";
        this.render();
      }
      return;
    }
    await this.refresh();
  }

  previewKey() {
    const p = this.preview || {};
    return `${p.active}|${p.mode}|${p.dataset}|${p.clip}|${p.kind}|${p.count}|${p.show_compare}|${p.show_path}|${p.show_feet}|${p.loop}`;
  }

  setTab(t) {
    this.tab = t;
    this.render();
  }

  // ------------------------------------------------------------------ render
  render() {
    const d = this.ds();
    const p = this.preview || {};
    this.frame.setTabs(tabs([
      { value: "library", label: "Library", count: d ? d.clips : null },
      { value: "clip", label: "Clip", count: p.active ? (p.mode === "grid" ? p.count : null) : null },
      { value: "convert", label: "Export" },
    ], this.tab, (t) => this.setTab(t)));
    if (this.tab === "library") this.renderLibrary(d);
    else if (this.tab === "clip") this.renderClip(p);
    else this.renderConvert(d);
    this.app.touch();
  }

  // ------------------------------------------------------------------ library
  renderLibrary(d) {
    const rows = this.datasets.map((x) =>
      listRow({
        lead: ring(x.raw || x.npz ? 1 : 0),
        title: x.name,
        sub: `${plural(x.clips, "clip")}${x.stats ? ` · ${x.stats.hours.toFixed(2)} h` : ""}${x.linked ? " · linked" : ""}`,
        trail: x.raw ? null : x.npz ? tag("converted only", "amber") : tag("empty", "amber"),
        active: x.name === this.dataset,
        onClick: () => this.selectDataset(x.name),
      }),
    );
    this.frame.setBody(
      group("Datasets", { aside: `data/${this.app.state.robot}/` },
        rows.length ? h("div.list", rows) : empty("database", "No datasets yet", "Import a folder of retargeted CSV / PKL clips.", button("Import", () => this.importDialog(), { cls: "primary small", icon: "plus" }))),
      d ? this.clipsGroup(d) : null,
    );
    this.frame.setFooter(
      d ? h("div.note", h("b", d.name), ` · ${plural(d.raw || d.npz, "clip")}`) : h("div.note"),
      button("Preset", () => this.app.select("preset"), { cls: "primary", icon: "arrow", disabled: !(d?.raw || d?.npz) }),
    );
    const input = this.frame.body.querySelector("[data-role=clip-search] input");
    if (this._searchFocus && input) {
      input.focus();
      input.setSelectionRange(this.q.length, this.q.length);
    }
  }

  clipsGroup(d) {
    const L = this.listing;
    const s = d.stats;
    const search = searchBox(this.q, "Search clips · *walk* wildcards", (v) => {
      this.q = v;
      this.page = 0;
      this._searchFocus = true;
      this._reloadList();
    });
    search.dataset.role = "clip-search";
    search.querySelector("input").addEventListener("blur", () => (this._searchFocus = false));
    const filter = chips([
      { value: "all", label: "All" },
      { value: "converted", label: "Exported" },
      { value: "missing", label: "Not exported" },
    ], this.status, (v) => { this.status = v; this.page = 0; this.loadListing().then(() => this.render()); });
    const activeClip = this.preview?.mode === "single" && this.preview?.dataset === this.dataset ? this.preview.clip : "";
    const rows = (L?.rows || []).map((r) => {
      const box = h("input", { type: "checkbox", title: "select for export" });
      box.checked = this.selected.has(r.name);
      box.addEventListener("click", (e) => e.stopPropagation());
      box.addEventListener("change", () => (box.checked ? this.selected.add(r.name) : this.selected.delete(r.name)));
      const tr = h(`tr${r.name === activeClip ? ".is-active" : ""}`, { on: { click: () => this.loadClip(r.name) }, title: r.error || (r.raw ? "source clip" + (r.npz ? " (npz exported)" : "") : "converted clip only") },
        h("td", box),
        h("td.name", dot(r.error ? "bad" : "ok"), r.name),
        h("td.n", r.duration ? fmtDuration(r.duration) : "—"),
        h("td.n", r.fps ? `${r.fps}` : "—"));
      tr.dataset.name = r.name;
      return tr;
    });
    const table = h("div.table-wrap",
      h("div.table-scroll", h("table.table",
        h("thead", h("tr", h("th", { style: { width: "28px" } }), h("th", "Clip"), h("th.n", "Length"), h("th.n", "Hz"))),
        h("tbody", rows))),
      L ? h("div.pager",
        h("span", `${L.total.toLocaleString()} of ${L.all.toLocaleString()}`),
        h("div.row",
          iconButton("back", () => { this.page = Math.max(0, this.page - 1); this.loadListing().then(() => this.render()); }, "Previous page"),
          h("span.num", `${L.page + 1}/${L.pages}`),
          iconButton("arrow", () => { this.page += 1; this.loadListing().then(() => this.render()); }, "Next page"))) : null);
    const grid = select(["4", "9", "16", "25", "36", "49"].map((v) => ({ value: v, label: `${v} clips` })), String(this.gridCount), (v) => (this.gridCount = Number(v)));
    grid.style.width = "auto";
    const hist = s
      ? h("div", { title: "clip lengths (s)" },
          h("div.histogram", s.hist.counts.map((c) => h("div", { title: `${c} clips`, style: { height: `${(100 * c) / Math.max(1, ...s.hist.counts)}%` } }))),
          h("div.histogram-axis", s.hist.edges.map((e, i) => h("span", i === s.hist.edges.length - 1 ? `${e}+` : `${e}`))))
      : null;
    return group("Clips", {
      action: h("button.link", { on: { click: () => post("/api/data/scan", { dataset: d.name }) }, title: "Read every clip header: hours, length histogram" }, svgIcon("scan"), s ? `${s.hours.toFixed(2)} h · rescan` : "Scan lengths"),
    },
      h("div.stack",
        hist,
        search,
        filter,
        rows.length ? table : empty("search", "No clips match", this.q ? "Try another search." : "This dataset is empty."),
        h("div.row", h("span.help.grow", "Play several at once:"), grid,
          button("Grid", () => post("/api/data/grid", { dataset: this.dataset, q: this.q, status: this.status, count: this.gridCount }), { cls: "small", icon: "grid" }))));
  }

  selectDataset(name) {
    this.dataset = name;
    this.page = 0;
    this.q = "";
    this.selected.clear();
    this.loadListing().then(() => this.render());
  }

  async loadListing() {
    if (!this.dataset) {
      this.listing = null;
      return;
    }
    this.listing = await api(`/api/data/clips?${qs({ dataset: this.dataset, q: this.q, status: this.status, page: this.page, size: PAGE })}`);
  }

  async loadClip(name, source = "auto") {
    try {
      this.preview = await post("/api/data/load", { dataset: this.dataset, clip: name, source });
      this.tab = "clip";
      this.render();
    } catch { /* toasted */ }
  }

  stepClip(d) {
    const names = (this.listing?.rows || []).map((r) => r.name);
    if (!names.length || this.preview?.mode !== "single") return;
    const i = names.indexOf(this.preview.clip);
    this.loadClip(names[(((i < 0 ? 0 : i + d) % names.length) + names.length) % names.length]);
  }

  // ------------------------------------------------------------------ clip
  renderClip(p) {
    if (!p.active) {
      this.frame.setBody(empty("pose", "No clip open", "Pick a clip in Library — raw sources open before conversion too.", button("Library", () => this.setTab("library"), { cls: "small" })));
      this.frame.setFooter();
      return;
    }
    if (p.mode === "grid") {
      this.frame.setBody(
        group(`${p.count} clips side by side`, {},
          help("All clips share one clock; shorter ones loop. Click a robot or its label to focus it, <kbd>,</kbd> <kbd>.</kbd> step the focus."),
          h("div.tags", { style: { marginTop: "10px" } }, p.names.map((n) => tag(n)))),
      );
      this.frame.setFooter(h("div.note", p.dataset), button("Close", () => post("/api/data/close"), { cls: "ghost", icon: "x" }));
      return;
    }
    const qa = p.qa || {};
    const head = h("div.row.between",
      h("div.row", { style: { minWidth: 0, flex: "1" } },
        h("b", { style: { fontSize: "0.95rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, title: p.path }, p.clip),
        tag(p.kind === "raw" ? "raw source" : "converted", p.kind === "raw" ? "amber" : "ok")),
      h("div.row", iconButton("back", () => this.stepClip(-1), "Previous clip  ["), iconButton("arrow", () => this.stepClip(1), "Next clip  ]")));
    const source = p.has_compare || p.kind === "raw"
      ? seg([{ value: "npz", label: "Converted" }, { value: "raw", label: "Raw source" }], p.kind, (v) => this.loadClip(p.clip, v), { block: true })
      : null;
    const vsRaw = qa.vs_raw
      ? help(`vs raw: ${qa.vs_raw.frames_raw} frames @ ${qa.vs_raw.fps_raw} Hz → ${qa.frames} @ ${qa.fps} Hz; root height ${(100 * qa.vs_raw.dz_mean).toFixed(1)} cm${Math.hypot(...qa.vs_raw.xy_offset) > 0.05 ? `; npz carries a (${qa.vs_raw.xy_offset.map((v) => v.toFixed(1)).join(", ")}) m pool offset (re-centred here)` : ""}.`)
      : null;
    const opt = (k) => (v) => post("/api/data/options", { [k]: v });
    this.frame.setBody(
      group("", {}, h("div.stack", head, source,
        strip([
          ["Length", fmtDuration(qa.duration), "amber"],
          ["Root z · m", qa.root_z ? `${qa.root_z[0].toFixed(1)}–${qa.root_z[1].toFixed(1)}` : "—"],
          ["Speed · m/s", qa.speed_mean !== undefined ? qa.speed_mean.toFixed(2) : "—"],
          ["Airborne", qa.feet?.airborne !== undefined ? `${Math.round(100 * qa.feet.airborne)}%` : "—"],
        ]))),
      group("Quality", {},
        h("div.issues", (qa.issues || []).map((i) => this.issueRow(i))),
        vsRaw ? h("div", { style: { marginTop: "8px" } }, vsRaw) : null),
      group("Display", { collapsible: true, closed: true, key: "data-display" },
        h("div.grid2",
          p.has_compare ? switchEl("Raw source ghost", p.show_compare, opt("show_compare")) : null,
          switchEl("Root path", p.show_path, opt("show_path")),
          switchEl("Foot contacts", p.show_feet, opt("show_feet")),
          switchEl("Loop", p.loop, opt("loop")))),
    );
    this.frame.setFooter(
      button("Close", () => post("/api/data/close"), { cls: "ghost", icon: "x" }),
      h("span.spacer"),
      button("Plots", () => this.app.plots.toggle(), { cls: "ghost", icon: "chart", title: "Curves of this clip: every joint, root, feet; the raw source dashed  j" }),
      button("Play with a policy", () => this.app.panes.play.prefillClip(this.dataset, p.clip), { cls: "", icon: "play", disabled: p.kind === "raw" && !p.has_compare, title: "Pin this clip in Play" }),
    );
  }

  /** One QA line; with a ``plot`` it opens the Plots window on its channels and first bad frame. */
  issueRow(i) {
    const mark = dot({ ok: "ok", warn: "warn", error: "bad", info: "accent" }[i.level]);
    if (!i.plot) return h("div.issue", mark, h("span", i.text));
    const at = Number.isInteger(i.plot.frame) ? ` at frame ${i.plot.frame}` : "";
    return h("button.issue.is-link", { type: "button", title: `Show in Plots${at}`, on: { click: () => this.app.plots.focusOn(i.plot) } },
      mark, h("span.issue-text", i.text), svgIcon("chart"));
  }

  // ------------------------------------------------------------------ convert
  renderConvert(d) {
    if (!d) {
      this.frame.setBody(empty("database", "No dataset", "Import one first."));
      this.frame.setFooter();
      return;
    }
    const counts = { missing: d.missing, all: d.raw, selected: this.selected.size };
    const o = this.opts;
    const set = (k) => (v) => { o[k] = v; if (k === "which") this.render(); };
    const jobs = this.jobs.slice(0, 8).map((j) => this.jobCard(j));
    this.frame.setBody(
      group("Clips", {},
        seg([
          { value: "missing", label: `Not exported · ${d.missing}` },
          { value: "all", label: `All · ${d.raw}` },
          { value: "selected", label: `Selected · ${this.selected.size}` },
        ], o.which, set("which"), { block: true }),
        h("p.help", { style: { marginTop: "8px" } }, "Optional: training reads the source clips directly (resampled, mirrored and debiased on the GPU, set in Train). Export writes npz/<clip>.npz — body poses and velocities — for tools that want converted clips. Tick clips in Library for “Selected”.")),
      group("Options", {},
        h("div.stack",
          h("div.grid2",
            field("GPU batch", input(o.batch_size, { type: "number", min: 1, onChange: set("batch_size") }), "parallel FK envs"),
            field("Output rate", input(o.output_fps, { type: "number", min: 10, onChange: set("output_fps") }), "Hz (policy rate)")),
          field("Source rate override", input(o.input_fps || "", { type: "number", placeholder: "from the file (CSV: 30 Hz)", onChange: set("input_fps") })),
          switchEl("Mirror — also export left/right mirrored clips", o.mirror, set("mirror")),
          switchEl("Debias z — put the feet on the ground", o.debias_z, set("debias_z")))),
      group("Jobs", { aside: jobs.length ? "" : "none yet" }, jobs.length ? h("div.jobs", jobs) : null),
    );
    const n = counts[o.which];
    this.frame.setFooter(
      h("div.note", h("b", d.name), ` · ${plural(n, "clip")}`),
      button(`Export ${n}`, () => this.convert(), { cls: "primary", icon: "bolt", disabled: !n }),
    );
  }

  async convert() {
    const o = this.opts;
    if (o.which === "all" && !confirm(`Export all source clips of ${this.dataset}? Existing npz files are overwritten.`)) return;
    const r = await post("/api/data/convert", { dataset: this.dataset, clips: [...this.selected], ...o });
    this.openJob = r.job;
    await this.loadJobs();
    this.render();
  }

  async loadJobs() {
    this.jobs = (await api("/api/jobs?kind=convert")).filter((j) => !j.meta.robot || j.meta.robot === this.app.state.robot);
  }

  jobCard(j) {
    const tone = { running: "accent", done: "ok", failed: "bad", stopped: "amber" }[j.status];
    const pr = j.progress || {};
    const open = j.name === this.openJob;
    const logHost = h("div");
    const card = h(`div.job${open ? ".is-active" : ""}`,
      h("div.job-head", h("b", `${j.meta.dataset} · ${plural(j.meta.clips, "clip")}`), tag(j.status, tone)),
      h("div.job-meta", `${j.started_text} · ${fmtDuration(j.elapsed)}${pr.total ? ` · ${pr.done}/${pr.total}` : ""}${j.meta.mirror ? " · mirror" : ""}${j.meta.debias_z ? " · debias" : ""}`),
      j.status === "running" ? progressBar(pr.fraction) : null,
      open ? logHost : null,
      open && j.status === "running" ? h("div.row.end", button("Stop", (e) => { e.stopPropagation(); post(`/api/jobs/${j.name}/stop`); }, { cls: "small danger", icon: "stop" })) : null);
    card.addEventListener("click", (e) => {
      if (e.target.closest("pre")) return;
      this.openJob = open ? "" : j.name;
      this.render();
    });
    if (open) api(`/api/jobs/${j.name}?tail=24`).then((d) => {
      logHost.replaceChildren(h("pre.log", d.log || "(no output yet)"));
      logHost.firstChild.scrollTop = 1e9;
    });
    return card;
  }

  // ------------------------------------------------------------------ import
  importDialog() {
    const name = input("", { placeholder: "e.g. my_clips" });
    const source = input("", { placeholder: "/path/to/folder" });
    let mode = "copy";
    const files = h("input", { type: "file", multiple: true, accept: ".csv,.pkl", style: { display: "none" } });
    const drop = h("div.drop", svgIcon("upload"), h("div", "Drop .csv / .pkl clips or ", h("a", { href: "#", on: { click: (e) => { e.preventDefault(); files.click(); } } }, "browse")));
    const status = h("div.help");
    const upload = async (list) => {
      const n = name.value.trim();
      if (!n) return toast("name the dataset first", "warn");
      const items = [...list].filter((f) => /\.(csv|pkl)$/i.test(f.name));
      let done = 0;
      for (const f of items) {
        const res = await fetch(`/api/data/upload/${encodeURIComponent(n)}/${encodeURIComponent(f.name)}`, { method: "PUT", body: f });
        if (!res.ok) toast(`${f.name}: ${(await res.json()).detail}`, "error");
        status.textContent = `uploaded ${++done}/${items.length}`;
      }
      await post("/api/data/uploaded", { dataset: n });
      toast(`${done} clip(s) uploaded to ${n}`, "ok");
      this.dataset = n;
      dlg.close();
    };
    files.addEventListener("change", () => upload(files.files));
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files); });
    const dlg = modal("New dataset", [
      field("Name", name, `data/${this.app.state.robot}/<name>/ — sources at the root, converted clips in npz/`),
      field("From a folder on the Studio machine", h("div.stack", source,
        seg([{ value: "copy", label: "Copy clips" }, { value: "link", label: "Link folder" }], mode, (v) => (mode = v), { block: true }))),
      drop, files, status,
      help("Clips are retargeted robot motion: CSV rows <code>root_pos · root_quat_xyzw · joint_pos</code> (LAFAN layout, 30 Hz) or GMR <code>.pkl</code>."),
    ], [
      button("Create empty", async () => { if (!name.value.trim()) return; await post("/api/data/new", { name: name.value.trim() }); this.dataset = name.value.trim(); dlg.close(); }, { cls: "ghost" }),
      button("Import folder", async () => {
        if (!name.value.trim() || !source.value.trim()) return toast("name and folder please", "warn");
        await post("/api/data/import", { source: source.value.trim(), name: name.value.trim(), link: mode === "link" });
        this.dataset = name.value.trim();
        dlg.close();
      }, { cls: "primary", icon: "folder" }),
    ]);
    setTimeout(() => name.focus(), 30);
  }
}
