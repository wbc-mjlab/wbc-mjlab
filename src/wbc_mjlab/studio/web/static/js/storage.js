// Disk: runs (checkpoints, curves) and the jobs that wrote them. One manager lists every run
// folder with its size; a run is deleted, or pruned to its last checkpoint. Removing a job can
// take its run along.
import { api, button, chips, del, h, modal, plural, post, select, switchEl, tag, toast } from "./ui.js";

export const fmtSize = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.max(n > 0 ? 1 : 0, Math.round(n / 1e6))} MB`);

const STATUS_TONE = { running: "accent", done: "ok", failed: "bad", stopped: "amber" };
const KEEP = [
  { value: "0", label: "last only" },
  { value: "1000", label: "last + every 1k it" },
  { value: "5000", label: "last + every 5k it" },
  { value: "10000", label: "last + every 10k it" },
];

/** Every run folder with size, checkpoints and its job; select runs, then delete them or
 * keep their last checkpoint. ``onChange`` runs after each change. */
export function openStorage({ onChange } = {}) {
  let runs = [];
  let total = 0;
  const picked = new Set();
  let quick = "";
  let keep = "0";
  const summary = h("div.help");
  const body = h("div.table-wrap");
  const count = h("span.terminal-count");
  const pruneBtn = button("Prune checkpoints", () => prune(), { cls: "ghost", icon: "layers", title: "Delete each selected run's older checkpoints" });
  const deleteBtn = button("Delete runs", () => remove(), { cls: "danger", icon: "trash" });
  const keepSel = select(KEEP, keep, (v) => { keep = v; draw(); });
  keepSel.style.width = "auto";

  const chosen = () => runs.filter((r) => picked.has(r.label));
  const QUICK = {
    empty: (r) => r.checkpoints <= 1,
    finished: (r) => r.status && r.status !== "done",
    all: () => true,
  };
  const quickHost = h("div");
  const toggle = (r) => {
    picked.has(r.label) ? picked.delete(r.label) : picked.add(r.label);
    quick = "";
    draw();
  };

  const draw = () => {
    quickHost.replaceChildren(chips([
      { value: "empty", label: "No progress (≤ 1 checkpoint)" },
      { value: "finished", label: "Failed / stopped" },
      { value: "all", label: "All" },
      { value: "", label: "None" },
    ], quick, (v) => {
      quick = v;
      picked.clear();
      if (v) for (const r of runs) if (!r.busy && QUICK[v](r)) picked.add(r.label);
      draw();
    }));
    summary.textContent = `${fmtSize(total)} in ${plural(runs.length, "run")} under logs/rsl_rl · older checkpoints ${fmtSize(runs.reduce((s, r) => s + r.prunable, 0))}`;
    const rows = runs.map((r) => {
      const box = h("input", { type: "checkbox", disabled: !!r.busy, title: r.busy || "select" });
      box.checked = picked.has(r.label);
      box.addEventListener("click", (e) => e.stopPropagation());
      box.addEventListener("change", () => toggle(r));
      const [exp, name] = r.label.split("/");
      const tr = h("tr", { title: r.busy ? `can't change: ${r.busy}` : r.label, on: { click: () => !r.busy && toggle(r) } },
        h("td", box),
        h("td.name", h("div", r.task || exp), h("div.job-meta", name)),
        h("td", r.algo === "wm" ? "world model" : r.algo || "—"),
        h("td", r.status ? tag(r.status, STATUS_TONE[r.status]) : h("span.job-meta", r.busy || "no job")),
        h("td.n", String(r.checkpoints)),
        h("td.n", r.last >= 0 ? r.last.toLocaleString() : "—"),
        h("td.n", fmtSize(r.size)),
        h("td.n", r.updated_text));
      if (picked.has(r.label)) tr.classList.add("is-active");
      return tr;
    });
    body.replaceChildren(h("div.table-scroll", { style: { maxHeight: "56vh" } }, h("table.table",
      h("thead", h("tr", h("th", { style: { width: "28px" } }), h("th", "Run"), h("th", "Method"), h("th", "Job"), h("th.n", "Ckpts"), h("th.n", "Last it"), h("th.n", "Size"), h("th.n", "Updated"))),
      h("tbody", rows))));
    const sel = chosen();
    const size = sel.reduce((s, r) => s + r.size, 0);
    const older = sel.reduce((s, r) => s + r.prunable, 0);
    count.textContent = sel.length ? `${plural(sel.length, "run")} selected · ${fmtSize(size)}` : "Select runs to delete or prune";
    deleteBtn.disabled = !sel.length;
    deleteBtn.lastChild.textContent = sel.length ? `Delete · ${fmtSize(size)}` : "Delete runs";
    pruneBtn.disabled = !sel.length || !older;
    pruneBtn.lastChild.textContent = sel.length && older ? `Prune${keep === "0" ? ` · ${fmtSize(older)}` : ""}` : "Prune checkpoints";
  };

  const load = async () => {
    const d = await api("/api/storage");
    runs = d.runs;
    total = d.total;
    for (const l of [...picked]) if (!runs.some((r) => r.label === l && !r.busy)) picked.delete(l);
    draw();
  };

  const remove = async () => {
    const sel = chosen();
    const size = sel.reduce((s, r) => s + r.size, 0);
    if (!confirm(`Delete ${plural(sel.length, "run")} (${fmtSize(size)}): checkpoints, curves and their jobs? This can't be undone.`)) return;
    try {
      const { freed } = await post("/api/runs/delete", { labels: sel.map((r) => r.label) });
      toast(`deleted ${plural(sel.length, "run")}, ${fmtSize(freed)} freed`, "ok");
    } catch (e) {
      toast(e.message, "error");
    }
    picked.clear();
    await load();
    onChange?.();
  };

  const prune = async () => {
    const sel = chosen().filter((r) => r.checkpoints > 1);
    const what = keep === "0" ? "the last checkpoint" : `the last checkpoint and every ${Number(keep).toLocaleString()}th iteration`;
    if (!confirm(`Keep only ${what} of ${plural(sel.length, "run")}? The other checkpoints are deleted.`)) return;
    try {
      const { freed } = await post("/api/runs/prune", { labels: sel.map((r) => r.label), every: Number(keep) });
      toast(`${fmtSize(freed)} freed`, "ok");
    } catch (e) {
      toast(e.message, "error");
    }
    await load();
    onChange?.();
  };

  modal("Runs on disk", [summary, quickHost, body],
    [count, h("span.spacer"), keepSel, pruneBtn, deleteBtn],
    { width: 980 });
  load();
}

/** Remove a finished job; optionally its run folder too (on by default when the run saved
 * no progress). */
export async function removeJob(job, title, { onDone } = {}) {
  const run = (await api("/api/storage")).runs.find((r) => r.job === job);
  let withRun = !!run && !run.busy && run.checkpoints <= 1;
  const body = [h("p.help", "The job's record and log are removed.")];
  if (run) {
    body.push(switchEl(`Also delete its run · ${plural(run.checkpoints, "checkpoint")} · ${fmtSize(run.size)}`, withRun, (v) => (withRun = v), run.busy ? `can't: ${run.busy}` : run.label));
  }
  const m = modal(`Remove ${title}`, body, [
    button("Cancel", () => m.close(), { cls: "ghost" }),
    button("Remove", async () => {
      m.close();
      const { freed } = await del(`/api/jobs/${encodeURIComponent(job)}${withRun ? "?run=true" : ""}`);
      if (freed) toast(`job and run removed, ${fmtSize(freed)} freed`, "ok");
      onDone?.();
    }, { cls: "danger", icon: "trash" }),
  ], { width: 460 });
}

/** Remove every finished job of a kind; for training jobs, optionally their runs too. */
export async function clearJobs(kind, jobs, { robot, onDone } = {}) {
  const names = new Set(jobs.filter((j) => j.status !== "running").map((j) => j.name));
  const runs = kind === "train" ? (await api("/api/storage")).runs.filter((r) => names.has(r.job) && !r.busy) : [];
  let withRuns = false;
  const body = [h("p.help", `${plural(names.size, "finished job")}: their records and logs are removed.`)];
  if (runs.length) {
    body.push(switchEl(`Also delete their runs · ${plural(runs.length, "run")} · ${fmtSize(runs.reduce((s, r) => s + r.size, 0))}`, withRuns, (v) => (withRuns = v)));
  }
  const m = modal("Clear finished jobs", body, [
    button("Cancel", () => m.close(), { cls: "ghost" }),
    button("Remove", async () => {
      m.close();
      const q = new URLSearchParams({ kind, ...(robot ? { robot } : {}), ...(withRuns ? { runs: "true" } : {}) });
      const { deleted, freed } = await del(`/api/jobs?${q}`);
      toast(`removed ${plural(deleted.length, "job")}${freed ? `, ${fmtSize(freed)} freed` : ""}`, "ok");
      onDone?.();
    }, { cls: "danger", icon: "trash" }),
  ], { width: 460 });
}
