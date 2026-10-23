// Job logs: the last lines inline, and a terminal-sized viewer of the whole output (filter,
// wrap, errors and warnings highlighted, copy, download) that follows the end while the job
// runs.
import { api, button, h, iconButton, modal, searchBox, svgIcon, switchEl, toast } from "./ui.js";

const LINES = 5000; // the viewer's window: the whole file is a download
const POLL_MS = 2000;

function tone(line) {
  if (/Traceback|Error\b|Exception\b|\bFAILED\b|\[job exit [1-9]/.test(line)) return "bad";
  if (/\[WARN|\bWarning\b|\bWARNING\b/.test(line)) return "warn";
  if (/^\[INFO\]|\[job exit 0\]|Learning iteration/.test(line)) return "info";
  return "";
}

/** The filter as a test and the spans it marks: text (any case) or ``/regex/``. */
function matcher(query) {
  if (!query) return null;
  const m = query.match(/^\/(.+)\/$/);
  let re;
  try {
    re = new RegExp(m ? m[1] : query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  } catch {
    return null;
  }
  return re;
}

function lineNode(text, re) {
  const el = h(`span.ln${tone(text) ? `.${tone(text)}` : ""}`);
  if (!re) {
    el.textContent = `${text}\n`;
    return el;
  }
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (!m[0]) break;
    el.append(text.slice(last, m.index), h("mark", m[0]));
    last = m.index + m[0].length;
  }
  el.append(`${text.slice(last)}\n`);
  return el;
}

const fmtBytes = (n) => (n > 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} kB`);

/** A job's last lines, scrolled to the end, with a button that opens the viewer. */
export function logBlock(job, text, { title } = {}) {
  const pre = h("pre.log", text || "(no output yet)");
  requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
  const open = iconButton("expand", (e) => { e.stopPropagation(); openLog(job, title); }, "Open the whole log");
  pre.addEventListener("dblclick", () => openLog(job, title));
  return h("div.log-wrap", pre, h("div.log-tools", open));
}

/** The viewer: up to the last 5000 lines, refreshed every 2 s while the job runs. */
export function openLog(job, title = job) {
  let lines = [];
  let size = 0;
  let status = "";
  let timer = 0;
  let closed = false;
  let re = null;
  const out = h("pre.terminal");
  const count = h("span.terminal-count");
  const end = button("End", () => { out.scrollTop = out.scrollHeight; }, { cls: "ghost small", icon: "down", title: "Jump to the newest line" });
  const atEnd = () => out.scrollHeight - out.scrollTop - out.clientHeight < 24;
  const syncEnd = () => { end.hidden = atEnd(); };
  out.addEventListener("scroll", syncEnd);

  const draw = (first = false) => {
    const follow = first || atEnd();
    const top = out.scrollTop;
    const shown = re ? lines.filter((l) => { re.lastIndex = 0; return re.test(l); }) : lines;
    const frag = document.createDocumentFragment();
    for (const l of shown) frag.append(lineNode(l, re));
    out.replaceChildren(frag);
    out.scrollTop = follow ? out.scrollHeight : top;
    const window_ = lines.length >= LINES ? ` (the last ${LINES}: download for all)` : "";
    count.textContent = `${re ? `${shown.length} of ` : ""}${lines.length} lines${window_} · ${fmtBytes(size)}${status === "running" ? " · live" : status ? ` · ${status}` : ""}`;
    syncEnd();
  };

  const load = async (first = false) => {
    let d;
    try {
      d = await api(`/api/jobs/${encodeURIComponent(job)}/log?lines=${LINES}`);
    } catch {
      return;
    }
    if (closed) return;
    lines = d.log ? d.log.split("\n") : [];
    size = d.size;
    status = d.status;
    draw(first);
    if (status === "running") timer = setTimeout(load, POLL_MS);
  };

  const search = searchBox("", "Filter: text, or /regex/  (Ctrl+F)", (v) => { re = matcher(v.trim()); draw(); });
  const filter = search.querySelector("input");
  const wrap = switchEl("Wrap", true, (v) => out.classList.toggle("nowrap", !v));
  const copy = button("Copy", async () => {
    const shown = re ? lines.filter((l) => { re.lastIndex = 0; return re.test(l); }) : lines;
    try {
      await navigator.clipboard.writeText(shown.join("\n"));
      toast(`${shown.length} lines copied`, "ok");
    } catch {
      toast("the browser didn't allow copying", "error");
    }
  }, { cls: "ghost small", icon: "copy" });
  const download = h("a.btn.ghost.small", { href: `/api/jobs/${encodeURIComponent(job)}/log.txt`, download: `${job}.log`, title: "The whole log as a file" }, svgIcon("download"), "Download");

  const dlg = modal(title, [
    h("div.terminal-bar", search, wrap, copy, download),
    h("div.terminal-host", out, h("div.terminal-end", end)),
  ], [count, h("span.spacer")], { width: 1280, cls: "terminal-modal", onClose: () => { closed = true; clearTimeout(timer); } });
  dlg.el.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "f") {
      e.preventDefault();
      filter.focus();
    }
  });
  out.textContent = "loading…";
  load(true);
  return dlg;
}
