// HTTP helpers, a tiny DOM builder, theme and the shared widgets every pane is built from.

export async function api(path, body, method) {
  const opts = body === undefined ? { method: method || "GET" } : { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || `${res.status} ${res.statusText}`;
    toast(typeof msg === "string" ? msg : JSON.stringify(msg), "error");
    throw new Error(msg);
  }
  return data;
}
export const post = (path, body = {}) => api(path, body);
export const put = (path, body = {}) => api(path, body, "PUT");
export const del = (path) => api(path, undefined, "DELETE");
export const qs = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")).toString();

// h("div.cls#id", {attrs, on: {click}}, ...children)
export function h(tag, attrs, ...children) {
  const [name, ...rest] = tag.split(/(?=[.#])/);
  const el = document.createElement(name || "div");
  for (const r of rest) {
    if (r[0] === ".") el.classList.add(r.slice(1));
    else el.id = r.slice(1);
  }
  if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "html") el.innerHTML = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k in el && k !== "list" && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

// ------------------------------------------------------------------ theme (shared key with wbc-demo)
const THEME_KEY = "wbc-demo-theme";
export const THEME_EVENT = "wbc-theme";
export function getTheme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}
export function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
  window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: theme }));
}
export function wireThemeToggle(btn) {
  const sync = () => {
    const dark = getTheme() === "dark";
    btn.textContent = dark ? "☀" : "☾";
    btn.title = dark ? "Light mode" : "Dark mode";
  };
  sync();
  btn.addEventListener("click", () => { setTheme(getTheme() === "dark" ? "light" : "dark"); sync(); });
}
export function token(name, fallback = "#888") {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
export const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(`wbc-studio.${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`wbc-studio.${key}`, JSON.stringify(value)); } catch { /* ignore */ }
  },
};

// ------------------------------------------------------------------ icons
const ICONS = {
  play: '<path d="M7 4l12 8-12 8z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  bolt: '<path d="M13 2L4 14h7l-1 8 9-12h-7z"/>',
  refresh: '<path d="M3 12a9 9 0 0 1 15.5-6.2L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.5 6.2L3 16"/><path d="M3 21v-5h5"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  upload: '<path d="M12 16V4M6 10l6-6 6 6"/><path d="M4 20h16"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  grid: '<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  check: '<path d="M5 12l5 5 9-10"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  robot: '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 2v4M8 13h.01M16 13h.01M9 17h6"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  back: '<path d="M19 12H5M11 6l-6 6 6 6"/>',
  search: '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
  expand: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  link: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
  chart: '<path d="M4 20V4M4 20h16"/><path d="M7 15l4-5 3 3 5-7"/>',
  file: '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>',
  scan: '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M7 12h10"/>',
  chev: '<path d="M9 6l6 6-6 6"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  save: '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v6h8M8 21v-7h8v7"/>',
  pose: '<circle cx="12" cy="4" r="2"/><path d="M12 6v7M4 9l8-1 8 1M12 13l-4 8M12 13l4 8"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  live: '<circle cx="12" cy="12" r="2"/><path d="M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7M5.6 5.6a9 9 0 0 0 0 12.8M18.4 5.6a9 9 0 0 1 0 12.8"/>',
};
export function svgIcon(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.innerHTML = ICONS[name] || "";
  return s;
}

// ------------------------------------------------------------------ pane frame
/** A pane: header (title, subtitle, optional tabs), scrolling body, sticky footer. */
export class PaneFrame {
  constructor(title, subtitle, { actions } = {}) {
    this.titleEl = h("h2", title);
    this.subEl = h("p", subtitle || "");
    this.actions = h("div.row", actions || []);
    this.tabsHost = h("div");
    this.header = h("div.pane-header", h("div.title-row", h("div", this.titleEl, this.subEl), this.actions), this.tabsHost);
    this.body = h("div.pane-body");
    this.footer = h("div.pane-footer");
    this.el = h("div.pane", this.header, this.body, this.footer);
    this.header.classList.add("no-tabs");
  }
  setTabs(el) {
    this.tabsHost.replaceChildren(el || "");
    this.header.classList.toggle("no-tabs", !el);
  }
  setBody(...nodes) {
    const top = this.body.scrollTop;
    this.body.replaceChildren(...nodes.flat().filter(Boolean));
    this.body.scrollTop = top;
  }
  setFooter(...nodes) {
    this.footer.replaceChildren(...nodes.flat().filter(Boolean));
  }
}

/** Underline tabs: items [{value, label, count, mod}] */
export function tabs(items, value, onChange) {
  const el = h("div.tabs");
  for (const it of items) {
    const b = h(`button${it.value === value ? ".is-active" : ""}`, { type: "button" }, it.label,
      it.count !== undefined && it.count !== null ? h(`span.count${it.mod ? ".mod" : ""}`, String(it.count)) : null);
    b.addEventListener("click", () => onChange(it.value));
    el.append(b);
  }
  return el;
}

const collapsed = new Map();
/** A titled group of a pane body; ``collapsible`` remembers its state per ``key``. */
export function group(title, { aside, action, collapsible, closed = false, key } = {}, ...body) {
  const el = h("section.group");
  if (collapsible) {
    const id = key || title;
    const isClosed = collapsed.has(id) ? collapsed.get(id) : store.get(`group.${id}`, closed);
    if (isClosed) el.classList.add("collapsed");
    const head = h("button.collapse-head", { type: "button" }, h("h3", title), aside ? h("span.aside", aside) : null, svgIcon("down"));
    head.lastChild.classList.add("chev");
    head.addEventListener("click", () => {
      el.classList.toggle("collapsed");
      const c = el.classList.contains("collapsed");
      collapsed.set(id, c);
      store.set(`group.${id}`, c);
    });
    el.append(head, h("div.group-content", ...body));
    return el;
  }
  if (title) el.append(h("div.group-head", h("h3", title), action || (aside ? h("span.aside", aside) : null)));
  el.append(...body.flat().filter(Boolean));
  return el;
}

// ------------------------------------------------------------------ inputs
export function field(label, input, hint) {
  return h("div.field", h("label", label), input, hint ? h("div.hint", hint) : null);
}

export function select(options, value, onChange) {
  const el = h("select");
  setOptions(el, options, value);
  if (onChange) el.addEventListener("change", () => onChange(el.value));
  return el;
}

export function setOptions(el, options, value) {
  const items = options.map((o) => (typeof o === "string" ? { value: o, label: o } : o));
  el.replaceChildren(
    ...items.map((o) =>
      o.group ? h("optgroup", { label: o.group }, o.items.map((x) => h("option", { value: x.value }, x.label))) : h("option", { value: o.value }, o.label),
    ),
  );
  const flat = items.flatMap((o) => (o.group ? o.items : [o]));
  if (value !== undefined && value !== null && flat.some((o) => o.value === value)) el.value = value;
}

export function input(value = "", { type = "text", placeholder, step, min, max, onChange, onInput, title } = {}) {
  const el = h("input", { type, placeholder, step, min, max, title });
  el.value = value ?? "";
  if (onChange) el.addEventListener("change", () => onChange(type === "number" ? Number(el.value) : el.value));
  if (onInput) el.addEventListener("input", () => onInput(type === "number" ? Number(el.value) : el.value));
  return el;
}

export function searchBox(value, placeholder, onInput) {
  const el = input(value, { placeholder, onInput });
  return h("div.search", svgIcon("search"), el);
}

export function switchEl(label, checked, onChange, title) {
  const box = h("input", { type: "checkbox" });
  box.checked = !!checked;
  if (onChange) box.addEventListener("change", () => onChange(box.checked));
  return h("label.switch", { title }, box, h("span.track"), label ? h("span", label) : null);
}

export function button(label, onClick, { cls = "", icon, title, disabled } = {}) {
  const el = h(`button.btn${cls ? `.${cls.split(" ").join(".")}` : ""}`, { type: "button", title, disabled }, icon ? svgIcon(icon) : null, label || null);
  if (onClick) el.addEventListener("click", onClick);
  return el;
}

export function iconButton(icon, onClick, title, { on } = {}) {
  const el = h(`button.icon-btn${on ? ".on" : ""}`, { type: "button", title }, svgIcon(icon));
  if (onClick) el.addEventListener("click", onClick);
  return el;
}

export function linkButton(label, href, { cls = "", icon, title } = {}) {
  return h(`a.btn${cls ? `.${cls.split(" ").join(".")}` : ""}`, { href, target: "_blank", rel: "noopener", title }, icon ? svgIcon(icon) : null, label);
}

export function seg(options, value, onChange, { block } = {}) {
  const el = h(`div.seg${block ? ".block" : ""}`);
  for (const o of options) {
    const opt = typeof o === "string" ? { value: o, label: o } : o;
    const b = h(`button${opt.value === value ? ".is-active" : ""}`, { type: "button", title: opt.title }, opt.label);
    b.addEventListener("click", () => {
      el.querySelectorAll("button").forEach((x) => x.classList.remove("is-active"));
      b.classList.add("is-active");
      onChange(opt.value);
    });
    el.append(b);
  }
  return el;
}

export function chips(options, value, onChange) {
  return h("div.chips", options.map((o) => {
    const b = h(`button${o.value === value ? ".is-active" : ""}`, { type: "button" }, o.label);
    b.addEventListener("click", () => onChange(o.value));
    return b;
  }));
}

// ------------------------------------------------------------------ display
export const tag = (text, tone = "", title) => h(`span.tag${tone ? `.${tone}` : ""}`, { title }, text);
export const dot = (tone = "") => h(`span.dot${tone ? `.${tone}` : ""}`);
export const help = (html) => h("p.help", { html });

export function empty(icon, title, text, ...actions) {
  return h("div.empty", h("div.icon", svgIcon(icon)), h("b", title), text ? h("div", text) : null, actions.length ? h("div.row", actions) : null);
}

export function kv(rows) {
  return h("dl.kv", rows.filter(Boolean).flatMap(([k, v]) => [h("dt", k), h("dd", v instanceof Node ? v : String(v))]));
}

/** Numbers in a row: [[label, value, tone]] */
export function strip(items) {
  return h("div.strip", items.filter(Boolean).map(([k, v, tone]) => h("div", h("small", k), h(`b${tone ? `.${tone}` : ""}`, { title: String(v) }, v))));
}

/** A selectable list row: {lead (icon name | node), title, sub, trail (node), active, onClick} */
export function listRow({ lead, title, sub, trail, active, onClick, extra, attrs }) {
  const leadEl = lead === undefined ? null : h("div.lead", typeof lead === "string" ? svgIcon(lead) : lead);
  const el = h(`div.list-row${active ? ".is-active" : ""}`, attrs || {},
    leadEl || h("span"),
    h("div.main", h("b", { title: typeof title === "string" ? title : null }, title), sub ? h("span", sub) : null),
    h("div.trail", trail || []),
    extra ? h("div.sub", extra) : null);
  if (onClick) el.addEventListener("click", onClick);
  return el;
}

export function ring(fraction) {
  const el = h(`div.mini-ring${fraction >= 0.999 ? ".full" : ""}`);
  el.style.setProperty("--p", String(Math.round(100 * Math.max(0, Math.min(1, fraction)))));
  return el;
}

export function bars(values, limit = 20) {
  const items = Object.entries(values || {}).filter(([, v]) => v != null).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, limit);
  if (!items.length) return help("No step yet.");
  const scale = Math.max(1e-6, ...items.map(([, v]) => Math.abs(v)));
  return h("div.bars", items.map(([k, v]) =>
    h(`div.bar-row${v < 0 ? ".neg" : ""}`,
      h("div.label", { title: k }, k),
      h("div.value", (v >= 0 ? "+" : "") + v.toFixed(3)),
      h("div.track", h("div", { style: { width: `${(100 * Math.abs(v)) / scale}%` } })))));
}

export function progressBar(fraction, tone = "") {
  return h(`div.progress${tone ? `.${tone}` : ""}`, h("div", { style: { width: `${Math.round(100 * Math.max(0, Math.min(1, fraction || 0)))}%` } }));
}

export function stepFor(v) {
  const a = Math.abs(v || 0);
  if (!a) return 0.1;
  return 10 ** Math.floor(Math.log10(a) - 1);
}

/** One editable setting ``{key, kind, value, help}`` (kinds: bool, int, number, choice, range,
 *  ints); ``ov`` its current override (undefined: default), marked when it differs. */
export function knob(f, ov, onChange, prefix = "", { pretty = (k) => k.replace(/_/g, " ") } = {}) {
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
    } else if (f.kind === "strings") {
      control = input(cur.join(", "), {});
      control.addEventListener("change", () => emit(control.value.split(/[\s,]+/).filter(Boolean)));
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

// ------------------------------------------------------------------ overlays
let openPop = null;
export function popover(anchor, content, { width = 380, align = "right" } = {}) {
  if (openPop && openPop.anchor === anchor) return closePopover();
  closePopover();
  const el = h("div.popover", { style: { width: `${width}px` } }, content);
  document.body.append(el);
  const r = anchor.getBoundingClientRect();
  const left = align === "right" ? r.right - width : r.left;
  el.style.left = `${Math.max(8, Math.min(left, window.innerWidth - width - 8))}px`;
  el.style.top = `${r.bottom + 6}px`;
  el.style.maxHeight = `${Math.max(160, window.innerHeight - r.bottom - 18)}px`;
  const away = (e) => { if (!el.contains(e.target) && !anchor.contains(e.target)) closePopover(); };
  const esc = (e) => e.key === "Escape" && closePopover();
  setTimeout(() => document.addEventListener("pointerdown", away), 0);
  document.addEventListener("keydown", esc);
  openPop = { el, anchor, off: () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); } };
  return el;
}
export function closePopover() {
  if (!openPop) return;
  openPop.off();
  openPop.el.remove();
  openPop = null;
}

export function modal(title, body, footer = [], { width = 520, onClose, cls } = {}) {
  const close = () => {
    back.remove();
    document.removeEventListener("keydown", esc);
    onClose?.();
  };
  const esc = (e) => e.key === "Escape" && close();
  const box = h(`div.modal${cls ? `.${cls}` : ""}`, { role: "dialog", style: { width: `min(${width}px, calc(100vw - 32px))` } },
    h("div.modal-head", h("h3", title), iconButton("x", close, "Close")),
    h("div.modal-body", ...[body].flat()),
    footer.length ? h("div.modal-foot", ...footer) : null);
  const back = h("div.modal-back", box);
  back.addEventListener("pointerdown", (e) => e.target === back && close());
  document.addEventListener("keydown", esc);
  document.body.append(back);
  return { el: box, close };
}

/** Ask for one line of text; resolves to the string or null. */
export function prompt(title, { label, value = "", placeholder, hint, ok = "Save" } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const inp = input(value, { placeholder });
    const finish = (v) => { if (done) return; done = true; m.close(); resolve(v); };
    const m = modal(title, [field(label || "Name", inp, hint)], [
      button("Cancel", () => finish(null), { cls: "ghost" }),
      button(ok, () => finish(inp.value.trim() || null), { cls: "primary" }),
    ], { width: 420, onClose: () => { if (!done) { done = true; resolve(null); } } });
    inp.addEventListener("keydown", (e) => e.key === "Enter" && finish(inp.value.trim() || null));
    setTimeout(() => inp.focus(), 30);
  });
}

export function toast(text, level = "info", ms = 4200) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const el = h(`div.toast.${level}`, h("span", text));
  host.append(el);
  setTimeout(() => el.remove(), ms);
  while (host.children.length > 3) host.firstChild.remove();
}

// ------------------------------------------------------------------ formatting
export const fmt = (v, digits = 3, unit = "") => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${Number(v).toFixed(digits)}${unit}`);
export function fmtNum(v) {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000 || a < 1e-3) return Number(v).toExponential(1);
  return String(Number(v.toPrecision(4)));
}
export function fmtDuration(s) {
  if (s === null || s === undefined) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
}
export function plural(n, word) {
  return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
}

export function debounce(fn, ms = 150) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
