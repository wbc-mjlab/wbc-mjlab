// Shared live GPU telemetry and selection for Train, Play and evaluation.
import { api, h } from "./ui.js";
let nextPickerId = 0;

export class GpuPicker {
  constructor({ multiple = false, value = null, onChange = () => {}, help = "" } = {}) {
    Object.assign(this, { multiple, value, onChange, help });
    this.gpus = [];
    this.error = "";
    this.timer = null;
  }

  setValue(value) { this.value = value; this.draw?.(); }

  async refresh() {
    try {
      const result = await api("/api/gpus");
      this.gpus = result.gpus;
      this.error = result.error;
      if (this.multiple && this.value === null && this.gpus.length) {
        this.value = [this.gpus.find((g) => g.cuda_index === 0)?.uuid || this.gpus[0].uuid];
        this.onChange(this.value);
      }
    } catch (error) { this.error = String(error); }
    this.draw?.();
  }

  element() {
    clearInterval(this.timer);
    // Play/Train rebuild their pane on background events. Keep the disclosure
    // and keyboard position instead of closing a menu the user is working in.
    const previous = this.picker;
    const focused = previous?.contains(document.activeElement) ? document.activeElement.dataset.gpu : null;
    const summary = h("summary");
    const rows = h("div.gpu-rows");
    const picker = h("details.gpu-picker", summary, rows);
    picker.open = previous?.open || false;
    this.picker = picker;
    const radioName = `gpu-${++nextPickerId}`;
    this.draw = () => {
      const focused = document.activeElement?.dataset.gpu;
      const selected = this.multiple ? (this.value || []) : [this.value];
      const chosen = this.gpus.find((g) => g.uuid === this.value);
      summary.textContent = this.multiple
        ? (selected.length ? `${selected.length} GPU${selected.length === 1 ? "" : "s"} selected · change` : "Choose training GPUs")
        : (this.value === null ? "Studio default device · change" : chosen ? `GPU ${chosen.index} · ${chosen.name}` : "Selected GPU is unavailable");
      const memory = (n) => n === null ? "?" : (n / 1024).toFixed(1);
      const option = (uuid, title, subtitle, status = "unknown") => {
        const control = h("input", {
          type: this.multiple ? "checkbox" : "radio", name: radioName,
          "data-gpu": uuid || "default", checked: selected.includes(uuid),
          on: { change: () => {
            if (this.multiple) {
              const ids = new Set(this.value || []);
              if (control.checked) ids.add(uuid); else ids.delete(uuid);
              this.value = [...ids];
            } else this.value = uuid;
            this.onChange(this.value);
            this.draw();
          } },
        });
        return h(`label.gpu-option.${status}`, { title: uuid || "Use Studio's configured device" }, control,
          h("span", h("b", title), h("small", subtitle)), h("span.gpu-state", uuid ? status : ""));
      };
      rows.replaceChildren(
        ...(!this.multiple ? [option(null, "Studio default device", "Uses Studio's launch device, including CPU mode.")] : []),
        ...this.gpus.map((g) => option(g.uuid, `GPU ${g.index} · ${g.name}`,
          `${g.utilization === null ? "?" : g.utilization}% load · ${memory(g.free_mib)}/${memory(g.total_mib)} GiB free`, g.status)),
        h("p.help", this.error || (this.gpus.length ? "Live every 3s while open. Colors show current load; free memory is not reserved. Hover for GPU UUID." : "No CUDA GPUs visible to Studio.")),
        ...(this.help ? [h("p.help", this.help)] : []),
      );
      if (focused) rows.querySelector(`[data-gpu="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
    };
    picker.addEventListener("toggle", async () => {
      if (this.picker !== picker) return;
      clearInterval(this.timer);
      if (!picker.open) return;
      await this.refresh();
      if (!picker.isConnected || !picker.open) return;
      let loading = false;
      const timer = setInterval(async () => {
        if (!picker.isConnected || !picker.open) { clearInterval(timer); return; }
        if (loading) return;
        loading = true;
        await this.refresh();
        loading = false;
      }, 3000);
      this.timer = timer;
    });
    this.draw();
    if (focused) queueMicrotask(() => {
      if (this.picker === picker && picker.isConnected) {
        rows.querySelector(`[data-gpu="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
      }
    });
    return picker;
  }
}
