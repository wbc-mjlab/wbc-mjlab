// Three.js stage: the robot model instanced N times (one draw call per visual geom, whatever N),
// a translucent second instance set for reference ghosts, overlays (points, segments, meshes),
// soft shadows, camera follow, click-to-focus and projected instance labels.
// World is MuJoCo's: Z up, metres. Scene colours come from the CSS tokens (wbc-demo palette).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { THEME_EVENT, token } from "./ui.js";

const HOME_OFFSET = new THREE.Vector3(2.6, -3.0, 1.1);
const FOCUS_HEIGHT = 0.75;
const Y = new THREE.Vector3(0, 1, 0);

function primitiveGeometry(g) {
  const s = g.size;
  let geo = null;
  switch (g.type) {
    case "sphere": geo = new THREE.SphereGeometry(s[0], 24, 16); break;
    case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case "cylinder": geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24); geo.rotateX(Math.PI / 2); break;
    case "capsule": geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16); geo.rotateX(Math.PI / 2); break;
    case "ellipsoid": geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0], s[1], s[2]); break;
    default: return null;
  }
  return geo;
}

// A set of instanced copies of the robot model (robots or ghosts).
class InstanceSet {
  constructor(parent, makeMaterial, { shadows }) {
    this.parent = parent;
    this.makeMaterial = makeMaterial;
    this.shadows = shadows;
    this.geoms = [];
    this.capacity = 0;
    this.count = 0;
  }

  reset() {
    for (const g of this.geoms) {
      this.parent.remove(g.mesh);
      g.mesh.dispose();
      g.mesh.material.dispose();
    }
    this.geoms = [];
    this.capacity = 0;
    this.count = 0;
  }

  ensure(n, defs, geometryFor) {
    if (n <= this.capacity) return;
    const cap = Math.max(4, 2 ** Math.ceil(Math.log2(n)));
    const old = this.geoms;
    for (const g of old) this.parent.remove(g.mesh);
    this.geoms = [];
    for (const def of defs) {
      const prev = old.find((g) => g.def === def);
      const geometry = prev?.mesh.geometry || geometryFor(def);
      if (!geometry) continue;
      const material = prev?.mesh.material || this.makeMaterial(def);
      const mesh = new THREE.InstancedMesh(geometry, material, cap);
      mesh.castShadow = this.shadows;
      mesh.receiveShadow = this.shadows;
      mesh.count = 0;
      mesh.frustumCulled = false;
      const local = new THREE.Matrix4().compose(
        new THREE.Vector3(...def.pos),
        new THREE.Quaternion(def.quat[1], def.quat[2], def.quat[3], def.quat[0]),
        new THREE.Vector3(1, 1, 1),
      );
      this.parent.add(mesh);
      this.geoms.push({ def, mesh, local, body: def.body });
    }
    for (const g of old) if (!this.geoms.some((x) => x.mesh.material === g.mesh.material)) g.mesh.material.dispose();
    this.capacity = cap;
  }

  pose(n, nbody, poses, offset, tmp) {
    const { body, m, p, q, s } = tmp;
    for (const g of this.geoms) {
      for (let i = 0; i < n; i++) {
        const o = offset + (i * nbody + g.body) * 7;
        p.set(poses[o], poses[o + 1], poses[o + 2]);
        q.set(poses[o + 4], poses[o + 5], poses[o + 6], poses[o + 3]);
        body.compose(p, q, s);
        m.multiplyMatrices(body, g.local);
        g.mesh.setMatrixAt(i, m);
      }
      g.mesh.count = n;
      g.mesh.instanceMatrix.needsUpdate = true;
    }
    this.count = n;
  }

  hide() {
    for (const g of this.geoms) g.mesh.count = 0;
    this.count = 0;
  }
}

export class Viewer {
  constructor(host, { onPick, labelsHost } = {}) {
    this.host = host;
    this.onPick = onPick;
    this.labelsHost = labelsHost;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping; // flat MuJoCo response, like wbc-demo
    host.append(this.renderer.domElement);

    const scene = (this.scene = new THREE.Scene());
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 300);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.resetCamera();

    this.hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 1.15);
    this.hemi.position.set(0, 0, 1);
    scene.add(this.hemi);
    const sun = (this.sun = new THREE.DirectionalLight(0xffffff, 1.7));
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    const sc = sun.shadow.camera;
    sc.left = sc.bottom = -6;
    sc.right = sc.top = 6;
    sc.near = 0.5;
    sc.far = 30;
    scene.add(sun, sun.target);
    this.headlight = new THREE.DirectionalLight(0xffffff, 0.45);
    scene.add(this.headlight, this.headlight.target);

    this.ground = new THREE.Group();
    scene.add(this.ground);
    this.applyTheme();
    window.addEventListener(THEME_EVENT, () => this.applyTheme());

    this.robotGroup = new THREE.Group();
    this.ghostGroup = new THREE.Group();
    this.overlayGroup = new THREE.Group();
    this.terrainGroup = new THREE.Group();
    scene.add(this.terrainGroup, this.robotGroup, this.ghostGroup, this.overlayGroup);
    this.robots = new InstanceSet(this.robotGroup, (def) => this._robotMaterial(def), { shadows: true });
    this.ghosts = new InstanceSet(this.ghostGroup, () => this._ghostMaterial(), { shadows: false });
    this.nbody = 0;
    this.focus = 0;
    this.count = 0;
    this.follow = true;
    this.showLabels = true;
    this.labels = [];
    this.overlays = new Map();
    this._focusRing = this._makeFocusRing();
    scene.add(this._focusRing);
    this._tmp = { body: new THREE.Matrix4(), m: new THREE.Matrix4(), p: new THREE.Vector3(), q: new THREE.Quaternion(), s: new THREE.Vector3(1, 1, 1) };
    this._roots = [];

    new ResizeObserver(() => this._resize()).observe(host);
    this._resize();
    this._bindPicking();
    const loop = () => {
      this._animate();
      this.renderer.render(scene, this.camera);
      this._placeLabels();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ theme / ground
  applyTheme() {
    const bg = new THREE.Color(token("--color-viewport-bg", "#16283a"));
    this.scene.background = bg;
    this.scene.fog = new THREE.Fog(bg, 22, 70);
    const floor = new THREE.Color(token("--color-floor", "#1d3447"));
    this.hemi.color.set(0xffffff);
    this.hemi.groundColor.copy(floor);
    const light = 0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b > 0.45;
    this.hemi.intensity = light ? 1.25 : 1.1;
    for (const c of [...this.ground.children]) {
      this.ground.remove(c);
      c.geometry?.dispose();
      c.material?.dispose();
    }
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ color: floor, roughness: 0.95, metalness: 0 }),
    );
    plane.receiveShadow = true;
    this.ground.add(plane);
    const grid = (size, div, color, opacity, z) => {
      const g = new THREE.GridHelper(size, div, color, color);
      g.rotation.x = Math.PI / 2;
      g.position.z = z;
      g.material.transparent = true;
      g.material.opacity = opacity;
      g.material.depthWrite = false;
      this.ground.add(g);
    };
    grid(120, 240, new THREE.Color(token("--color-grid-minor")), light ? 0.35 : 0.22, 0.0012);
    grid(120, 120, new THREE.Color(token("--color-grid-major")), light ? 0.45 : 0.32, 0.0016);
    this._ghostColor = new THREE.Color(token("--color-ghost", "#36cfe0"));
    for (const g of this.ghosts?.geoms || []) g.mesh.material.color.copy(this._ghostColor);
    if (this._focusRing) this._focusRing.material.color.set(token("--color-accent"));
  }

  _robotMaterial(def) {
    const [r, g, b, a] = def.rgba;
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(r, g, b),
      roughness: 0.6,
      metalness: 0.1,
      transparent: a < 1,
      opacity: a,
    });
  }

  _ghostMaterial() {
    return new THREE.MeshStandardMaterial({
      color: this._ghostColor || new THREE.Color(0x36cfe0),
      roughness: 0.5,
      metalness: 0,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      emissive: this._ghostColor || new THREE.Color(0x36cfe0),
      emissiveIntensity: 0.25,
    });
  }

  _makeFocusRing() {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.36, 0.42, 48),
      new THREE.MeshBasicMaterial({ color: token("--color-accent"), transparent: true, opacity: 0.8, depthWrite: false }),
    );
    ring.position.z = 0.004;
    ring.visible = false;
    return ring;
  }

  _resize() {
    const w = this.host.clientWidth || 1;
    const hgt = this.host.clientHeight || 1;
    this.renderer.setSize(w, hgt, false);
    this.camera.aspect = w / hgt;
    this.camera.updateProjectionMatrix();
  }

  resetCamera() {
    const t = this.focusPoint ? this.focusPoint.clone() : new THREE.Vector3(0, 0, FOCUS_HEIGHT);
    const k = this.count > 4 ? 1 + Math.sqrt(this.count) * 0.45 : 1;
    this.controls.target.copy(t);
    this.camera.position.copy(t).add(HOME_OFFSET.clone().multiplyScalar(k));
    this.controls.update();
  }

  // ------------------------------------------------------------------ robot model
  /** Terrain under the robots: ``[u32 nv][u32 nf][f32 verts][u8 colours, pad 4][u32 faces]``
   * (nv = 0: none). */
  setTerrain(buffer) {
    for (const o of [...this.terrainGroup.children]) {
      this.terrainGroup.remove(o);
      o.geometry.dispose();
      o.material.dispose();
    }
    const head = new Uint32Array(buffer, 0, 2);
    const nv = head[0];
    const nf = head[1];
    // the floor steps below the terrain tiles (terrain dips under z = 0)
    this.ground.position.z = nv ? -0.6 : 0;
    if (!nv) return;
    let off = 8;
    const verts = new Float32Array(buffer, off, nv * 3);
    off += nv * 12;
    const cols = new Uint8Array(buffer, off, nv * 3);
    off += Math.ceil((nv * 3) / 4) * 4;
    const faces = new Uint32Array(buffer, off, nf * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(cols, 3, true));
    geo.setIndex(new THREE.BufferAttribute(faces, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    this.terrainGroup.add(mesh);
  }

  setModel(meta, buffer) {
    this.robots.reset();
    this.ghosts.reset();
    for (const geo of this._meshGeo?.values() || []) geo.dispose();
    this.nbody = meta.nbody;
    this.meta = meta;
    this.buffer = buffer;
    this._meshGeo = new Map();
  }

  _geometryFor(g) {
    if (g.type !== "mesh") return primitiveGeometry(g);
    if (this._meshGeo.has(g.mesh)) return this._meshGeo.get(g.mesh);
    const m = this.meta.meshes[String(g.mesh)];
    const verts = new Float32Array(this.buffer, m.vertices[0], m.vertices[1]);
    const faces = new Uint32Array(this.buffer, m.faces[0], m.faces[1]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(faces, 1));
    geo.computeVertexNormals();
    this._meshGeo.set(g.mesh, geo);
    return geo;
  }

  // poses: Float32Array ((n + ng) * nbody * 7): pos xyz + quat wxyz; robots then ghosts
  setPoses(n, ng, nbody, poses, focus, labels) {
    if (!this.meta || nbody !== this.nbody) return;
    const geometryFor = (def) => this._geometryFor(def);
    this.robots.ensure(Math.max(1, n), this.meta.geoms, geometryFor);
    this.robots.pose(n, nbody, poses, 0, this._tmp);
    if (ng > 0 && this.showGhost !== false) {
      this.ghosts.ensure(ng, this.meta.geoms, geometryFor);
      this.ghosts.pose(ng, nbody, poses, n * nbody * 7, this._tmp);
    } else {
      this.ghosts.hide();
    }
    const firstFrame = n > 0 && n !== this.count; // new framing when the instance count changes
    this.count = n;
    this.focus = Math.min(focus, Math.max(0, n - 1));
    this.labels = labels || [];
    this._roots = [];
    for (let i = 0; i < n; i++) {
      const o = (i * nbody + 1) * 7;
      this._roots.push([poses[o], poses[o + 1], poses[o + 2]]);
    }
    if (n > 0) {
      const r = this._roots[this.focus];
      this.focusPoint = new THREE.Vector3(r[0], r[1], FOCUS_HEIGHT);
      this._focusRing.visible = n > 1;
      this._focusRing.position.set(r[0], r[1], 0.004);
    }
    if (firstFrame) {
      this.focusPoint = n > 1 ? this._centroid() : this.focusPoint;
      this.resetCamera();
    }
  }

  _centroid() {
    const c = new THREE.Vector3();
    for (const r of this._roots) c.add(new THREE.Vector3(r[0], r[1], 0));
    c.multiplyScalar(1 / Math.max(1, this._roots.length));
    c.z = FOCUS_HEIGHT;
    return c;
  }

  // ------------------------------------------------------------------ labels
  _placeLabels() {
    const host = this.labelsHost;
    if (!host) return;
    const want = this.showLabels && this.labels.length > 1 ? Math.min(this.labels.length, this._roots.length, 64) : 0;
    while (host.children.length > want) host.lastChild.remove();
    while (host.children.length < want) {
      const el = document.createElement("div");
      el.className = "lbl";
      el.addEventListener("click", () => this.onPick?.(Number(el.dataset.i)));
      host.append(el);
    }
    if (!want) return;
    const w = host.clientWidth;
    const hgt = host.clientHeight;
    const v = new THREE.Vector3();
    for (let i = 0; i < want; i++) {
      const el = host.children[i];
      const r = this._roots[i];
      v.set(r[0], r[1], r[2] + 0.75).project(this.camera);
      const visible = v.z < 1 && Math.abs(v.x) < 1.1 && Math.abs(v.y) < 1.1;
      el.style.display = visible ? "" : "none";
      if (!visible) continue;
      el.style.left = `${((v.x + 1) / 2) * w}px`;
      el.style.top = `${((1 - v.y) / 2) * hgt}px`;
      if (el.textContent !== this.labels[i]) el.textContent = this.labels[i];
      el.dataset.i = i;
      el.title = `${this.labels[i]} — click to focus`;
      el.classList.toggle("focus", i === this.focus);
    }
  }

  // ------------------------------------------------------------------ overlays
  setOverlayVisible(visible) {
    this.overlayGroup.visible = visible;
  }

  setOverlays(items, blob) {
    const keep = new Set(items.map((it) => it.key));
    for (const [key, obj] of this.overlays) {
      if (!keep.has(key)) {
        this.overlayGroup.remove(obj.object);
        obj.object.geometry.dispose();
        obj.object.material.dispose();
        this.overlays.delete(key);
      }
    }
    for (const it of items) {
      const data = blob.subarray(it.data[0], it.data[0] + it.data[1]);
      const colors = it.colors ? blob.subarray(it.colors[0], it.colors[0] + it.colors[1]) : null;
      if (it.kind === "points") this._points(it, data, colors);
      else if (it.kind === "segments") this._segments(it, data, colors);
      else if (it.kind === "mesh") this._mesh(it, data, blob.subarray(it.faces[0], it.faces[0] + it.faces[1]));
    }
  }

  _instanced(it, count, makeGeometry) {
    const prev = this.overlays.get(it.key);
    if (prev && prev.kind === it.kind && prev.object.instanceMatrix.count >= count) {
      prev.object.count = count;
      prev.object.material.opacity = it.opacity;
      prev.object.material.transparent = it.opacity < 1 || !!it.on_top;
      prev.object.material.depthTest = !it.on_top;
      prev.object.renderOrder = it.on_top ? 10 : 0;
      return prev.object;
    }
    if (prev) {
      this.overlayGroup.remove(prev.object);
      prev.object.geometry.dispose();
      prev.object.material.dispose();
    }
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0, color: 0xffffff, transparent: it.opacity < 1 || !!it.on_top, opacity: it.opacity, depthTest: !it.on_top });
    const obj = new THREE.InstancedMesh(makeGeometry(), mat, Math.max(16, count * 2));
    obj.renderOrder = it.on_top ? 10 : 0;
    obj.count = count;
    obj.frustumCulled = false;
    obj.castShadow = false;
    this.overlayGroup.add(obj);
    this.overlays.set(it.key, { kind: it.kind, object: obj });
    return obj;
  }

  _applyColors(obj, it, colors, n, perItem = 1) {
    const c = new THREE.Color();
    if (!colors) {
      c.setRGB(it.color[0] / 255, it.color[1] / 255, it.color[2] / 255, THREE.SRGBColorSpace);
      obj.material.color.copy(c);
      if (obj.instanceColor) {
        for (let i = 0; i < n; i++) obj.setColorAt(i, new THREE.Color(1, 1, 1));
        obj.instanceColor.needsUpdate = true;
      }
      return;
    }
    obj.material.color.set(0xffffff);
    for (let i = 0; i < n; i++) {
      const k = i * perItem * 3;
      c.setRGB(colors[k], colors[k + 1], colors[k + 2], THREE.SRGBColorSpace);
      obj.setColorAt(i, c);
    }
    obj.instanceColor.needsUpdate = true;
  }

  _points(it, data, colors) {
    const n = data.length / 3;
    const obj = this._instanced(it, n, () => new THREE.IcosahedronGeometry(1, 2));
    const { m, p, q } = this._tmp;
    const r = Math.max(0.004, it.size * 0.5);
    const s = new THREE.Vector3(r, r, r);
    q.identity();
    for (let i = 0; i < n; i++) {
      p.set(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
      obj.setMatrixAt(i, m.compose(p, q, s));
    }
    obj.instanceMatrix.needsUpdate = true;
    this._applyColors(obj, it, colors, n);
  }

  _segments(it, data, colors) {
    const n = data.length / 6;
    const obj = this._instanced(it, n, () => new THREE.CylinderGeometry(1, 1, 1, 8));
    const { m, p, q } = this._tmp;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const d = new THREE.Vector3();
    const s = new THREE.Vector3();
    const r = Math.max(0.003, it.size * 0.5);
    for (let i = 0; i < n; i++) {
      a.set(data[6 * i], data[6 * i + 1], data[6 * i + 2]);
      b.set(data[6 * i + 3], data[6 * i + 4], data[6 * i + 5]);
      d.subVectors(b, a);
      const len = d.length();
      p.addVectors(a, b).multiplyScalar(0.5);
      if (len > 1e-6) q.setFromUnitVectors(Y, d.normalize());
      else q.identity();
      s.set(r, Math.max(len, 1e-4), r);
      obj.setMatrixAt(i, m.compose(p, q, s));
    }
    obj.instanceMatrix.needsUpdate = true;
    this._applyColors(obj, it, colors, n, 2);
  }

  _mesh(it, data, faces) {
    const prev = this.overlays.get(it.key);
    if (prev) {
      this.overlayGroup.remove(prev.object);
      prev.object.geometry.dispose();
      prev.object.material.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(data), 3));
    geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(faces), 1));
    geo.computeVertexNormals();
    const c = new THREE.Color().setRGB(it.color[0] / 255, it.color[1] / 255, it.color[2] / 255, THREE.SRGBColorSpace);
    const mat = new THREE.MeshStandardMaterial({ color: c, transparent: it.opacity < 1, opacity: it.opacity, depthWrite: it.opacity >= 1, roughness: 0.7 });
    const obj = new THREE.Mesh(geo, mat);
    this.overlayGroup.add(obj);
    this.overlays.set(it.key, { kind: "mesh", object: obj });
  }

  // ------------------------------------------------------------------ camera / picking
  _animate() {
    if (this.follow && this.focusPoint) {
      const d = this.focusPoint.clone().sub(this.controls.target).multiplyScalar(0.15);
      if (d.lengthSq() > 1e-10) {
        this.controls.target.add(d);
        this.camera.position.add(d);
      }
    }
    const t = this.controls.target;
    this.sun.position.set(t.x + 2.5, t.y - 2, 7);
    this.sun.target.position.set(t.x, t.y, 0);
    this.headlight.position.copy(this.camera.position);
    this.headlight.target.position.copy(t);
    this.controls.update();
  }

  _bindPicking() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4 || this.count < 2) return;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      const ray = new THREE.Raycaster();
      ray.setFromCamera(ndc, this.camera);
      const hit = ray.intersectObjects(this.robots.geoms.map((g) => g.mesh), false)[0];
      if (hit && hit.instanceId !== undefined && this.onPick) this.onPick(hit.instanceId);
    });
  }
}
