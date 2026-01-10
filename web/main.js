// Build wasm with:
//   wasm-pack build wasm --release --target web --out-dir web/pkg
// Then serve the repo root (e.g. python -m http.server) and open /web/

import init, { Simulation, GrayScottParams } from "../wasm/web/pkg/abiogenesis.js";

const canvas = document.querySelector("#c");
const statsEl = document.querySelector("#stats");

function resizeCanvasToDisplaySize(c) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.floor(c.clientWidth * dpr);
  const h = Math.floor(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
    return true;
  }
  return false;
}

function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

function mat4Mul(a, b) {
  // Column-major (OpenGL/WebGL) matrix multiply: out = a * b.
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

function mat4Perspective(fovyRad, aspect, near, far) {
  const f = 1.0 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    (2 * far * near) * nf,
    0,
  ]);
}

function mat4Translation(x, y, z) {
  const m = mat4Identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

function mat4RotationYX(yaw, pitch) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);

  // Column-major.
  return new Float32Array([
    cy,
    0,
    -sy,
    0,
    sy * sp,
    cp,
    cy * sp,
    0,
    sy * cp,
    -sp,
    cy * cp,
    0,
    0,
    0,
    0,
    1,
  ]);
}

function vec3Add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function vec3Scale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function vec3Normalize(a) {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (len < 1e-8) return [0, 0, 1];
  return [a[0] / len, a[1] / len, a[2] / len];
}

function vec3Dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vec3Cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) || "shader compile failed");
  }
  return sh;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) || "program link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return p;
}

const vsSource = `#version 300 es
precision highp float;

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
layout(location=2) in vec4 aCol;

uniform mat4 uViewProj;

out vec3 vNor;
out vec4 vCol;
out vec3 vPos;

void main() {
  vNor = aNor;
  vCol = aCol;
  vPos = aPos;
  gl_Position = uViewProj * vec4(aPos, 1.0);
}
`;

const fsSource = `#version 300 es
precision highp float;

in vec3 vNor;
in vec4 vCol;
in vec3 vPos;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;

out vec4 outColor;

void main() {
  vec3 n = normalize(vNor);
  float ndl = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 base = vCol.rgb;
  vec3 lit = base * (0.25 + 0.75 * ndl);

  float d = length(vPos - uCamPos);
  float fog = 1.0 - exp(-uFogDensity * d * d);
  vec3 rgb = mix(lit, uFogColor, fog);

  outColor = vec4(rgb, vCol.a);
}
`;

class FlyCamera {
  constructor() {
    // Simulation volume is normalized to roughly [-0.5, 0.5] in each axis.
    // Start the camera 25% into the volume on +Z.
    this.pos = [0, 0, 0.25];
    this.yaw = 0;
    this.pitch = 0;
    this.moveSpeed = 0.75;
  }

  viewMatrix() {
    const f = this.forward();
    const r = this.right();
    const u = vec3Normalize(vec3Cross(r, f));

    const tx = -vec3Dot(r, this.pos);
    const ty = -vec3Dot(u, this.pos);
    const tz = vec3Dot(f, this.pos);

    // Column-major view matrix.
    return new Float32Array([
      r[0],
      r[1],
      r[2],
      0,
      u[0],
      u[1],
      u[2],
      0,
      -f[0],
      -f[1],
      -f[2],
      0,
      tx,
      ty,
      tz,
      1,
    ]);
  }

  forward() {
    // OpenGL-style: look down -Z at yaw=0, pitch=0.
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    return vec3Normalize([sy * cp, -sp, -cy * cp]);
  }

  right() {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    return vec3Normalize([cy, 0, sy]);
  }
}

function createMeshGpu(gl) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const posBuf = gl.createBuffer();
  const norBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const idxBuf = gl.createBuffer();

  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, norBuf);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);

  gl.bindVertexArray(null);

  return {
    vao,
    posBuf,
    norBuf,
    colBuf,
    idxBuf,
    indexCount: 0,
  };
}

function uploadMeshFromWasm(gl, wasm, sim, gpuMesh) {
  // Positions
  const posPtr = sim.mesh_positions_ptr();
  const posLen = sim.mesh_positions_len();
  const pos = new Float32Array(wasm.memory.buffer, posPtr, posLen);

  const norPtr = sim.mesh_normals_ptr();
  const norLen = sim.mesh_normals_len();
  const nor = new Float32Array(wasm.memory.buffer, norPtr, norLen);

  const colPtr = sim.mesh_colors_ptr();
  const colLen = sim.mesh_colors_len();
  const col = new Float32Array(wasm.memory.buffer, colPtr, colLen);

  const idxPtr = sim.mesh_indices_ptr();
  const idxLen = sim.mesh_indices_len();
  const idx = new Uint32Array(wasm.memory.buffer, idxPtr, idxLen);

  gl.bindVertexArray(gpuMesh.vao);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.norBuf);
  gl.bufferData(gl.ARRAY_BUFFER, nor, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, gpuMesh.colBuf);
  gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gpuMesh.idxBuf);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);

  gl.bindVertexArray(null);

  gpuMesh.indexCount = idx.length;
}

async function main() {
  const wasm = await init();
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) throw new Error("WebGL2 not supported");

  const program = createProgram(gl, vsSource, fsSource);
  const uViewProjLoc = gl.getUniformLocation(program, "uViewProj");
  const uLightDirLoc = gl.getUniformLocation(program, "uLightDir");
  const uCamPosLoc = gl.getUniformLocation(program, "uCamPos");
  const uFogColorLoc = gl.getUniformLocation(program, "uFogColor");
  const uFogDensityLoc = gl.getUniformLocation(program, "uFogDensity");

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const cam = new FlyCamera();

  const viewRadius = 0.35;
  const fogColor = [0.04, 0.06, 0.08];
  const fogDensity = 8.0;

  const params = new GrayScottParams();
  // A reasonably lively Gray–Scott regime.
  params.set_du(0.16);
  params.set_dv(0.08);
  params.set_feed(0.037);
  params.set_kill(0.06);

  const sim = new Simulation(192, 192, 192, 1337, params);
  // Smaller dt => more stable, slower evolution.
  sim.set_dt(0.1);
  // Choose one seed method:
  // sim.seed_sphere(0.25);
  sim.seed_perlin(6.0, 4, 0.0, 1.0);

  const gpuMesh = createMeshGpu(gl);

  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (e.code === "Escape") {
      document.exitPointerLock?.();
      return;
    }
    keys.add(e.code);
  });
  window.addEventListener("keyup", (e) => keys.delete(e.code));

  // Mouse-look via Pointer Lock.
  canvas.addEventListener("click", async () => {
    if (document.pointerLockElement !== canvas) {
      await canvas.requestPointerLock();
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    cam.yaw += e.movementX * 0.002;
    cam.pitch += e.movementY * 0.002;
    cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch));
  });

  window.addEventListener("wheel", (e) => {
    const sign = Math.sign(e.deltaY);
    cam.moveSpeed = Math.max(0.1, Math.min(12.0, cam.moveSpeed * (sign > 0 ? 0.9 : 1.1)));
  });

  let lastT = performance.now();
  let accum = 0;
  let frame = 0;

  function step(dt) {
    // Fixed-ish sim step.
    accum += dt;
    const tick = 1 / 6;
    while (accum >= tick) {
      sim.step(1);
      accum -= tick;
    }

    // Mesh less frequently.
    frame++;
    if (frame % 10 === 0) {
      const iso = 0.5;
      sim.generate_isosurface_mesh_visible(
        cam.pos[0],
        cam.pos[1],
        cam.pos[2],
        viewRadius,
        iso,
        0.15,
        0.65,
        0.9,
        0.9,
      );
      uploadMeshFromWasm(gl, wasm, sim, gpuMesh);
    }
  }

  function moveCam(dt) {
    const v = cam.moveSpeed * dt;
    let delta = [0, 0, 0];
    const f = cam.forward();
    const r = cam.right();

    if (keys.has("KeyW")) delta = vec3Add(delta, vec3Scale(f, v));
    if (keys.has("KeyS")) delta = vec3Add(delta, vec3Scale(f, -v));
    if (keys.has("KeyD")) delta = vec3Add(delta, vec3Scale(r, v));
    if (keys.has("KeyA")) delta = vec3Add(delta, vec3Scale(r, -v));

    cam.pos = vec3Add(cam.pos, delta);
  }

  function render(tNow) {
    const dt = Math.min(0.05, (tNow - lastT) / 1000);
    lastT = tNow;

    resizeCanvasToDisplaySize(canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);

    moveCam(dt);
    step(dt);

    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective((60 * Math.PI) / 180, aspect, 0.01, 100.0);
    const view = cam.viewMatrix();
    const viewProj = mat4Mul(proj, view);

    gl.clearColor(fogColor[0], fogColor[1], fogColor[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(program);
    gl.uniformMatrix4fv(uViewProjLoc, false, viewProj);
    gl.uniform3f(uLightDirLoc, -0.4, 0.8, 0.2);
    gl.uniform3f(uCamPosLoc, cam.pos[0], cam.pos[1], cam.pos[2]);
    gl.uniform3f(uFogColorLoc, fogColor[0], fogColor[1], fogColor[2]);
    gl.uniform1f(uFogDensityLoc, fogDensity);

    gl.bindVertexArray(gpuMesh.vao);
    if (gpuMesh.indexCount > 0) {
      gl.drawElements(gl.TRIANGLES, gpuMesh.indexCount, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);

    const verts = sim.mesh_vertex_count();
    const vMin = sim.v_min();
    const vMax = sim.v_max();
    const ge = sim.v_count_ge(0.5);
    const vRange = vMax - vMin;
    const dtSim = sim.dt();
    statsEl.textContent = `verts ${verts.toLocaleString()}  iso 0.5  r ${viewRadius.toFixed(2)}  V[${vMin.toFixed(6)},${vMax.toFixed(6)}]  dV ${vRange.toExponential(2)}  ge0.5 ${ge}  dt ${dtSim.toFixed(3)}  speed ${cam.moveSpeed.toFixed(2)}`;

    requestAnimationFrame(render);
  }

  // Build an initial mesh immediately.
  const iso0 = 0.5;
  sim.generate_isosurface_mesh_visible(
    cam.pos[0],
    cam.pos[1],
    cam.pos[2],
    viewRadius,
    iso0,
    0.15,
    0.65,
    0.9,
    0.9,
  );
  uploadMeshFromWasm(gl, wasm, sim, gpuMesh);

  requestAnimationFrame(render);
}

main().catch((e) => {
  console.error(e);
  statsEl.textContent = String(e);
});
