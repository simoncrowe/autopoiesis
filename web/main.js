const canvas = document.querySelector("#c");
const statsEl = document.querySelector("#stats");
const seedInput = document.querySelector("#seed");
const restartBtn = document.querySelector("#restart");

function normalizeSeed(value) {
  const n = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(n)) return Date.now() >>> 0;
  return n >>> 0;
}

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
    this.pos = [0, 0, 0.25];
    this.yaw = 0;
    this.pitch = 0;
    this.moveSpeed = 0.75;
  }

  viewMatrix() {
    const f = this.forward();

    // Build an orthonormal basis (right-handed).
    const worldUp = [0, 1, 0];
    let r = vec3Cross(f, worldUp);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-6) {
      // Forward is too close to worldUp; pick a different up.
      r = vec3Cross(f, [1, 0, 0]);
    }
    r = vec3Normalize(r);
    const u = vec3Normalize(vec3Cross(r, f));

    const tx = -vec3Dot(r, this.pos);
    const ty = -vec3Dot(u, this.pos);
    const tz = vec3Dot(f, this.pos);

    // Column-major storage, but these are ROWS of the view matrix.
    // (This is easy to get wrong and causes "orbiting" when looking around.)
    return new Float32Array([
      r[0],
      u[0],
      -f[0],
      0,
      r[1],
      u[1],
      -f[1],
      0,
      r[2],
      u[2],
      -f[2],
      0,
      tx,
      ty,
      tz,
      1,
    ]);
  }

  forward() {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    return vec3Normalize([sy * cp, -sp, -cy * cp]);
  }

  right() {
    // Used for movement; derived from current forward direction.
    const f = this.forward();
    const worldUp = [0, 1, 0];
    let r = vec3Cross(f, worldUp);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-6) {
      r = vec3Cross(f, [1, 0, 0]);
    }
    return vec3Normalize(r);
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
    vertexCount: 0,
  };
}

function uploadMeshFromBuffers(gl, gpuMesh, msg) {
  const pos = new Float32Array(msg.positions);
  const nor = new Float32Array(msg.normals);
  const col = new Float32Array(msg.colors);
  const idx = new Uint32Array(msg.indices);

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
  gpuMesh.vertexCount = msg.vertexCount || 0;
}


async function main() {
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
  const iso = 0.5;
  const meshColor = [0.15, 0.65, 0.9, 0.9];

  const fogColor = [0.04, 0.06, 0.08];
  const fogDensity = 4.0;

  const gpuMesh = createMeshGpu(gl);

  const keys = new Set();
  window.addEventListener("keydown", (e) => {
    if (e.code === "Escape") {
      document.exitPointerLock?.();
      return;
    }
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    keys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }
    keys.delete(e.code);
  });

  canvas.addEventListener("click", async () => {
    if (document.pointerLockElement !== canvas) {
      await canvas.requestPointerLock();
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    // Non-inverted: moving mouse right => look right, mouse up => look up.
    cam.yaw += e.movementX * 0.0015;
    cam.pitch += e.movementY * 0.0015;
    cam.pitch = Math.max(-1.55, Math.min(1.55, cam.pitch));
  });

  window.addEventListener("wheel", (e) => {
    const sign = Math.sign(e.deltaY);
    cam.moveSpeed = Math.max(0.1, Math.min(8.0, cam.moveSpeed * (sign > 0 ? 0.9 : 1.1)));
  });

  if (!globalThis.crossOriginIsolated || typeof SharedArrayBuffer === "undefined") {
    throw new Error(
      "SharedArrayBuffer requires crossOriginIsolated (COOP/COEP headers).",
    );
  }

  const dims = 192;
  const periodMs = 500;

  const n = dims * dims * dims;

  let simWorker = null;
  let meshWorker = null;

  let lastMeshMs = 0;
  let lastMeshEpoch = 0;
  let lastSimStepsPerSec = 0;
  let lastSimTotalSteps = 0;
  let meshReady = false;

  function stopWorkers() {
    simWorker?.terminate();
    meshWorker?.terminate();
    simWorker = null;
    meshWorker = null;
    meshReady = false;
  }

  function startWorkers(seed) {
    stopWorkers();

    // Reset UI + mesh.
    statsEl.textContent = "";
    gpuMesh.indexCount = 0;
    gpuMesh.vertexCount = 0;
    lastMeshMs = 0;
    lastMeshEpoch = 0;
    lastSimStepsPerSec = 0;
    lastSimTotalSteps = 0;
    lastCamSendAt = 0;

    const vSabs = [
      new SharedArrayBuffer(n * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(n * Float32Array.BYTES_PER_ELEMENT),
    ];

    // Must match `CHUNK` in `wasm/src/sim.rs`.
    const chunkSize = 16;
    const cubes = dims - 1;
    const chunkN = Math.ceil(cubes / chunkSize);
    const chunkTotal = chunkN * chunkN * chunkN;

    const chunkMinSabs = [
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
    ];
    const chunkMaxSabs = [
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
      new SharedArrayBuffer(chunkTotal * Float32Array.BYTES_PER_ELEMENT),
    ];

    // Shared control layout:
    // ctrlI32[0] = epoch
    // ctrlI32[1] = frontIndex (0/1)
    const ctrl = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2);
    const ctrlI32 = new Int32Array(ctrl);
    ctrlI32[0] = 0;
    ctrlI32[1] = 0;

    // timingI64[0] = last publish Date.now() (ms)
    // timingI64[1] = target period (ms)
    const timing = new SharedArrayBuffer(BigInt64Array.BYTES_PER_ELEMENT * 2);
    const timingI64 = new BigInt64Array(timing);
    timingI64[0] = BigInt(Date.now());
    timingI64[1] = BigInt(periodMs);

    simWorker = new Worker(new URL("./sim_worker.js", import.meta.url), { type: "module" });
    meshWorker = new Worker(new URL("./mesh_worker.js", import.meta.url), { type: "module" });

    simWorker.postMessage({
      type: "init",
      ctrl,
      vSabs,
      chunkMinSabs,
      chunkMaxSabs,
      timing,
      seed,
      dims,
      periodMs,
    });
    meshWorker.postMessage({ type: "init", ctrl, vSabs, chunkMinSabs, chunkMaxSabs, timing });

    simWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "sim_ready") {
        return;
      }

      if (msg.type === "sim_stats") {
        lastSimStepsPerSec = msg.stepsPerSec || 0;
        lastSimTotalSteps = msg.totalSteps || 0;
        return;
      }

      if (msg.type === "error") {
        console.error(msg.message);
        statsEl.textContent = String(msg.message);
      }
    };

    meshWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg !== "object") return;

      if (msg.type === "mesh_ready") {
        meshReady = true;
        return;
      }

      if (msg.type === "mesh") {
        uploadMeshFromBuffers(gl, gpuMesh, msg);
        lastMeshMs = msg.meshMs || 0;
        lastMeshEpoch = msg.epoch || 0;
        return;
      }

      if (msg.type === "error") {
        console.error(msg.message);
        statsEl.textContent = String(msg.message);
      }
    };
  }

  function restartFromUi() {
    const seed = normalizeSeed(seedInput?.value);
    if (seedInput) seedInput.value = String(seed);
    startWorkers(seed);
  }

  const urlSeed = new URLSearchParams(globalThis.location?.search ?? "").get("seed");
  const initialSeed = normalizeSeed(urlSeed ?? seedInput?.value ?? 1337);
  if (seedInput) seedInput.value = String(initialSeed);

  restartBtn?.addEventListener("click", () => {
    document.exitPointerLock?.();
    restartFromUi();
  });

  seedInput?.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    restartFromUi();
  });

  let lastCamSendAt = 0;
  startWorkers(initialSeed);
  function sendCamera() {
    const now = performance.now();
    if (now - lastCamSendAt < 33) return;
    lastCamSendAt = now;

    if (!meshReady || !meshWorker) return;
    meshWorker.postMessage({
      type: "camera",
      pos: cam.pos,
      radius: viewRadius,
      iso,
      color: meshColor,
    });
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

  let lastT = performance.now();
  function render(tNow) {
    const dt = Math.min(0.05, (tNow - lastT) / 1000);
    lastT = tNow;

    resizeCanvasToDisplaySize(canvas);
    gl.viewport(0, 0, canvas.width, canvas.height);

    moveCam(dt);
    sendCamera();

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

    const vtx = gpuMesh.vertexCount;
    statsEl.textContent = `verts ${vtx.toLocaleString()}  iso ${iso.toFixed(2)}  r ${viewRadius.toFixed(2)}  sim ${lastSimStepsPerSec.toFixed(1)} steps/s  mesh ${lastMeshMs.toFixed(1)}ms  epoch ${lastMeshEpoch}  steps ${Math.floor(lastSimTotalSteps).toLocaleString()}`;

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

main().catch((e) => {
  console.error(e);
  statsEl.textContent = String(e);
});
