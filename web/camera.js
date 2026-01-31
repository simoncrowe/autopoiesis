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

function quatMul(a, b) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function quatNormalize(q) {
  const len = Math.hypot(q[0], q[1], q[2], q[3]);
  if (len < 1e-8) return [0, 0, 0, 1];
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

function quatFromAxisAngle(axis, angle) {
  const a = vec3Normalize(axis);
  const half = angle * 0.5;
  const s = Math.sin(half);
  return [a[0] * s, a[1] * s, a[2] * s, Math.cos(half)];
}

function quatRotateVec3(q, v) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];

  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);

  return [
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  ];
}

export function mat4Mul(a, b) {
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

export function mat4Perspective(fovyRad, aspect, near, far) {
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

export function mat4Invert(m) {
  const inv = new Float32Array(16);

  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return null;

  det = 1.0 / det;
  for (let i = 0; i < 16; i++) inv[i] *= det;
  return inv;
}

export class FlyCamera {
  constructor() {
    this.pos = [0, 0, 0.25];
    // Quaternion (x, y, z, w)
    this.q = [0, 0, 0, 1];
    this.moveSpeed = 0.75;
  }

  viewMatrix() {
    const f = this.forward();
    const r = this.right();
    const u = this.up();

    const tx = -vec3Dot(r, this.pos);
    const ty = -vec3Dot(u, this.pos);
    const tz = vec3Dot(f, this.pos);

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
    return vec3Normalize(quatRotateVec3(this.q, [0, 0, -1]));
  }

  up() {
    return vec3Normalize(quatRotateVec3(this.q, [0, 1, 0]));
  }

  right() {
    return vec3Normalize(quatRotateVec3(this.q, [1, 0, 0]));
  }

  yaw(deltaRad) {
    const axis = this.up();
    this.q = quatNormalize(quatMul(quatFromAxisAngle(axis, deltaRad), this.q));
  }

  pitch(deltaRad) {
    const axis = this.right();
    this.q = quatNormalize(quatMul(quatFromAxisAngle(axis, deltaRad), this.q));
  }
}

export function createMouseFlightController({
  canvas,
  camera,
  lookSensitivity = 0.0015 * 0.75,
  moveSpeedScale = 0.5,
  responsiveness = 10.0,
}) {
  let isThrusting = false;
  let pendingThrust = false;
  let forwardSpeed = 0;

  canvas.addEventListener("pointerdown", async (e) => {
    if (e.button !== 0) return;
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

    pendingThrust = true;

    if (document.pointerLockElement !== canvas) {
      try {
        await canvas.requestPointerLock();
      } catch {
        // ignore
      }
    }

    if (document.pointerLockElement === canvas) {
      isThrusting = true;
      pendingThrust = false;
    }
  });

  window.addEventListener("pointerup", (e) => {
    if (e.button !== 0) return;
    isThrusting = false;
    pendingThrust = false;
  });

  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas) {
      if (pendingThrust) {
        isThrusting = true;
        pendingThrust = false;
      }
      return;
    }

    isThrusting = false;
    pendingThrust = false;
  });

  window.addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    camera.yaw(-e.movementX * lookSensitivity);
    camera.pitch(-e.movementY * lookSensitivity);
  });

  window.addEventListener("wheel", (e) => {
    const sign = Math.sign(e.deltaY);
    camera.moveSpeed = Math.max(0.1, Math.min(8.0, camera.moveSpeed * (sign > 0 ? 0.9 : 1.1)));
  });

  function update(dt) {
    const targetSpeed = isThrusting ? (camera.moveSpeed * moveSpeedScale) : 0;
    const alpha = 1.0 - Math.exp(-responsiveness * dt);
    forwardSpeed += (targetSpeed - forwardSpeed) * alpha;

    if (Math.abs(forwardSpeed) < 1e-6) return;

    const f = camera.forward();
    camera.pos = vec3Add(camera.pos, vec3Scale(f, forwardSpeed * dt));
  }

  return {
    update,
  };
}
