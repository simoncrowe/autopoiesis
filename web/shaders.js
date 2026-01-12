function compileShader(gl, type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
  }
  return shader;
}

function createProgram(gl, vsSrc, fsSrc) {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "program link failed");
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

const VS_MESH = `#version 300 es
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

const FS_MESH = `#version 300 es
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

  // Premultiply alpha.
  outColor = vec4(rgb * vCol.a, vCol.a);
}
`;

export function createMeshProgram(gl) {
  const program = createProgram(gl, VS_MESH, FS_MESH);
  return {
    program,
    uniforms: {
      uViewProj: gl.getUniformLocation(program, "uViewProj"),
      uLightDir: gl.getUniformLocation(program, "uLightDir"),
      uCamPos: gl.getUniformLocation(program, "uCamPos"),
      uFogColor: gl.getUniformLocation(program, "uFogColor"),
      uFogDensity: gl.getUniformLocation(program, "uFogDensity"),
    },
  };
}

function createOitTargets(gl, width, height) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);

  const accumTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, accumTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, accumTex, 0);

  const revealTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, revealTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, width, height, 0, gl.RED, gl.HALF_FLOAT, null);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, revealTex, 0);

  const depthRb = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
  gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, width, height);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthRb);

  gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.bindRenderbuffer(gl.RENDERBUFFER, null);

  if (!ok) {
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(accumTex);
    gl.deleteTexture(revealTex);
    gl.deleteRenderbuffer(depthRb);
    return null;
  }

  return { fbo, accumTex, revealTex, depthRb, width, height };
}

function deleteOitTargets(gl, t) {
  if (!t) return;
  gl.deleteFramebuffer(t.fbo);
  gl.deleteTexture(t.accumTex);
  gl.deleteTexture(t.revealTex);
  gl.deleteRenderbuffer(t.depthRb);
}

const FS_OIT = `#version 300 es
precision highp float;

in vec3 vNor;
in vec4 vCol;
in vec3 vPos;

uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uOitWeight;

layout(location=0) out vec4 outAccum;
layout(location=1) out float outReveal;

void main() {
  vec3 n = normalize(vNor);
  float ndl = max(dot(n, normalize(uLightDir)), 0.0);
  vec3 base = vCol.rgb;
  vec3 lit = base * (0.25 + 0.75 * ndl);

  float d = length(vPos - uCamPos);
  float fog = 1.0 - exp(-uFogDensity * d * d);
  vec3 rgb = mix(lit, uFogColor, fog);

  float a = clamp(vCol.a, 0.0, 1.0);

  float z = gl_FragCoord.z;
  float w = clamp(uOitWeight / (1e-5 + pow(z, 4.0)), 0.01, 3000.0);

  outAccum = vec4(rgb * a, a) * w;
  outReveal = a;
}
`;

const VS_RESOLVE = `#version 300 es
precision highp float;

out vec2 vUv;

void main() {
  vec2 p = vec2(
    (gl_VertexID == 2) ? 3.0 : -1.0,
    (gl_VertexID == 1) ? 3.0 : -1.0
  );
  vUv = 0.5 * (p + 1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}
`;

const FS_RESOLVE = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uAccum;
uniform sampler2D uReveal;

out vec4 outColor;

void main() {
  vec4 accum = texture(uAccum, vUv);
  float reveal = texture(uReveal, vUv).r;

  float a = clamp(1.0 - reveal, 0.0, 1.0);
  if (a < 1e-6) {
    outColor = vec4(0.0);
    return;
  }

  vec3 rgb = accum.rgb / max(accum.a, 1e-5);
  outColor = vec4(rgb * a, a);
}
`;

export function createWeightedOitRenderer(gl) {
  const floatExt = gl.getExtension("EXT_color_buffer_float");
  if (!floatExt) return null;

  const drawBuffersIndexedExt = gl.getExtension("EXT_draw_buffers_indexed");

  const meshProgram = createProgram(gl, VS_MESH, FS_OIT);
  const resolveProgram = createProgram(gl, VS_RESOLVE, FS_RESOLVE);

  const meshUniforms = {
    uViewProj: gl.getUniformLocation(meshProgram, "uViewProj"),
    uLightDir: gl.getUniformLocation(meshProgram, "uLightDir"),
    uCamPos: gl.getUniformLocation(meshProgram, "uCamPos"),
    uFogColor: gl.getUniformLocation(meshProgram, "uFogColor"),
    uFogDensity: gl.getUniformLocation(meshProgram, "uFogDensity"),
    uOitWeight: gl.getUniformLocation(meshProgram, "uOitWeight"),
  };

  const resolveUniforms = {
    uAccum: gl.getUniformLocation(resolveProgram, "uAccum"),
    uReveal: gl.getUniformLocation(resolveProgram, "uReveal"),
  };

  const resolveVao = gl.createVertexArray();
  gl.bindVertexArray(resolveVao);
  gl.bindVertexArray(null);

  let targets = null;

  function resize(width, height) {
    if (targets && targets.width === width && targets.height === height) return true;
    deleteOitTargets(gl, targets);
    targets = createOitTargets(gl, width, height);
    return !!targets;
  }

  function render({
    canvasWidth,
    canvasHeight,
    vao,
    indexCount,
    viewProj,
    camPos,
    lightDir,
    fogColor,
    fogDensity,
    oitWeight,
  }) {
    if (!targets || targets.width !== canvasWidth || targets.height !== canvasHeight) {
      if (!resize(canvasWidth, canvasHeight)) return;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.fbo);
    gl.viewport(0, 0, targets.width, targets.height);

    gl.clearBufferfv(gl.COLOR, 0, new Float32Array([0, 0, 0, 0]));
    gl.clearBufferfv(gl.COLOR, 1, new Float32Array([1, 0, 0, 0]));
    gl.clear(gl.DEPTH_BUFFER_BIT);

    gl.useProgram(meshProgram);
    gl.uniformMatrix4fv(meshUniforms.uViewProj, false, viewProj);
    gl.uniform3f(meshUniforms.uLightDir, lightDir[0], lightDir[1], lightDir[2]);
    gl.uniform3f(meshUniforms.uCamPos, camPos[0], camPos[1], camPos[2]);
    gl.uniform3f(meshUniforms.uFogColor, fogColor[0], fogColor[1], fogColor[2]);
    gl.uniform1f(meshUniforms.uFogDensity, fogDensity);
    gl.uniform1f(meshUniforms.uOitWeight, oitWeight);

    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);

    gl.bindVertexArray(vao);

    if (typeof gl.blendFunci === "function" && typeof gl.blendEquationi === "function") {
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

      gl.blendEquationi(0, gl.FUNC_ADD);
      gl.blendFunci(0, gl.ONE, gl.ONE);

      gl.blendEquationi(1, gl.FUNC_ADD);
      gl.blendFunci(1, gl.ZERO, gl.ONE_MINUS_SRC_COLOR);

      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
    } else if (
      drawBuffersIndexedExt &&
      typeof drawBuffersIndexedExt.blendFunciOES === "function" &&
      typeof drawBuffersIndexedExt.blendEquationiOES === "function"
    ) {
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

      drawBuffersIndexedExt.blendEquationiOES(0, gl.FUNC_ADD);
      drawBuffersIndexedExt.blendFunciOES(0, gl.ONE, gl.ONE);

      drawBuffersIndexedExt.blendEquationiOES(1, gl.FUNC_ADD);
      drawBuffersIndexedExt.blendFunciOES(1, gl.ZERO, gl.ONE_MINUS_SRC_COLOR);

      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);
    } else {
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.NONE]);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);

      gl.drawBuffers([gl.NONE, gl.COLOR_ATTACHMENT1]);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_COLOR);
      gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_INT, 0);

      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    }

    gl.bindVertexArray(null);
    gl.depthMask(true);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasWidth, canvasHeight);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(resolveProgram);
    gl.bindVertexArray(resolveVao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, targets.accumTex);
    gl.uniform1i(resolveUniforms.uAccum, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, targets.revealTex);
    gl.uniform1i(resolveUniforms.uReveal, 1);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(null);
  }

  return {
    resize,
    render,
  };
}
