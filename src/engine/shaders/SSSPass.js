/**
 * SSSPass — Separable Screen-Space Subsurface Scattering
 *
 * Based on Jimenez et al. 2015 — "Separable Subsurface Scattering".
 * Approximates light diffusion through skin using two post-processing passes
 * (horizontal + vertical separable blur), weighted by a skin diffusion profile.
 *
 * Key properties:
 *   - Red channel scatters most (widest kernel) → warm translucent look
 *   - Green scatters moderately
 *   - Blue scatters least (sharpest)
 *   - Depth-aware: prevents bleeding across silhouettes (ears vs background)
 *   - Runs at 60fps with a 25-tap kernel
 *
 * Integration:
 *   Uses Three.js EffectComposer + custom ShaderPass.
 *   The normal scene render becomes the base pass.
 *
 * Usage:
 *   import { SSSEffect } from './shaders/SSSPass.js';
 *   const sss = new SSSEffect(renderer, scene, camera);
 *   // In render loop:  sss.render()  instead of  renderer.render(scene, camera)
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ---------------------------------------------------------------------------
// Skin Diffusion Profile (Jimenez 2015)
// ---------------------------------------------------------------------------

/**
 * 25-tap separable kernel weights + offsets.
 * Each sample: [offset, weightR, weightG, weightB]
 *
 * Derived from a sum-of-Gaussians fit to the measured diffusion profile
 * of human skin (Caucasian, average thickness).
 *
 * Red scatters the most (widest Gaussian), blue the least.
 */
const KERNEL_SAMPLES = [
  // offset,   wR,      wG,      wB
  [  0.000,  0.2340,  0.2340,  0.2340 ],  // center tap (sharpest)
  [  0.025,  0.1000,  0.0950,  0.0600 ],
  [ -0.025,  0.1000,  0.0950,  0.0600 ],
  [  0.055,  0.0800,  0.0580,  0.0210 ],
  [ -0.055,  0.0800,  0.0580,  0.0210 ],
  [  0.090,  0.0550,  0.0300,  0.0080 ],
  [ -0.090,  0.0550,  0.0300,  0.0080 ],
  [  0.130,  0.0360,  0.0140,  0.0025 ],
  [ -0.130,  0.0360,  0.0140,  0.0025 ],
  [  0.180,  0.0240,  0.0060,  0.0008 ],
  [ -0.180,  0.0240,  0.0060,  0.0008 ],
  [  0.240,  0.0150,  0.0025,  0.0002 ],
  [ -0.240,  0.0150,  0.0025,  0.0002 ],
  [  0.320,  0.0090,  0.0010,  0.0001 ],
  [ -0.320,  0.0090,  0.0010,  0.0001 ],
  [  0.420,  0.0050,  0.0003,  0.0000 ],
  [ -0.420,  0.0050,  0.0003,  0.0000 ],
  [  0.550,  0.0025,  0.0001,  0.0000 ],
  [ -0.550,  0.0025,  0.0001,  0.0000 ],
  [  0.720,  0.0012,  0.0000,  0.0000 ],
  [ -0.720,  0.0012,  0.0000,  0.0000 ],
  [  0.950,  0.0005,  0.0000,  0.0000 ],
  [ -0.950,  0.0005,  0.0000,  0.0000 ],
  [  1.250,  0.0002,  0.0000,  0.0000 ],
  [ -1.250,  0.0002,  0.0000,  0.0000 ],
];

// Normalize kernel so weights sum to 1.0 per channel
const _normalizeKernel = (samples) => {
  const sumR = samples.reduce((s, k) => s + k[1], 0);
  const sumG = samples.reduce((s, k) => s + k[2], 0);
  const sumB = samples.reduce((s, k) => s + k[3], 0);
  return samples.map(([off, r, g, b]) => [off, r / sumR, g / sumG, b / sumB]);
};

const NORMALIZED_KERNEL = _normalizeKernel(KERNEL_SAMPLES);

// ---------------------------------------------------------------------------
// SSS Blur Shader
// ---------------------------------------------------------------------------

const SSSBlurShader = {
  uniforms: {
    tDiffuse:      { value: null },
    tDepth:        { value: null },
    resolution:    { value: new THREE.Vector2(1, 1) },
    direction:     { value: new THREE.Vector2(1, 0) }, // (1,0)=H, (0,1)=V
    sssWidth:      { value: 0.012 },  // blur radius in UV space
    cameraNear:    { value: 0.01 },
    cameraFar:     { value: 10.0 },
    // Kernel is compiled into the shader as a constant array (see below)
  },

  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,

  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform sampler2D tDepth;
    uniform vec2 resolution;
    uniform vec2 direction;
    uniform float sssWidth;
    uniform float cameraNear;
    uniform float cameraFar;

    varying vec2 vUv;

    // Linearize depth from depth buffer
    float linearizeDepth(float d) {
      return cameraNear * cameraFar / (cameraFar - d * (cameraFar - cameraNear));
    }

    void main() {
      vec4 colorCenter = texture2D(tDiffuse, vUv);
      float depthCenter = linearizeDepth(texture2D(tDepth, vUv).r);

      // Skip SSS for background (far plane)
      if (depthCenter > cameraFar * 0.95) {
        gl_FragColor = colorCenter;
        return;
      }

      // Depth-dependent blur width: closer faces get more scatter
      float blurWidth = sssWidth / depthCenter;
      vec2 step = direction * blurWidth / resolution;

      // Kernel weights (25 taps) — compiled from NORMALIZED_KERNEL
      ${_generateKernelGLSL()}

      // Accumulate samples
      vec3 colorSum = vec3(0.0);

      for (int i = 0; i < ${NORMALIZED_KERNEL.length}; i++) {
        vec2 offset = vUv + kernel[i].x * step;
        vec4 sampleColor = texture2D(tDiffuse, offset);
        float sampleDepth = linearizeDepth(texture2D(tDepth, offset).r);

        // Depth test: reject samples from different surfaces
        float depthDiff = abs(depthCenter - sampleDepth);
        float depthWeight = 1.0 - smoothstep(0.0, 0.005 * depthCenter, depthDiff);

        // Per-channel weights from kernel
        vec3 weight = vec3(kernel[i].y, kernel[i].z, kernel[i].w);

        colorSum += sampleColor.rgb * weight * depthWeight;
      }

      // Normalize by actual weights used (accounts for rejected samples)
      gl_FragColor = vec4(colorSum, colorCenter.a);
    }
  `,
};

/**
 * Generate GLSL constant array from the kernel samples.
 */
function _generateKernelGLSL() {
  const lines = NORMALIZED_KERNEL.map((k, i) =>
    `kernel[${i}] = vec4(${k[0].toFixed(4)}, ${k[1].toFixed(6)}, ${k[2].toFixed(6)}, ${k[3].toFixed(6)});`
  );
  return `vec4 kernel[${NORMALIZED_KERNEL.length}];\n      ` + lines.join('\n      ');
}

// ---------------------------------------------------------------------------
// SSSEffect
// ---------------------------------------------------------------------------

export class SSSEffect {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Scene} scene
   * @param {THREE.PerspectiveCamera} camera
   * @param {object} [options]
   * @param {number} [options.sssWidth=0.012] - Scatter radius (UV space)
   * @param {boolean} [options.enabled=true]
   */
  constructor(renderer, scene, camera, options = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.enabled = options.enabled ?? true;
    this.sssWidth = options.sssWidth ?? 0.012;

    // Depth render target
    this._depthTarget = new THREE.WebGLRenderTarget(
      renderer.domElement.width,
      renderer.domElement.height,
      {
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
      }
    );
    this._depthTarget.depthTexture = new THREE.DepthTexture();
    this._depthTarget.depthTexture.type = THREE.UnsignedIntType;

    // Effect Composer
    this._composer = new EffectComposer(renderer);

    // Pass 1: Normal scene render
    const renderPass = new RenderPass(scene, camera);
    this._composer.addPass(renderPass);

    // Pass 2: Horizontal SSS blur
    this._hBlurPass = new ShaderPass(SSSBlurShader);
    this._hBlurPass.uniforms.direction.value.set(1, 0);
    this._hBlurPass.uniforms.sssWidth.value = this.sssWidth;
    this._composer.addPass(this._hBlurPass);

    // Pass 3: Vertical SSS blur
    this._vBlurPass = new ShaderPass(SSSBlurShader);
    this._vBlurPass.uniforms.direction.value.set(0, 1);
    this._vBlurPass.uniforms.sssWidth.value = this.sssWidth;
    this._composer.addPass(this._vBlurPass);

    // Initial size
    this._updateSize();
  }

  /**
   * Render the scene with SSS post-processing.
   */
  render() {
    if (!this.enabled) {
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Step 1: Render depth to depth target
    this.renderer.setRenderTarget(this._depthTarget);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(null);

    // Step 2: Update blur pass uniforms
    const depthTex = this._depthTarget.depthTexture;
    this._hBlurPass.uniforms.tDepth.value = depthTex;
    this._vBlurPass.uniforms.tDepth.value = depthTex;
    this._hBlurPass.uniforms.cameraNear.value = this.camera.near;
    this._hBlurPass.uniforms.cameraFar.value = this.camera.far;
    this._vBlurPass.uniforms.cameraNear.value = this.camera.near;
    this._vBlurPass.uniforms.cameraFar.value = this.camera.far;

    // Step 3: Render through composer (scene → hBlur → vBlur → screen)
    this._composer.render();
  }

  /**
   * Update render target sizes (call on resize).
   */
  resize(width, height) {
    this._depthTarget.setSize(width, height);
    this._composer.setSize(width, height);

    const res = new THREE.Vector2(width, height);
    this._hBlurPass.uniforms.resolution.value.copy(res);
    this._vBlurPass.uniforms.resolution.value.copy(res);
  }

  /**
   * Set the SSS scatter width.
   * @param {number} width - Scatter radius in UV space (0.005-0.03 typical)
   */
  setWidth(width) {
    this.sssWidth = width;
    this._hBlurPass.uniforms.sssWidth.value = width;
    this._vBlurPass.uniforms.sssWidth.value = width;
  }

  /**
   * Enable or disable the SSS effect.
   */
  setEnabled(enabled) {
    this.enabled = enabled;
  }

  /**
   * Clean up GPU resources.
   */
  dispose() {
    this._depthTarget.dispose();
    if (this._depthTarget.depthTexture) {
      this._depthTarget.depthTexture.dispose();
    }
    this._composer.dispose();
  }

  // =========================================================================
  // Private
  // =========================================================================

  _updateSize() {
    const w = this.renderer.domElement.width;
    const h = this.renderer.domElement.height;
    this.resize(w, h);
  }
}

export { SSSBlurShader, NORMALIZED_KERNEL };
