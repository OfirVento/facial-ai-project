/**
 * FaceRenderer.js
 *
 * Professional Three.js PBR face renderer for a clinical facial design platform.
 * Provides realistic skin rendering with subsurface scattering approximation,
 * clinical 3-point lighting, camera controls, morph deformation, region
 * highlighting, comparison mode, expression animation, and screenshot capture.
 *
 * Usage:
 *   const renderer = new FaceRenderer(document.getElementById('viewport'));
 *   renderer.init();
 *   renderer.loadFromGLB('/models/face.glb');
 */

import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { RectAreaLightHelper } from 'three/addons/helpers/RectAreaLightHelper.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_SKIN_COLOR = 0xe8b89d;
const BG_COLOR_TOP = new THREE.Color(0x16162e);
const BG_COLOR_BOTTOM = new THREE.Color(0x0a0a1a);

const CAMERA_FOV = 30;
const CAMERA_NEAR = 0.01;
const CAMERA_FAR = 100;

/** Preset camera orientations (spherical: theta, phi, radius). */
const VIEW_PRESETS = {
  front:                { theta: 0,               phi: Math.PI / 2, radius: 3.0 },
  'profile-left':       { theta: -Math.PI / 2,    phi: Math.PI / 2, radius: 3.0 },
  'profile-right':      { theta: Math.PI / 2,     phi: Math.PI / 2, radius: 3.0 },
  'three-quarter-left': { theta: -Math.PI / 4,    phi: Math.PI / 2, radius: 3.0 },
  'three-quarter-right':{ theta: Math.PI / 4,     phi: Math.PI / 2, radius: 3.0 },
  above:                { theta: 0,               phi: Math.PI / 4, radius: 3.0 },
  below:                { theta: 0,               phi: 3 * Math.PI / 4, radius: 3.0 },
};

const LERP_SPEED = 0.08;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compute the bounding-box center and a reasonable camera distance for a mesh.
 */
function computeFitParams(geometry) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);
  const radius = maxDim * 1.8;
  return { center, radius };
}

/**
 * Creates a gradient background plane placed far behind the face.
 */
function createBackgroundPlane() {
  const geo = new THREE.PlaneGeometry(100, 100, 1, 64);
  const colors = new Float32Array(geo.attributes.position.count * 3);
  const positions = geo.attributes.position.array;
  for (let i = 0; i < geo.attributes.position.count; i++) {
    const y = positions[i * 3 + 1];
    const t = THREE.MathUtils.clamp((y + 50) / 100, 0, 1);
    const c = new THREE.Color().copy(BG_COLOR_BOTTOM).lerp(BG_COLOR_TOP, t);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, depthWrite: false });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.z = -20;
  mesh.renderOrder = -1;
  return mesh;
}

// ---------------------------------------------------------------------------
// FaceRenderer
// ---------------------------------------------------------------------------

export class FaceRenderer {
  /**
   * @param {HTMLElement} container - DOM element to mount the renderer into.
   */
  constructor(container) {
    if (!container) throw new Error('FaceRenderer requires a container element.');
    this.container = container;

    // Core Three objects
    this.scene = null;
    this.camera = null;
    this.renderer = null;

    // Face mesh state
    this.faceMesh = null;
    this.faceMaterial = null;
    this.originalPositions = null; // Float32Array clone

    // SSS post-processing (null until enabled)
    this._sssEffect = null;

    // Morph animation state
    this._morphAnimation = null;

    // Comparison overlay
    this._ghostMesh = null;
    this._comparisonEnabled = false;
    this._clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
    this._comparisonSlider = 0.5;

    // Camera orbit state (spherical coordinates)
    this._orbitTarget = new THREE.Vector3(0, 0, 0);
    this._spherical = { theta: 0, phi: Math.PI / 2, radius: 3 };
    this._sphericalTarget = { ...this._spherical };

    // Interaction
    this._isDragging = false;
    this._previousMouse = { x: 0, y: 0 };
    this._touchStartDist = 0;
    this._boundOnPointerDown = this._onPointerDown.bind(this);
    this._boundOnPointerMove = this._onPointerMove.bind(this);
    this._boundOnPointerUp = this._onPointerUp.bind(this);
    this._boundOnWheel = this._onWheel.bind(this);
    this._boundOnDblClick = this._onDblClick.bind(this);
    this._boundOnTouchStart = this._onTouchStart.bind(this);
    this._boundOnTouchMove = this._onTouchMove.bind(this);
    this._boundOnTouchEnd = this._onTouchEnd.bind(this);
    this._boundOnResize = this._onResize.bind(this);

    // Animation
    this._rafId = null;
    this._clock = new THREE.Clock();
    this._expressionMixer = null;
    this._expressionActions = [];

    // Lights (stored for potential external tweaking)
    this.lights = {};

    // Photo render mode: hybrid = emissive (faithful photo) + specular (3D depth)
    this._photoRenderMode = 'hybrid';     // 'hybrid' | 'pbr' | 'emissive'
    this._photoTexture = null;            // Stored for mode switching
    this._photoExposure = 1.0;            // neutral default — hook for future UI
    this._photoWhiteBalance = 0xffffff;   // neutral default — hook for future UI
  }

  // -----------------------------------------------------------------------
  // Initialisation
  // -----------------------------------------------------------------------

  init() {
    this._initRenderer();
    this._initScene();
    this._initCamera();
    this._initLights();
    this._initMaterial();
    this._bindEvents();
    this._startLoop();
    return this;
  }

  // -- renderer -----------------------------------------------------------

  _initRenderer() {
    // WebGPU readiness check — Three.js v0.170+ has experimental WebGPURenderer.
    // Currently fall back to WebGL which is universally supported.
    // When WebGPU is stable, the commented block below activates automatically.
    //
    // if (navigator.gpu && THREE.WebGPURenderer) {
    //   this.renderer = new THREE.WebGPURenderer({ antialias: true });
    //   await this.renderer.init();
    //   this._isWebGPU = true;
    //   console.log('FaceRenderer: Using WebGPU renderer');
    // }

    this._isWebGPU = false;
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true, // needed for screenshots
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    // Log GPU capabilities for debugging
    const gl = this.renderer.getContext();
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    if (debugInfo) {
      console.log('FaceRenderer: GPU:', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
    }
    console.log(`FaceRenderer: WebGPU available: ${!!navigator.gpu}, using: ${this._isWebGPU ? 'WebGPU' : 'WebGL'}`);
  }

  // -- scene --------------------------------------------------------------

  _initScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BG_COLOR_BOTTOM);
    this._bgPlane = createBackgroundPlane();
    this.scene.add(this._bgPlane);
  }

  // -- camera -------------------------------------------------------------

  _initCamera() {
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(CAMERA_FOV, aspect, CAMERA_NEAR, CAMERA_FAR);
    this._applyCameraSpherical(this._spherical);
  }

  // -- lights (3-point clinical studio) -----------------------------------

  _initLights() {
    // Initialize RectAreaLight uniforms (required before creating RectAreaLights)
    RectAreaLightUniformsLib.init();

    // Key light -- warm white from upper-right
    const keyLight = new THREE.DirectionalLight(0xfff5e6, 2.0);
    keyLight.position.set(2, 3, 2);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 0.1;
    keyLight.shadow.camera.far = 20;
    keyLight.shadow.bias = -0.001;
    this.scene.add(keyLight);
    this.lights.key = keyLight;

    // Fill light -- cool blue from left
    const fillLight = new THREE.DirectionalLight(0xe6f0ff, 0.7);
    fillLight.position.set(-3, 1, 1);
    this.scene.add(fillLight);
    this.lights.fill = fillLight;

    // Rim / back light -- white from behind
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
    rimLight.position.set(0, 2, -3);
    this.scene.add(rimLight);
    this.lights.rim = rimLight;

    // Ambient -- subtle purple-ish
    const ambient = new THREE.AmbientLight(0x404060, 0.4);
    this.scene.add(ambient);
    this.lights.ambient = ambient;

    // Soft clinical RectAreaLight (simulating a large softbox)
    const rectArea = new THREE.RectAreaLight(0xffffff, 1.2, 4, 4);
    rectArea.position.set(0, 3, 3);
    rectArea.lookAt(0, 0, 0);
    this.scene.add(rectArea);
    this.lights.rectArea = rectArea;
  }

  /**
   * Switch to neutral even lighting for photo-textured models.
   * Removes directional bias so the delighted texture shows faithfully.
   * The photo already contains its own shading information.
   */
  _setPhotoLighting() {
    const hasHDRI = this.scene && this.scene.environment;

    if (hasHDRI) {
      // Gray-sphere calibrated: HDRI is the SOLE light source in photo mode.
      // All direct lights OFF — any additional light doubles the illumination
      // and washes out the photo texture (confirmed by gray sphere test).
      if (this.lights.key) {
        this.lights.key.intensity = 0.0;
      }
      if (this.lights.fill) {
        this.lights.fill.intensity = 0.0;
      }
      if (this.lights.rim) {
        this.lights.rim.intensity = 0.0;
      }
      if (this.lights.ambient) {
        this.lights.ambient.intensity = 0.0;
      }
      if (this.lights.rectArea) {
        this.lights.rectArea.intensity = 0.0;
      }
      console.log('FaceRenderer: Photo lighting (HDRI-only, all direct lights OFF)');
    } else {
      // Fallback: no HDRI — use nearly flat lighting (Phase 10 legacy)
      if (this.lights.key) {
        this.lights.key.intensity = 0.25;
        this.lights.key.color.set(0xffffff);
        this.lights.key.castShadow = false;
      }
      if (this.lights.fill) {
        this.lights.fill.intensity = 0.25;
        this.lights.fill.color.set(0xffffff);
      }
      if (this.lights.rim) {
        this.lights.rim.intensity = 0.1;
      }
      if (this.lights.ambient) {
        this.lights.ambient.intensity = 1.05;
        this.lights.ambient.color.set(0xffffff);
      }
      if (this.lights.rectArea) {
        this.lights.rectArea.intensity = 0.15;
      }
      console.log('FaceRenderer: Photo lighting (flat, no HDRI)');
    }
  }

  // -- HDRI Environment (Phase 0) ------------------------------------------

  /**
   * Apply an HDRI environment map for Image-Based Lighting (IBL).
   * This enables natural irradiance and specular reflections on all PBR materials
   * without setting the environment as the scene background (keeps gradient bg).
   *
   * @param {THREE.Texture} envMap - Prefiltered environment map from HDRIManager.
   * @param {number} [intensity=1.0] - Environment map intensity for the face material.
   */
  setEnvironment(envMap, intensity = 1.0) {
    if (!this.scene) return;

    // Set scene environment for IBL on all PBR materials
    this.scene.environment = envMap;

    // Fine-tune envMap intensity on the face material
    if (this.faceMaterial) {
      this.faceMaterial.envMapIntensity = intensity;
      this.faceMaterial.needsUpdate = true;
    }

    console.log(`FaceRenderer: Environment map applied (intensity=${intensity})`);
  }

  // -- Camera State (Phase 0 — for Lab sync) --------------------------------

  /**
   * Get the current spherical camera state (for syncing to another renderer).
   * @returns {{ theta: number, phi: number, radius: number, target: {x:number, y:number, z:number} }}
   */
  getSphericalState() {
    return {
      theta: this._spherical.theta,
      phi: this._spherical.phi,
      radius: this._spherical.radius,
      target: {
        x: this._orbitTarget.x,
        y: this._orbitTarget.y,
        z: this._orbitTarget.z,
      },
    };
  }

  /**
   * Set the camera spherical state (for syncing from another renderer).
   * Directly sets both current and target to avoid lerp delay.
   * @param {{ theta: number, phi: number, radius: number, target?: {x:number, y:number, z:number} }} state
   */
  setSphericalState(state) {
    if (state.theta !== undefined) {
      this._spherical.theta = state.theta;
      this._sphericalTarget.theta = state.theta;
    }
    if (state.phi !== undefined) {
      this._spherical.phi = state.phi;
      this._sphericalTarget.phi = state.phi;
    }
    if (state.radius !== undefined) {
      this._spherical.radius = state.radius;
      this._sphericalTarget.radius = state.radius;
    }
    if (state.target) {
      this._orbitTarget.set(state.target.x, state.target.y, state.target.z);
    }
    this._applyCameraSpherical(this._spherical);
  }

  // -- Photo render mode (Phase 2: PBR default, emissive fallback) ----------

  /**
   * Switch material to PBR photo mode (Phase 2).
   * Photo texture goes through the standard diffuse `map` channel.
   * HDRI environment provides natural irradiance + specular reflections.
   * Tone mapping stays ON for proper color management.
   */
  _setPbrPhotoMode(texture) {
    const mat = this.faceMaterial;
    this._photoTexture = texture;

    // Clear any emissive state from previous mode
    if (mat.emissiveMap) { mat.emissiveMap.dispose(); mat.emissiveMap = null; }
    mat.emissive.set(0x000000);
    mat.emissiveIntensity = 1.0;

    // Photo through standard diffuse channel — HDRI provides lighting
    mat.map = texture;
    mat.color.set(0xffffff);      // don't tint the texture

    // Expert-mandated PBR values for photo-textured face:
    // sheen=0 (cloth/fuzz, not skin), clearcoat=0 (car paint, creates waxy double-specular)
    // Let HDRI handle natural skin reflections via roughness alone.
    mat.roughness = 0.65;         // 0.6-0.75 range per expert. Let HDRI handle highlights.
    mat.metalness = 0.0;
    mat.sheen = 0.0;              // Expert: disabled — sheen is for cloth, not skin
    mat.clearcoat = 0.0;          // Expert: disabled — creates waxy double-specular
    mat.transmission = 0;         // no SSS on photo-textured mesh
    mat.thickness = 0;

    // Tone mapping ON — proper ACES color management
    mat.toneMapped = true;
    // Expert: don't fight 1/π — set envMapIntensity=1.0, calibrate with gray sphere
    mat.envMapIntensity = 1.0;
    mat.side = THREE.FrontSide;

    mat.needsUpdate = true;

    // Adjust lighting for HDRI photo mode
    this._setPhotoLighting();

    // Gray-sphere calibrated exposure: ACES filmic at 1.0 over-exposes with
    // studio HDRI. 0.55 makes 0x808080 sphere render as middle gray.
    if (this.renderer) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 0.55;
    }

    console.log('FaceRenderer: PBR photo mode (envMap=1.0, exposure=0.55, roughness=0.65)');
  }

  /**
   * Switch material to emissive photo mode (Phase 10 legacy — for A/B testing).
   * Routes photo texture through the emissive path (bypasses BRDF 1/π darkening).
   */
  _setEmissivePhotoMode(texture) {
    const mat = this.faceMaterial;
    this._photoTexture = texture;

    // Route photo through emissive path → bypasses BRDF 1/π darkening
    mat.emissiveMap = texture;
    mat.emissive.set(this._photoWhiteBalance ?? 0xffffff);
    mat.emissiveIntensity = this._photoExposure ?? 1.0;

    // Kill diffuse path
    mat.map = null;
    mat.color.set(0x000000);

    // Disable PBR extras
    mat.sheen = 0.0;
    mat.clearcoat = 0.0;
    mat.transmission = 0;
    mat.thickness = 0;
    mat.roughness = 1.0;
    mat.metalness = 0.0;

    // Bypass tone mapping and HDRI
    mat.toneMapped = false;
    mat.envMapIntensity = 0;
    mat.side = THREE.FrontSide;

    mat.needsUpdate = true;

    this._setPhotoLighting();

    console.log('FaceRenderer: Emissive photo mode (legacy, toneMapped=false)');
  }

  /**
   * Hybrid photo mode: emissive for faithful color + subtle PBR specular for 3D depth.
   * Photo → emissiveMap bypasses BRDF 1/π darkening (photo shown at ~94% brightness).
   * HDRI specular adds subtle highlights that shift with viewing angle.
   * Expert-mandated: roughness=0.7 (not lower — reads plastic), envMap conservative.
   */
  _setHybridPhotoMode(texture) {
    const mat = this.faceMaterial;
    this._photoTexture = texture;

    // Clear any existing diffuse map
    if (mat.map) { mat.map = null; }

    // Expert round 3: disable normal map in hybrid mode.
    if (mat.normalMap) {
      mat.normalMap.dispose();
      mat.normalMap = null;
    }

    // ---- EMISSIVE-ONLY: photo shown at full fidelity ----
    // Photo → emissiveMap bypasses BRDF 1/π, shown at full brightness.
    // The photo already contains baked lighting from the camera.
    mat.emissiveMap = texture;
    mat.emissive.set(0xffffff);
    mat.emissiveIntensity = 1.0;

    // Kill ALL other shading channels — photo is the sole visual
    mat.color.set(0x000000);          // No Lambertian diffuse
    mat.roughness = 1.0;              // Fully rough (no specular lobe)
    mat.metalness = 0.0;
    mat.sheen = 0.0;
    mat.clearcoat = 0.0;
    mat.transmission = 0;
    mat.thickness = 0;

    // ---- KILL SPECULAR COMPLETELY ----
    // Root cause of "plastic layer": even envMapIntensity=0.02 combined with
    // HDR peak brightness (studio lights at 50-100x) + Fresnel at grazing angles
    // creates a visible glossy film over the entire face.
    // High roughness (0.88) just SPREADS the specular into a uniform sheen
    // rather than eliminating it — making it worse (plastic wrap effect).
    // Fix: zero out ALL specular pathways.
    mat.envMapIntensity = 0;           // No HDRI reflections at all
    mat.specularIntensity = 0;         // Kill Fresnel/F0 BRDF lobe
    mat.reflectivity = 0;              // Belt-and-suspenders: zero reflectivity

    mat.toneMapped = false;            // Skip tone mapping entirely
    mat.side = THREE.FrontSide;

    mat.needsUpdate = true;

    // All direct lights OFF
    this._setPhotoLighting();

    // No tone mapping for hybrid — photo values pass through as-is
    if (this.renderer) {
      this.renderer.toneMapping = THREE.NoToneMapping;
      this.renderer.toneMappingExposure = 1.0;
    }

    console.log('FaceRenderer: Hybrid photo mode (emissive-only, zero specular, NoToneMapping)');
  }

  /**
   * Switch photo render mode and re-apply stored texture in new mode.
   * Enables instant A/B/C comparison without re-generating the texture.
   * @param {'hybrid'|'pbr'|'emissive'} mode
   */
  setPhotoRenderMode(mode) {
    if (!['hybrid', 'pbr', 'emissive'].includes(mode)) {
      console.warn(`FaceRenderer: Unknown photo render mode "${mode}"`);
      return;
    }
    this._photoRenderMode = mode;

    if (this._photoTexture) {
      switch (mode) {
        case 'hybrid':  this._setHybridPhotoMode(this._photoTexture); break;
        case 'pbr':     this._setPbrPhotoMode(this._photoTexture); break;
        case 'emissive': this._setEmissivePhotoMode(this._photoTexture); break;
      }
      console.log(`FaceRenderer: Switched to "${mode}" photo render mode`);
    } else {
      console.log(`FaceRenderer: Mode set to "${mode}" (will apply on next texture load)`);
    }
  }

  /** @returns {'hybrid'|'pbr'|'emissive'} */
  getPhotoRenderMode() { return this._photoRenderMode; }

  /**
   * Restore material to default PBR mode (for FLAME without photo).
   */
  _restoreDefaultMode() {
    const mat = this.faceMaterial;
    this._photoTexture = null;

    // Clear emissive
    if (mat.emissiveMap) {
      mat.emissiveMap.dispose();
      mat.emissiveMap = null;
    }
    mat.emissive.set(0x000000);
    mat.emissiveIntensity = 1.0;

    // Restore PBR defaults
    mat.map = null;
    mat.color.set(DEFAULT_SKIN_COLOR);
    mat.roughness = 0.55;
    mat.metalness = 0.0;
    mat.transmission = 0.05;
    mat.thickness = 0.8;
    mat.sheen = 0.25;
    mat.clearcoat = 0.05;

    mat.toneMapped = true;
    mat.envMapIntensity = 1.0;   // Restore HDRI for default PBR mesh
    mat.side = THREE.DoubleSide;

    mat.needsUpdate = true;

    // Reset tone mapping to default ACES
    if (this.renderer) {
      this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer.toneMappingExposure = 1.0;
    }

    console.log('FaceRenderer: Restored default PBR mode');
  }

  // -- PBR skin material --------------------------------------------------

  _initMaterial() {
    this.faceMaterial = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(DEFAULT_SKIN_COLOR),
      roughness: 0.55,
      metalness: 0.0,

      // Subsurface scattering approximation via transmission + thickness
      transmission: 0.05,
      thickness: 0.8,
      attenuationColor: new THREE.Color(0xd4836b),
      attenuationDistance: 0.4,

      // Sheen for skin micro-fiber look
      sheen: 0.25,
      sheenRoughness: 0.5,
      sheenColor: new THREE.Color(0xffddcc),

      // Clearcoat can mimic oily forehead / nose highlights
      clearcoat: 0.05,
      clearcoatRoughness: 0.4,

      // Vertex colors will be used for region highlighting
      vertexColors: false,

      side: THREE.DoubleSide,
    });
  }

  // -----------------------------------------------------------------------
  // Mesh Loading
  // -----------------------------------------------------------------------

  /**
   * Load a BufferGeometry directly (e.g. from FLAME mesh generator).
   * @param {THREE.BufferGeometry} geometry
   */
  loadFromGeometry(geometry) {
    this._removeFaceMesh();

    geometry.computeVertexNormals();
    geometry.computeBoundingBox();

    // Prepare vertex color attribute (white = no highlight)
    this._ensureVertexColors(geometry);

    this.faceMesh = new THREE.Mesh(geometry, this.faceMaterial);
    this.faceMesh.castShadow = true;
    this.faceMesh.receiveShadow = true;
    this.scene.add(this.faceMesh);

    this._storeOriginalPositions(geometry);
    this._fitCameraToMesh(geometry);
  }

  /**
   * Load a face mesh from an OBJ file URL.
   * @param {string} url
   * @returns {Promise<THREE.Mesh>}
   */
  async loadFromOBJ(url) {
    const loader = new OBJLoader();
    const group = await loader.loadAsync(url);
    let geometry = null;
    group.traverse((child) => {
      if (child.isMesh && !geometry) {
        geometry = child.geometry;
      }
    });
    if (!geometry) throw new Error('No mesh found in OBJ file.');
    this.loadFromGeometry(geometry);
    return this.faceMesh;
  }

  /**
   * Load a face mesh from a GLB/glTF file URL.
   * @param {string} url
   * @returns {Promise<THREE.Mesh>}
   */
  async loadFromGLB(url) {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url);
    let mesh = null;
    gltf.scene.traverse((child) => {
      if (child.isMesh && !mesh) {
        mesh = child;
      }
    });
    if (!mesh) throw new Error('No mesh found in GLB file.');

    // Re-use any embedded material textures but switch to our PBR material
    if (mesh.material.map) this.faceMaterial.map = mesh.material.map;
    if (mesh.material.normalMap) this.faceMaterial.normalMap = mesh.material.normalMap;
    if (mesh.material.roughnessMap) this.faceMaterial.roughnessMap = mesh.material.roughnessMap;
    this.faceMaterial.needsUpdate = true;

    this.loadFromGeometry(mesh.geometry);

    // If the GLB has morph targets, forward them
    if (mesh.geometry.morphAttributes && Object.keys(mesh.geometry.morphAttributes).length > 0) {
      this.faceMesh.morphTargetInfluences = mesh.morphTargetInfluences
        ? [...mesh.morphTargetInfluences]
        : new Array(Object.keys(mesh.geometry.morphAttributes).length).fill(0);
      this.faceMesh.morphTargetDictionary = mesh.morphTargetDictionary
        ? { ...mesh.morphTargetDictionary }
        : undefined;
    }

    return this.faceMesh;
  }

  // -----------------------------------------------------------------------
  // Texture Loading
  // -----------------------------------------------------------------------

  /**
   * Load a texture and assign it to the face material.
   * @param {string} url - Texture image URL.
   * @param {'albedo'|'normal'|'roughness'} type
   * @param {{ flipY?: boolean }} [options]
   * @returns {Promise<THREE.Texture>}
   */
  async loadTexture(url, type = 'albedo', options = {}) {
    const loader = new THREE.TextureLoader();
    const texture = await loader.loadAsync(url);
    texture.colorSpace = type === 'albedo' ? THREE.SRGBColorSpace : THREE.LinearSRGBColorSpace;
    texture.flipY = options.flipY !== undefined ? options.flipY : true;
    texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

    switch (type) {
      case 'albedo':
        if (this._photoRenderMode === 'hybrid') {
          this._setHybridPhotoMode(texture);
        } else if (this._photoRenderMode === 'emissive') {
          this._setEmissivePhotoMode(texture);
        } else {
          this._setPbrPhotoMode(texture);
        }
        break;
      case 'normal':
        this.faceMaterial.normalMap = texture;
        this.faceMaterial.normalScale = new THREE.Vector2(1.0, 1.0);
        break;
      case 'roughness':
        this.faceMaterial.roughnessMap = texture;
        break;
      default:
        console.warn(`FaceRenderer.loadTexture: unknown type "${type}"`);
    }

    this.faceMaterial.needsUpdate = true;
    return texture;
  }

  /**
   * Apply a diagnostic overlay texture onto the 3D head (replaces emissive/albedo temporarily).
   * Call restoreDiagOverlay() to switch back to the real photo texture.
   * @param {string} dataUrl - PNG data URL of the diagnostic overlay
   */
  async applyDiagOverlay(dataUrl) {
    if (!this._diagBackup && this._photoTexture) {
      this._diagBackup = {
        emissiveMap: this.faceMaterial.emissiveMap,
        emissive: this.faceMaterial.emissive.clone(),
        emissiveIntensity: this.faceMaterial.emissiveIntensity,
        color: this.faceMaterial.color.clone(),
        envMapIntensity: this.faceMaterial.envMapIntensity,
      };
    }
    const loader = new THREE.TextureLoader();
    const tex = await loader.loadAsync(dataUrl);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.flipY = false;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

    this.faceMaterial.emissiveMap = tex;
    this.faceMaterial.emissive.set(0xffffff);
    this.faceMaterial.emissiveIntensity = 1.0;
    this.faceMaterial.color.set(0x000000);
    this.faceMaterial.envMapIntensity = 0;
    this.faceMaterial.needsUpdate = true;
    console.log('FaceRenderer: Diagnostic overlay applied');
  }

  /**
   * Restore the original photo texture after a diagnostic overlay was shown.
   */
  restoreDiagOverlay() {
    if (!this._diagBackup) return;
    const mat = this.faceMaterial;
    mat.emissiveMap = this._diagBackup.emissiveMap;
    mat.emissive.copy(this._diagBackup.emissive);
    mat.emissiveIntensity = this._diagBackup.emissiveIntensity;
    mat.color.copy(this._diagBackup.color);
    mat.envMapIntensity = this._diagBackup.envMapIntensity;
    mat.needsUpdate = true;
    this._diagBackup = null;
    console.log('FaceRenderer: Restored photo texture from diagnostic overlay');
  }

  /**
   * Generate and apply a procedural skin-like albedo texture via Canvas2D.
   * Useful as a quick demo / placeholder.
   */
  applyDemoTexture() {
    // If we have a real albedo texture, don't overwrite it with the demo texture
    if (this._hasRealAlbedo) return;

    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Base skin gradient
    const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    grad.addColorStop(0, '#f2c4a8');
    grad.addColorStop(0.5, '#e8b89d');
    grad.addColorStop(1.0, '#c9906e');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);

    // Subtle pore-like noise
    const imageData = ctx.getImageData(0, 0, size, size);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 12;
      data[i] = Math.min(255, Math.max(0, data[i] + noise));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
    }
    ctx.putImageData(imageData, 0, 0);

    // Small freckle dots
    for (let k = 0; k < 60; k++) {
      const x = Math.random() * size;
      const y = Math.random() * size;
      const r = 1 + Math.random() * 2;
      ctx.fillStyle = `rgba(160, 100, 70, ${0.05 + Math.random() * 0.08})`;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    this.faceMaterial.map = texture;
    this.faceMaterial.needsUpdate = true;
    return texture;
  }

  /**
   * Apply a real FLAME albedo texture from raw Uint8Array RGB data.
   *
   * @param {Uint8Array} diffuseData - Raw RGB bytes (row-major, already vertically flipped).
   *   Expected length: width * height * 3.
   * @param {Uint8Array|null} [specularData=null] - Raw RGB bytes for specular map, or null.
   * @param {Object|null} [meta=null] - Optional metadata object.
   * @param {number[]} [meta.albedo_resolution] - [width, height], defaults to [512, 512].
   * @param {number} [meta.specular_scale_factor] - Scale factor for specular, defaults to 0.26.
   */
  applyAlbedoTexture(diffuseData, specularData = null, meta = null) {
    if (!this.faceMesh || !diffuseData) return;

    const width = meta?.albedo_resolution?.[0] ?? 512;
    const height = meta?.albedo_resolution?.[1] ?? 512;

    // Three.js >= r152 removed THREE.RGBFormat; convert RGB → RGBA
    const rgbaData = this._rgbToRgba(diffuseData, width, height);

    // Create diffuse texture from raw RGBA uint8 data
    const diffuseTex = new THREE.DataTexture(
      rgbaData,
      width,
      height,
      THREE.RGBAFormat,
    );
    diffuseTex.needsUpdate = true;
    diffuseTex.colorSpace = THREE.SRGBColorSpace;
    diffuseTex.wrapS = THREE.ClampToEdgeWrapping;
    diffuseTex.wrapT = THREE.ClampToEdgeWrapping;
    diffuseTex.minFilter = THREE.LinearMipmapLinearFilter;
    diffuseTex.magFilter = THREE.LinearFilter;
    diffuseTex.generateMipmaps = true;

    // Apply to material
    const material = this.faceMaterial;
    if (material.map) material.map.dispose();
    material.map = diffuseTex;
    material.needsUpdate = true;

    // Apply specular texture if available
    if (specularData) {
      const specRgba = this._rgbToRgba(specularData, width, height);
      const specTex = new THREE.DataTexture(
        specRgba,
        width,
        height,
        THREE.RGBAFormat,
      );
      specTex.needsUpdate = true;
      specTex.wrapS = THREE.ClampToEdgeWrapping;
      specTex.wrapT = THREE.ClampToEdgeWrapping;
      specTex.minFilter = THREE.LinearMipmapLinearFilter;
      specTex.magFilter = THREE.LinearFilter;
      specTex.generateMipmaps = true;

      // Use specular map for roughness/metalness
      if (material.roughnessMap) material.roughnessMap.dispose();
      material.roughnessMap = specTex;
      material.roughness = 0.7;
      material.needsUpdate = true;
    }

    // Adjust material for realistic skin rendering with real texture
    material.color.set(0xffffff); // Don't tint the texture
    if (material.isMeshPhysicalMaterial) {
      material.clearcoat = 0.05;
      material.clearcoatRoughness = 0.4;
      material.sheen = 0.3;
      material.sheenColor.set(0xcc8866);
      material.sheenRoughness = 0.6;
    }

    this._hasRealAlbedo = true;
    console.log('FaceRenderer: Applied FLAME albedo texture (%dx%d)', width, height);
  }

  /**
   * Convert a raw RGB Uint8Array to RGBA (opaque alpha = 255).
   * @param {Uint8Array} rgb - Input RGB data.
   * @param {number} width
   * @param {number} height
   * @returns {Uint8Array} RGBA data.
   */
  _rgbToRgba(rgb, width, height) {
    const pixelCount = width * height;
    const rgba = new Uint8Array(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      rgba[i * 4]     = rgb[i * 3];
      rgba[i * 4 + 1] = rgb[i * 3 + 1];
      rgba[i * 4 + 2] = rgb[i * 3 + 2];
      rgba[i * 4 + 3] = 255;
    }
    return rgba;
  }

  // -----------------------------------------------------------------------
  // Camera Controls
  // -----------------------------------------------------------------------

  /**
   * Animate the camera to a preset view.
   * @param {'front'|'profile-left'|'profile-right'|'three-quarter-left'|'three-quarter-right'|'above'|'below'} name
   */
  setView(name) {
    const preset = VIEW_PRESETS[name];
    if (!preset) {
      console.warn(`FaceRenderer.setView: unknown preset "${name}"`);
      return;
    }
    this._sphericalTarget = {
      theta: preset.theta,
      phi: preset.phi,
      radius: preset.radius * (this._spherical.radius / 3), // scale to current zoom
    };
  }

  /** Reset the camera to front view (also triggered by double-click). */
  resetView() {
    this.setView('front');
  }

  // -----------------------------------------------------------------------
  // Sub-Region Highlighting
  // -----------------------------------------------------------------------

  /**
   * Highlight a set of vertices by painting their vertex color.
   * @param {number[]} vertexIndices - Array of vertex indices to highlight.
   * @param {THREE.Color|number|string} color - Highlight color.
   */
  highlightRegion(vertexIndices, color = 0x44aaff) {
    if (!this.faceMesh) return;
    const geo = this.faceMesh.geometry;
    this._ensureVertexColors(geo);

    // Enable vertex colors on material if not already
    if (!this.faceMaterial.vertexColors) {
      this.faceMaterial.vertexColors = true;
      this.faceMaterial.needsUpdate = true;
    }

    const colAttr = geo.attributes.color;
    const c = new THREE.Color(color);

    for (const idx of vertexIndices) {
      if (idx >= 0 && idx < colAttr.count) {
        colAttr.setXYZ(idx, c.r, c.g, c.b);
      }
    }
    colAttr.needsUpdate = true;
  }

  /**
   * Reset all vertex colors to white (neutral).
   */
  clearHighlight() {
    if (!this.faceMesh) return;
    const geo = this.faceMesh.geometry;
    const colAttr = geo.attributes.color;
    if (!colAttr) return;

    for (let i = 0; i < colAttr.count; i++) {
      colAttr.setXYZ(i, 1, 1, 1);
    }
    colAttr.needsUpdate = true;

    // Disable vertex color blending so skin color is clean
    this.faceMaterial.vertexColors = false;
    this.faceMaterial.needsUpdate = true;
  }

  // -----------------------------------------------------------------------
  // Morph / Deformation
  // -----------------------------------------------------------------------

  /**
   * Move specific vertices by a displacement vector.
   * @param {number[]} vertexIndices
   * @param {THREE.Vector3|{x:number,y:number,z:number}} displacement
   */
  deformVertices(vertexIndices, displacement) {
    if (!this.faceMesh) return;
    const pos = this.faceMesh.geometry.attributes.position;
    const dx = displacement.x || 0;
    const dy = displacement.y || 0;
    const dz = displacement.z || 0;

    for (const idx of vertexIndices) {
      if (idx >= 0 && idx < pos.count) {
        pos.setXYZ(
          idx,
          pos.getX(idx) + dx,
          pos.getY(idx) + dy,
          pos.getZ(idx) + dz,
        );
      }
    }
    pos.needsUpdate = true;
    this.faceMesh.geometry.computeVertexNormals();
  }

  /**
   * Apply parameterized region deformation.
   * @param {number[]} vertexIndices
   * @param {{ inflate?: number, translate?: {x:number,y:number,z:number}, smooth?: number }} params
   */
  deformRegion(vertexIndices, params = {}) {
    if (!this.faceMesh) return;
    const geo = this.faceMesh.geometry;
    const pos = geo.attributes.position;
    const norm = geo.attributes.normal;

    // Compute mesh-proportional scale: inflate values (-0.5..+0.5) should
    // produce displacements proportional to the mesh's bounding radius,
    // not absolute world-space units.
    const meshRadius = this._meshBoundingRadius || 0.085;
    const INFLATE_SCALE = meshRadius * 0.15; // max slider → ~7.5% of radius

    // --- inflate: push vertices along their normals ---
    if (params.inflate && norm) {
      const amount = params.inflate * INFLATE_SCALE;
      for (const idx of vertexIndices) {
        if (idx < 0 || idx >= pos.count) continue;
        const nx = norm.getX(idx);
        const ny = norm.getY(idx);
        const nz = norm.getZ(idx);
        pos.setXYZ(
          idx,
          pos.getX(idx) + nx * amount,
          pos.getY(idx) + ny * amount,
          pos.getZ(idx) + nz * amount,
        );
      }
    }

    // --- translate ---
    if (params.translate) {
      const tx = (params.translate.x || 0) * INFLATE_SCALE;
      const ty = (params.translate.y || 0) * INFLATE_SCALE;
      const tz = (params.translate.z || 0) * INFLATE_SCALE;
      for (const idx of vertexIndices) {
        if (idx < 0 || idx >= pos.count) continue;
        pos.setXYZ(
          idx,
          pos.getX(idx) + tx,
          pos.getY(idx) + ty,
          pos.getZ(idx) + tz,
        );
      }
    }

    // --- smooth (Laplacian-like averaging) ---
    if (params.smooth && params.smooth > 0) {
      this._smoothVertices(vertexIndices, params.smooth);
    }

    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }

  /**
   * Apply a complete morph state to the mesh using a region map.
   * @param {Object<string, { inflate?: number, translate?: {x:number,y:number,z:number} }>} morphState
   *   Map of regionName -> deformation params.
   * @param {Object<string, number[]>} regionMap
   *   Map of regionName -> array of vertex indices.
   */
  applyMorphState(morphState, regionMap) {
    if (!this.faceMesh || !morphState || !regionMap) return;

    // Reset to original first
    this.resetDeformation();

    for (const [regionName, params] of Object.entries(morphState)) {
      const indices = regionMap[regionName];
      if (!indices || indices.length === 0) continue;
      this.deformRegion(indices, params);
    }
  }

  /**
   * Reset all vertex positions to the original stored state.
   */
  resetDeformation() {
    if (!this.faceMesh || !this.originalPositions) return;
    const pos = this.faceMesh.geometry.attributes.position;
    pos.array.set(this.originalPositions);
    pos.needsUpdate = true;
    this.faceMesh.geometry.computeVertexNormals();
  }

  // -----------------------------------------------------------------------
  // Comparison Mode
  // -----------------------------------------------------------------------

  /**
   * Toggle comparison overlay showing the original (undeformed) mesh.
   * @param {boolean} enabled
   */
  setComparisonMode(enabled) {
    this._comparisonEnabled = enabled;

    if (enabled && !this._ghostMesh && this.faceMesh && this.originalPositions) {
      // Create a translucent copy of the original mesh
      const ghostGeo = this.faceMesh.geometry.clone();
      ghostGeo.attributes.position.array.set(this.originalPositions);
      ghostGeo.attributes.position.needsUpdate = true;
      ghostGeo.computeVertexNormals();

      const ghostMat = new THREE.MeshPhysicalMaterial({
        color: 0x88bbff,
        roughness: 0.6,
        metalness: 0.0,
        transparent: true,
        opacity: 0.35,
        depthWrite: false,
        side: THREE.DoubleSide,
        clippingPlanes: [this._clipPlane],
        clipShadows: true,
      });

      this._ghostMesh = new THREE.Mesh(ghostGeo, ghostMat);
      this._ghostMesh.renderOrder = 1;
      this.scene.add(this._ghostMesh);

      // Enable clipping on the renderer
      this.renderer.localClippingEnabled = true;
    }

    if (!enabled && this._ghostMesh) {
      this.scene.remove(this._ghostMesh);
      this._ghostMesh.geometry.dispose();
      this._ghostMesh.material.dispose();
      this._ghostMesh = null;
    }
  }

  /**
   * Adjust the comparison clip plane position (0 = fully original, 1 = fully modified).
   * @param {number} value - 0..1
   */
  setComparisonSlider(value) {
    this._comparisonSlider = THREE.MathUtils.clamp(value, 0, 1);

    if (this.faceMesh) {
      const box = this.faceMesh.geometry.boundingBox;
      if (box) {
        const xRange = box.max.x - box.min.x;
        const clipX = box.min.x + xRange * this._comparisonSlider;
        this._clipPlane.set(new THREE.Vector3(-1, 0, 0), clipX);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Expression Animation
  // -----------------------------------------------------------------------

  /**
   * Animate expression blend shapes on the mesh.
   * @param {Object<string, number>} blendshapeData - Map of blendshape names to target weights (0..1).
   * @param {number} [duration=0.5] - Animation duration in seconds.
   */
  playExpression(blendshapeData, duration = 0.5) {
    if (!this.faceMesh) return;

    const mesh = this.faceMesh;
    if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) {
      console.warn('FaceRenderer.playExpression: mesh has no morph targets.');
      return;
    }

    // If there is an existing mixer, stop it
    if (this._expressionMixer) {
      this._expressionMixer.stopAllAction();
    }

    this._expressionMixer = new THREE.AnimationMixer(mesh);

    const times = [0, duration];
    const tracks = [];

    for (const [name, targetWeight] of Object.entries(blendshapeData)) {
      const index = mesh.morphTargetDictionary[name];
      if (index === undefined) {
        console.warn(`FaceRenderer.playExpression: blendshape "${name}" not found.`);
        continue;
      }
      const currentWeight = mesh.morphTargetInfluences[index] || 0;
      tracks.push(
        new THREE.NumberKeyframeTrack(
          `.morphTargetInfluences[${index}]`,
          times,
          [currentWeight, targetWeight],
        ),
      );
    }

    if (tracks.length === 0) return;

    const clip = new THREE.AnimationClip('expression', duration, tracks);
    const action = this._expressionMixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce);
    action.clampWhenFinished = true;
    action.play();

    this._expressionActions.push(action);
  }

  // -----------------------------------------------------------------------
  // Screenshot
  // -----------------------------------------------------------------------

  /**
   * Capture the current view as a PNG data URL.
   * @param {number} [width] - Override width (defaults to current size).
   * @param {number} [height] - Override height (defaults to current size).
   * @returns {string} Data URL (image/png).
   */
  captureScreenshot(width, height) {
    // Render one frame to make sure the buffer is current
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  // -----------------------------------------------------------------------
  // Diagnostic Tools (Expert Calibration)
  // -----------------------------------------------------------------------

  /**
   * A/B Quad Test: Place a flat plane in the scene with the current face
   * texture applied via MeshBasicMaterial (no lighting, no PBR).
   * If the quad looks correct but the head looks pale → lighting/material issue.
   * If the quad also looks pale → texture pipeline issue.
   */
  showQuadTest() {
    // Remove existing quad if any
    this.hideQuadTest();

    if (!this.faceMaterial || !this.faceMaterial.map) {
      console.warn('FaceRenderer: No texture loaded — cannot show quad test');
      return;
    }

    const texture = this.faceMaterial.map;

    // Create a flat plane with MeshBasicMaterial (unlit, no PBR, no tone mapping)
    const quadGeo = new THREE.PlaneGeometry(0.12, 0.12);
    const quadMat = new THREE.MeshBasicMaterial({
      map: texture,
      toneMapped: false, // raw pixel passthrough — no ACES darkening
      side: THREE.DoubleSide,
    });
    this._quadTestMesh = new THREE.Mesh(quadGeo, quadMat);

    // Position to the right of the face
    const offset = this._orbitTarget.clone();
    offset.x += 0.15;
    offset.y += 0.05;
    this._quadTestMesh.position.copy(offset);
    this._quadTestMesh.renderOrder = 10;

    this.scene.add(this._quadTestMesh);
    console.log('FaceRenderer: Quad test shown (MeshBasicMaterial, toneMapped=false)');
  }

  /** Remove the A/B quad test plane. */
  hideQuadTest() {
    if (this._quadTestMesh) {
      this.scene.remove(this._quadTestMesh);
      this._quadTestMesh.geometry.dispose();
      this._quadTestMesh.material.dispose();
      this._quadTestMesh = null;
    }
  }

  /**
   * 0.18 Gray Sphere: Spawn a middle-gray sphere (sRGB 0x808080 ≈ 18% reflectance)
   * with roughness=1.0 for lighting calibration.
   * Adjust envMapIntensity and toneMappingExposure until this looks correct.
   * Then lock the values — do not adjust for the face.
   */
  showGraySphere() {
    this.hideGraySphere();

    const sphereGeo = new THREE.SphereGeometry(0.04, 32, 32);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: 0x808080,       // middle gray (sRGB)
      roughness: 1.0,        // fully matte — pure Lambertian
      metalness: 0.0,
      envMapIntensity: this.faceMaterial?.envMapIntensity ?? 1.0,
    });
    this._graySphere = new THREE.Mesh(sphereGeo, sphereMat);

    // Position to the left of the face
    const offset = this._orbitTarget.clone();
    offset.x -= 0.15;
    offset.y += 0.05;
    this._graySphere.position.copy(offset);

    this.scene.add(this._graySphere);
    console.log('FaceRenderer: Gray sphere shown (0x808080, roughness=1.0)');
  }

  /** Remove the gray sphere. */
  hideGraySphere() {
    if (this._graySphere) {
      this.scene.remove(this._graySphere);
      this._graySphere.geometry.dispose();
      this._graySphere.material.dispose();
      this._graySphere = null;
    }
  }

  // -----------------------------------------------------------------------
  // Render Loop
  // -----------------------------------------------------------------------

  _startLoop() {
    const loop = () => {
      this._rafId = requestAnimationFrame(loop);
      this._update();

      // Use SSS composer if enabled, otherwise standard render
      if (this._sssEffect && this._sssEffect.enabled) {
        this._sssEffect.render();
      } else {
        this.renderer.render(this.scene, this.camera);
      }
    };
    loop();
  }

  _update() {
    const dt = this._clock.getDelta();

    // Smooth spherical interpolation for camera orbit
    this._spherical.theta += (this._sphericalTarget.theta - this._spherical.theta) * LERP_SPEED;
    this._spherical.phi += (this._sphericalTarget.phi - this._spherical.phi) * LERP_SPEED;
    this._spherical.radius += (this._sphericalTarget.radius - this._spherical.radius) * LERP_SPEED;

    this._applyCameraSpherical(this._spherical);

    // Expression animation mixer
    if (this._expressionMixer) {
      this._expressionMixer.update(dt);
    }

    // Morph animation (treatment preview lerp)
    if (this._morphAnimation) {
      this._updateMorphAnimation();
    }

    // Keep background plane facing the camera
    if (this._bgPlane) {
      this._bgPlane.position.copy(this._orbitTarget);
      this._bgPlane.position.z -= 20;
      this._bgPlane.lookAt(this.camera.position);
    }
  }

  _applyCameraSpherical(s) {
    const x = s.radius * Math.sin(s.phi) * Math.sin(s.theta);
    const y = s.radius * Math.cos(s.phi);
    const z = s.radius * Math.sin(s.phi) * Math.cos(s.theta);

    this.camera.position.set(
      this._orbitTarget.x + x,
      this._orbitTarget.y + y,
      this._orbitTarget.z + z,
    );
    this.camera.lookAt(this._orbitTarget);
  }

  // -----------------------------------------------------------------------
  // Input Handling
  // -----------------------------------------------------------------------

  _bindEvents() {
    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', this._boundOnPointerDown);
    el.addEventListener('pointermove', this._boundOnPointerMove);
    el.addEventListener('pointerup', this._boundOnPointerUp);
    el.addEventListener('pointerleave', this._boundOnPointerUp);
    el.addEventListener('wheel', this._boundOnWheel, { passive: false });
    el.addEventListener('dblclick', this._boundOnDblClick);

    // Touch
    el.addEventListener('touchstart', this._boundOnTouchStart, { passive: false });
    el.addEventListener('touchmove', this._boundOnTouchMove, { passive: false });
    el.addEventListener('touchend', this._boundOnTouchEnd);

    window.addEventListener('resize', this._boundOnResize);
  }

  _unbindEvents() {
    const el = this.renderer.domElement;
    el.removeEventListener('pointerdown', this._boundOnPointerDown);
    el.removeEventListener('pointermove', this._boundOnPointerMove);
    el.removeEventListener('pointerup', this._boundOnPointerUp);
    el.removeEventListener('pointerleave', this._boundOnPointerUp);
    el.removeEventListener('wheel', this._boundOnWheel);
    el.removeEventListener('dblclick', this._boundOnDblClick);
    el.removeEventListener('touchstart', this._boundOnTouchStart);
    el.removeEventListener('touchmove', this._boundOnTouchMove);
    el.removeEventListener('touchend', this._boundOnTouchEnd);
    window.removeEventListener('resize', this._boundOnResize);
  }

  _onPointerDown(e) {
    this._isDragging = true;
    this._previousMouse.x = e.clientX;
    this._previousMouse.y = e.clientY;
  }

  _onPointerMove(e) {
    if (!this._isDragging) return;
    const dx = e.clientX - this._previousMouse.x;
    const dy = e.clientY - this._previousMouse.y;
    this._previousMouse.x = e.clientX;
    this._previousMouse.y = e.clientY;

    const rotateSpeed = 0.005;
    this._sphericalTarget.theta -= dx * rotateSpeed;
    this._sphericalTarget.phi += dy * rotateSpeed;

    // Clamp phi to avoid flipping
    this._sphericalTarget.phi = THREE.MathUtils.clamp(
      this._sphericalTarget.phi,
      0.1,
      Math.PI - 0.1,
    );
  }

  _onPointerUp() {
    this._isDragging = false;
  }

  _onWheel(e) {
    e.preventDefault();
    const zoomSpeed = 0.001;
    this._sphericalTarget.radius *= 1 + e.deltaY * zoomSpeed;
    this._sphericalTarget.radius = THREE.MathUtils.clamp(
      this._sphericalTarget.radius,
      0.5,
      20,
    );
  }

  _onDblClick() {
    this.resetView();
  }

  // -- Touch pinch-zoom ---------------------------------------------------

  _onTouchStart(e) {
    if (e.touches.length === 1) {
      this._isDragging = true;
      this._previousMouse.x = e.touches[0].clientX;
      this._previousMouse.y = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      this._touchStartDist = Math.sqrt(dx * dx + dy * dy);
    }
  }

  _onTouchMove(e) {
    if (e.touches.length === 1 && this._isDragging) {
      const dx = e.touches[0].clientX - this._previousMouse.x;
      const dy = e.touches[0].clientY - this._previousMouse.y;
      this._previousMouse.x = e.touches[0].clientX;
      this._previousMouse.y = e.touches[0].clientY;

      const rotateSpeed = 0.005;
      this._sphericalTarget.theta -= dx * rotateSpeed;
      this._sphericalTarget.phi += dy * rotateSpeed;
      this._sphericalTarget.phi = THREE.MathUtils.clamp(
        this._sphericalTarget.phi,
        0.1,
        Math.PI - 0.1,
      );
    } else if (e.touches.length === 2) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const scale = this._touchStartDist / dist;
      this._sphericalTarget.radius *= scale;
      this._sphericalTarget.radius = THREE.MathUtils.clamp(
        this._sphericalTarget.radius,
        0.5,
        20,
      );
      this._touchStartDist = dist;
    }
  }

  _onTouchEnd() {
    this._isDragging = false;
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);

    // Resize SSS render targets if active
    if (this._sssEffect) {
      this._sssEffect.resize(w * this.renderer.getPixelRatio(), h * this.renderer.getPixelRatio());
    }
  }

  // -----------------------------------------------------------------------
  // Internal Helpers
  // -----------------------------------------------------------------------

  /**
   * Ensure the geometry has a `color` attribute for vertex coloring.
   */
  _ensureVertexColors(geometry) {
    if (geometry.attributes.color) return;
    const count = geometry.attributes.position.count;
    const colors = new Float32Array(count * 3);
    colors.fill(1); // white = neutral
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }

  /**
   * Store a clone of the original vertex positions for reset / comparison.
   */
  _storeOriginalPositions(geometry) {
    const pos = geometry.attributes.position;
    this.originalPositions = new Float32Array(pos.array.length);
    this.originalPositions.set(pos.array);
  }

  /**
   * Adjust orbit target and camera radius to fit the loaded mesh.
   */
  _fitCameraToMesh(geometry) {
    const { center, radius } = computeFitParams(geometry);
    this._orbitTarget.copy(center);
    this._spherical.radius = radius;
    this._sphericalTarget.radius = radius;
    this._sphericalTarget.theta = 0;
    this._sphericalTarget.phi = Math.PI / 2;
    this._applyCameraSpherical(this._spherical);

    // Store mesh bounding radius for proportional deformation scaling
    geometry.computeBoundingSphere();
    this._meshBoundingRadius = geometry.boundingSphere
      ? geometry.boundingSphere.radius
      : 0.085;
  }

  /**
   * Remove the current face mesh from the scene and dispose resources.
   */
  _removeFaceMesh() {
    if (this.faceMesh) {
      this.scene.remove(this.faceMesh);
      this.faceMesh.geometry.dispose();
      this.faceMesh = null;
    }
    if (this._ghostMesh) {
      this.scene.remove(this._ghostMesh);
      this._ghostMesh.geometry.dispose();
      this._ghostMesh.material.dispose();
      this._ghostMesh = null;
    }
    this.originalPositions = null;
  }

  /**
   * Simple Laplacian smoothing over the specified vertices.
   * @param {number[]} vertexIndices
   * @param {number} iterations
   */
  _smoothVertices(vertexIndices, iterations = 1) {
    const geo = this.faceMesh.geometry;
    const pos = geo.attributes.position;
    const index = geo.index;
    const indexSet = new Set(vertexIndices);

    // Build adjacency from the index buffer
    const adjacency = new Map();
    for (const idx of vertexIndices) {
      adjacency.set(idx, []);
    }

    if (index) {
      const arr = index.array;
      for (let i = 0; i < arr.length; i += 3) {
        const a = arr[i], b = arr[i + 1], c = arr[i + 2];
        const tri = [a, b, c];
        for (let j = 0; j < 3; j++) {
          const v = tri[j];
          if (indexSet.has(v)) {
            const neighbors = adjacency.get(v);
            for (let k = 0; k < 3; k++) {
              if (k !== j) neighbors.push(tri[k]);
            }
          }
        }
      }
    }

    for (let iter = 0; iter < iterations; iter++) {
      const newPos = new Float32Array(pos.array); // copy

      for (const idx of vertexIndices) {
        const neighbors = adjacency.get(idx);
        if (!neighbors || neighbors.length === 0) continue;

        let sx = 0, sy = 0, sz = 0;
        for (const n of neighbors) {
          sx += pos.getX(n);
          sy += pos.getY(n);
          sz += pos.getZ(n);
        }
        const count = neighbors.length;
        // Blend 50% toward neighbor average
        const ox = pos.getX(idx);
        const oy = pos.getY(idx);
        const oz = pos.getZ(idx);
        newPos[idx * 3] = ox * 0.5 + (sx / count) * 0.5;
        newPos[idx * 3 + 1] = oy * 0.5 + (sy / count) * 0.5;
        newPos[idx * 3 + 2] = oz * 0.5 + (sz / count) * 0.5;
      }

      pos.array.set(newPos);
    }

    pos.needsUpdate = true;
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  /**
   * Tear down the renderer, remove event listeners, dispose GPU resources.
   */
  destroy() {
    // Stop render loop
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    this._unbindEvents();

    // Clear environment reference (don't dispose — may be shared)
    if (this.scene) this.scene.environment = null;

    // Dispose face mesh
    this._removeFaceMesh();

    // Dispose material and its textures
    if (this.faceMaterial) {
      if (this.faceMaterial.map) this.faceMaterial.map.dispose();
      if (this.faceMaterial.normalMap) this.faceMaterial.normalMap.dispose();
      if (this.faceMaterial.roughnessMap) this.faceMaterial.roughnessMap.dispose();
      this.faceMaterial.dispose();
      this.faceMaterial = null;
    }

    // Dispose lights that have dispose methods
    for (const light of Object.values(this.lights)) {
      if (light.dispose) light.dispose();
    }
    this.lights = {};

    // Dispose background
    if (this._bgPlane) {
      this._bgPlane.geometry.dispose();
      this._bgPlane.material.dispose();
      this.scene.remove(this._bgPlane);
      this._bgPlane = null;
    }

    // Dispose scene children
    this.scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m) => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });

    // Dispose renderer
    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.domElement && this.renderer.domElement.parentElement) {
        this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
      }
      this.renderer = null;
    }

    this.scene = null;
    this.camera = null;
  }

  // -----------------------------------------------------------------------
  // SSS (Subsurface Scattering) Post-Processing
  // -----------------------------------------------------------------------

  /**
   * Enable screen-space subsurface scattering.
   * Dynamically imports SSSEffect on first call.
   *
   * @param {boolean} [enabled=true]
   * @param {object} [options]
   * @param {number} [options.sssWidth=0.012] - Scatter radius
   */
  async enableSSS(enabled = true, options = {}) {
    if (enabled && !this._sssEffect) {
      try {
        const { SSSEffect } = await import('./shaders/SSSPass.js');
        this._sssEffect = new SSSEffect(this.renderer, this.scene, this.camera, {
          sssWidth: options.sssWidth ?? 0.012,
          enabled: true,
        });
        console.log('FaceRenderer: SSS post-processing enabled');
      } catch (err) {
        console.warn('FaceRenderer: Failed to load SSS effect:', err);
        return;
      }
    }

    if (this._sssEffect) {
      this._sssEffect.setEnabled(enabled);
      if (options.sssWidth !== undefined) {
        this._sssEffect.setWidth(options.sssWidth);
      }
    }
  }

  /**
   * Disable SSS and dispose GPU resources.
   */
  disableSSS() {
    if (this._sssEffect) {
      this._sssEffect.dispose();
      this._sssEffect = null;
      console.log('FaceRenderer: SSS post-processing disabled');
    }
  }

  // -----------------------------------------------------------------------
  // Treatment Preview Animation (Morph Lerp)
  // -----------------------------------------------------------------------

  /**
   * Animate from current morph state to a target state over time.
   *
   * @param {Object} targetMorphState - Target morph state { region: { inflate, ... } }
   * @param {Object} regions - Region vertex index maps
   * @param {object} [options]
   * @param {number} [options.duration=500] - Animation duration in ms
   * @param {Function} [options.onComplete] - Called when animation finishes
   */
  animateMorphTo(targetMorphState, regions, options = {}) {
    const duration = options.duration ?? 500;
    const onComplete = options.onComplete ?? null;

    // Capture the current positions as "from" state
    if (!this.faceMesh) return;
    const posAttr = this.faceMesh.geometry.attributes.position;
    const fromPositions = new Float32Array(posAttr.array);

    // Apply target morph state to get "to" positions
    this.resetDeformation();
    this.applyMorphState(targetMorphState, regions);
    const toPositions = new Float32Array(posAttr.array);

    // Reset back to "from" positions to start animation
    posAttr.array.set(fromPositions);
    posAttr.needsUpdate = true;

    // Animation state
    const startTime = performance.now();
    this._morphAnimation = {
      from: fromPositions,
      to: toPositions,
      startTime,
      duration,
      onComplete,
    };
  }

  /**
   * Called from _update() to advance morph animation.
   */
  _updateMorphAnimation() {
    const anim = this._morphAnimation;
    if (!anim || !this.faceMesh) return;

    const elapsed = performance.now() - anim.startTime;
    let t = Math.min(1, elapsed / anim.duration);

    // Smooth easing: ease-in-out cubic
    t = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

    const posAttr = this.faceMesh.geometry.attributes.position;
    const arr = posAttr.array;
    const from = anim.from;
    const to = anim.to;

    for (let i = 0; i < arr.length; i++) {
      arr[i] = from[i] + (to[i] - from[i]) * t;
    }

    posAttr.needsUpdate = true;
    this.faceMesh.geometry.computeVertexNormals();

    if (t >= 1) {
      this._morphAnimation = null;
      anim.onComplete?.();
    }
  }
}

export default FaceRenderer;
