/**
 * MediaPipeBridge.js
 *
 * Bridges MediaPipe Face Mesh 478-landmark detection to the FLAME parametric
 * head model.  Provides real-time webcam face tracking, single-image detection,
 * landmark-to-FLAME parameter estimation, head-pose extraction, and semantic
 * facial-region grouping.
 *
 * MediaPipe is loaded dynamically from CDN so the module works inside a Vite
 * project without bundling the WASM/TFLite artefacts.  If loading fails the
 * module degrades gracefully.
 *
 * Usage:
 *   import { MediaPipeBridge } from './MediaPipeBridge.js';
 *
 *   const bridge = new MediaPipeBridge({
 *     onFaceDetected: (landmarks, params) => { ... },
 *   });
 *   await bridge.init();
 *   await bridge.startCamera(document.getElementById('webcam'));
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** CDN base for MediaPipe Vision tasks bundle (WASM + model). */
const VISION_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

/** CDN entry point for the MediaPipe Vision JS module. */
const VISION_MODULE_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';

/**
 * Semantic landmark definitions.
 * Maps human-readable names to MediaPipe Face Mesh landmark indices.
 *
 * Reference: https://github.com/google/mediapipe/blob/master/mediapipe/
 *            modules/face_geometry/data/canonical_face_model_uv_visualization.png
 */
const LANDMARK_DEFINITIONS = Object.freeze({
  // Nose
  nose_tip:              1,
  nose_bridge_top:       6,
  nose_bridge_mid:       197,
  nose_bottom:           2,
  nose_right_alar:       98,
  nose_left_alar:        327,

  // Right eye (subject right, image left)
  right_eye_inner:       133,
  right_eye_outer:       33,
  right_eye_upper:       159,
  right_eye_lower:       145,
  right_pupil:           468,  // iris centre (478-landmark model)

  // Left eye (subject left, image right)
  left_eye_inner:        362,
  left_eye_outer:        263,
  left_eye_upper:        386,
  left_eye_lower:        374,
  left_pupil:            473,  // iris centre

  // Right eyebrow
  right_brow_inner:      107,
  right_brow_mid:        66,
  right_brow_outer:      46,

  // Left eyebrow
  left_brow_inner:       336,
  left_brow_mid:         296,
  left_brow_outer:       276,

  // Lips
  upper_lip_top:         13,
  upper_lip_bottom:      14,
  lower_lip_top:         14,
  lower_lip_bottom:      17,
  lip_right_corner:      61,
  lip_left_corner:       291,
  upper_lip_right:       40,
  upper_lip_left:        270,
  lower_lip_right:       88,
  lower_lip_left:        318,

  // Jaw / chin
  chin_tip:              152,
  jaw_right:             172,
  jaw_left:              397,
  jaw_right_mid:         136,
  jaw_left_mid:          365,

  // Forehead
  forehead_center:       10,

  // Cheeks
  right_cheek:           123,
  left_cheek:            352,
});

/**
 * Indices used for head-pose estimation (PnP-style).
 * Chosen for stability and spread across the face.
 */
const POSE_LANDMARK_INDICES = [
  LANDMARK_DEFINITIONS.nose_tip,           // 1
  LANDMARK_DEFINITIONS.chin_tip,           // 152
  LANDMARK_DEFINITIONS.left_eye_outer,     // 263
  LANDMARK_DEFINITIONS.right_eye_outer,    // 33
  LANDMARK_DEFINITIONS.lip_left_corner,    // 291
  LANDMARK_DEFINITIONS.lip_right_corner,   // 61
];

/**
 * Approximate 3-D reference positions (in a canonical, unit-scale face) for
 * the landmarks listed in POSE_LANDMARK_INDICES.  These are rough but good
 * enough for a single-camera Euler-angle estimate.
 */
const POSE_MODEL_POINTS = [
  [0.0,    0.0,    0.0],     // nose tip (origin)
  [0.0,   -0.33,  -0.065],   // chin
  [-0.225,  0.17,  -0.135],   // left eye outer
  [0.225,   0.17,  -0.135],   // right eye outer
  [-0.15,  -0.15,  -0.125],   // left lip corner
  [0.15,   -0.15,  -0.125],   // right lip corner
];

/**
 * Facial region groupings.  Each key maps to an array of MediaPipe landmark
 * indices that belong to that region.  Using the canonical Face Mesh topology.
 */
const FACE_REGIONS = Object.freeze({
  right_eye: [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  ],
  left_eye: [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
  ],
  right_eyebrow: [
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
  ],
  left_eyebrow: [
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
  ],
  nose: [
    1, 2, 3, 4, 5, 6, 197, 195, 5, 4, 45, 220, 115, 48, 64, 98,
    60, 75, 59, 166, 219, 218, 237, 44, 1, 274, 275, 440, 344, 278,
    294, 327, 290, 305, 289, 392, 439, 438, 457,
  ],
  upper_lip: [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
    314, 17, 84, 181, 91, 146,
  ],
  lower_lip: [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95,
  ],
  jaw: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  ],
  right_cheek: [
    123, 50, 36, 137, 205, 206, 187, 147, 213, 192, 138, 135, 169, 170, 140,
  ],
  left_cheek: [
    352, 280, 266, 366, 425, 426, 411, 376, 433, 416, 367, 364, 394, 395, 369,
  ],
  forehead: [
    10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 151, 108, 69, 104,
    68, 71, 139, 70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 368,
  ],
});

/**
 * Exponential moving average smoothing factor.
 * Higher = more responsive but jitterier; lower = smoother but laggier.
 */
const DEFAULT_SMOOTH_FACTOR = 0.6;

/** Number of FLAME shape PCA components to estimate. */
const NUM_SHAPE_PARAMS = 10;
/** Number of FLAME expression PCA components to estimate. */
const NUM_EXPRESSION_PARAMS = 10;

// ---------------------------------------------------------------------------
// Helper utilities (exported for external use)
// ---------------------------------------------------------------------------

/**
 * Normalize a set of 478 MediaPipe landmarks so the face is centred at the
 * origin and scaled to a canonical inter-pupillary distance of 1.0.
 *
 * @param {Array<{x:number, y:number, z:number}>} landmarks - Raw MP landmarks
 *   (normalised 0-1 image coordinates).
 * @returns {Array<{x:number, y:number, z:number}>} Centred, scaled landmarks.
 */
export function normalizeAndCenter(landmarks) {
  if (!landmarks || landmarks.length === 0) return landmarks;

  // Compute centre from all landmarks
  let cx = 0, cy = 0, cz = 0;
  for (const lm of landmarks) {
    cx += lm.x;
    cy += lm.y;
    cz += lm.z;
  }
  const n = landmarks.length;
  cx /= n;
  cy /= n;
  cz /= n;

  // Compute inter-pupillary distance for scale
  const ipd = computeInterPupillaryDistance(landmarks);
  const scale = ipd > 1e-6 ? 1.0 / ipd : 1.0;

  return landmarks.map(lm => ({
    x: (lm.x - cx) * scale,
    y: (lm.y - cy) * scale,
    z: (lm.z - cz) * scale,
  }));
}

/**
 * Compute the inter-pupillary distance from MediaPipe landmarks.
 * Uses left_pupil (473) and right_pupil (468) if available, otherwise falls
 * back to eye-inner/outer midpoints.
 *
 * @param {Array<{x:number, y:number, z:number}>} landmarks
 * @returns {number}
 */
export function computeInterPupillaryDistance(landmarks) {
  if (!landmarks || landmarks.length < 468) return 0;

  let lx, ly, lz, rx, ry, rz;

  if (landmarks.length >= 478) {
    // Iris landmarks available
    const lp = landmarks[473];
    const rp = landmarks[468];
    lx = lp.x; ly = lp.y; lz = lp.z;
    rx = rp.x; ry = rp.y; rz = rp.z;
  } else {
    // Fallback: midpoint of inner/outer corners
    const li = landmarks[LANDMARK_DEFINITIONS.left_eye_inner];
    const lo = landmarks[LANDMARK_DEFINITIONS.left_eye_outer];
    const ri = landmarks[LANDMARK_DEFINITIONS.right_eye_inner];
    const ro = landmarks[LANDMARK_DEFINITIONS.right_eye_outer];
    lx = (li.x + lo.x) / 2;
    ly = (li.y + lo.y) / 2;
    lz = (li.z + lo.z) / 2;
    rx = (ri.x + ro.x) / 2;
    ry = (ri.y + ro.y) / 2;
    rz = (ri.z + ro.z) / 2;
  }

  const dx = lx - rx;
  const dy = ly - ry;
  const dz = lz - rz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Simple exponential moving average smoother for a flat array of numbers.
 */
class LandmarkSmoother {
  /**
   * @param {number} numLandmarks
   * @param {number} alpha - EMA factor (0-1)
   */
  constructor(numLandmarks = 478, alpha = DEFAULT_SMOOTH_FACTOR) {
    this._alpha = alpha;
    this._numLandmarks = numLandmarks;
    /** @type {Float64Array|null} */
    this._prev = null;
  }

  /**
   * Smooth the incoming landmarks.
   * @param {Array<{x:number, y:number, z:number}>} landmarks
   * @returns {Array<{x:number, y:number, z:number}>}
   */
  smooth(landmarks) {
    if (!landmarks || landmarks.length === 0) return landmarks;

    const n = landmarks.length;
    const flat = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      flat[i * 3]     = landmarks[i].x;
      flat[i * 3 + 1] = landmarks[i].y;
      flat[i * 3 + 2] = landmarks[i].z;
    }

    if (this._prev === null) {
      this._prev = flat;
    } else {
      const a = this._alpha;
      const b = 1 - a;
      for (let i = 0; i < flat.length; i++) {
        flat[i] = a * flat[i] + b * this._prev[i];
      }
      this._prev = flat;
    }

    const result = new Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = {
        x: flat[i * 3],
        y: flat[i * 3 + 1],
        z: flat[i * 3 + 2],
      };
    }
    return result;
  }

  reset() {
    this._prev = null;
  }
}

/**
 * Dynamically load the MediaPipe Vision Tasks module from CDN.
 * Returns the vision module namespace or null on failure.
 */
async function loadMediaPipeVision() {
  try {
    // Attempt ESM dynamic import (works in modern browsers / Vite dev)
    const vision = await import(
      /* @vite-ignore */
      `${VISION_MODULE_URL}`
    );
    return vision;
  } catch (_e1) {
    // Fallback: inject a script tag and pull from globalThis
    return new Promise((resolve) => {
      const script = document.createElement('script');
      script.src = `${VISION_MODULE_URL}`;
      script.type = 'module';
      script.onload = () => {
        // The UMD / module-script variant exposes on globalThis
        const vision = globalThis.vision ?? globalThis.FaceLandmarker ?? null;
        resolve(vision);
      };
      script.onerror = () => {
        console.warn('MediaPipeBridge: failed to load MediaPipe Vision from CDN.');
        resolve(null);
      };
      document.head.appendChild(script);
    });
  }
}

/**
 * Solve a small ordinary least-squares problem using the normal-equation
 * pseudo-inverse:  x = (A^T A)^{-1} A^T b
 *
 * A is (m x n), b is (m x 1).  Returns x as Float64Array(n).
 * If the system is singular a zero vector is returned.
 *
 * This is intentionally naive (O(n^2 m + n^3)) — fine for n <= 20.
 */
function solveLeastSquares(A, b, m, n) {
  // AtA = A^T * A  (n x n)
  const AtA = new Float64Array(n * n);
  // Atb = A^T * b  (n x 1)
  const Atb = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) {
        sum += A[k * n + i] * A[k * n + j];
      }
      AtA[i * n + j] = sum;
    }
    let s = 0;
    for (let k = 0; k < m; k++) {
      s += A[k * n + i] * b[k];
    }
    Atb[i] = s;
  }

  // Add small Tikhonov regularisation for stability
  for (let i = 0; i < n; i++) {
    AtA[i * n + i] += 1e-4;
  }

  // Solve via Cholesky (AtA is symmetric positive semi-definite + regularised)
  const L = new Float64Array(n * n);

  // Cholesky decomposition AtA = L L^T
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = AtA[i * n + j];
      for (let k = 0; k < j; k++) {
        sum -= L[i * n + k] * L[j * n + k];
      }
      if (i === j) {
        if (sum <= 0) {
          // Not positive-definite — return zeros
          return new Float64Array(n);
        }
        L[i * n + j] = Math.sqrt(sum);
      } else {
        L[i * n + j] = sum / L[j * n + j];
      }
    }
  }

  // Forward substitution: L y = Atb
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = Atb[i];
    for (let k = 0; k < i; k++) {
      s -= L[i * n + k] * y[k];
    }
    y[i] = s / L[i * n + i];
  }

  // Back substitution: L^T x = y
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) {
      s -= L[k * n + i] * x[k];
    }
    x[i] = s / L[i * n + i];
  }

  return x;
}

// ---------------------------------------------------------------------------
// MediaPipeBridge class
// ---------------------------------------------------------------------------

export class MediaPipeBridge {
  /**
   * @param {object} [options]
   * @param {string} [options.mappingUrl]       - URL to FLAME-MediaPipe mapping JSON
   * @param {HTMLVideoElement} [options.videoElement] - Optional video element
   * @param {Function} [options.onFaceDetected] - Callback (landmarks, params)
   * @param {number}  [options.smoothFactor]    - EMA alpha, 0-1 (default 0.6)
   */
  constructor(options = {}) {
    this._mappingUrl = options.mappingUrl
      ?? '/models/flame/web/flame_mediapipe_mapping.json';

    this._videoElement = options.videoElement ?? null;
    this._onFaceDetected = options.onFaceDetected ?? null;
    this._smoothFactor = options.smoothFactor ?? DEFAULT_SMOOTH_FACTOR;

    // Internal state
    /** @type {object|null} MediaPipe Vision module namespace */
    this._vision = null;
    /** @type {object|null} FaceLandmarker instance */
    this._faceLandmarker = null;
    /** @type {object|null} Loaded FLAME-MP mapping */
    this._flameMapping = null;
    /** @type {MediaStream|null} */
    this._stream = null;
    /** @type {boolean} */
    this._running = false;
    /** @type {number|null} */
    this._rafId = null;
    /** @type {LandmarkSmoother} */
    this._smoother = new LandmarkSmoother(478, this._smoothFactor);

    /** @type {boolean} Whether a face is currently visible */
    this._faceVisible = false;

    /** @type {Map<string, Set<Function>>} Event listeners */
    this._listeners = new Map();

    /** @type {Array<{x:number,y:number,z:number}>|null} Last detected landmarks */
    this._lastLandmarks = null;
    /** @type {object|null} Last estimated FLAME params */
    this._lastParams = null;
  }

  // -----------------------------------------------------------------------
  // Initialisation
  // -----------------------------------------------------------------------

  /**
   * Load MediaPipe Face Mesh and the FLAME-MediaPipe mapping file.
   * Must be called before any detection.
   *
   * @returns {Promise<boolean>} true if ready, false if MediaPipe failed to load
   */
  async init() {
    // Load MediaPipe Vision Tasks
    this._vision = await loadMediaPipeVision();
    if (!this._vision) {
      console.error(
        'MediaPipeBridge: MediaPipe Vision could not be loaded. '
        + 'Face detection will be unavailable.'
      );
      return false;
    }

    try {
      const { FaceLandmarker, FilesetResolver } = this._vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(VISION_CDN);

      this._faceLandmarker = await FaceLandmarker.createFromOptions(
        filesetResolver,
        {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        },
      );
    } catch (err) {
      console.error('MediaPipeBridge: failed to create FaceLandmarker:', err);
      return false;
    }

    // Load the FLAME-MediaPipe mapping (non-critical)
    await this._loadMapping();

    return true;
  }

  // -----------------------------------------------------------------------
  // Camera
  // -----------------------------------------------------------------------

  /**
   * Start webcam capture and face detection loop.
   *
   * @param {HTMLVideoElement} [videoElement] - If supplied, overrides the
   *   element given in the constructor.
   * @returns {Promise<void>}
   */
  async startCamera(videoElement) {
    if (videoElement) this._videoElement = videoElement;

    if (!this._videoElement) {
      this._videoElement = document.createElement('video');
      this._videoElement.setAttribute('playsinline', '');
      this._videoElement.setAttribute('autoplay', '');
      this._videoElement.style.display = 'none';
      document.body.appendChild(this._videoElement);
    }

    // Request camera
    this._stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });

    this._videoElement.srcObject = this._stream;
    await this._videoElement.play();

    // Switch to VIDEO running mode if needed
    if (this._faceLandmarker) {
      try {
        this._faceLandmarker.setOptions({ runningMode: 'VIDEO' });
      } catch (_e) {
        // Already in VIDEO mode — ignore
      }
    }

    this._running = true;
    this._smoother.reset();
    this._detectLoop();
  }

  /**
   * Stop webcam capture and face detection.
   */
  stopCamera() {
    this._running = false;

    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }

    if (this._stream) {
      for (const track of this._stream.getTracks()) {
        track.stop();
      }
      this._stream = null;
    }

    if (this._videoElement) {
      this._videoElement.srcObject = null;
    }

    if (this._faceVisible) {
      this._faceVisible = false;
      this._emit('face_lost');
    }

    this._smoother.reset();
  }

  // -----------------------------------------------------------------------
  // Single-image detection
  // -----------------------------------------------------------------------

  /**
   * Detect face landmarks from a static image or canvas.
   *
   * @param {HTMLImageElement|HTMLCanvasElement|HTMLVideoElement} imageSource
   * @returns {Promise<Array<{x:number,y:number,z:number}>|null>} Landmarks or null
   */
  async detectFromImage(imageSource) {
    if (!this._faceLandmarker) {
      console.warn('MediaPipeBridge: not initialised. Call init() first.');
      return null;
    }

    try {
      // Switch to IMAGE mode
      this._faceLandmarker.setOptions({ runningMode: 'IMAGE' });
    } catch (_e) {
      // May already be in IMAGE mode
    }

    const result = this._faceLandmarker.detect(imageSource);

    if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) {
      return null;
    }

    const landmarks = result.faceLandmarks[0];
    this._lastLandmarks = landmarks;
    return landmarks;
  }

  // -----------------------------------------------------------------------
  // Landmark -> FLAME parameter conversion
  // -----------------------------------------------------------------------

  /**
   * Convert MediaPipe 478 landmarks to approximate FLAME parameters.
   *
   * The estimation uses a least-squares fit when a mapping file and FLAME
   * shape basis are available, otherwise falls back to a heuristic approach
   * based on landmark distances.
   *
   * @param {Array<{x:number, y:number, z:number}>} landmarks
   * @returns {{
   *   shape_params: number[],
   *   expression_params: number[],
   *   head_pose: { rotation: [number,number,number], translation: [number,number,number] }
   * }}
   */
  landmarksToFLAMEParams(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return this._emptyParams();
    }

    const normed = normalizeAndCenter(landmarks);
    const pose = this.landmarksToHeadPose(landmarks);

    // If we have the full mapping + shape basis, use least-squares
    if (this._flameMapping && this._flameMapping.shapedirs) {
      return this._estimateParamsLeastSquares(normed, pose);
    }

    // Fallback: heuristic shape/expression estimation
    return this._estimateParamsHeuristic(normed, pose);
  }

  /**
   * Extract head rotation and position from landmarks.
   *
   * Uses a lightweight perspective-n-point approach: project known 3-D model
   * points onto the observed 2-D (+ depth) landmark positions, then recover
   * Euler angles via SVD-free rotation fitting.
   *
   * @param {Array<{x:number, y:number, z:number}>} landmarks
   * @returns {{ pitch: number, yaw: number, roll: number, position: {x:number, y:number, z:number} }}
   */
  landmarksToHeadPose(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return { pitch: 0, yaw: 0, roll: 0, position: { x: 0, y: 0, z: 0 } };
    }

    // Gather observed 2-D/3-D positions for pose landmarks
    const observed = POSE_LANDMARK_INDICES.map(idx => landmarks[idx]);

    // Estimate translation as offset of nose tip from image centre
    const noseTip = observed[0];
    const tx = noseTip.x - 0.5;
    const ty = noseTip.y - 0.5;
    const tz = noseTip.z;

    // Estimate yaw from horizontal displacement between eye outer corners
    const leftEyeOuter = observed[2];   // 263
    const rightEyeOuter = observed[3];  // 33
    const eyeDx = leftEyeOuter.x - rightEyeOuter.x;
    const eyeDz = leftEyeOuter.z - rightEyeOuter.z;
    const yaw = Math.atan2(eyeDz, Math.abs(eyeDx) + 1e-8);

    // Estimate pitch from vertical displacement nose-chin vs reference
    const chin = observed[1]; // 152
    const noseVec = { x: chin.x - noseTip.x, y: chin.y - noseTip.y };
    const noseLen = Math.sqrt(noseVec.x * noseVec.x + noseVec.y * noseVec.y) || 1;
    // When head tilts back, the chin rises relative to nose tip
    const pitch = Math.asin(Math.max(-1, Math.min(1, -noseVec.y / noseLen))) - Math.PI / 2;

    // Estimate roll from the angle of the line connecting eye outer corners
    const roll = Math.atan2(
      rightEyeOuter.y - leftEyeOuter.y,
      rightEyeOuter.x - leftEyeOuter.x,
    );

    return {
      pitch,
      yaw,
      roll,
      position: { x: tx, y: ty, z: tz },
    };
  }

  /**
   * Group landmarks by facial region.
   *
   * @param {Array<{x:number, y:number, z:number}>} landmarks
   * @returns {Record<string, Array<{index:number, x:number, y:number, z:number}>>}
   */
  getLandmarkRegions(landmarks) {
    if (!landmarks || landmarks.length === 0) return {};

    const result = {};
    for (const [region, indices] of Object.entries(FACE_REGIONS)) {
      result[region] = indices
        .filter(idx => idx < landmarks.length)
        .map(idx => ({
          index: idx,
          x: landmarks[idx].x,
          y: landmarks[idx].y,
          z: landmarks[idx].z,
        }));
    }
    return result;
  }

  // -----------------------------------------------------------------------
  // Events / Callbacks
  // -----------------------------------------------------------------------

  /**
   * Register a listener for face tracking events.
   *
   * Supported events:
   *   - `face_detected`     — fires when a face first appears
   *   - `face_lost`         — fires when the face disappears
   *   - `landmarks_updated` — fires every frame with (landmarks)
   *   - `params_estimated`  — fires every frame with (flameParams)
   *
   * @param {string} event
   * @param {Function} callback
   * @returns {Function} Unsubscribe function
   */
  onChange(event, callback) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event).add(callback);

    return () => {
      const set = this._listeners.get(event);
      if (set) set.delete(callback);
    };
  }

  // -----------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------

  /** @returns {Array<{x:number,y:number,z:number}>|null} Last detected landmarks */
  get lastLandmarks() {
    return this._lastLandmarks;
  }

  /** @returns {object|null} Last estimated FLAME params */
  get lastParams() {
    return this._lastParams;
  }

  /** @returns {boolean} Whether a face is currently being tracked */
  get isFaceVisible() {
    return this._faceVisible;
  }

  /** @returns {boolean} Whether the bridge is initialised and ready */
  get isReady() {
    return this._faceLandmarker !== null;
  }

  /**
   * @returns {object|null} FLAME-MediaPipe mapping data:
   *   { landmark_indices: number[], lmk_face_idx: number[], lmk_b_coords: number[][] }
   */
  get flameMapping() {
    return this._flameMapping ?? null;
  }

  /** @returns {Readonly<typeof LANDMARK_DEFINITIONS>} Semantic landmark map */
  static get LANDMARK_DEFINITIONS() {
    return LANDMARK_DEFINITIONS;
  }

  /** @returns {Readonly<typeof FACE_REGIONS>} Region groupings */
  static get FACE_REGIONS() {
    return FACE_REGIONS;
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  /**
   * Release all resources.
   */
  destroy() {
    this.stopCamera();

    if (this._faceLandmarker) {
      try { this._faceLandmarker.close(); } catch (_e) { /* ignore */ }
      this._faceLandmarker = null;
    }

    this._listeners.clear();
    this._lastLandmarks = null;
    this._lastParams = null;
    this._flameMapping = null;
    this._vision = null;
  }

  // -----------------------------------------------------------------------
  // Private: detection loop
  // -----------------------------------------------------------------------

  /** @private */
  _detectLoop() {
    if (!this._running) return;

    this._rafId = requestAnimationFrame(() => this._detectLoop());

    if (
      !this._faceLandmarker ||
      !this._videoElement ||
      this._videoElement.readyState < 2
    ) {
      return;
    }

    const timestamp = performance.now();
    let result;
    try {
      result = this._faceLandmarker.detectForVideo(
        this._videoElement,
        timestamp,
      );
    } catch (err) {
      // Occasionally MediaPipe throws on bad frames — skip
      return;
    }

    if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
      let landmarks = result.faceLandmarks[0];

      // Smooth
      landmarks = this._smoother.smooth(landmarks);

      this._lastLandmarks = landmarks;

      if (!this._faceVisible) {
        this._faceVisible = true;
        this._emit('face_detected', landmarks);
      }

      this._emit('landmarks_updated', landmarks);

      // Estimate FLAME params
      const params = this.landmarksToFLAMEParams(landmarks);
      this._lastParams = params;
      this._emit('params_estimated', params);

      // User callback
      if (this._onFaceDetected) {
        try {
          this._onFaceDetected(landmarks, params);
        } catch (err) {
          console.error('MediaPipeBridge: onFaceDetected callback error:', err);
        }
      }
    } else {
      if (this._faceVisible) {
        this._faceVisible = false;
        this._emit('face_lost');
      }
    }
  }

  // -----------------------------------------------------------------------
  // Private: mapping loader
  // -----------------------------------------------------------------------

  /** @private */
  async _loadMapping() {
    try {
      const resp = await fetch(this._mappingUrl);
      if (!resp.ok) {
        console.warn(
          `MediaPipeBridge: mapping file not found at ${this._mappingUrl} `
          + `(${resp.status}). Parameter estimation will use heuristics.`
        );
        return;
      }
      this._flameMapping = await resp.json();
    } catch (err) {
      console.warn(
        'MediaPipeBridge: could not load FLAME-MediaPipe mapping:', err.message,
        '— falling back to heuristic parameter estimation.'
      );
    }
  }

  // -----------------------------------------------------------------------
  // Private: parameter estimation
  // -----------------------------------------------------------------------

  /** @private */
  _emptyParams() {
    return {
      shape_params: new Array(NUM_SHAPE_PARAMS).fill(0),
      expression_params: new Array(NUM_EXPRESSION_PARAMS).fill(0),
      head_pose: {
        rotation: [0, 0, 0],
        translation: [0, 0, 0],
      },
    };
  }

  /**
   * Least-squares FLAME parameter estimation.
   *
   * Given the mapping file which provides correspondence between MediaPipe
   * landmarks and FLAME vertices, we solve:
   *
   *   min || V_template[indices] + ShapeDirs[indices] * beta - Observed ||^2
   *
   * where beta is the vector of shape parameters.
   *
   * For expression parameters we do the same with the expression basis.
   *
   * @private
   */
  _estimateParamsLeastSquares(normedLandmarks, pose) {
    const mapping = this._flameMapping;

    // mapping.landmark_indices: MediaPipe idx -> FLAME vertex idx
    // mapping.shapedirs: (numVertices, 3, numShapeParams) flattened
    // mapping.v_template: (numVertices, 3) flattened

    const landmarkIndices = mapping.landmark_indices ?? mapping.lmk_indices ?? [];
    const vTemplate = mapping.v_template;
    const shapedirs = mapping.shapedirs;
    const numShapePC = mapping.num_shape_params ?? NUM_SHAPE_PARAMS;
    const numExprPC = mapping.num_expression_params ?? NUM_EXPRESSION_PARAMS;

    const numLM = Math.min(landmarkIndices.length, normedLandmarks.length);
    if (numLM === 0 || !vTemplate || !shapedirs) {
      return this._estimateParamsHeuristic(normedLandmarks, pose);
    }

    // Build the (m x n) design matrix A and (m x 1) residual b
    // m = numLM * 3 (x, y, z for each landmark)
    // n = numShapePC
    const m = numLM * 3;
    const nShape = Math.min(numShapePC, NUM_SHAPE_PARAMS);

    const A = new Float64Array(m * nShape);
    const b = new Float64Array(m);

    for (let li = 0; li < numLM; li++) {
      const mpIdx = li; // index into normedLandmarks
      const flameVtxIdx = landmarkIndices[li];
      const obs = normedLandmarks[mpIdx];

      for (let axis = 0; axis < 3; axis++) {
        const row = li * 3 + axis;
        // Template vertex position
        const templateVal = vTemplate[flameVtxIdx * 3 + axis] ?? 0;
        // Residual: observed - template
        const obsVal = axis === 0 ? obs.x : axis === 1 ? obs.y : obs.z;
        b[row] = obsVal - templateVal;

        // Shape basis vectors
        for (let pc = 0; pc < nShape; pc++) {
          // shapedirs layout: [vtx][axis][pc]
          const sdIdx = (flameVtxIdx * 3 + axis) * numShapePC + pc;
          A[row * nShape + pc] = shapedirs[sdIdx] ?? 0;
        }
      }
    }

    const shapeParams = Array.from(solveLeastSquares(A, b, m, nShape));

    // Expression params: similar approach if expression basis is provided
    let expressionParams;
    if (mapping.expressiondirs) {
      const nExpr = Math.min(numExprPC, NUM_EXPRESSION_PARAMS);
      const Ae = new Float64Array(m * nExpr);
      const be = new Float64Array(m);

      // Subtract shape contribution from observations
      for (let li = 0; li < numLM; li++) {
        const flameVtxIdx = landmarkIndices[li];
        const obs = normedLandmarks[li];

        for (let axis = 0; axis < 3; axis++) {
          const row = li * 3 + axis;
          const templateVal = vTemplate[flameVtxIdx * 3 + axis] ?? 0;
          const obsVal = axis === 0 ? obs.x : axis === 1 ? obs.y : obs.z;

          // Subtract shape contribution
          let shapeContrib = 0;
          for (let pc = 0; pc < shapeParams.length; pc++) {
            const sdIdx = (flameVtxIdx * 3 + axis) * numShapePC + pc;
            shapeContrib += (shapedirs[sdIdx] ?? 0) * shapeParams[pc];
          }
          be[row] = obsVal - templateVal - shapeContrib;

          for (let pc = 0; pc < nExpr; pc++) {
            const edIdx = (flameVtxIdx * 3 + axis) * numExprPC + pc;
            Ae[row * nExpr + pc] = mapping.expressiondirs[edIdx] ?? 0;
          }
        }
      }

      expressionParams = Array.from(solveLeastSquares(Ae, be, m, nExpr));
    } else {
      expressionParams = this._heuristicExpressionParams(normedLandmarks);
    }

    return {
      shape_params: shapeParams,
      expression_params: expressionParams,
      head_pose: {
        rotation: [pose.pitch, pose.yaw, pose.roll],
        translation: [pose.position.x, pose.position.y, pose.position.z],
      },
    };
  }

  /**
   * Heuristic FLAME parameter estimation when no mapping file is available.
   *
   * Derives approximate shape and expression parameters from landmark
   * distances and ratios.
   *
   * @private
   */
  _estimateParamsHeuristic(normedLandmarks, pose) {
    const params = this._emptyParams();

    if (!normedLandmarks || normedLandmarks.length < 468) return params;

    const lm = normedLandmarks;

    // --- Shape heuristics ---
    // We map a handful of geometric ratios to the first few PCA components.
    // This is an approximation — real fitting requires the actual basis.

    // Face width-to-height ratio
    const jawRight = lm[LANDMARK_DEFINITIONS.jaw_right];
    const jawLeft = lm[LANDMARK_DEFINITIONS.jaw_left];
    const forehead = lm[LANDMARK_DEFINITIONS.forehead_center];
    const chin = lm[LANDMARK_DEFINITIONS.chin_tip];

    const faceWidth = dist3(jawRight, jawLeft);
    const faceHeight = dist3(forehead, chin);
    const widthHeightRatio = faceWidth / (faceHeight + 1e-8);

    // Nose length
    const noseBridge = lm[LANDMARK_DEFINITIONS.nose_bridge_top];
    const noseTip = lm[LANDMARK_DEFINITIONS.nose_tip];
    const noseLen = dist3(noseBridge, noseTip);

    // Lip width
    const lipR = lm[LANDMARK_DEFINITIONS.lip_right_corner];
    const lipL = lm[LANDMARK_DEFINITIONS.lip_left_corner];
    const lipWidth = dist3(lipR, lipL);

    // Eye spacing
    const rEyeIn = lm[LANDMARK_DEFINITIONS.right_eye_inner];
    const lEyeIn = lm[LANDMARK_DEFINITIONS.left_eye_inner];
    const eyeSpacing = dist3(rEyeIn, lEyeIn);

    // Jaw width
    const jawRMid = lm[LANDMARK_DEFINITIONS.jaw_right_mid];
    const jawLMid = lm[LANDMARK_DEFINITIONS.jaw_left_mid];
    const jawWidth = dist3(jawRMid, jawLMid);

    // Map to shape params (rough — these are arbitrary but directional)
    params.shape_params[0] = (widthHeightRatio - 0.75) * 3.0;  // face roundness
    params.shape_params[1] = (noseLen - 0.15) * 5.0;           // nose length
    params.shape_params[2] = (lipWidth - 0.12) * 5.0;          // lip width
    params.shape_params[3] = (eyeSpacing - 0.08) * 6.0;        // eye spacing
    params.shape_params[4] = (jawWidth - 0.3) * 3.0;           // jaw width

    // Clamp all shape params
    for (let i = 0; i < params.shape_params.length; i++) {
      params.shape_params[i] = Math.max(-3, Math.min(3, params.shape_params[i]));
    }

    // --- Expression heuristics ---
    params.expression_params = this._heuristicExpressionParams(normedLandmarks);

    // --- Head pose ---
    params.head_pose = {
      rotation: [pose.pitch, pose.yaw, pose.roll],
      translation: [pose.position.x, pose.position.y, pose.position.z],
    };

    return params;
  }

  /**
   * Heuristic expression parameter estimation.
   * @private
   */
  _heuristicExpressionParams(normedLandmarks) {
    const exprParams = new Array(NUM_EXPRESSION_PARAMS).fill(0);
    if (!normedLandmarks || normedLandmarks.length < 468) return exprParams;

    const lm = normedLandmarks;

    // Mouth open
    const upperLip = lm[LANDMARK_DEFINITIONS.upper_lip_bottom];
    const lowerLip = lm[LANDMARK_DEFINITIONS.lower_lip_bottom];
    const mouthOpen = dist3(upperLip, lowerLip);
    exprParams[0] = Math.min(3, mouthOpen * 15);

    // Mouth width (smile proxy)
    const lipR = lm[LANDMARK_DEFINITIONS.lip_right_corner];
    const lipL = lm[LANDMARK_DEFINITIONS.lip_left_corner];
    const mouthWidth = dist3(lipR, lipL);
    exprParams[1] = (mouthWidth - 0.12) * 8;

    // Brow raise
    const rBrowMid = lm[LANDMARK_DEFINITIONS.right_brow_mid];
    const rEyeUpper = lm[LANDMARK_DEFINITIONS.right_eye_upper];
    const browEyeGap = rBrowMid.y - rEyeUpper.y;
    exprParams[2] = browEyeGap * 20;

    // Eye openness (right)
    const rEyeU = lm[LANDMARK_DEFINITIONS.right_eye_upper];
    const rEyeL = lm[LANDMARK_DEFINITIONS.right_eye_lower];
    const eyeOpen = dist3(rEyeU, rEyeL);
    exprParams[3] = eyeOpen * 20;

    // Left eye openness
    const lEyeU = lm[LANDMARK_DEFINITIONS.left_eye_upper];
    const lEyeL = lm[LANDMARK_DEFINITIONS.left_eye_lower];
    exprParams[4] = dist3(lEyeU, lEyeL) * 20;

    // Clamp expression params
    for (let i = 0; i < exprParams.length; i++) {
      exprParams[i] = Math.max(-3, Math.min(3, exprParams[i]));
    }

    return exprParams;
  }

  // -----------------------------------------------------------------------
  // Private: event emitter
  // -----------------------------------------------------------------------

  /** @private */
  _emit(event, ...args) {
    const set = this._listeners.get(event);
    if (!set) return;
    for (const fn of set) {
      try {
        fn(...args);
      } catch (err) {
        console.error(`MediaPipeBridge: error in '${event}' listener:`, err);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/**
 * Euclidean distance between two 3-D points.
 * @param {{x:number,y:number,z:number}} a
 * @param {{x:number,y:number,z:number}} b
 * @returns {number}
 */
function dist3(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export {
  LANDMARK_DEFINITIONS,
  FACE_REGIONS,
  POSE_LANDMARK_INDICES,
  POSE_MODEL_POINTS,
  LandmarkSmoother,
};

export default MediaPipeBridge;
