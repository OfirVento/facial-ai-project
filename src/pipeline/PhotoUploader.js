/**
 * Photo Upload & Reconstruction Pipeline — Orchestrator
 *
 * Handles photo capture/upload, coordinates FLAME shape fitting, UV texture
 * projection, and post-processing via three delegate modules:
 *
 *   ShapeFitter       — FLAME PCA fitting, camera estimation
 *   TextureProjector  — TPS / dense / multi-view UV projection, rasterisation
 *   TexturePostProcess — Delighting, alpha erosion, Laplacian blending
 *
 * Public API preserved:
 *   setPhoto(), captureFromCamera(), getStatus(), generateTextureFromPhoto(),
 *   generateNormalMapFromPhoto(), loadReconstructionResult(), loadFromFiles(),
 *   clear(), onChange()
 *
 * Decomposed from original 5128-line monolith.
 */

import { ShapeFitter } from './ShapeFitter.js';
import { TextureProjector } from './TextureProjector.js';
import { TexturePostProcess } from './TexturePostProcess.js';

export class PhotoUploader {
  /**
   * @param {object} [options]
   * @param {import('../engine/FlameMeshGenerator.js').FlameMeshGenerator} [options.meshGenerator]
   * @param {import('../engine/MediaPipeBridge.js').MediaPipeBridge} [options.mediaPipeBridge]
   */
  constructor(options = {}) {
    this.photos = { front: null, left45: null, right45: null };
    this.reconstructionStatus = 'idle'; // idle | uploading | reconstructing | done | error
    this.listeners = new Set();
    this.reconstructionResult = null;

    /** @type {import('../engine/FlameMeshGenerator.js').FlameMeshGenerator|null} */
    this._meshGenerator = options.meshGenerator ?? null;
    /** @type {import('../engine/MediaPipeBridge.js').MediaPipeBridge|null} */
    this._mediaPipeBridge = options.mediaPipeBridge ?? null;

    // Delegate modules (instantiated lazily or eagerly)
    this._shapeFitter = new ShapeFitter();
    this._postProcessor = new TexturePostProcess();
    this._projector = new TextureProjector({
      postProcessor: this._postProcessor,
      shapeFitter: this._shapeFitter,
    });
  }

  // -----------------------------------------------------------------------
  // Photo management
  // -----------------------------------------------------------------------

  /**
   * Set a photo for a specific angle.
   * @param {string} angle - 'front' | 'left45' | 'right45'
   * @param {File|Blob} file - The image file
   */
  async setPhoto(angle, file) {
    if (!['front', 'left45', 'right45'].includes(angle)) {
      throw new Error(`Invalid angle: ${angle}. Use 'front', 'left45', or 'right45'`);
    }

    const url = URL.createObjectURL(file);
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });

    this.photos[angle] = {
      file,
      url,
      width: img.naturalWidth,
      height: img.naturalHeight,
      name: file.name,
      size: file.size,
    };

    this._notify({ type: 'photo_set', angle, photo: this.photos[angle] });
    return this.photos[angle];
  }

  /**
   * Capture photo from webcam.
   */
  async captureFromCamera(angle) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: 'user',
        },
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      await new Promise((r) => { video.onloadedmetadata = r; });
      await video.play();
      await new Promise((r) => setTimeout(r, 500));

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      stream.getTracks().forEach((t) => t.stop());

      const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.95));
      const file = new File([blob], `capture_${angle}.jpg`, { type: 'image/jpeg' });

      return this.setPhoto(angle, file);
    } catch (err) {
      this._notify({ type: 'error', error: `Camera access failed: ${err.message}` });
      throw err;
    }
  }

  /**
   * Check if we have the minimum required photos.
   */
  getStatus() {
    const hasFront = !!this.photos.front;
    const hasLeft = !!this.photos.left45;
    const hasRight = !!this.photos.right45;
    const photoCount = [hasFront, hasLeft, hasRight].filter(Boolean).length;

    return {
      hasFront,
      hasLeft,
      hasRight,
      photoCount,
      canReconstruct: hasFront,
      isOptimal: hasFront && hasLeft && hasRight,
      reconstructionStatus: this.reconstructionStatus,
    };
  }

  // -----------------------------------------------------------------------
  // Texture generation — main entry point
  // -----------------------------------------------------------------------

  /**
   * Generate a texture from the front photo by projecting it onto FLAME UV space.
   * Uses dense or multi-view projection when FLAME data + MediaPipe are available,
   * otherwise falls back to a naive centre-crop.
   */
  async generateTextureFromPhoto(options = {}) {
    // Store render mode hint for delighting strength adjustment
    this._renderModeHint = options.renderMode || 'hybrid';
    this._postProcessor._renderModeHint = this._renderModeHint;
    this._projector._renderModeHint = this._renderModeHint;

    if (!this.photos.front) {
      throw new Error('Front photo required');
    }

    // Load photo into an Image element
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = this.photos.front.url;
    });

    // Check whether we have everything for proper UV projection
    const meshGen = this._meshGenerator;
    const bridge = this._mediaPipeBridge;

    const canProject = meshGen
      && meshGen.isFlameLoaded
      && meshGen.flameUVCoords
      && meshGen.flameUVFaces
      && bridge;

    if (!canProject) {
      console.warn('PhotoUploader: Missing FLAME UV data or MediaPipeBridge — falling back to naive projection');
      return this._naiveTextureProjection(img);
    }

    // Ensure MediaPipe is initialised (lazy init)
    if (!bridge.isReady) {
      console.log('PhotoUploader: Initialising MediaPipe for face detection…');
      const ok = await bridge.init();
      if (!ok) {
        console.warn('PhotoUploader: MediaPipe init failed — falling back to naive projection');
        return this._naiveTextureProjection(img);
      }
    }

    // Detect landmarks in the photo
    const landmarks = await bridge.detectFromImage(img);
    if (!landmarks || landmarks.length < 468) {
      console.warn('PhotoUploader: No face detected in photo — falling back to naive projection');
      return this._naiveTextureProjection(img);
    }

    // Get the FLAME-MediaPipe mapping
    const mapping = bridge.flameMapping;
    if (!mapping || !mapping.landmark_indices || !mapping.lmk_face_idx || !mapping.lmk_b_coords) {
      console.warn('PhotoUploader: Mapping data unavailable — falling back to naive projection');
      return this._naiveTextureProjection(img);
    }

    console.log(`PhotoUploader: Detected ${landmarks.length} landmarks, projecting onto FLAME UV space…`);
    console.log(`PhotoUploader: Image ${img.naturalWidth}×${img.naturalHeight}`);

    // Fit FLAME shape to match user's face proportions before texture baking
    try {
      this._shapeFitter._fitShapeFromLandmarks(landmarks, mapping, meshGen);
    } catch (err) {
      console.warn('PhotoUploader: Shape fitting failed, using mean shape:', err.message);
    }

    // Fit FLAME expression to match the photo's mouth/eyes/brows
    try {
      this._shapeFitter._fitExpressionFromLandmarks(landmarks, mapping, meshGen);
    } catch (err) {
      console.warn('PhotoUploader: Expression fitting failed, using neutral:', err.message);
    }

    // Estimate ONE camera from the final fitted mesh — used for ALL projection
    const fittedCamera = this._shapeFitter._estimateCameraFromLandmarks(landmarks, mapping, meshGen);
    if (fittedCamera) {
      console.log(`PhotoUploader: Fitted camera: sx=${fittedCamera.sx.toFixed(4)}, sy=${fittedCamera.sy.toFixed(4)}, tx=${fittedCamera.tx.toFixed(4)}, ty=${fittedCamera.ty.toFixed(4)}`);
    } else {
      console.error('PhotoUploader: Camera estimation failed on fitted mesh!');
    }

    // Check for multi-view: if left45 or right45 photos exist, use multi-view pipeline
    const hasMultiView = this.photos.left45 || this.photos.right45;

    if (hasMultiView) {
      console.log('PhotoUploader: Multi-view photos detected, using multi-view pipeline');
      const result = await this._projector._multiViewProjectTextureToUV(
        img, landmarks, mapping, meshGen, fittedCamera, bridge, this.photos
      );
      if (result) {
        // Copy diagnostic state for public diagnostic generators
        this._syncDiagnostics();
        console.log('PhotoUploader: Multi-view projection succeeded');
        return result;
      }
      console.warn('PhotoUploader: Multi-view failed, falling back to single front view');
    }

    // Dense Projection (camera-based texture baking)
    const result = this._projector._denseProjectTextureToUV(img, landmarks, mapping, meshGen, fittedCamera);
    if (result) {
      this._syncDiagnostics();
      console.log('PhotoUploader: Dense projection succeeded');
      return result;
    }

    // If dense projection fails, it's a real problem — log it clearly
    console.error('PhotoUploader: Dense projection returned null! Falling back to TPS.');
    const tpsResult = this._projector._warpTextureToUV(img, landmarks, mapping, meshGen);
    this._syncDiagnostics();
    return tpsResult;
  }

  // -----------------------------------------------------------------------
  // Diagnostic passthrough — public API delegates to TextureProjector
  // -----------------------------------------------------------------------

  /**
   * Generate a UV coverage heatmap from the last projection.
   */
  generateCoverageHeatmap() {
    return this._projector.generateCoverageHeatmap();
  }

  /**
   * Generate an N·V visibility heatmap from the last projection.
   */
  generateNVHeatmap() {
    return this._projector.generateNVHeatmap();
  }

  /**
   * Generate a UV landmark overlay from the last projection.
   */
  generateLandmarkOverlay() {
    return this._projector.generateLandmarkOverlay();
  }

  /**
   * Generate a labelled UV checkerboard (static utility).
   */
  static generateUVCheckerboard() {
    return TextureProjector.generateUVCheckerboard();
  }

  // -----------------------------------------------------------------------
  // Normal map generation
  // -----------------------------------------------------------------------

  /**
   * Generate a normal map from the photo for surface detail.
   */
  async generateNormalMapFromPhoto() {
    if (!this.photos.front) return null;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = this.photos.front.url;
    });

    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size);
    const pixels = imageData.data;

    const heights = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      heights[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    }

    const normalData = ctx.createImageData(size, size);
    const strength = 2.0;

    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const idx = y * size + x;
        const left = heights[idx - 1];
        const right = heights[idx + 1];
        const up = heights[idx - size];
        const down = heights[idx + size];

        let nx = (left - right) * strength;
        let ny = (up - down) * strength;
        let nz = 1.0;

        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        nx /= len; ny /= len; nz /= len;

        const pi = idx * 4;
        normalData.data[pi]     = Math.round((nx * 0.5 + 0.5) * 255);
        normalData.data[pi + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        normalData.data[pi + 2] = Math.round((nz * 0.5 + 0.5) * 255);
        normalData.data[pi + 3] = 255;
      }
    }

    ctx.putImageData(normalData, 0, 0);

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/png'),
      width: size,
      height: size,
    };
  }

  // -----------------------------------------------------------------------
  // Naive fallback (original approach)
  // -----------------------------------------------------------------------

  /**
   * Naive centre-crop texture projection (fallback when FLAME/MediaPipe unavailable).
   */
  _naiveTextureProjection(img) {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = '#e8b89d';
    ctx.fillRect(0, 0, size, size);

    const faceRegion = { x: size * 0.2, y: size * 0.05, w: size * 0.6, h: size * 0.8 };
    const srcFaceX = img.width * 0.2;
    const srcFaceY = img.height * 0.05;
    const srcFaceW = img.width * 0.6;
    const srcFaceH = img.height * 0.85;

    ctx.drawImage(img, srcFaceX, srcFaceY, srcFaceW, srcFaceH,
      faceRegion.x, faceRegion.y, faceRegion.w, faceRegion.h);

    const gradient = ctx.createRadialGradient(
      size / 2, size * 0.45, size * 0.25,
      size / 2, size * 0.45, size * 0.5
    );
    gradient.addColorStop(0, 'rgba(232, 184, 157, 0)');
    gradient.addColorStop(1, 'rgba(232, 184, 157, 1)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.9),
      width: size,
      height: size,
    };
  }

  // -----------------------------------------------------------------------
  // Reconstruction loading (cloud / file-based)
  // -----------------------------------------------------------------------

  async loadReconstructionResult(result) {
    this.reconstructionResult = result;
    this.reconstructionStatus = 'done';
    this._notify({ type: 'reconstruction_complete', result });
    return result;
  }

  async loadFromFiles(meshFile, textureFile, normalMapFile) {
    const result = {};
    if (meshFile) {
      result.meshUrl = URL.createObjectURL(meshFile);
      result.meshType = meshFile.name.endsWith('.glb') ? 'glb' : 'obj';
    }
    if (textureFile) result.textureUrl = URL.createObjectURL(textureFile);
    if (normalMapFile) result.normalMapUrl = URL.createObjectURL(normalMapFile);
    return this.loadReconstructionResult(result);
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------

  clear() {
    for (const angle of Object.keys(this.photos)) {
      if (this.photos[angle]?.url) URL.revokeObjectURL(this.photos[angle].url);
      this.photos[angle] = null;
    }
    this.reconstructionStatus = 'idle';
    this.reconstructionResult = null;
    this._notify({ type: 'cleared' });
  }

  onChange(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  _notify(event) {
    for (const listener of this.listeners) {
      try { listener(event); } catch (e) { console.error(e); }
    }
  }

  // -----------------------------------------------------------------------
  // Internal: sync diagnostic state from projector for backward compat
  // -----------------------------------------------------------------------

  /**
   * Copy diagnostic state from the TextureProjector so that external code
   * that reads `photoUploader._debug*` or `_diag*` properties still works.
   */
  _syncDiagnostics() {
    const p = this._projector;
    // Mirror diagnostic state for backward compatibility
    this._lastDebugData = p._lastDebugData;
    this._diagCameraParams = p._diagCameraParams;
    this._diagFaceVisibility = p._diagFaceVisibility;
    this._diagProjectedCoords = p._diagProjectedCoords;
    this._diagMeshGen = p._diagMeshGen;
    this._diagPreErosionAlpha = p._diagPreErosionAlpha;
    this._diagTextureSize = p._diagTextureSize;
    this._debugPhotoLandmarks = p._debugPhotoLandmarks;
    this._debugUVLandmarks = p._debugUVLandmarks;
    this._debugPreFillTexture = p._debugPreFillTexture;
    this._debugFinalTexture = p._debugFinalTexture;
    this._debugOverlayUrl = p._debugOverlayUrl;
    this._debugCoverageMask = p._debugCoverageMask;
    this._debugAlphaCoverage = p._debugAlphaCoverage;
    this._debugPhotoUV_raw = p._debugPhotoUV_raw;
    this._debugAlbedoTinted = p._debugAlbedoTinted;
  }
}

export default PhotoUploader;
