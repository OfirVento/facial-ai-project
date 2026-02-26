/**
 * Photo Upload & Reconstruction Pipeline
 * Handles photo capture/upload, landmark-guided UV projection,
 * and processes the returned FLAME mesh + texture.
 *
 * When a FLAME mesh generator and MediaPipe bridge are available,
 * generateTextureFromPhoto() uses Thin-Plate Spline (TPS) interpolation
 * to map 105 UV↔image landmark pairs to all 5118 FLAME UV vertices,
 * then rasterises all 9976 mesh faces following actual face topology.
 * Otherwise falls back to a naive centre-crop.
 */

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
  }

  // -----------------------------------------------------------------------
  // Photo management
  // -----------------------------------------------------------------------

  /**
   * Set a photo for a specific angle
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
      size: file.size
    };

    this._notify({ type: 'photo_set', angle, photo: this.photos[angle] });
    return this.photos[angle];
  }

  /**
   * Capture photo from webcam
   */
  async captureFromCamera(angle) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: 'user'
        }
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      await new Promise(r => { video.onloadedmetadata = r; });
      await video.play();
      await new Promise(r => setTimeout(r, 500));

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      stream.getTracks().forEach(t => t.stop());

      const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.95));
      const file = new File([blob], `capture_${angle}.jpg`, { type: 'image/jpeg' });

      return this.setPhoto(angle, file);
    } catch (err) {
      this._notify({ type: 'error', error: `Camera access failed: ${err.message}` });
      throw err;
    }
  }

  /**
   * Check if we have the minimum required photos
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
      reconstructionStatus: this.reconstructionStatus
    };
  }

  // -----------------------------------------------------------------------
  // Texture generation — main entry point
  // -----------------------------------------------------------------------

  /**
   * Generate a texture from the front photo by projecting it onto FLAME UV space.
   * Uses piecewise affine warp when FLAME data + MediaPipe are available,
   * otherwise falls back to a naive centre-crop.
   */
  async generateTextureFromPhoto() {
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

    // Store image dimensions for expression fitting lip gap detection
    this._lastImgH = img.naturalHeight;

    // Estimate focal length and camera distance from IPD
    let perspParams = null;
    try {
      perspParams = this._estimateFocalLength(landmarks, img.naturalWidth, img.naturalHeight, meshGen);
    } catch (err) {
      console.warn('PhotoUploader: Focal length estimation failed:', err.message);
    }

    // Fit FLAME shape to match user's face proportions before texture baking
    try {
      this._fitShapeFromLandmarks(landmarks, mapping, meshGen, perspParams);
    } catch (err) {
      console.warn('PhotoUploader: Shape fitting failed, using mean shape:', err.message);
    }

    // Fit FLAME expression to match the photo's mouth/eyes/brows
    try {
      this._fitExpressionFromLandmarks(landmarks, mapping, meshGen, perspParams);
    } catch (err) {
      console.warn('PhotoUploader: Expression fitting failed, using neutral:', err.message);
    }

    // Try Dense Projection first (camera-based), fall back to TPS
    try {
      const result = this._denseProjectTextureToUV(img, landmarks, mapping, meshGen, perspParams);
      if (result) {
        console.log('PhotoUploader: Dense projection succeeded');
        return result;
      }
    } catch (err) {
      console.warn('PhotoUploader: Dense projection failed, falling back to TPS:', err.message);
    }

    console.log('PhotoUploader: Using TPS fallback');
    return this._warpTextureToUV(img, landmarks, mapping, meshGen);
  }

  // -----------------------------------------------------------------------
  // TPS-based UV-to-image mapping — core pipeline
  // -----------------------------------------------------------------------

  /**
   * Project the photo onto FLAME UV space using Thin-Plate Spline (TPS)
   * interpolation + full FLAME mesh rasterisation.
   *
   * Algorithm (inspired by DECA's world2uv approach):
   *
   * 1. For each of 105 mapped landmarks, compute UV coords (via barycentric
   *    interpolation on FLAME UV faces) and pair with MediaPipe image coords.
   *    This gives 105 (u,v) ↔ (imgX, imgY) correspondences.
   *
   * 2. Fit two TPS functions:  UV → image_x  and  UV → image_y.
   *    TPS provides smooth, globally consistent interpolation through all
   *    105 control points, handling non-linear warps gracefully.
   *
   * 3. Evaluate TPS at ALL 5118 FLAME UV vertex positions to obtain
   *    per-vertex image coordinates.  This is the key difference from the
   *    Delaunay approach: every single UV vertex gets a smooth, high-quality
   *    image mapping — not just the 105 control points.
   *
   * 4. Rasterise ALL 9976 FLAME UV faces (the actual mesh topology) using
   *    per-vertex image coordinates and barycentric interpolation.
   *    This follows the real face surface structure rather than arbitrary
   *    Delaunay triangles.
   */
  _warpTextureToUV(img, landmarks, mapping, meshGen) {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // --- source image ---
    const maxSrc = 2048;
    const sc = Math.min(1, maxSrc / Math.max(img.naturalWidth, img.naturalHeight));
    const srcW = Math.round(img.naturalWidth * sc);
    const srcH = Math.round(img.naturalHeight * sc);
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext('2d').drawImage(img, 0, 0, srcW, srcH);
    const srcData = srcCanvas.getContext('2d').getImageData(0, 0, srcW, srcH).data;

    // --- 1. Build UV ↔ image control points from 105 mapped landmarks ---
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces  = meshGen.flameUVFaces;
    const mpIndices   = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords  = mapping.lmk_b_coords;

    const ctrlU = [], ctrlV = [], ctrlIX = [], ctrlIY = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const uvi0 = uvFaces[fi * 3], uvi1 = uvFaces[fi * 3 + 1], uvi2 = uvFaces[fi * 3 + 2];

      const u = bc[0] * uvCoords[uvi0 * 2]     + bc[1] * uvCoords[uvi1 * 2]     + bc[2] * uvCoords[uvi2 * 2];
      const v = bc[0] * uvCoords[uvi0 * 2 + 1] + bc[1] * uvCoords[uvi1 * 2 + 1] + bc[2] * uvCoords[uvi2 * 2 + 1];

      ctrlU.push(u);
      ctrlV.push(v);
      ctrlIX.push(lm.x);
      ctrlIY.push(lm.y);
    }

    const N = ctrlU.length;
    console.log(`PhotoUploader: ${N} UV↔image control points for TPS`);
    for (let i = 0; i < Math.min(5, N); i++) {
      console.log(`  [${i}] UV(${ctrlU[i].toFixed(4)}, ${ctrlV[i].toFixed(4)}) → Img(${ctrlIX[i].toFixed(4)}, ${ctrlIY[i].toFixed(4)})`);
    }

    if (N < 10) {
      console.warn('PhotoUploader: Too few control points — falling back');
      return this._naiveTextureProjection(img);
    }

    // ===== DIAGNOSTIC: Draw debug overlays =====
    this._lastDebugData = {
      srcCanvas, srcW, srcH, img,
      ctrlU, ctrlV, ctrlIX, ctrlIY, N,
      landmarks, mpIndices, mapping
    };
    this._drawDebugOverlays(srcCanvas, srcW, srcH, ctrlU, ctrlV, ctrlIX, ctrlIY, N);

    // --- 2. Fit TPS: UV → image_x and UV → image_y ---
    console.log('PhotoUploader: Fitting TPS interpolation…');
    const tpsX = this._fitTPS(ctrlU, ctrlV, ctrlIX);
    const tpsY = this._fitTPS(ctrlU, ctrlV, ctrlIY);
    console.log('PhotoUploader: TPS fitted successfully');

    // ===== DIAGNOSTIC: TPS interpolation error at control points =====
    let maxErrX = 0, maxErrY = 0;
    for (let i = 0; i < N; i++) {
      const predX = this._evalTPS(tpsX, ctrlU[i], ctrlV[i]);
      const predY = this._evalTPS(tpsY, ctrlU[i], ctrlV[i]);
      maxErrX = Math.max(maxErrX, Math.abs(predX - ctrlIX[i]));
      maxErrY = Math.max(maxErrY, Math.abs(predY - ctrlIY[i]));
    }
    console.log(`PhotoUploader DIAG: TPS max control-point error: X=${maxErrX.toFixed(6)}, Y=${maxErrY.toFixed(6)}`);

    // --- 3. Evaluate TPS at all 5118 UV vertex positions ---
    const nUV = uvCoords.length / 2;
    const uvImgCoords = new Float32Array(nUV * 2);
    let minIX = Infinity, maxIX = -Infinity, minIY = Infinity, maxIY = -Infinity;
    for (let i = 0; i < nUV; i++) {
      const u = uvCoords[i * 2], v = uvCoords[i * 2 + 1];
      const ix = this._evalTPS(tpsX, u, v);
      const iy = this._evalTPS(tpsY, u, v);
      uvImgCoords[i * 2] = ix;
      uvImgCoords[i * 2 + 1] = iy;
      if (ix < minIX) minIX = ix; if (ix > maxIX) maxIX = ix;
      if (iy < minIY) minIY = iy; if (iy > maxIY) maxIY = iy;
    }
    console.log(`PhotoUploader DIAG: TPS output range: X=[${minIX.toFixed(4)}, ${maxIX.toFixed(4)}], Y=[${minIY.toFixed(4)}, ${maxIY.toFixed(4)}]`);
    console.log(`PhotoUploader: TPS evaluated at ${nUV} UV vertices`);

    // --- 4. Rasterise all 9976 FLAME UV faces ---
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH);

    // ===== DIAGNOSTIC: Count mapped vs unmapped pixels =====
    let mappedPixels = 0;
    const totalPixels = size * size;
    for (let i = 0; i < totalPixels; i++) {
      if (outImageData.data[i * 4 + 3] === 255) mappedPixels++;
    }
    console.log(`PhotoUploader DIAG: Mapped ${mappedPixels}/${totalPixels} pixels (${(mappedPixels/totalPixels*100).toFixed(1)}%) before fill`);

    // ===== DIAGNOSTIC: Save pre-fill texture =====
    const preFillCanvas = document.createElement('canvas');
    preFillCanvas.width = size;
    preFillCanvas.height = size;
    preFillCanvas.getContext('2d').putImageData(outImageData, 0, 0);
    this._debugPreFillTexture = preFillCanvas.toDataURL('image/png');

    // --- 5. Fill unmapped areas ---
    this._fillUnmappedRegions(outImageData);
    ctx.putImageData(outImageData, 0, 0);

    console.log('PhotoUploader: ✅ TPS + mesh texture projection complete');

    // ===== DIAGNOSTIC: Save final texture =====
    this._debugFinalTexture = canvas.toDataURL('image/png');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.92),
      width: size,
      height: size
    };
  }

  // -----------------------------------------------------------------------
  // DIAGNOSTIC: Visual landmark overlays (Tests 1 & 2)
  // -----------------------------------------------------------------------

  /**
   * Draw debug overlays:
   * - Debug A: source photo with 105 landmark dots
   * - Debug B: 1024×1024 UV atlas with 105 UV control point dots
   */
  _drawDebugOverlays(srcCanvas, srcW, srcH, ctrlU, ctrlV, ctrlIX, ctrlIY, N) {
    // Known landmark labels for key points
    const KEY_LANDMARKS = {
      168: 'nose_tip', 6: 'nose_bridge', 33: 'R_eye_in', 263: 'L_eye_in',
      0: 'upper_lip', 17: 'lower_lip', 133: 'R_eye_out', 362: 'L_eye_out',
      70: 'R_brow', 300: 'L_brow'
    };

    // --- Debug A: Landmarks on source photo ---
    const photoDbg = document.createElement('canvas');
    photoDbg.width = 512;
    photoDbg.height = Math.round(512 * srcH / srcW);
    const pCtx = photoDbg.getContext('2d');
    pCtx.drawImage(srcCanvas, 0, 0, photoDbg.width, photoDbg.height);

    for (let i = 0; i < N; i++) {
      const px = ctrlIX[i] * photoDbg.width;
      const py = ctrlIY[i] * photoDbg.height;
      pCtx.fillStyle = i < 20 ? '#ff0000' : i < 40 ? '#00ff00' : i < 60 ? '#0088ff' : i < 80 ? '#ff8800' : '#ff00ff';
      pCtx.beginPath();
      pCtx.arc(px, py, 3, 0, Math.PI * 2);
      pCtx.fill();

      // Label key landmarks
      const mpIdx = this._lastDebugData?.mpIndices?.[i];
      if (mpIdx !== undefined && KEY_LANDMARKS[mpIdx]) {
        pCtx.fillStyle = '#ffffff';
        pCtx.font = '9px monospace';
        pCtx.fillText(KEY_LANDMARKS[mpIdx], px + 5, py - 3);
      }
    }
    this._debugPhotoLandmarks = photoDbg.toDataURL('image/png');

    // --- Debug B: UV control points on atlas ---
    const uvDbg = document.createElement('canvas');
    uvDbg.width = 512;
    uvDbg.height = 512;
    const uCtx = uvDbg.getContext('2d');

    // Background grid
    uCtx.fillStyle = '#1a1a2e';
    uCtx.fillRect(0, 0, 512, 512);
    uCtx.strokeStyle = '#333355';
    uCtx.lineWidth = 0.5;
    for (let g = 0; g <= 10; g++) {
      const p = g * 51.2;
      uCtx.beginPath(); uCtx.moveTo(p, 0); uCtx.lineTo(p, 512); uCtx.stroke();
      uCtx.beginPath(); uCtx.moveTo(0, p); uCtx.lineTo(512, p); uCtx.stroke();
    }
    // Axis labels
    uCtx.fillStyle = '#888';
    uCtx.font = '10px monospace';
    uCtx.fillText('U=0', 2, 510);
    uCtx.fillText('U=1', 490, 510);
    uCtx.fillText('V=0 (top=chin)', 2, 12);
    uCtx.fillText('V=1 (bot=scalp)', 2, 508);

    // Draw UV points (V mapped directly: v=0→top, v=1→bottom of canvas)
    for (let i = 0; i < N; i++) {
      const px = ctrlU[i] * 512;
      const py = ctrlV[i] * 512;
      uCtx.fillStyle = i < 20 ? '#ff0000' : i < 40 ? '#00ff00' : i < 60 ? '#0088ff' : i < 80 ? '#ff8800' : '#ff00ff';
      uCtx.beginPath();
      uCtx.arc(px, py, 3, 0, Math.PI * 2);
      uCtx.fill();

      const mpIdx = this._lastDebugData?.mpIndices?.[i];
      if (mpIdx !== undefined && KEY_LANDMARKS[mpIdx]) {
        uCtx.fillStyle = '#ffffff';
        uCtx.font = '9px monospace';
        uCtx.fillText(KEY_LANDMARKS[mpIdx], px + 5, py - 3);
      }
    }
    this._debugUVLandmarks = uvDbg.toDataURL('image/png');

    console.log('PhotoUploader DIAG: Debug overlays generated (photo + UV atlas)');
  }

  /**
   * Generate a labeled UV checkerboard texture for orientation testing.
   * Returns a data URL that can be loaded via renderer.loadTexture().
   */
  static generateUVCheckerboard() {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Checkerboard: 8x8 grid
    const cellSize = size / 8;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const isLight = (row + col) % 2 === 0;
        ctx.fillStyle = isLight ? '#dddddd' : '#444444';
        ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }

    // Label each corner quadrant with UV info
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    // Top-left corner of canvas → (U≈0, V≈0 in canvas space)
    ctx.fillStyle = '#ff0000';
    ctx.fillText('TL: U=0, V=0', size * 0.25, 40);
    ctx.fillText('(should=chin-right)', size * 0.25, 70);
    // Top-right → (U≈1, V≈0)
    ctx.fillStyle = '#00cc00';
    ctx.fillText('TR: U=1, V=0', size * 0.75, 40);
    ctx.fillText('(should=chin-left)', size * 0.75, 70);
    // Bottom-left → (U≈0, V≈1)
    ctx.fillStyle = '#0066ff';
    ctx.fillText('BL: U=0, V=1', size * 0.25, size - 20);
    ctx.fillText('(should=scalp-right)', size * 0.25, size - 50);
    // Bottom-right → (U≈1, V≈1)
    ctx.fillStyle = '#ff8800';
    ctx.fillText('BR: U=1, V=1', size * 0.75, size - 20);
    ctx.fillText('(should=scalp-left)', size * 0.75, size - 50);

    // Center crosshair
    ctx.strokeStyle = '#ff00ff';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(size/2, 0); ctx.lineTo(size/2, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, size/2); ctx.lineTo(size, size/2); ctx.stroke();
    ctx.fillStyle = '#ff00ff';
    ctx.fillText('CENTER', size/2, size/2 - 10);
    ctx.fillText('(should=nose)', size/2, size/2 + 30);

    // Red arrow pointing right = +U direction
    ctx.fillStyle = '#ff0000';
    ctx.font = 'bold 20px monospace';
    ctx.fillText('→ +U', size - 70, size / 2 - 5);
    // Green arrow pointing down = +V direction
    ctx.fillStyle = '#00cc00';
    ctx.fillText('↓ +V', size / 2 + 20, size - 80);

    return canvas.toDataURL('image/png');
  }

  // -----------------------------------------------------------------------
  // Dense Projection — camera-based texture baking (replaces TPS)
  // -----------------------------------------------------------------------

  /**
   * Project the photo onto FLAME UV space using Dense Projection.
   *
   * Instead of TPS interpolation from 105 sparse landmarks, this:
   * 1. Estimates a camera pose from 105 2D-3D landmark correspondences
   * 2. Projects ALL 5023 FLAME position vertices into image space
   * 3. Culls back-facing faces
   * 4. Rasterises visible faces using the existing mesh rasteriser
   * 5. Fills non-visible regions from the FLAME albedo texture
   */
  _denseProjectTextureToUV(img, landmarks, mapping, meshGen, perspParams = null) {
    const size = 2048;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // --- Source image ---
    const maxSrc = 2048;
    const sc = Math.min(1, maxSrc / Math.max(img.naturalWidth, img.naturalHeight));
    const srcW = Math.round(img.naturalWidth * sc);
    const srcH = Math.round(img.naturalHeight * sc);
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext('2d').drawImage(img, 0, 0, srcW, srcH);
    const srcData = srcCanvas.getContext('2d').getImageData(0, 0, srcW, srcH).data;

    // --- 1. Estimate camera from 105 2D-3D correspondences ---
    const cameraParams = this._estimateCameraFromLandmarks(landmarks, mapping, meshGen, perspParams);
    if (!cameraParams) {
      console.warn('PhotoUploader: Camera estimation failed');
      return null;
    }

    // --- 2. Project all position vertices to image space ---
    const projectedCoords = this._projectAllVertices(meshGen, cameraParams);

    // --- 3. Compute per-face visibility (backface culling) ---
    const faceVisibility = this._computeFaceVisibility(meshGen, cameraParams);

    // --- 4. Map position-vertex projections to UV-vertex coords ---
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;
    const uvImgCoords = this._mapProjectionToUV(projectedCoords, meshGen, faceVisibility);

    // --- Reuse diagnostic overlays from TPS path ---
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const ctrlU = [], ctrlV = [], ctrlIX = [], ctrlIY = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const uvi0 = uvFaces[fi * 3], uvi1 = uvFaces[fi * 3 + 1], uvi2 = uvFaces[fi * 3 + 2];
      const u = bc[0] * uvCoords[uvi0 * 2] + bc[1] * uvCoords[uvi1 * 2] + bc[2] * uvCoords[uvi2 * 2];
      const v = bc[0] * uvCoords[uvi0 * 2 + 1] + bc[1] * uvCoords[uvi1 * 2 + 1] + bc[2] * uvCoords[uvi2 * 2 + 1];
      ctrlU.push(u); ctrlV.push(v); ctrlIX.push(lm.x); ctrlIY.push(lm.y);
    }
    this._lastDebugData = {
      srcCanvas, srcW, srcH, img,
      ctrlU, ctrlV, ctrlIX, ctrlIY, N: ctrlU.length,
      landmarks, mpIndices, mapping
    };
    this._drawDebugOverlays(srcCanvas, srcW, srcH, ctrlU, ctrlV, ctrlIX, ctrlIY, ctrlU.length);

    // --- 4b. Build UV mask (valid UV footprint only) ---
    const uvMask = this._buildUVMask(size, uvCoords, uvFaces);

    // --- 4c. Build inner mouth exclusion mask ---
    const excludedFaces = this._buildInnerMouthMask(meshGen);

    // --- 4d. Build face boundary mask on source photo ---
    const faceBoundaryMask = this._buildFaceBoundaryMask(landmarks, srcW, srcH);

    // --- 5. Rasterise all visible faces (excluding inner mouth, face boundary masked) ---
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH, faceVisibility, excludedFaces, faceBoundaryMask);

    // --- 5a. Apply UV mask: clear pixels outside valid UV footprint ---
    this._applyUVMask(outImageData, size, uvMask);

    // --- Diagnostic: pixel coverage ---
    let mappedPixels = 0;
    const totalPixels = size * size;
    for (let i = 0; i < totalPixels; i++) {
      if (outImageData.data[i * 4 + 3] > 0) mappedPixels++;
    }
    console.log(`PhotoUploader DENSE: Mapped ${mappedPixels}/${totalPixels} pixels (${(mappedPixels / totalPixels * 100).toFixed(1)}%) before fill`);

    // --- 5b. Delight the texture: remove baked-in photo shading ---
    // Divides out low-frequency luminance so 3D lighting can create proper shadows.
    this._delightTexture(outImageData, size);

    // --- Save pre-fill texture for debug ---
    const preFillCanvas = document.createElement('canvas');
    preFillCanvas.width = size;
    preFillCanvas.height = size;
    preFillCanvas.getContext('2d').putImageData(outImageData, 0, 0);
    this._debugPreFillTexture = preFillCanvas.toDataURL('image/png');

    // --- 6. Mask-normalized Laplacian pyramid blending ---
    // Build full-coverage albedo layer, then blend with photo using 5-level pyramid
    const albedoLayer = this._prepareAlbedoLayer(outImageData, meshGen, size, landmarks, mapping);
    const blendedData = this._laplacianBlend(outImageData, albedoLayer, size, 5);
    ctx.putImageData(blendedData, 0, 0);

    console.log('PhotoUploader: Dense projection texture complete');
    this._debugFinalTexture = canvas.toDataURL('image/png');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.92),
      width: size,
      height: size
    };
  }

  /**
   * Estimate camera parameters from 105 2D-3D landmark correspondences.
   *
   * Uses weak-perspective fit:
   *   imgX ≈ s * Xrot + tx
   *   imgY ≈ s * Yrot + ty
   *
   * where Xrot,Yrot are rotated 3D coords (rotation from MediaPipe head pose).
   *
   * @returns {{ R: Float64Array, fx: number, fy: number, cx: number, cy: number,
   *             tx: number, ty: number, tz: number }} | null
   */
  _estimateCameraFromLandmarks(landmarks, mapping, meshGen, perspParams = null) {
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;

    // --- Extract 105 3D-2D correspondences ---
    const pts3d = [];
    const pts2d = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      const x = bc[0] * verts[pi0 * 3] + bc[1] * verts[pi1 * 3] + bc[2] * verts[pi2 * 3];
      const y = bc[0] * verts[pi0 * 3 + 1] + bc[1] * verts[pi1 * 3 + 1] + bc[2] * verts[pi2 * 3 + 1];
      const z = bc[0] * verts[pi0 * 3 + 2] + bc[1] * verts[pi1 * 3 + 2] + bc[2] * verts[pi2 * 3 + 2];

      pts3d.push([x, y, z]);
      pts2d.push([lm.x, lm.y]);
    }

    const N = pts3d.length;
    if (N < 10) return null;
    console.log(`PhotoUploader DENSE: ${N} 2D-3D correspondences for camera estimation`);

    // --- Get rotation from MediaPipe head pose ---
    const R = this._estimateRotationFromLandmarks(pts3d, pts2d);

    // --- Rotate all 3D points ---
    const xr = new Float64Array(N);
    const yr = new Float64Array(N);
    const zr = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const [x, y, z] = pts3d[i];
      xr[i] = R[0] * x + R[1] * y + R[2] * z;
      yr[i] = R[3] * x + R[4] * y + R[5] * z;
      zr[i] = R[6] * x + R[7] * y + R[8] * z;
    }

    // --- Weak-perspective least squares with separate X/Y scales ---
    // imgX = sx * xr + tx  (2x2 system)
    // imgY = sy * yr + ty  (2x2 system, separate — handles Y-flip naturally)

    // Solve X: [xr_i  1] [sx] = [imgX_i]
    let Axx = 0, Ax1 = 0, A11x = 0, bx0 = 0, bx1 = 0;
    for (let i = 0; i < N; i++) {
      Axx += xr[i] * xr[i];
      Ax1 += xr[i];
      bx0 += xr[i] * pts2d[i][0];
      bx1 += pts2d[i][0];
    }
    A11x = N;
    const detX = Axx * A11x - Ax1 * Ax1;
    if (Math.abs(detX) < 1e-15) return null;
    const sx = (bx0 * A11x - bx1 * Ax1) / detX;
    const tx = (Axx * bx1 - Ax1 * bx0) / detX;

    // Solve Y: [yr_i  1] [sy] = [imgY_i]
    let Ayy = 0, Ay1 = 0, A11y = 0, by0 = 0, by1 = 0;
    for (let i = 0; i < N; i++) {
      Ayy += yr[i] * yr[i];
      Ay1 += yr[i];
      by0 += yr[i] * pts2d[i][1];
      by1 += pts2d[i][1];
    }
    A11y = N;
    const detY = Ayy * A11y - Ay1 * Ay1;
    if (Math.abs(detY) < 1e-15) return null;
    const sy = (by0 * A11y - by1 * Ay1) / detY;
    const ty = (Ayy * by1 - Ay1 * by0) / detY;

    if (!isFinite(sx) || !isFinite(sy) || Math.abs(sx) < 1e-6) return null;

    console.log(`PhotoUploader DENSE: Weak perspective: sx=${sx.toFixed(4)}, sy=${sy.toFixed(4)}, tx=${tx.toFixed(4)}, ty=${ty.toFixed(4)}`);

    // --- Convert to perspective parameters ---
    let meanZ = 0;
    for (let i = 0; i < N; i++) meanZ += zr[i];
    meanZ /= N;
    // Use IPD-estimated Z_face if available; otherwise fall back to heuristic
    const tz = (perspParams && perspParams.Z_face) ? perspParams.Z_face : Math.max(Math.abs(meanZ), 0.01);
    // sx ≈ fx / tz  →  fx = sx * tz
    const fx = sx * tz;
    const fy = sy * tz;
    if (perspParams && perspParams.Z_face) {
      console.log(`PhotoUploader DENSE: Using perspective Z_face=${tz.toFixed(3)}m (IPD-estimated)`);
    }

    // --- Validate: reprojection error using weak perspective model ---
    let totalErr = 0;
    for (let i = 0; i < N; i++) {
      const predX = sx * xr[i] + tx;
      const predY = sy * yr[i] + ty;
      const ex = predX - pts2d[i][0];
      const ey = predY - pts2d[i][1];
      totalErr += Math.sqrt(ex * ex + ey * ey);
    }
    const meanErr = totalErr / N;
    console.log(`PhotoUploader DENSE: Mean reprojection error = ${meanErr.toFixed(6)} (${N} points)`);

    if (meanErr > 0.1) {
      console.warn('PhotoUploader DENSE: Reprojection error too high, falling back');
      return null;
    }

    return { R, fx, fy, tx, ty, tz, meanZ, sx, sy };
  }

  /**
   * Estimate rotation matrix from 2D-3D correspondences using Procrustes alignment.
   * Aligns the centred 3D X,Y to the centred 2D positions, then computes R.
   * @returns {Float64Array} 3x3 rotation matrix (row-major)
   */
  _estimateRotationFromLandmarks(pts3d, pts2d) {
    const N = pts3d.length;

    // Centre both point sets
    let cx3 = 0, cy3 = 0, cz3 = 0, cx2 = 0, cy2 = 0;
    for (let i = 0; i < N; i++) {
      cx3 += pts3d[i][0]; cy3 += pts3d[i][1]; cz3 += pts3d[i][2];
      cx2 += pts2d[i][0]; cy2 += pts2d[i][1];
    }
    cx3 /= N; cy3 /= N; cz3 /= N; cx2 /= N; cy2 /= N;

    // Build cross-covariance matrix H (2x3) between centred 2D and centred 3D
    // We only have 2D observations, so we fit the X,Y projection
    // Compute an approximate yaw from the z-depth variation
    let H00 = 0, H01 = 0, H02 = 0;
    let H10 = 0, H11 = 0, H12 = 0;
    for (let i = 0; i < N; i++) {
      const dx2 = pts2d[i][0] - cx2;
      const dy2 = pts2d[i][1] - cy2;
      const dx3 = pts3d[i][0] - cx3;
      const dy3 = pts3d[i][1] - cy3;
      const dz3 = pts3d[i][2] - cz3;
      H00 += dx2 * dx3; H01 += dx2 * dy3; H02 += dx2 * dz3;
      H10 += dy2 * dx3; H11 += dy2 * dy3; H12 += dy2 * dz3;
    }

    // Estimate yaw from the ratio of z-correlation to x-correlation
    const yaw = Math.atan2(H02, H00 + 1e-10);
    // Estimate pitch from the ratio of z-correlation to y-correlation
    const pitch = Math.atan2(-H12, -(H11 + 1e-10));
    // Estimate roll from the skew between x and y
    const roll = Math.atan2(H10, H00 + 1e-10);

    // Clamp to reasonable ranges for near-frontal faces
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const yawC = clamp(yaw, -0.8, 0.8);
    const pitchC = clamp(pitch, -0.6, 0.6);
    const rollC = clamp(roll, -0.4, 0.4);

    return this._eulerToRotationMatrix(pitchC, yawC, rollC);
  }

  /**
   * Euler angles (pitch, yaw, roll) to 3x3 rotation matrix (row-major).
   * Convention: R = Ry(yaw) * Rx(pitch) * Rz(roll)
   * @returns {Float64Array} 9-element row-major rotation matrix
   */
  _eulerToRotationMatrix(pitch, yaw, roll) {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cr = Math.cos(roll), sr = Math.sin(roll);

    return new Float64Array([
      cy * cr + sy * sp * sr,   -cy * sr + sy * sp * cr,  sy * cp,
      cp * sr,                   cp * cr,                  -sp,
      -sy * cr + cy * sp * sr,   sy * sr + cy * sp * cr,   cy * cp,
    ]);
  }

  // -----------------------------------------------------------------------
  // FLAME shape fitting — landmark-only PCA optimisation
  // -----------------------------------------------------------------------

  /**
   * Solve A·x = b for symmetric positive-definite A via Cholesky decomposition.
   * @param {Float64Array} A - n×n SPD matrix (row-major, overwritten with L)
   * @param {Float64Array} b - n×1 right-hand side
   * @param {number} n      - dimension
   * @returns {Float64Array} n×1 solution vector x
   */
  _solveCholesky(A, b, n) {
    // Cholesky: A = L * L^T  (in-place, lower triangle)
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += A[j * n + k] * A[j * n + k];
      const diag = A[j * n + j] - sum;
      if (diag <= 0) {
        // Not SPD — add small ridge and retry
        console.warn('Cholesky: matrix not SPD, adding ridge');
        for (let i = 0; i < n; i++) A[i * n + i] += 1e-6;
        return this._solveCholesky(A, b, n);
      }
      A[j * n + j] = Math.sqrt(diag);
      for (let i = j + 1; i < n; i++) {
        let s = 0;
        for (let k = 0; k < j; k++) s += A[i * n + k] * A[j * n + k];
        A[i * n + j] = (A[i * n + j] - s) / A[j * n + j];
      }
    }

    // Forward substitution: L · y = b
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < i; k++) s += A[i * n + k] * y[k];
      y[i] = (b[i] - s) / A[i * n + i];
    }

    // Back substitution: L^T · x = y
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = 0;
      for (let k = i + 1; k < n; k++) s += A[k * n + i] * x[k];
      x[i] = (y[i] - s) / A[i * n + i];
    }

    return x;
  }

  /**
   * Estimate focal length and camera-to-face distance from inter-pupillary distance.
   *
   * Uses MediaPipe iris landmarks (468, 473) to measure pixel IPD, then
   * combines with FLAME mesh IPD to estimate the camera distance (Z_face).
   * Assumes a typical smartphone focal length (~26-32mm equiv).
   *
   * @param {Array} landmarks - MediaPipe 478 landmarks (normalised 0-1)
   * @param {number} imgW - image width in pixels
   * @param {number} imgH - image height in pixels
   * @param {object} meshGen - FlameMeshGenerator instance
   * @returns {{ fxNorm: number, fyNorm: number, Z_face: number }}
   */
  _estimateFocalLength(landmarks, imgW, imgH, meshGen) {
    // 1. Measure IPD from MediaPipe iris landmarks (in normalised coords)
    let lmL = landmarks[473]; // left iris center
    let lmR = landmarks[468]; // right iris center

    // Fallback to outer eye corners if iris not available
    if (!lmL || isNaN(lmL.x) || !lmR || isNaN(lmR.x)) {
      lmL = landmarks[263]; // left eye outer
      lmR = landmarks[33];  // right eye outer
    }
    if (!lmL || isNaN(lmL.x) || !lmR || isNaN(lmR.x)) {
      console.warn('Focal length: No eye landmarks, using default Z_face=0.5');
      return { fxNorm: 1.0, fyNorm: -1.0, Z_face: 0.5 };
    }

    const ipdPixX = (lmL.x - lmR.x) * imgW;
    const ipdPixY = (lmL.y - lmR.y) * imgH;
    const ipdPixels = Math.sqrt(ipdPixX * ipdPixX + ipdPixY * ipdPixY);

    if (ipdPixels < 5) {
      console.warn('Focal length: IPD too small, using default Z_face=0.5');
      return { fxNorm: 1.0, fyNorm: -1.0, Z_face: 0.5 };
    }

    // 2. Measure FLAME mesh IPD from eye region centroids
    const regions = meshGen._flameRegions;
    const verts = meshGen._flameShapedVertices ?? meshGen._flameCurrentVertices ?? meshGen._flameTemplateVertices;
    let ipdFlame = 0.063; // default anatomical IPD (63mm)

    if (regions && verts) {
      const leftEye = regions['eye_left_upper'] || regions['eye_left_lower'];
      const rightEye = regions['eye_right_upper'] || regions['eye_right_lower'];
      if (leftEye && rightEye) {
        let lx = 0, ly = 0, lz = 0, ln = 0;
        let rx = 0, ry = 0, rz = 0, rn = 0;
        for (const vi of leftEye) {
          lx += verts[vi * 3]; ly += verts[vi * 3 + 1]; lz += verts[vi * 3 + 2]; ln++;
        }
        for (const vi of rightEye) {
          rx += verts[vi * 3]; ry += verts[vi * 3 + 1]; rz += verts[vi * 3 + 2]; rn++;
        }
        if (ln > 0 && rn > 0) {
          const dx = lx / ln - rx / rn;
          const dy = ly / ln - ry / rn;
          const dz = lz / ln - rz / rn;
          ipdFlame = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
    }

    // 3. Estimate focal length and distance
    // Assume fx_pixels ≈ max(imgW, imgH) (typical smartphone ~26-32mm equiv)
    const fxPixels = Math.max(imgW, imgH);
    const Z_face = Math.max(0.15, Math.min(1.5, fxPixels * ipdFlame / ipdPixels));

    // Normalised focal length (in [0,1] image coordinate space)
    const fxNorm = fxPixels / imgW;
    const fyNorm = -(fxPixels / imgH); // negative for Y-flip (image Y is downward)

    console.log(`Focal length: IPD_px=${ipdPixels.toFixed(1)}, IPD_3d=${(ipdFlame * 1000).toFixed(1)}mm, ` +
      `fx=${fxPixels}px, Z_face=${Z_face.toFixed(3)}m`);

    return { fxNorm, fyNorm, Z_face };
  }

  /**
   * Estimate camera for shape/expression fitting.
   *
   * When perspective params (Z_face, fxNorm, fyNorm) are provided, uses a pinhole
   * camera model. Otherwise falls back to isotropic weak-perspective.
   *
   * Perspective model:  imgX = fxNorm * (R·p).x / ((R·p).z + Z_face) + cx
   *                     imgY = fyNorm * (R·p).y / ((R·p).z + Z_face) + cy
   *
   * Weak-perspective:   imgX =  s * (R·p).x + tx
   *                     imgY = -s * (R·p).y + ty
   */
  _estimateShapeFitCamera(lmk3D, pts2D, Z_face = null, fxNorm = null, fyNorm = null) {
    const N = lmk3D.length;

    // Estimate rotation via Procrustes (reuse existing method)
    let R = this._estimateRotationFromLandmarks(lmk3D, pts2D);

    // --- Perspective mode ---
    if (Z_face != null && fxNorm != null && fyNorm != null) {
      // Iterative depth-corrected Procrustes (3 refinement iterations)
      for (let pIter = 0; pIter < 3; pIter++) {
        // For each point, compute per-vertex depth and undo perspective
        const pts3dCorrected = [];
        const pts2dCorrected = [];
        for (let i = 0; i < N; i++) {
          const [x, y, z] = lmk3D[i];
          const zr = R[6] * x + R[7] * y + R[8] * z;
          const Zk = zr + Z_face;
          if (Zk < 0.01) continue;
          // "Undo" perspective: scale 2D obs by Zk/fx to get 3D-like X
          pts3dCorrected.push(lmk3D[i]);
          pts2dCorrected.push(pts2D[i]);
        }
        if (pts3dCorrected.length >= 10) {
          R = this._estimateRotationFromLandmarks(pts3dCorrected, pts2dCorrected);
        }
      }

      // Compute perspective projection for all points and solve for cx, cy
      const xr = new Float64Array(N);
      const yr = new Float64Array(N);
      const zr = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const [x, y, z] = lmk3D[i];
        xr[i] = R[0] * x + R[1] * y + R[2] * z;
        yr[i] = R[3] * x + R[4] * y + R[5] * z;
        zr[i] = R[6] * x + R[7] * y + R[8] * z;
      }

      // Solve for cx, cy: obsX_i = fxNorm * xr_i / (zr_i + Z_face) + cx
      let cxSum = 0, cySum = 0, cnt = 0;
      for (let i = 0; i < N; i++) {
        const Zk = zr[i] + Z_face;
        if (Zk < 0.01) continue;
        cxSum += pts2D[i][0] - fxNorm * xr[i] / Zk;
        cySum += pts2D[i][1] - fyNorm * yr[i] / Zk;
        cnt++;
      }
      const cx = cnt > 0 ? cxSum / cnt : 0.5;
      const cy = cnt > 0 ? cySum / cnt : 0.5;

      return { R, sx: fxNorm, sy: fyNorm, tx: cx, ty: cy, Z_face, perspective: true };
    }

    // --- Weak-perspective fallback (original code) ---
    const xr = new Float64Array(N);
    const yr = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const [x, y, z] = lmk3D[i];
      xr[i] = R[0] * x + R[1] * y + R[2] * z;
      yr[i] = R[3] * x + R[4] * y + R[5] * z;
    }

    // Isotropic scale: solve for single |s|, tx, ty jointly
    let A00 = 0, A01 = 0, A02 = 0;
    let A11 = 0, A12 = 0, A22 = 0;
    let b0 = 0, b1 = 0, b2 = 0;

    for (let i = 0; i < N; i++) {
      const xi = xr[i], yi = -yr[i];
      const obsX = pts2D[i][0], obsY = pts2D[i][1];
      A00 += xi * xi; A01 += xi; b0 += xi * obsX; b1 += obsX;
      A00 += yi * yi; A02 += yi; A22 += 1; b0 += yi * obsY; b2 += obsY;
      A11 += 1;
    }

    const det = A00 * (A11 * A22 - A12 * A12)
              - A01 * (A01 * A22 - A12 * A02)
              + A02 * (A01 * A12 - A11 * A02);
    if (Math.abs(det) < 1e-15) return null;

    const invDet = 1 / det;
    const s  = invDet * (b0 * (A11 * A22 - A12 * A12) - A01 * (b1 * A22 - A12 * b2) + A02 * (b1 * A12 - A11 * b2));
    const tx = invDet * (A00 * (b1 * A22 - A12 * b2) - b0 * (A01 * A22 - A12 * A02) + A02 * (A01 * b2 - b1 * A02));
    const ty = invDet * (A00 * (A11 * b2 - b1 * A12) - A01 * (A01 * b2 - b1 * A02) + b0 * (A01 * A12 - A11 * A02));

    if (!isFinite(s) || Math.abs(s) < 1e-6) return null;

    return { R, sx: s, sy: -s, tx, ty };
  }

  /**
   * Fit FLAME shape PCA parameters (β) to 105 MediaPipe landmarks.
   *
   * Uses alternating linear least-squares:
   *   1) Fix β → estimate camera (R, sx, sy, tx, ty)
   *   2) Fix camera → solve for β via regularised normal equations
   * Repeats for 4 iterations. All sub-steps are closed-form (no gradient descent).
   *
   * @param {Array} landmarks  - MediaPipe 478 landmarks
   * @param {object} mapping   - flameMapping { landmark_indices, lmk_face_idx, lmk_b_coords }
   * @param {object} meshGen   - FlameMeshGenerator instance
   * @returns {Float64Array} fitted shape parameters (20 components)
   */
  _fitShapeFromLandmarks(landmarks, mapping, meshGen, perspParams = null) {
    const NUM_COMPONENTS = 20;
    const NUM_ITERATIONS = 5;
    const LAMBDA = 0.01;

    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const templateVerts = meshGen._flameTemplateVertices ?? meshGen.flameTemplateVertices;
    const shapeBasis = meshGen._flameShapeBasis;
    const { shapeComponents } = meshGen._flameMeta;

    if (!shapeBasis || !templateVerts) {
      console.warn('Shape fitting: FLAME data not available');
      return null;
    }

    const C = Math.min(NUM_COMPONENTS, shapeComponents);

    // --- Extract valid 2D-3D landmark correspondences ---
    const validIndices = []; // indices into mpIndices
    const pts2D = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      validIndices.push(i);
      pts2D.push([lm.x, lm.y]);
    }

    const K = validIndices.length;
    if (K < 10) {
      console.warn(`Shape fitting: Only ${K} valid landmarks, need ≥10`);
      return null;
    }
    console.log(`Shape fitting: ${K} valid landmarks, ${C} PCA components`);

    // --- Pre-compute per-landmark template positions and shape basis Jacobian ---
    // templateLmk[k] = barycentric interpolation of template vertices at landmark k
    // J[k][coord][c] = barycentric interpolation of shapeBasis for component c at landmark k
    const templateLmk = new Array(K); // K × 3
    const J = new Array(K);           // K × 3 × C

    for (let ki = 0; ki < K; ki++) {
      const i = validIndices[ki];
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      // Template position (barycentric interpolation)
      const tx = bc[0] * templateVerts[pi0 * 3]     + bc[1] * templateVerts[pi1 * 3]     + bc[2] * templateVerts[pi2 * 3];
      const ty = bc[0] * templateVerts[pi0 * 3 + 1] + bc[1] * templateVerts[pi1 * 3 + 1] + bc[2] * templateVerts[pi2 * 3 + 1];
      const tz = bc[0] * templateVerts[pi0 * 3 + 2] + bc[1] * templateVerts[pi1 * 3 + 2] + bc[2] * templateVerts[pi2 * 3 + 2];
      templateLmk[ki] = [tx, ty, tz];

      // Shape basis Jacobian: J[k][coord][c] = interpolated basis for this landmark
      // basis layout: basis[(vertex*3 + coord) * shapeComponents + component]
      const Jk = [new Float64Array(C), new Float64Array(C), new Float64Array(C)]; // x, y, z
      for (let c = 0; c < C; c++) {
        for (let coord = 0; coord < 3; coord++) {
          Jk[coord][c] =
            bc[0] * shapeBasis[(pi0 * 3 + coord) * shapeComponents + c] +
            bc[1] * shapeBasis[(pi1 * 3 + coord) * shapeComponents + c] +
            bc[2] * shapeBasis[(pi2 * 3 + coord) * shapeComponents + c];
        }
      }
      J[ki] = Jk;
    }

    // --- Alternating optimisation ---
    let beta = new Float64Array(C); // start from mean shape

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      // 1. Compute current 3D landmark positions: lmk3D[k] = templateLmk[k] + J[k] · β
      const lmk3D = new Array(K);
      for (let ki = 0; ki < K; ki++) {
        const p = [templateLmk[ki][0], templateLmk[ki][1], templateLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * beta[c];
          p[1] += J[ki][1][c] * beta[c];
          p[2] += J[ki][2][c] * beta[c];
        }
        lmk3D[ki] = p;
      }

      // 2. Estimate camera (fix β)
      const cam = this._estimateShapeFitCamera(lmk3D, pts2D,
        perspParams?.Z_face, perspParams?.fxNorm, perspParams?.fyNorm);
      if (!cam) {
        console.warn(`Shape fitting: Camera estimation failed at iter ${iter}`);
        break;
      }
      const { R, sx, sy, tx, ty, Z_face: camZ, perspective: isPersp } = cam;

      // 3. Build design matrix M (2K × C) and target vector d (2K × 1)
      const rows = 2 * K;
      const M = new Float64Array(rows * C);
      const d = new Float64Array(rows);

      for (let ki = 0; ki < K; ki++) {
        if (isPersp) {
          // --- Perspective Gauss-Newton: linearise around current lmk3D ---
          const [px, py, pz] = lmk3D[ki]; // current position (template + J·β)
          const xr = R[0] * px + R[1] * py + R[2] * pz;
          const yr = R[3] * px + R[4] * py + R[5] * pz;
          const zr = R[6] * px + R[7] * py + R[8] * pz;
          const Zk = zr + camZ;
          const Zk2 = Zk * Zk;

          // Residual: observed - perspective projection of current position
          d[2 * ki]     = pts2D[ki][0] - (sx * xr / Zk + tx);
          d[2 * ki + 1] = pts2D[ki][1] - (sy * yr / Zk + ty);

          // Perspective Jacobian (quotient rule: d/dβ [f·x/(z+Z)] = f·(Z·dx - x·dz)/Z²)
          for (let c = 0; c < C; c++) {
            const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
            const Jx_cam = R[0] * jx + R[1] * jy + R[2] * jz;
            const Jy_cam = R[3] * jx + R[4] * jy + R[5] * jz;
            const Jz_cam = R[6] * jx + R[7] * jy + R[8] * jz;
            M[(2 * ki) * C + c]     = sx * (Zk * Jx_cam - xr * Jz_cam) / Zk2;
            M[(2 * ki + 1) * C + c] = sy * (Zk * Jy_cam - yr * Jz_cam) / Zk2;
          }
        } else {
          // --- Weak-perspective: linear in β ---
          const [tmx, tmy, tmz] = templateLmk[ki];
          const txr = R[0] * tmx + R[1] * tmy + R[2] * tmz;
          const tyr = R[3] * tmx + R[4] * tmy + R[5] * tmz;

          d[2 * ki]     = pts2D[ki][0] - (sx * txr + tx);
          d[2 * ki + 1] = pts2D[ki][1] - (sy * tyr + ty);

          for (let c = 0; c < C; c++) {
            const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
            M[(2 * ki) * C + c]     = sx * (R[0] * jx + R[1] * jy + R[2] * jz);
            M[(2 * ki + 1) * C + c] = sy * (R[3] * jx + R[4] * jy + R[5] * jz);
          }
        }
      }

      // 4. Normal equations: (M^T M + λΛ) β = M^T d
      // Build M^T M (C × C)
      const MtM = new Float64Array(C * C);
      const Mtd = new Float64Array(C);
      for (let a = 0; a < C; a++) {
        for (let b = a; b < C; b++) {
          let sum = 0;
          for (let r = 0; r < rows; r++) sum += M[r * C + a] * M[r * C + b];
          MtM[a * C + b] = sum;
          MtM[b * C + a] = sum; // symmetric
        }
        let rhs = 0;
        for (let r = 0; r < rows; r++) rhs += M[r * C + a] * d[r];
        Mtd[a] = rhs;
      }

      // Add Tikhonov regularisation: λ * Λ where Λ[c,c] = 2^(c/5)
      for (let c = 0; c < C; c++) {
        MtM[c * C + c] += LAMBDA * Math.pow(2, c / 5);
      }

      // 5. Solve via Cholesky
      if (isPersp) {
        // Gauss-Newton: solve for δβ (increment), since perspective is nonlinear
        const deltaBeta = this._solveCholesky(MtM, Mtd, C);
        for (let c = 0; c < C; c++) beta[c] += deltaBeta[c];
      } else {
        // Weak-perspective: solve for full β (problem is linear)
        beta = this._solveCholesky(MtM, Mtd, C);
      }

      // Log progress
      let reproj = 0;
      for (let ki = 0; ki < K; ki++) {
        const p = [templateLmk[ki][0], templateLmk[ki][1], templateLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * beta[c];
          p[1] += J[ki][1][c] * beta[c];
          p[2] += J[ki][2][c] * beta[c];
        }
        if (isPersp) {
          const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
          const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
          const zrP = R[6] * p[0] + R[7] * p[1] + R[8] * p[2];
          const Zk = zrP + camZ;
          const ex = (sx * xrP / Zk + tx) - pts2D[ki][0];
          const ey = (sy * yrP / Zk + ty) - pts2D[ki][1];
          reproj += Math.sqrt(ex * ex + ey * ey);
        } else {
          const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
          const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
          const ex = (sx * xrP + tx) - pts2D[ki][0];
          const ey = (sy * yrP + ty) - pts2D[ki][1];
          reproj += Math.sqrt(ex * ex + ey * ey);
        }
      }
      console.log(`Shape fitting: iter ${iter}, reproj error = ${(reproj / K).toFixed(6)}${isPersp ? ' (perspective)' : ''}`);
    }

    // --- Clamp β to valid range ---
    for (let c = 0; c < C; c++) {
      const limit = c < 10 ? 3.0 : 2.0;
      beta[c] = Math.max(-limit, Math.min(limit, beta[c]));
    }

    // --- Apply to mesh ---
    const shapeParams = new Array(shapeComponents).fill(0);
    for (let c = 0; c < C; c++) shapeParams[c] = beta[c];
    meshGen.applyShapeParams(shapeParams);

    console.log(`Shape fitting: β[0..4] = [${beta.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
    console.log('Shape fitting: Applied shape params to mesh');

    return beta;
  }

  /**
   * Fit FLAME expression PCA parameters (ε) to 105 MediaPipe landmarks.
   *
   * Same alternating least-squares as shape fitting, but:
   *   - Uses expression basis (not shape basis)
   *   - Starts from shaped vertices (not template)
   *   - Fits 10 components with higher regularisation (λ=0.1)
   *   - Adds symmetry penalty to prevent lopsided expressions
   *
   * Must be called AFTER _fitShapeFromLandmarks().
   */
  _fitExpressionFromLandmarks(landmarks, mapping, meshGen, perspParams = null) {
    const NUM_COMPONENTS = 20;
    const NUM_ITERATIONS = 5;
    const LAMBDA = 0.05;
    const LAMBDA_SYM = 0.05;
    const W_LIP = 3.0; // Weight boost for inner lip landmarks

    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    // Use shaped vertices (after shape fitting) as the base
    const shapedVerts = meshGen._flameShapedVertices ?? meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const exprBasis = meshGen._flameExpressionBasis;
    const { expressionComponents } = meshGen._flameMeta;

    if (!exprBasis || !shapedVerts) {
      console.warn('Expression fitting: FLAME expression data not available');
      return null;
    }

    const C = Math.min(NUM_COMPONENTS, expressionComponents);

    // --- Extract valid 2D-3D landmark correspondences ---
    const validIndices = [];
    const pts2D = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      validIndices.push(i);
      pts2D.push([lm.x, lm.y]);
    }

    const K = validIndices.length;
    if (K < 10) {
      console.warn(`Expression fitting: Only ${K} valid landmarks, need ≥10`);
      return null;
    }
    console.log(`Expression fitting: ${K} landmarks, ${C} PCA components`);

    // --- Pre-compute per-landmark shaped positions and expression Jacobian ---
    const baseLmk = new Array(K); // K × 3 (shaped vertex positions at landmarks)
    const J = new Array(K);       // K × 3 × C (expression basis Jacobian)

    for (let ki = 0; ki < K; ki++) {
      const i = validIndices[ki];
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      // Shaped position (barycentric interpolation on shaped vertices)
      const bx = bc[0] * shapedVerts[pi0 * 3]     + bc[1] * shapedVerts[pi1 * 3]     + bc[2] * shapedVerts[pi2 * 3];
      const by = bc[0] * shapedVerts[pi0 * 3 + 1] + bc[1] * shapedVerts[pi1 * 3 + 1] + bc[2] * shapedVerts[pi2 * 3 + 1];
      const bz = bc[0] * shapedVerts[pi0 * 3 + 2] + bc[1] * shapedVerts[pi1 * 3 + 2] + bc[2] * shapedVerts[pi2 * 3 + 2];
      baseLmk[ki] = [bx, by, bz];

      // Expression Jacobian (same layout as shape basis)
      const Jk = [new Float64Array(C), new Float64Array(C), new Float64Array(C)];
      for (let c = 0; c < C; c++) {
        for (let coord = 0; coord < 3; coord++) {
          Jk[coord][c] =
            bc[0] * exprBasis[(pi0 * 3 + coord) * expressionComponents + c] +
            bc[1] * exprBasis[(pi1 * 3 + coord) * expressionComponents + c] +
            bc[2] * exprBasis[(pi2 * 3 + coord) * expressionComponents + c];
        }
      }
      J[ki] = Jk;
    }

    // --- Build symmetry pairs for regularisation ---
    // Find left/right landmark pairs by checking X-coordinate symmetry
    // Landmarks with similar Y but opposite X are symmetric pairs
    const symPairs = [];
    const used = new Set();
    for (let a = 0; a < K; a++) {
      if (used.has(a)) continue;
      const xa = baseLmk[a][0], ya = baseLmk[a][1];
      if (Math.abs(xa) < 0.005) continue; // Skip midline landmarks
      let bestB = -1, bestDist = 0.01;
      for (let b = a + 1; b < K; b++) {
        if (used.has(b)) continue;
        const xb = baseLmk[b][0], yb = baseLmk[b][1];
        // Mirror: similar Y, opposite X
        const dist = Math.abs(xa + xb) + Math.abs(ya - yb);
        if (dist < bestDist) { bestDist = dist; bestB = b; }
      }
      if (bestB >= 0) {
        symPairs.push([a, bestB]);
        used.add(a);
        used.add(bestB);
      }
    }
    console.log(`Expression fitting: ${symPairs.length} symmetric landmark pairs`);

    // --- Identify inner lip landmarks for weighting ---
    // MediaPipe 13 (upper lip top), 14 (lower lip bottom), 78, 308 (lip corners)
    const LIP_MP_INDICES = new Set([13, 14, 78, 308]);
    const lipKiSet = new Set();
    for (let ki = 0; ki < K; ki++) {
      const mpIdx = mpIndices[validIndices[ki]];
      if (LIP_MP_INDICES.has(mpIdx)) lipKiSet.add(ki);
    }

    // --- Detect closed mouth for conditional clamping ---
    const lipTop = landmarks[13], lipBot = landmarks[14];
    const lipGapNorm = Math.abs(lipBot.y - lipTop.y); // normalized [0-1]
    const lipGapPx = lipGapNorm * Math.max(1, (this._lastImgH || 512));
    const mouthClosed = lipGapPx < 8;
    console.log(`Expression fitting: lip gap = ${lipGapPx.toFixed(1)}px, mouth ${mouthClosed ? 'CLOSED' : 'OPEN'}`);

    // --- Alternating optimisation ---
    let epsilon = new Float64Array(C);

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      // 1. Compute current 3D landmark positions: lmk3D = baseLmk + J · ε
      const lmk3D = new Array(K);
      for (let ki = 0; ki < K; ki++) {
        const p = [baseLmk[ki][0], baseLmk[ki][1], baseLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * epsilon[c];
          p[1] += J[ki][1][c] * epsilon[c];
          p[2] += J[ki][2][c] * epsilon[c];
        }
        lmk3D[ki] = p;
      }

      // 2. Estimate camera (fix ε)
      const cam = this._estimateShapeFitCamera(lmk3D, pts2D,
        perspParams?.Z_face, perspParams?.fxNorm, perspParams?.fyNorm);
      if (!cam) {
        console.warn(`Expression fitting: Camera failed at iter ${iter}`);
        break;
      }
      const { R, sx, sy, tx, ty, Z_face: camZ, perspective: isPersp } = cam;

      // 3. Build design matrix M and target vector d
      const rows = 2 * K;
      const M = new Float64Array(rows * C);
      const d = new Float64Array(rows);

      for (let ki = 0; ki < K; ki++) {
        // Inner lip landmarks get higher weight to prevent mouth-opening artifacts
        const w = lipKiSet.has(ki) ? W_LIP : 1.0;

        if (isPersp) {
          // --- Perspective Gauss-Newton: linearise around current lmk3D ---
          const [px, py, pz] = lmk3D[ki]; // current position (base + J·ε)
          const xr = R[0] * px + R[1] * py + R[2] * pz;
          const yr = R[3] * px + R[4] * py + R[5] * pz;
          const zr = R[6] * px + R[7] * py + R[8] * pz;
          const Zk = zr + camZ;
          const Zk2 = Zk * Zk;

          d[2 * ki]     = w * (pts2D[ki][0] - (sx * xr / Zk + tx));
          d[2 * ki + 1] = w * (pts2D[ki][1] - (sy * yr / Zk + ty));

          for (let c = 0; c < C; c++) {
            const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
            const Jx_cam = R[0] * jx + R[1] * jy + R[2] * jz;
            const Jy_cam = R[3] * jx + R[4] * jy + R[5] * jz;
            const Jz_cam = R[6] * jx + R[7] * jy + R[8] * jz;
            M[(2 * ki) * C + c]     = w * sx * (Zk * Jx_cam - xr * Jz_cam) / Zk2;
            M[(2 * ki + 1) * C + c] = w * sy * (Zk * Jy_cam - yr * Jz_cam) / Zk2;
          }
        } else {
          // --- Weak-perspective: linear in ε ---
          const [bmx, bmy, bmz] = baseLmk[ki];
          const bxr = R[0] * bmx + R[1] * bmy + R[2] * bmz;
          const byr = R[3] * bmx + R[4] * bmy + R[5] * bmz;

          d[2 * ki]     = w * (pts2D[ki][0] - (sx * bxr + tx));
          d[2 * ki + 1] = w * (pts2D[ki][1] - (sy * byr + ty));

          for (let c = 0; c < C; c++) {
            const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
            M[(2 * ki) * C + c]     = w * sx * (R[0] * jx + R[1] * jy + R[2] * jz);
            M[(2 * ki + 1) * C + c] = w * sy * (R[3] * jx + R[4] * jy + R[5] * jz);
          }
        }
      }

      // 4. Normal equations: (M^T M + λΛ + λ_sym S) ε = M^T d
      const MtM = new Float64Array(C * C);
      const Mtd = new Float64Array(C);
      for (let a = 0; a < C; a++) {
        for (let b = a; b < C; b++) {
          let sum = 0;
          for (let r = 0; r < rows; r++) sum += M[r * C + a] * M[r * C + b];
          MtM[a * C + b] = sum;
          MtM[b * C + a] = sum;
        }
        let rhs = 0;
        for (let r = 0; r < rows; r++) rhs += M[r * C + a] * d[r];
        Mtd[a] = rhs;
      }

      // Tikhonov: λ * 2^(c/5) (match shape fitting growth rate)
      for (let c = 0; c < C; c++) {
        MtM[c * C + c] += LAMBDA * Math.pow(2, c / 5);
      }

      // Symmetry regularisation: penalise asymmetric expression effects
      // For each symmetric pair (a, b), penalty: || (J[a] - J[b]_mirrored) · ε ||²
      // This adds the outer product S[ca][cb] = Σ_pairs (diff_ca · diff_cb) to MtM
      for (const [a, b] of symPairs) {
        for (let ca = 0; ca < C; ca++) {
          // diff_ca = J[a][:,ca] - mirror(J[b][:,ca])  (mirror = negate X)
          const dXa = J[a][0][ca] + J[b][0][ca]; // J[a].x - (-J[b].x) = J[a].x + J[b].x
          const dYa = J[a][1][ca] - J[b][1][ca];
          const dZa = J[a][2][ca] - J[b][2][ca];
          for (let cb = ca; cb < C; cb++) {
            const dXb = J[a][0][cb] + J[b][0][cb];
            const dYb = J[a][1][cb] - J[b][1][cb];
            const dZb = J[a][2][cb] - J[b][2][cb];
            const pen = LAMBDA_SYM * (dXa * dXb + dYa * dYb + dZa * dZb);
            MtM[ca * C + cb] += pen;
            if (ca !== cb) MtM[cb * C + ca] += pen; // symmetric matrix
          }
        }
      }

      // 5. Solve via Cholesky
      if (isPersp) {
        // Gauss-Newton: solve for δε (increment), since perspective is nonlinear
        const deltaEps = this._solveCholesky(MtM, Mtd, C);
        for (let c = 0; c < C; c++) epsilon[c] += deltaEps[c];
      } else {
        // Weak-perspective: solve for full ε (problem is linear)
        epsilon = this._solveCholesky(MtM, Mtd, C);
      }

      // Log progress
      let reproj = 0;
      for (let ki = 0; ki < K; ki++) {
        const p = [baseLmk[ki][0], baseLmk[ki][1], baseLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * epsilon[c];
          p[1] += J[ki][1][c] * epsilon[c];
          p[2] += J[ki][2][c] * epsilon[c];
        }
        if (isPersp) {
          const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
          const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
          const zrP = R[6] * p[0] + R[7] * p[1] + R[8] * p[2];
          const Zk = zrP + camZ;
          const ex = (sx * xrP / Zk + tx) - pts2D[ki][0];
          const ey = (sy * yrP / Zk + ty) - pts2D[ki][1];
          reproj += Math.sqrt(ex * ex + ey * ey);
        } else {
          const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
          const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
          const ex = (sx * xrP + tx) - pts2D[ki][0];
          const ey = (sy * yrP + ty) - pts2D[ki][1];
          reproj += Math.sqrt(ex * ex + ey * ey);
        }
      }
      console.log(`Expression fitting: iter ${iter}, reproj error = ${(reproj / K).toFixed(6)}${isPersp ? ' (perspective)' : ''}`);
    }

    // --- Clamp ε to valid range ---
    for (let c = 0; c < C; c++) {
      epsilon[c] = Math.max(-2.0, Math.min(2.0, epsilon[c]));
    }
    // Conditional clamping: if mouth is closed in photo, restrict jaw/mouth components
    if (mouthClosed) {
      epsilon[0] = Math.max(-1.0, Math.min(1.0, epsilon[0])); // jaw open
      epsilon[1] = Math.max(-1.0, Math.min(1.0, epsilon[1])); // secondary jaw
      console.log('Expression fitting: mouth closed → clamped ε[0], ε[1] to ±1.0');
    }

    // --- Apply expression to mesh ---
    const exprParams = new Array(expressionComponents).fill(0);
    for (let c = 0; c < C; c++) exprParams[c] = epsilon[c];
    meshGen.applyExpressionParams(exprParams);

    console.log(`Expression fitting: ε[0..4] = [${epsilon.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
    console.log('Expression fitting: Applied expression params to mesh');

    return epsilon;
  }

  /**
   * Project all 5023 FLAME position vertices into image space.
   * @returns {Float32Array} 5023*2 array of [imgX, imgY] pairs (normalized 0-1)
   */
  _projectAllVertices(meshGen, cam) {
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const nVerts = verts.length / 3;
    const result = new Float32Array(nVerts * 2);
    const { R, fx, fy, tx, ty, tz, meanZ } = cam;

    for (let i = 0; i < nVerts; i++) {
      const x = verts[i * 3], y = verts[i * 3 + 1], z = verts[i * 3 + 2];
      // Rotate
      const xr = R[0] * x + R[1] * y + R[2] * z;
      const yr = R[3] * x + R[4] * y + R[5] * z;
      const zr = R[6] * x + R[7] * y + R[8] * z;
      // Perspective projection
      const zCam = zr - meanZ + tz;
      if (zCam > 0.001) {
        result[i * 2] = fx * xr / zCam + tx;
        result[i * 2 + 1] = fy * yr / zCam + ty;
      } else {
        result[i * 2] = -999;
        result[i * 2 + 1] = -999;
      }
    }

    // Diagnostic: count in-bounds vertices
    let inBounds = 0;
    for (let i = 0; i < nVerts; i++) {
      const ix = result[i * 2], iy = result[i * 2 + 1];
      if (ix >= 0 && ix < 1 && iy >= 0 && iy < 1) inBounds++;
    }
    console.log(`PhotoUploader DENSE: ${inBounds}/${nVerts} vertices project into image bounds`);

    return result;
  }

  /**
   * Compute per-face visibility via backface culling.
   * @returns {Float32Array} 9976 visibility values: 0=back-facing, 1=visible, smooth at silhouette
   */
  _computeFaceVisibility(meshGen, cam) {
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const nFaces = posFaces.length / 3;
    const visibility = new Float32Array(nFaces);
    const { R, meanZ, tz } = cam;

    // Camera position in model space: looking along -Z in camera space,
    // so camera is at R^T * [0, 0, -tz] in model space
    // For face culling we just need the view direction in model space
    // View direction = R^T * [0, 0, 1] (camera looks along +Z in camera space)
    const viewDirX = R[2]; // R^T column 2 = R row 2... actually row 0 col 2, row 1 col 2, row 2 col 2
    const viewDirY = R[5];
    const viewDirZ = R[8];

    let visibleCount = 0;
    for (let f = 0; f < nFaces; f++) {
      const i0 = posFaces[f * 3], i1 = posFaces[f * 3 + 1], i2 = posFaces[f * 3 + 2];

      const x0 = verts[i0 * 3], y0 = verts[i0 * 3 + 1], z0 = verts[i0 * 3 + 2];
      const x1 = verts[i1 * 3], y1 = verts[i1 * 3 + 1], z1 = verts[i1 * 3 + 2];
      const x2 = verts[i2 * 3], y2 = verts[i2 * 3 + 1], z2 = verts[i2 * 3 + 2];

      // Edge vectors
      const ex1 = x1 - x0, ey1 = y1 - y0, ez1 = z1 - z0;
      const ex2 = x2 - x0, ey2 = y2 - y0, ez2 = z2 - z0;

      // Face normal (cross product)
      const nx = ey1 * ez2 - ez1 * ey2;
      const ny = ez1 * ex2 - ex1 * ez2;
      const nz = ex1 * ey2 - ey1 * ex2;

      // Dot with view direction (how much the face points toward camera)
      const dot = nx * viewDirX + ny * viewDirY + nz * viewDirZ;
      const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
      const cosAngle = nLen > 1e-10 ? dot / nLen : 0;

      // Smooth visibility: fully visible when cosAngle > 0.15, fade at silhouette
      if (cosAngle > 0.25) {
        visibility[f] = 1.0;
        visibleCount++;
      } else if (cosAngle > -0.10) {
        // Wide smooth transition at silhouette for gradual blending
        visibility[f] = (cosAngle + 0.10) / 0.35;
        visibleCount++;
      } else {
        visibility[f] = 0;
      }
    }

    console.log(`PhotoUploader DENSE: ${visibleCount}/${nFaces} faces visible`);
    return visibility;
  }

  /**
   * Build a set of face indices to exclude from rasterization (inner mouth, eyeballs).
   * Identifies unlabeled vertices near the lip region and behind the lip surface.
   * @returns {Set<number>} face indices to skip during rasterization
   */
  _buildInnerMouthMask(meshGen) {
    const regions = meshGen._flameRegions;
    const verts = meshGen._flameCurrentVertices ?? meshGen._flameTemplateVertices;
    const faces = meshGen.flameFaces;
    const nVerts = verts.length / 3;
    const nFaces = faces.length / 3;

    if (!regions || !regions.zones) {
      console.warn('Inner mouth mask: No region data available');
      return new Set();
    }

    // Collect all vertices that belong to any labeled region
    const labeledVerts = new Set();
    const lipVerts = new Set();
    for (const [name, data] of Object.entries(regions.zones)) {
      const vs = Array.isArray(data) ? data : (data.vertex_indices || []);
      vs.forEach(v => labeledVerts.add(v));
      if (name.startsWith('lip_') || name.startsWith('lip_corner')) {
        vs.forEach(v => lipVerts.add(v));
      }
    }

    // Compute lip bounding box
    let lipXmin = 1e9, lipXmax = -1e9, lipYmin = 1e9, lipYmax = -1e9, lipZsum = 0;
    for (const v of lipVerts) {
      const x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2];
      if (x < lipXmin) lipXmin = x; if (x > lipXmax) lipXmax = x;
      if (y < lipYmin) lipYmin = y; if (y > lipYmax) lipYmax = y;
      lipZsum += z;
    }
    const lipZmean = lipZsum / lipVerts.size;

    // Find unlabeled vertices near the lip region and behind the lip surface
    const innerMouthVerts = new Set();
    const margin = 0.02; // 2cm margin around lip bbox
    for (let v = 0; v < nVerts; v++) {
      if (labeledVerts.has(v)) continue;
      const x = verts[v * 3], y = verts[v * 3 + 1], z = verts[v * 3 + 2];
      if (x >= lipXmin - margin && x <= lipXmax + margin &&
          y >= lipYmin - margin && y <= lipYmax + margin &&
          z < lipZmean + 0.005) {
        innerMouthVerts.add(v);
      }
    }

    // Find faces that touch any inner mouth vertex
    const excludedFaces = new Set();
    for (let f = 0; f < nFaces; f++) {
      const v0 = faces[f * 3], v1 = faces[f * 3 + 1], v2 = faces[f * 3 + 2];
      if (innerMouthVerts.has(v0) || innerMouthVerts.has(v1) || innerMouthVerts.has(v2)) {
        excludedFaces.add(f);
      }
    }

    console.log(`Inner mouth mask: ${innerMouthVerts.size} vertices, ${excludedFaces.size} faces excluded`);
    return excludedFaces;
  }

  /**
   * Build a binary mask on the source photo that marks pixels inside the face boundary.
   * Uses MediaPipe's face oval landmarks to create a polygon, then rasterises it.
   *
   * @param {Array} landmarks - MediaPipe 478 landmarks (normalised 0-1)
   * @param {number} srcW - source image width in pixels
   * @param {number} srcH - source image height in pixels
   * @returns {Uint8Array} srcW*srcH binary mask (1 = inside face, 0 = outside)
   */
  _buildFaceBoundaryMask(landmarks, srcW, srcH) {
    // MediaPipe standard face oval contour (ordered: forehead → jaw → forehead)
    const FACE_OVAL = [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
      365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
      132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ];

    // Extract polygon vertices in pixel coordinates
    const polyX = [];
    const polyY = [];
    for (const idx of FACE_OVAL) {
      if (idx >= landmarks.length) continue;
      const lm = landmarks[idx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      polyX.push(lm.x * srcW);
      polyY.push(lm.y * srcH);
    }

    if (polyX.length < 10) {
      console.warn('Face boundary mask: insufficient landmarks, disabling');
      const mask = new Uint8Array(srcW * srcH);
      mask.fill(1);
      return mask;
    }

    // Expand polygon outward by a small margin to avoid cutting face edges
    let cx = 0, cy = 0;
    for (let i = 0; i < polyX.length; i++) { cx += polyX[i]; cy += polyY[i]; }
    cx /= polyX.length;
    cy /= polyY.length;

    const MARGIN_PX = 5;
    for (let i = 0; i < polyX.length; i++) {
      const dx = polyX[i] - cx;
      const dy = polyY[i] - cy;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0.01) {
        polyX[i] += (dx / len) * MARGIN_PX;
        polyY[i] += (dy / len) * MARGIN_PX;
      }
    }

    // Scanline polygon rasterisation
    const mask = new Uint8Array(srcW * srcH);
    const nVerts = polyX.length;

    let minY = srcH, maxY = 0;
    for (let i = 0; i < nVerts; i++) {
      if (polyY[i] < minY) minY = Math.floor(polyY[i]);
      if (polyY[i] > maxY) maxY = Math.ceil(polyY[i]);
    }
    minY = Math.max(0, minY);
    maxY = Math.min(srcH - 1, maxY);

    for (let y = minY; y <= maxY; y++) {
      const intersections = [];
      for (let i = 0; i < nVerts; i++) {
        const j = (i + 1) % nVerts;
        const yi = polyY[i], yj = polyY[j];
        if ((yi <= y && yj > y) || (yj <= y && yi > y)) {
          const t = (y - yi) / (yj - yi);
          intersections.push(polyX[i] + t * (polyX[j] - polyX[i]));
        }
      }
      intersections.sort((a, b) => a - b);

      for (let k = 0; k < intersections.length - 1; k += 2) {
        const xStart = Math.max(0, Math.ceil(intersections[k]));
        const xEnd = Math.min(srcW - 1, Math.floor(intersections[k + 1]));
        for (let x = xStart; x <= xEnd; x++) {
          mask[y * srcW + x] = 1;
        }
      }
    }

    let count = 0;
    for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
    console.log(`Face boundary mask: ${count}/${srcW * srcH} pixels inside face (${(count / (srcW * srcH) * 100).toFixed(1)}%)`);

    return mask;
  }

  /**
   * Map per-position-vertex image coords to per-UV-vertex image coords.
   * @returns {Float32Array} 5118*2 array of UV-vertex image coordinates
   */
  _mapProjectionToUV(projectedCoords, meshGen, faceVisibility) {
    const uvToPosMap = meshGen.uvToPosMap;
    const nUV = meshGen.flameUVVertexCount;
    const result = new Float32Array(nUV * 2);

    for (let i = 0; i < nUV; i++) {
      const posIdx = uvToPosMap[i];
      result[i * 2] = projectedCoords[posIdx * 2];
      result[i * 2 + 1] = projectedCoords[posIdx * 2 + 1];
    }

    return result;
  }

  /**
   * Build a boolean UV mask: for each pixel in the texture, is it inside any UV triangle?
   * Prevents writing texture data to invalid UV regions (seam gaps, outside face area).
   * @returns {Uint8Array} size*size mask (1=valid, 0=invalid)
   */
  _buildUVMask(size, uvCoords, uvFaces) {
    const mask = new Uint8Array(size * size);
    const nFaces = uvFaces.length / 3;

    for (let f = 0; f < nFaces; f++) {
      const vi0 = uvFaces[f * 3], vi1 = uvFaces[f * 3 + 1], vi2 = uvFaces[f * 3 + 2];
      const u0 = uvCoords[vi0 * 2], v0 = uvCoords[vi0 * 2 + 1];
      const u1 = uvCoords[vi1 * 2], v1 = uvCoords[vi1 * 2 + 1];
      const u2 = uvCoords[vi2 * 2], v2 = uvCoords[vi2 * 2 + 1];

      const minPx = Math.max(0, Math.floor(Math.min(u0, u1, u2) * size));
      const maxPx = Math.min(size - 1, Math.ceil(Math.max(u0, u1, u2) * size));
      const minPy = Math.max(0, Math.floor(Math.min(v0, v1, v2) * size));
      const maxPy = Math.min(size - 1, Math.ceil(Math.max(v0, v1, v2) * size));

      const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
      if (Math.abs(denom) < 1e-12) continue;
      const invDenom = 1.0 / denom;

      for (let py = minPy; py <= maxPy; py++) {
        const vv = (py + 0.5) / size;
        for (let px = minPx; px <= maxPx; px++) {
          const uu = (px + 0.5) / size;
          const w0 = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) * invDenom;
          const w1 = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) * invDenom;
          const w2 = 1.0 - w0 - w1;
          if (w0 >= -0.001 && w1 >= -0.001 && w2 >= -0.001) {
            mask[py * size + px] = 1;
          }
        }
      }
    }
    return mask;
  }

  /**
   * Zero out pixels that fall outside the valid UV footprint.
   */
  _applyUVMask(imageData, size, uvMask) {
    const data = imageData.data;
    let cleared = 0;
    for (let i = 0; i < size * size; i++) {
      if (data[i * 4 + 3] > 0 && !uvMask[i]) {
        data[i * 4] = 0;
        data[i * 4 + 1] = 0;
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = 0;
        cleared++;
      }
    }
    if (cleared > 0) console.log(`PhotoUploader DENSE: UV mask cleared ${cleared} out-of-footprint pixels`);
  }

  /**
   * Remove baked-in photo shading from the texture (Tier A delighting).
   *
   * Computes a low-frequency luminance field (heavy box blur on the luminance channel),
   * then divides each pixel's RGB by this field to produce a flat-lit albedo.
   * This removes the photo's original shadows so 3D lighting creates consistent ones.
   *
   * Only operates on mapped pixels (alpha > 0).
   */
  _delightTexture(imageData, size) {
    const data = imageData.data;
    const total = size * size;

    // --- 1. Extract luminance for mapped pixels ---
    const lum = new Float32Array(total);
    let lumSum = 0, lumCount = 0;
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) continue;
      const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      lum[i] = l;
      lumSum += l;
      lumCount++;
    }
    if (lumCount < 100) return; // Not enough data

    const avgLum = lumSum / lumCount;

    // --- 2. Create low-frequency shading map (heavy blur, radius=size/16) ---
    // Fill unmapped pixels with average luminance for blur boundary handling
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) lum[i] = avgLum;
    }

    const blurRadius = Math.max(8, Math.round(size / 16));
    const blurred = new Float32Array(total);

    // Horizontal box blur
    const tmp = new Float32Array(total);
    for (let y = 0; y < size; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x < Math.min(blurRadius, size); x++) {
        sum += lum[y * size + x]; count++;
      }
      for (let x = 0; x < size; x++) {
        if (x + blurRadius < size) { sum += lum[y * size + x + blurRadius]; count++; }
        if (x - blurRadius - 1 >= 0) { sum -= lum[y * size + x - blurRadius - 1]; count--; }
        tmp[y * size + x] = sum / count;
      }
    }
    // Vertical box blur
    for (let x = 0; x < size; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y < Math.min(blurRadius, size); y++) {
        sum += tmp[y * size + x]; count++;
      }
      for (let y = 0; y < size; y++) {
        if (y + blurRadius < size) { sum += tmp[(y + blurRadius) * size + x]; count++; }
        if (y - blurRadius - 1 >= 0) { sum -= tmp[(y - blurRadius - 1) * size + x]; count--; }
        blurred[y * size + x] = sum / count;
      }
    }

    // Second pass for smoother result
    for (let y = 0; y < size; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x < Math.min(blurRadius, size); x++) {
        sum += blurred[y * size + x]; count++;
      }
      for (let x = 0; x < size; x++) {
        if (x + blurRadius < size) { sum += blurred[y * size + x + blurRadius]; count++; }
        if (x - blurRadius - 1 >= 0) { sum -= blurred[y * size + x - blurRadius - 1]; count--; }
        tmp[y * size + x] = sum / count;
      }
    }
    for (let x = 0; x < size; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y < Math.min(blurRadius, size); y++) {
        sum += tmp[y * size + x]; count++;
      }
      for (let y = 0; y < size; y++) {
        if (y + blurRadius < size) { sum += tmp[(y + blurRadius) * size + x]; count++; }
        if (y - blurRadius - 1 >= 0) { sum -= tmp[(y - blurRadius - 1) * size + x]; count--; }
        blurred[y * size + x] = sum / count;
      }
    }

    // --- 3. Divide out shading: pixel / lowFreq * avgLum ---
    // This normalizes each pixel to remove directional lighting while preserving
    // the overall brightness level (avgLum).
    // Use a mild strength factor to avoid over-flattening.
    const strength = 0.7; // 0=no delighting, 1=full delighting
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) continue;

      const shade = blurred[i];
      if (shade < 5) continue; // Avoid division by near-zero

      // Correction factor: how much to brighten/darken to remove shading
      const correction = avgLum / shade;
      // Blend between original and delighted based on strength
      const factor = 1.0 + strength * (correction - 1.0);

      data[i * 4]     = Math.min(255, Math.max(0, Math.round(data[i * 4] * factor)));
      data[i * 4 + 1] = Math.min(255, Math.max(0, Math.round(data[i * 4 + 1] * factor)));
      data[i * 4 + 2] = Math.min(255, Math.max(0, Math.round(data[i * 4 + 2] * factor)));
    }

    console.log(`PhotoUploader DENSE: Delighting applied (avgLum=${avgLum.toFixed(1)}, radius=${blurRadius}, strength=${strength})`);
  }

  /**
   * Blur the alpha channel at the boundary between mapped (alpha>0) and unmapped (alpha=0)
   * pixels. This creates a smooth feathered edge for blending with FLAME albedo.
   * Uses a box blur on the alpha channel only, applied at boundary pixels.
   */
  _blurAlphaBoundary(imageData, size) {
    const data = imageData.data;
    const total = size * size;
    const alpha = new Uint8Array(total);

    // Copy alpha channel
    for (let i = 0; i < total; i++) alpha[i] = data[i * 4 + 3];

    // 3-pass box blur on alpha (radius=4 for ~8px feather at 2048)
    const radius = 4;
    const blurred = new Uint8Array(total);
    for (let pass = 0; pass < 3; pass++) {
      const src = pass === 0 ? alpha : blurred;
      const dst = blurred;

      // Horizontal pass
      const tmp = new Uint8Array(total);
      for (let y = 0; y < size; y++) {
        let sum = 0, count = 0;
        for (let x = 0; x < Math.min(radius, size); x++) { sum += src[y * size + x]; count++; }
        for (let x = 0; x < size; x++) {
          if (x + radius < size) { sum += src[y * size + x + radius]; count++; }
          if (x - radius - 1 >= 0) { sum -= src[y * size + x - radius - 1]; count--; }
          tmp[y * size + x] = Math.round(sum / count);
        }
      }
      // Vertical pass
      for (let x = 0; x < size; x++) {
        let sum = 0, count = 0;
        for (let y = 0; y < Math.min(radius, size); y++) { sum += tmp[y * size + x]; count++; }
        for (let y = 0; y < size; y++) {
          if (y + radius < size) { sum += tmp[(y + radius) * size + x]; count++; }
          if (y - radius - 1 >= 0) { sum -= tmp[(y - radius - 1) * size + x]; count--; }
          dst[y * size + x] = Math.round(sum / count);
        }
      }
    }

    // Apply: only modify alpha at boundary pixels (partially blurred)
    // Keep fully mapped pixels at original alpha, only soften edges
    for (let i = 0; i < total; i++) {
      if (alpha[i] > 0 && alpha[i] < 255) {
        // Silhouette edge — use blurred alpha for smoother transition
        data[i * 4 + 3] = Math.min(alpha[i], blurred[i]);
      } else if (alpha[i] === 255 && blurred[i] < 255) {
        // Boundary of fully mapped region — keep fully opaque
        // (the albedo fill will handle the unmapped side)
      } else if (alpha[i] === 0 && blurred[i] > 0) {
        // Just outside mapped region — extend photo slightly via blur
        // Sample neighboring mapped pixel colors for the extension
        const x = i % size, y = Math.floor(i / size);
        let rSum = 0, gSum = 0, bSum = 0, wSum = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
            const ni = ny * size + nx;
            if (alpha[ni] > 0) {
              const w = alpha[ni] / 255;
              rSum += data[ni * 4] * w;
              gSum += data[ni * 4 + 1] * w;
              bSum += data[ni * 4 + 2] * w;
              wSum += w;
            }
          }
        }
        if (wSum > 0) {
          data[i * 4] = Math.round(rSum / wSum);
          data[i * 4 + 1] = Math.round(gSum / wSum);
          data[i * 4 + 2] = Math.round(bSum / wSum);
          data[i * 4 + 3] = blurred[i];
        }
      }
    }
  }

  /**
   * Fill unmapped pixels (alpha < 255) with FLAME albedo texture,
   * color-corrected to match the photo's skin tone.
   * Blends at silhouette edges.
   */
  _fillWithFlameAlbedo(outImageData, meshGen) {
    const data = outImageData.data;
    const size = outImageData.width;
    const total = data.length / 4;

    const albedo = meshGen.albedoDiffuseData; // Uint8Array, 512*512*3 RGB
    const albedoSize = 512;

    if (!albedo) {
      this._fillUnmappedRegions(outImageData);
      return;
    }

    // --- Compute color correction: match FLAME albedo to photo skin tone ---
    // Sample both photo pixels (alpha=255) and their corresponding FLAME albedo
    // to compute a color ratio for seamless blending.
    let photoR = 0, photoG = 0, photoB = 0;
    let albR = 0, albG = 0, albB = 0;
    let sampleCount = 0;

    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] !== 255) continue; // only fully-mapped photo pixels

      const px = i % size;
      const py = Math.floor(i / size);
      const u = (px + 0.5) / size;
      const v = (py + 0.5) / size;

      // Sample albedo at same UV
      const ax0 = Math.max(0, Math.min(Math.floor(u * albedoSize), albedoSize - 1));
      const ay0 = Math.max(0, Math.min(Math.floor(v * albedoSize), albedoSize - 1));
      const ai = (ay0 * albedoSize + ax0) * 3;

      photoR += data[i * 4];
      photoG += data[i * 4 + 1];
      photoB += data[i * 4 + 2];
      albR += albedo[ai];
      albG += albedo[ai + 1];
      albB += albedo[ai + 2];
      sampleCount++;
    }

    // Color correction ratios (photo avg / albedo avg)
    let crR = 1, crG = 1, crB = 1;
    if (sampleCount > 100) {
      const avgPR = photoR / sampleCount;
      const avgPG = photoG / sampleCount;
      const avgPB = photoB / sampleCount;
      const avgAR = albR / sampleCount;
      const avgAG = albG / sampleCount;
      const avgAB = albB / sampleCount;
      // Clamp ratios to avoid extreme values
      crR = avgAR > 10 ? Math.min(Math.max(avgPR / avgAR, 0.3), 3.0) : 1;
      crG = avgAG > 10 ? Math.min(Math.max(avgPG / avgAG, 0.3), 3.0) : 1;
      crB = avgAB > 10 ? Math.min(Math.max(avgPB / avgAB, 0.3), 3.0) : 1;
      console.log(`PhotoUploader DENSE: Albedo color correction: R×${crR.toFixed(2)}, G×${crG.toFixed(2)}, B×${crB.toFixed(2)} (from ${sampleCount} samples)`);
    }

    // --- Fill unmapped pixels with color-corrected albedo ---
    for (let i = 0; i < total; i++) {
      const alpha = data[i * 4 + 3];
      if (alpha === 255) continue;

      const px = i % size;
      const py = Math.floor(i / size);
      const u = (px + 0.5) / size;
      const v = (py + 0.5) / size;

      // Bilinear sample FLAME albedo
      const ax = u * albedoSize;
      const ay = v * albedoSize;
      const ax0 = Math.max(0, Math.min(Math.floor(ax), albedoSize - 1));
      const ay0 = Math.max(0, Math.min(Math.floor(ay), albedoSize - 1));
      const ax1 = Math.min(ax0 + 1, albedoSize - 1);
      const ay1 = Math.min(ay0 + 1, albedoSize - 1);
      const atx = Math.max(0, ax - ax0);
      const aty = Math.max(0, ay - ay0);

      const ai00 = (ay0 * albedoSize + ax0) * 3;
      const ai10 = (ay0 * albedoSize + ax1) * 3;
      const ai01 = (ay1 * albedoSize + ax0) * 3;
      const ai11 = (ay1 * albedoSize + ax1) * 3;

      const w00 = (1 - atx) * (1 - aty);
      const w10 = atx * (1 - aty);
      const w01 = (1 - atx) * aty;
      const w11 = atx * aty;

      // Apply color correction to albedo samples
      const aR = Math.min(255, Math.round((w00 * albedo[ai00] + w10 * albedo[ai10] + w01 * albedo[ai01] + w11 * albedo[ai11]) * crR));
      const aG = Math.min(255, Math.round((w00 * albedo[ai00 + 1] + w10 * albedo[ai10 + 1] + w01 * albedo[ai01 + 1] + w11 * albedo[ai11 + 1]) * crG));
      const aB = Math.min(255, Math.round((w00 * albedo[ai00 + 2] + w10 * albedo[ai10 + 2] + w01 * albedo[ai01 + 2] + w11 * albedo[ai11 + 2]) * crB));

      if (alpha === 0) {
        data[i * 4] = aR;
        data[i * 4 + 1] = aG;
        data[i * 4 + 2] = aB;
      } else {
        // Silhouette blend
        const t = alpha / 255;
        data[i * 4] = Math.round(t * data[i * 4] + (1 - t) * aR);
        data[i * 4 + 1] = Math.round(t * data[i * 4 + 1] + (1 - t) * aG);
        data[i * 4 + 2] = Math.round(t * data[i * 4 + 2] + (1 - t) * aB);
      }
      data[i * 4 + 3] = 255;
    }
  }

  // -----------------------------------------------------------------------
  // Mask-Normalized Laplacian Pyramid Blending
  // -----------------------------------------------------------------------

  /**
   * Separable Gaussian blur on a Float32Array image.
   * @param {Float32Array} data - size*size*channels flat array
   * @param {number} w - width
   * @param {number} h - height
   * @param {number} ch - channels (3 or 4)
   * @param {number} radius - blur radius
   * @returns {Float32Array} blurred copy
   */
  _gaussBlur(data, w, h, ch, radius) {
    const out = new Float32Array(data.length);
    const temp = new Float32Array(data.length);
    const r = Math.max(1, Math.round(radius));
    const diam = 2 * r + 1;

    // Horizontal pass: data → temp
    for (let y = 0; y < h; y++) {
      for (let c = 0; c < ch; c++) {
        let sum = 0;
        // Init window
        for (let dx = -r; dx <= r; dx++) {
          const x = Math.max(0, Math.min(w - 1, dx));
          sum += data[(y * w + x) * ch + c];
        }
        temp[(y * w + 0) * ch + c] = sum / diam;
        for (let x = 1; x < w; x++) {
          const addX = Math.min(w - 1, x + r);
          const remX = Math.max(0, x - r - 1);
          sum += data[(y * w + addX) * ch + c] - data[(y * w + remX) * ch + c];
          temp[(y * w + x) * ch + c] = sum / diam;
        }
      }
    }

    // Vertical pass: temp → out
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < ch; c++) {
        let sum = 0;
        for (let dy = -r; dy <= r; dy++) {
          const y = Math.max(0, Math.min(h - 1, dy));
          sum += temp[(y * w + x) * ch + c];
        }
        out[(0 * w + x) * ch + c] = sum / diam;
        for (let y = 1; y < h; y++) {
          const addY = Math.min(h - 1, y + r);
          const remY = Math.max(0, y - r - 1);
          sum += temp[(addY * w + x) * ch + c] - temp[(remY * w + x) * ch + c];
          out[(y * w + x) * ch + c] = sum / diam;
        }
      }
    }

    return out;
  }

  /**
   * Downsample image by 2x (average 2x2 blocks).
   * @param {Float32Array} data - w*h*ch
   * @returns {{ data: Float32Array, w: number, h: number }}
   */
  _downsample2x(data, w, h, ch) {
    const nw = Math.floor(w / 2), nh = Math.floor(h / 2);
    const out = new Float32Array(nw * nh * ch);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        for (let c = 0; c < ch; c++) {
          const v =
            data[((y * 2) * w + x * 2) * ch + c] +
            data[((y * 2) * w + x * 2 + 1) * ch + c] +
            data[((y * 2 + 1) * w + x * 2) * ch + c] +
            data[((y * 2 + 1) * w + x * 2 + 1) * ch + c];
          out[(y * nw + x) * ch + c] = v / 4;
        }
      }
    }
    return { data: out, w: nw, h: nh };
  }

  /**
   * Upsample image by 2x (bilinear).
   * @param {Float32Array} data - w*h*ch
   * @returns {{ data: Float32Array, w: number, h: number }}
   */
  _upsample2x(data, w, h, ch, targetW, targetH) {
    const out = new Float32Array(targetW * targetH * ch);
    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        const srcX = (x + 0.5) / 2 - 0.5;
        const srcY = (y + 0.5) / 2 - 0.5;
        const x0 = Math.max(0, Math.min(w - 1, Math.floor(srcX)));
        const y0 = Math.max(0, Math.min(h - 1, Math.floor(srcY)));
        const x1 = Math.min(w - 1, x0 + 1);
        const y1 = Math.min(h - 1, y0 + 1);
        const fx = srcX - x0, fy = srcY - y0;
        for (let c = 0; c < ch; c++) {
          const v00 = data[(y0 * w + x0) * ch + c];
          const v10 = data[(y0 * w + x1) * ch + c];
          const v01 = data[(y1 * w + x0) * ch + c];
          const v11 = data[(y1 * w + x1) * ch + c];
          out[(y * targetW + x) * ch + c] =
            (1 - fy) * ((1 - fx) * v00 + fx * v10) +
            fy * ((1 - fx) * v01 + fx * v11);
        }
      }
    }
    return { data: out, w: targetW, h: targetH };
  }

  /**
   * Mask-normalized Laplacian pyramid blend of photo texture and albedo.
   *
   * Uses mask-normalized convolution: at each pyramid level,
   * blur(img * mask) / blur(mask) — ensures only valid skin pixels
   * contribute, preventing background wall/hair bleed.
   *
   * @param {ImageData} photoData  - Rasterized photo texture (with alpha = coverage mask)
   * @param {ImageData} albedoData - Color-corrected FLAME albedo (full coverage)
   * @param {number} size - Texture dimension (2048)
   * @param {number} levels - Pyramid levels (5)
   * @returns {ImageData} blended result
   */
  _laplacianBlend(photoData, albedoData, size, levels = 5) {
    const ch = 3; // Work in RGB only
    const N = size * size;

    // Extract RGB float arrays and mask from alpha
    const photoRGB = new Float32Array(N * ch);
    const albedoRGB = new Float32Array(N * ch);
    const mask = new Float32Array(N); // 0..1 blend weight

    const pd = photoData.data;
    const ad = albedoData.data;
    for (let i = 0; i < N; i++) {
      const alpha = pd[i * 4 + 3] / 255; // photo coverage
      mask[i] = alpha;
      photoRGB[i * 3]     = pd[i * 4];
      photoRGB[i * 3 + 1] = pd[i * 4 + 1];
      photoRGB[i * 3 + 2] = pd[i * 4 + 2];
      albedoRGB[i * 3]     = ad[i * 4];
      albedoRGB[i * 3 + 1] = ad[i * 4 + 1];
      albedoRGB[i * 3 + 2] = ad[i * 4 + 2];
    }

    // --- Mask-normalized photo: fill unmapped regions with normalized blur ---
    // This prevents background bleed: blur(photo * mask) / blur(mask)
    const maskedPhoto = new Float32Array(N * ch);
    for (let i = 0; i < N; i++) {
      maskedPhoto[i * 3]     = photoRGB[i * 3] * mask[i];
      maskedPhoto[i * 3 + 1] = photoRGB[i * 3 + 1] * mask[i];
      maskedPhoto[i * 3 + 2] = photoRGB[i * 3 + 2] * mask[i];
    }
    const blurredMasked = this._gaussBlur(maskedPhoto, size, size, ch, 8);
    const blurredMask = this._gaussBlur(mask, size, size, 1, 8);
    // Normalize: fill unmapped photo pixels with mask-normalized values
    const photoFilled = new Float32Array(photoRGB);
    for (let i = 0; i < N; i++) {
      if (mask[i] < 0.5 && blurredMask[i] > 0.01) {
        photoFilled[i * 3]     = blurredMasked[i * 3]     / blurredMask[i];
        photoFilled[i * 3 + 1] = blurredMasked[i * 3 + 1] / blurredMask[i];
        photoFilled[i * 3 + 2] = blurredMasked[i * 3 + 2] / blurredMask[i];
      }
    }

    // --- Build Gaussian pyramids ---
    const photoGauss = [{ data: photoFilled, w: size, h: size }];
    const albedoGauss = [{ data: albedoRGB, w: size, h: size }];
    const maskGauss = [{ data: mask, w: size, h: size }];

    for (let l = 1; l <= levels; l++) {
      const prev = photoGauss[l - 1];
      const blurR = 2; // blur before downsample
      const pBlur = this._gaussBlur(prev.data, prev.w, prev.h, ch, blurR);
      photoGauss.push(this._downsample2x(pBlur, prev.w, prev.h, ch));

      const aprev = albedoGauss[l - 1];
      const aBlur = this._gaussBlur(aprev.data, aprev.w, aprev.h, ch, blurR);
      albedoGauss.push(this._downsample2x(aBlur, aprev.w, aprev.h, ch));

      const mprev = maskGauss[l - 1];
      const mBlur = this._gaussBlur(mprev.data, mprev.w, mprev.h, 1, blurR);
      maskGauss.push(this._downsample2x(mBlur, mprev.w, mprev.h, 1));
    }

    // --- Build Laplacian pyramids (L[l] = G[l] - upsample(G[l+1])) ---
    const photoLap = [];
    const albedoLap = [];
    for (let l = 0; l < levels; l++) {
      const pCur = photoGauss[l];
      const pNext = photoGauss[l + 1];
      const pUp = this._upsample2x(pNext.data, pNext.w, pNext.h, ch, pCur.w, pCur.h);

      const lapP = new Float32Array(pCur.data.length);
      for (let i = 0; i < lapP.length; i++) lapP[i] = pCur.data[i] - pUp.data[i];
      photoLap.push({ data: lapP, w: pCur.w, h: pCur.h });

      const aCur = albedoGauss[l];
      const aNext = albedoGauss[l + 1];
      const aUp = this._upsample2x(aNext.data, aNext.w, aNext.h, ch, aCur.w, aCur.h);

      const lapA = new Float32Array(aCur.data.length);
      for (let i = 0; i < lapA.length; i++) lapA[i] = aCur.data[i] - aUp.data[i];
      albedoLap.push({ data: lapA, w: aCur.w, h: aCur.h });
    }

    // --- Blend each Laplacian level using mask at that level ---
    const blendedLap = [];
    for (let l = 0; l < levels; l++) {
      const pL = photoLap[l];
      const aL = albedoLap[l];
      const mL = maskGauss[l];
      const blended = new Float32Array(pL.data.length);
      const pw = pL.w;
      for (let y = 0; y < pL.h; y++) {
        for (let x = 0; x < pw; x++) {
          const mi = mL.data[y * pw + x]; // mask at this level
          const t = Math.max(0, Math.min(1, mi)); // clamp
          for (let c = 0; c < ch; c++) {
            const idx = (y * pw + x) * ch + c;
            blended[idx] = t * pL.data[idx] + (1 - t) * aL.data[idx];
          }
        }
      }
      blendedLap.push({ data: blended, w: pw, h: pL.h });
    }

    // Blend coarsest Gaussian level too
    const pCoarse = photoGauss[levels];
    const aCoarse = albedoGauss[levels];
    const mCoarse = maskGauss[levels];
    const blendedCoarse = new Float32Array(pCoarse.data.length);
    for (let y = 0; y < pCoarse.h; y++) {
      for (let x = 0; x < pCoarse.w; x++) {
        const t = Math.max(0, Math.min(1, mCoarse.data[y * pCoarse.w + x]));
        for (let c = 0; c < ch; c++) {
          const idx = (y * pCoarse.w + x) * ch + c;
          blendedCoarse[idx] = t * pCoarse.data[idx] + (1 - t) * aCoarse.data[idx];
        }
      }
    }

    // --- Reconstruct: sum Laplacian levels bottom-up ---
    let current = { data: blendedCoarse, w: pCoarse.w, h: pCoarse.h };
    for (let l = levels - 1; l >= 0; l--) {
      const bL = blendedLap[l];
      const up = this._upsample2x(current.data, current.w, current.h, ch, bL.w, bL.h);
      const result = new Float32Array(bL.data.length);
      for (let i = 0; i < result.length; i++) result[i] = up.data[i] + bL.data[i];
      current = { data: result, w: bL.w, h: bL.h };
    }

    // --- Write back to ImageData ---
    const outData = new ImageData(size, size);
    const od = outData.data;
    for (let i = 0; i < N; i++) {
      od[i * 4]     = Math.max(0, Math.min(255, Math.round(current.data[i * 3])));
      od[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(current.data[i * 3 + 1])));
      od[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(current.data[i * 3 + 2])));
      od[i * 4 + 3] = 255; // fully opaque final result
    }

    console.log(`Laplacian blend: ${levels} levels, mask-normalized, ${size}x${size}`);
    return outData;
  }

  /**
   * Prepare the full-coverage FLAME albedo layer for Laplacian blending.
   * Returns an ImageData with color-corrected albedo at every pixel.
   */
  _prepareAlbedoLayer(photoImageData, meshGen, size, landmarks = null, mapping = null) {
    const albedoData = new ImageData(size, size);
    const ad = albedoData.data;
    const pd = photoImageData.data;

    // Get FLAME albedo data
    const albedo = meshGen.albedoDiffuseData;
    if (!albedo) {
      // No albedo → fill with average photo skin color
      let rSum = 0, gSum = 0, bSum = 0, count = 0;
      for (let i = 0; i < size * size; i++) {
        if (pd[i * 4 + 3] > 200) {
          rSum += pd[i * 4]; gSum += pd[i * 4 + 1]; bSum += pd[i * 4 + 2];
          count++;
        }
      }
      const avgR = count > 0 ? Math.round(rSum / count) : 180;
      const avgG = count > 0 ? Math.round(gSum / count) : 150;
      const avgB = count > 0 ? Math.round(bSum / count) : 130;
      for (let i = 0; i < size * size; i++) {
        ad[i * 4] = avgR; ad[i * 4 + 1] = avgG; ad[i * 4 + 2] = avgB; ad[i * 4 + 3] = 255;
      }
      return albedoData;
    }

    const albedoRes = meshGen.albedoResolution || 512;

    // Compute robust color correction using trimmed mean
    // Sample from forehead/cheek regions only (avoid beard, nose shadows, etc.)
    // Landmarks: 8 (nose bridge), 10 (forehead), 117 (right cheek), 346 (left cheek)

    // Build UV-space bounding boxes for sampling regions
    let sampleBoxes = null;
    if (landmarks && mapping) {
      const SAMPLE_MP = [8, 10, 117, 346];
      const uvCoords = meshGen.flameUVCoords;
      const uvFaces = meshGen.flameUVFaces;
      const mpIndices = mapping.landmark_indices;
      const faceIndices = mapping.lmk_face_idx;
      const baryCoords = mapping.lmk_b_coords;

      const sampleUVs = [];
      for (const targetMp of SAMPLE_MP) {
        for (let mi = 0; mi < mpIndices.length; mi++) {
          if (mpIndices[mi] === targetMp) {
            const fi = faceIndices[mi];
            const bc = baryCoords[mi];
            const uvi0 = uvFaces[fi * 3], uvi1 = uvFaces[fi * 3 + 1], uvi2 = uvFaces[fi * 3 + 2];
            const lu = bc[0] * uvCoords[uvi0 * 2] + bc[1] * uvCoords[uvi1 * 2] + bc[2] * uvCoords[uvi2 * 2];
            const lv = bc[0] * uvCoords[uvi0 * 2 + 1] + bc[1] * uvCoords[uvi1 * 2 + 1] + bc[2] * uvCoords[uvi2 * 2 + 1];
            sampleUVs.push({ u: lu, v: lv });
            break;
          }
        }
      }

      if (sampleUVs.length >= 2) {
        const MARGIN = 0.06; // 6% of UV space around each landmark
        sampleBoxes = sampleUVs.map(s => ({
          uMin: s.u - MARGIN, uMax: s.u + MARGIN,
          vMin: s.v - MARGIN, vMax: s.v + MARGIN
        }));
        console.log(`Albedo sampling: ${sampleBoxes.length} regions from forehead/cheeks`);
      }
    }

    const samples = [];
    for (let i = 0; i < size * size; i++) {
      if (pd[i * 4 + 3] < 200) continue;
      const u = (i % size) / size;
      const v = Math.floor(i / size) / size;

      // If we have sampling regions, only include pixels within them
      if (sampleBoxes) {
        let inRegion = false;
        for (const box of sampleBoxes) {
          if (u >= box.uMin && u <= box.uMax && v >= box.vMin && v <= box.vMax) {
            inRegion = true;
            break;
          }
        }
        if (!inRegion) continue;
      }

      const ax = Math.min(albedoRes - 1, Math.floor(u * albedoRes));
      const ay = Math.min(albedoRes - 1, Math.floor(v * albedoRes));
      const ai = (ay * albedoRes + ax) * 3;

      // Filter out extreme luminance (shadows, highlights, artifacts)
      const pLum = pd[i * 4] * 0.2126 + pd[i * 4 + 1] * 0.7152 + pd[i * 4 + 2] * 0.0722;
      const aLum = albedo[ai] * 0.2126 + albedo[ai + 1] * 0.7152 + albedo[ai + 2] * 0.0722;
      if (pLum < 20 || pLum > 240 || aLum < 20 || aLum > 240) continue;

      samples.push({
        pr: pd[i * 4], pg: pd[i * 4 + 1], pb: pd[i * 4 + 2],
        ar: albedo[ai], ag: albedo[ai + 1], ab: albedo[ai + 2],
        lum: pLum
      });
    }

    // Sort by luminance and trim 10% from each end
    samples.sort((a, b) => a.lum - b.lum);
    const trimStart = Math.floor(samples.length * 0.1);
    const trimEnd = Math.ceil(samples.length * 0.9);
    const trimmed = samples.slice(trimStart, trimEnd);

    let prSum = 0, pgSum = 0, pbSum = 0;
    let arSum = 0, agSum = 0, abSum = 0;
    for (const s of trimmed) {
      prSum += s.pr; pgSum += s.pg; pbSum += s.pb;
      arSum += s.ar; agSum += s.ag; abSum += s.ab;
    }
    const sampleCount = trimmed.length;

    const crR = sampleCount > 0 ? Math.max(0.3, Math.min(3.0, prSum / (arSum + 1))) : 1;
    const crG = sampleCount > 0 ? Math.max(0.3, Math.min(3.0, pgSum / (agSum + 1))) : 1;
    const crB = sampleCount > 0 ? Math.max(0.3, Math.min(3.0, pbSum / (abSum + 1))) : 1;
    console.log(`Albedo color correction: R=${crR.toFixed(3)}, G=${crG.toFixed(3)}, B=${crB.toFixed(3)}, samples=${sampleCount}`);

    // Fill every pixel with color-corrected albedo
    for (let i = 0; i < size * size; i++) {
      const u = (i % size) / size;
      const v = Math.floor(i / size) / size;
      const ax = Math.min(albedoRes - 1, Math.floor(u * albedoRes));
      const ay = Math.min(albedoRes - 1, Math.floor(v * albedoRes));
      const ai = (ay * albedoRes + ax) * 3;
      ad[i * 4]     = Math.min(255, Math.round(albedo[ai] * crR));
      ad[i * 4 + 1] = Math.min(255, Math.round(albedo[ai + 1] * crG));
      ad[i * 4 + 2] = Math.min(255, Math.round(albedo[ai + 2] * crB));
      ad[i * 4 + 3] = 255;
    }

    return albedoData;
  }

  // -----------------------------------------------------------------------
  // Thin-Plate Spline (TPS) interpolation
  // -----------------------------------------------------------------------

  /**
   * Fit a 2D Thin-Plate Spline through N control points.
   *
   * Solves the (N+3)×(N+3) linear system:
   *   [K + λI  P] [w]   [f]
   *   [P'      0] [a] = [0]
   *
   * where K_ij = U(r_ij), U(r) = r²·ln(r), P = [1, u, v] polynomial.
   *
   * @param {number[]} cu - control U coordinates
   * @param {number[]} cv - control V coordinates
   * @param {number[]} vals - function values at control points
   * @returns {{ weights: Float64Array, a0: number, a1: number, a2: number,
   *             cu: number[], cv: number[], N: number }}
   */
  _fitTPS(cu, cv, vals) {
    const N = cu.length;
    const M = N + 3;
    const lambda = 1e-4; // small regularisation for numerical stability

    // TPS radial basis: U(r) = r²·ln(r); U(0) = 0
    function tpsU(r2) {
      return r2 > 1e-20 ? r2 * Math.log(Math.sqrt(r2)) : 0;
    }

    // Build augmented matrix [A | b] of size M × (M+1)
    const mat = [];
    for (let i = 0; i < M; i++) mat[i] = new Float64Array(M + 1);

    // Upper-left: K + λI
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const du = cu[i] - cu[j], dv = cv[i] - cv[j];
        mat[i][j] = tpsU(du * du + dv * dv);
      }
      mat[i][i] += lambda; // regularisation
    }

    // Upper-right: P  and  lower-left: P'
    for (let i = 0; i < N; i++) {
      mat[i][N]     = 1;      mat[N][i]     = 1;
      mat[i][N + 1] = cu[i];  mat[N + 1][i] = cu[i];
      mat[i][N + 2] = cv[i];  mat[N + 2][i] = cv[i];
    }

    // Lower-right: zeros (already initialised)

    // RHS
    for (let i = 0; i < N; i++) mat[i][M] = vals[i];
    // Last 3 rows RHS = 0 (already initialised)

    // Solve via Gauss elimination with partial pivoting
    const x = this._solveLinearSystem(mat, M);

    return {
      weights: x.slice(0, N),
      a0: x[N],
      a1: x[N + 1],
      a2: x[N + 2],
      cu, cv, N,
    };
  }

  /**
   * Evaluate a fitted TPS at point (u, v).
   */
  _evalTPS(tps, u, v) {
    let result = tps.a0 + tps.a1 * u + tps.a2 * v;
    for (let i = 0; i < tps.N; i++) {
      const du = u - tps.cu[i], dv = v - tps.cv[i];
      const r2 = du * du + dv * dv;
      if (r2 > 1e-20) {
        result += tps.weights[i] * r2 * Math.log(Math.sqrt(r2));
      }
    }
    return result;
  }

  /**
   * Solve an n×n linear system (given as n×(n+1) augmented matrix)
   * via Gauss elimination with partial pivoting.
   * Returns a Float64Array of n solution values.
   */
  _solveLinearSystem(mat, n) {
    // Forward elimination
    for (let col = 0; col < n; col++) {
      let maxRow = col, maxVal = Math.abs(mat[col][col]);
      for (let row = col + 1; row < n; row++) {
        const v = Math.abs(mat[row][col]);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      if (maxRow !== col) { const tmp = mat[col]; mat[col] = mat[maxRow]; mat[maxRow] = tmp; }
      if (maxVal < 1e-15) continue;

      for (let row = col + 1; row < n; row++) {
        const factor = mat[row][col] / mat[col][col];
        for (let j = col; j <= n; j++) mat[row][j] -= factor * mat[col][j];
      }
    }

    // Back substitution
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = mat[i][n];
      for (let j = i + 1; j < n; j++) sum -= mat[i][j] * x[j];
      x[i] = Math.abs(mat[i][i]) > 1e-15 ? sum / mat[i][i] : 0;
    }
    return x;
  }

  // -----------------------------------------------------------------------
  // Rasteriser — all 9976 FLAME UV mesh faces
  // -----------------------------------------------------------------------

  /**
   * Rasterise all FLAME UV faces, sampling the source photo via
   * TPS-interpolated per-vertex image coordinates.
   */
  _rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH, faceVisibility = null, excludedFaces = null, faceBoundaryMask = null) {
    const out = outImageData.data;
    const nFaces = uvFaces.length / 3;

    for (let f = 0; f < nFaces; f++) {
      // Skip inner mouth / excluded faces
      if (excludedFaces && excludedFaces.has(f)) continue;

      // Optional backface culling via visibility mask
      const vis = faceVisibility ? faceVisibility[f] : 1.0;
      if (vis <= 0) continue;
      const faceAlpha = Math.round(vis * 255);

      const vi0 = uvFaces[f * 3], vi1 = uvFaces[f * 3 + 1], vi2 = uvFaces[f * 3 + 2];

      // UV-space triangle vertices (0-1)
      const u0 = uvCoords[vi0 * 2], v0 = uvCoords[vi0 * 2 + 1];
      const u1 = uvCoords[vi1 * 2], v1 = uvCoords[vi1 * 2 + 1];
      const u2 = uvCoords[vi2 * 2], v2 = uvCoords[vi2 * 2 + 1];

      // Image-space coordinates for each UV vertex
      const ix0 = uvImgCoords[vi0 * 2], iy0 = uvImgCoords[vi0 * 2 + 1];
      const ix1 = uvImgCoords[vi1 * 2], iy1 = uvImgCoords[vi1 * 2 + 1];
      const ix2 = uvImgCoords[vi2 * 2], iy2 = uvImgCoords[vi2 * 2 + 1];

      // Skip faces whose image coords are entirely far outside image
      if (Math.min(ix0, ix1, ix2) > 1.5 || Math.max(ix0, ix1, ix2) < -0.5 ||
          Math.min(iy0, iy1, iy2) > 1.5 || Math.max(iy0, iy1, iy2) < -0.5) continue;

      // Bounding box in pixel space
      const minPx = Math.max(0, Math.floor(Math.min(u0, u1, u2) * size));
      const maxPx = Math.min(size - 1, Math.ceil(Math.max(u0, u1, u2) * size));
      const minPy = Math.max(0, Math.floor(Math.min(v0, v1, v2) * size));
      const maxPy = Math.min(size - 1, Math.ceil(Math.max(v0, v1, v2) * size));

      // Barycentric denominator
      const denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2);
      if (Math.abs(denom) < 1e-12) continue;
      const invDenom = 1.0 / denom;

      for (let py = minPy; py <= maxPy; py++) {
        const vv = (py + 0.5) / size;
        for (let px = minPx; px <= maxPx; px++) {
          const uu = (px + 0.5) / size;

          const w0 = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) * invDenom;
          const w1 = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) * invDenom;
          const w2 = 1.0 - w0 - w1;

          if (w0 < -0.001 || w1 < -0.001 || w2 < -0.001) continue;

          // Interpolate image coordinates via barycentric weights
          const imgX = w0 * ix0 + w1 * ix1 + w2 * ix2;
          const imgY = w0 * iy0 + w1 * iy1 + w2 * iy2;

          // Clamp to image bounds (extends edge pixels outward for better coverage)
          const clampX = Math.max(0, Math.min(imgX, 0.9999));
          const clampY = Math.max(0, Math.min(imgY, 0.9999));
          // Skip pixels far outside (>15% beyond edge) — those are extrapolation artifacts
          if (imgX < -0.15 || imgX > 1.15 || imgY < -0.15 || imgY > 1.15) continue;

          // Skip source pixels outside the face boundary polygon
          if (faceBoundaryMask) {
            const mx = Math.min(srcW - 1, Math.max(0, Math.round(clampX * (srcW - 1))));
            const my = Math.min(srcH - 1, Math.max(0, Math.round(clampY * (srcH - 1))));
            if (!faceBoundaryMask[my * srcW + mx]) continue;
          }

          const outIdx = (py * size + px) * 4;
          const rgba = this._bilinearSample(srcData, srcW, srcH, clampX * srcW, clampY * srcH);
          out[outIdx]     = rgba[0];
          out[outIdx + 1] = rgba[1];
          out[outIdx + 2] = rgba[2];
          out[outIdx + 3] = faceAlpha;
        }
      }
    }
  }

  /**
   * Bilinear sample from RGBA image data.
   */
  _bilinearSample(data, w, h, fx, fy) {
    const x0 = Math.max(0, Math.min(Math.floor(fx), w - 1));
    const y0 = Math.max(0, Math.min(Math.floor(fy), h - 1));
    const x1 = Math.min(x0 + 1, w - 1);
    const y1 = Math.min(y0 + 1, h - 1);
    const tx = Math.max(0, fx - x0);
    const ty = Math.max(0, fy - y0);

    const i00 = (y0 * w + x0) * 4;
    const i10 = (y0 * w + x1) * 4;
    const i01 = (y1 * w + x0) * 4;
    const i11 = (y1 * w + x1) * 4;

    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;

    return [
      Math.round(w00 * data[i00]     + w10 * data[i10]     + w01 * data[i01]     + w11 * data[i11]),
      Math.round(w00 * data[i00 + 1] + w10 * data[i10 + 1] + w01 * data[i01 + 1] + w11 * data[i11 + 1]),
      Math.round(w00 * data[i00 + 2] + w10 * data[i10 + 2] + w01 * data[i01 + 2] + w11 * data[i11 + 2]),
    ];
  }

  // -----------------------------------------------------------------------
  // Fill unmapped regions
  // -----------------------------------------------------------------------

  /**
   * Fill unmapped (alpha=0) pixels with averaged skin colour from mapped pixels.
   */
  _fillUnmappedRegions(outImageData) {
    const data = outImageData.data;
    const total = data.length / 4;

    // Compute average colour from mapped pixels
    let rSum = 0, gSum = 0, bSum = 0, count = 0;
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 255) {
        rSum += data[i * 4];
        gSum += data[i * 4 + 1];
        bSum += data[i * 4 + 2];
        count++;
      }
    }

    const avgR = count > 0 ? Math.round(rSum / count) : 210;
    const avgG = count > 0 ? Math.round(gSum / count) : 170;
    const avgB = count > 0 ? Math.round(bSum / count) : 145;

    // Fill unmapped pixels
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) {
        data[i * 4]     = avgR;
        data[i * 4 + 1] = avgG;
        data[i * 4 + 2] = avgB;
        data[i * 4 + 3] = 255;
      }
    }
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
      height: size
    };
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
      height: size
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
}

export default PhotoUploader;
