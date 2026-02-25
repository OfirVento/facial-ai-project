/**
 * Photo Upload & Reconstruction Pipeline
 * Handles photo capture/upload, landmark-guided UV projection,
 * and processes the returned FLAME mesh + texture.
 *
 * When a FLAME mesh generator and MediaPipe bridge are available,
 * generateTextureFromPhoto() uses a piecewise affine warp to
 * project the photo into FLAME UV space using 105 landmark
 * correspondences.  Otherwise falls back to a naive centre-crop.
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
    return this._warpTextureToUV(img, landmarks, mapping, meshGen);
  }

  // -----------------------------------------------------------------------
  // Piecewise affine warp — core pipeline
  // -----------------------------------------------------------------------

  /**
   * Project the photo onto FLAME UV space using camera projection.
   *
   * 1. Solve an affine 3D→2D camera from 105 landmark correspondences
   * 2. Project all FLAME template vertices to image space
   * 3. Rasterise every FLAME UV face: for each UV pixel, map to image via
   *    barycentric interpolation of the projected vertex positions.
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

    // --- 1. Build correspondences: 3D FLAME position ↔ 2D image (normalised 0-1) ---
    const posVerts = meshGen.flameTemplateVertices; // check existence
    const posFaces = meshGen.flameFaces;
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces  = meshGen.flameUVFaces;
    const uvToPosMap = meshGen.uvToPosMap;

    // We need the raw position vertices (5023×3). They're stored on the generator.
    // Build correspondence from the 105 mapped landmarks.
    const mpIndices   = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords  = mapping.lmk_b_coords;

    // Gather (3D position, 2D image) pairs
    const pairs = []; // { px, py, pz, ix, iy }
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const v0 = posFaces[fi * 3], v1 = posFaces[fi * 3 + 1], v2 = posFaces[fi * 3 + 2];

      // Interpolate 3D position using barycentric coords
      const px = bc[0] * posVerts[v0 * 3]     + bc[1] * posVerts[v1 * 3]     + bc[2] * posVerts[v2 * 3];
      const py = bc[0] * posVerts[v0 * 3 + 1] + bc[1] * posVerts[v1 * 3 + 1] + bc[2] * posVerts[v2 * 3 + 1];
      const pz = bc[0] * posVerts[v0 * 3 + 2] + bc[1] * posVerts[v1 * 3 + 2] + bc[2] * posVerts[v2 * 3 + 2];

      pairs.push({ px, py, pz, ix: lm.x, iy: lm.y });
    }
    console.log(`PhotoUploader: ${pairs.length} landmark correspondences for camera solve`);

    if (pairs.length < 6) {
      console.warn('PhotoUploader: Too few correspondences — falling back');
      return this._naiveTextureProjection(img);
    }

    // --- 2. Solve affine camera: img_x = a*X + b*Y + c*Z + d, img_y = e*X + f*Y + g*Z + h ---
    const proj = this._solveAffineCamera(pairs);
    console.log(`PhotoUploader: Camera solved — residual ${proj.residual.toFixed(5)}`);

    // --- 3. Project all 5023 position vertices to 2D image space ---
    const nPos = posVerts.length / 3;
    const projected = new Float32Array(nPos * 2);
    for (let i = 0; i < nPos; i++) {
      const X = posVerts[i * 3], Y = posVerts[i * 3 + 1], Z = posVerts[i * 3 + 2];
      projected[i * 2]     = proj.a * X + proj.b * Y + proj.c * Z + proj.d; // img_x (0-1)
      projected[i * 2 + 1] = proj.e * X + proj.f * Y + proj.g * Z + proj.h; // img_y (0-1)
    }

    // --- 4. Build per-UV-vertex image coordinates via uvToPosMap ---
    const nUV = uvCoords.length / 2;
    const uvImgCoords = new Float32Array(nUV * 2);
    for (let i = 0; i < nUV; i++) {
      const posIdx = uvToPosMap[i];
      uvImgCoords[i * 2]     = projected[posIdx * 2];
      uvImgCoords[i * 2 + 1] = projected[posIdx * 2 + 1];
    }

    // --- 5. Rasterise all 9976 UV faces ---
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH);

    // --- 6. Fill unmapped areas ---
    this._fillUnmappedRegions(outImageData);
    ctx.putImageData(outImageData, 0, 0);

    console.log('PhotoUploader: ✅ Mesh-based texture projection complete');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.92),
      width: size,
      height: size
    };
  }

  // -----------------------------------------------------------------------
  // Camera solver — affine 3D→2D projection
  // -----------------------------------------------------------------------

  /**
   * Solve an affine camera model from (3D, 2D) correspondences.
   *   img_x = a*X + b*Y + c*Z + d
   *   img_y = e*X + f*Y + g*Z + h
   *
   * Uses least-squares via normal equations.
   * @returns {{ a,b,c,d,e,f,g,h, residual:number }}
   */
  _solveAffineCamera(pairs) {
    const n = pairs.length;

    // Build A (n×4) and bx, by (n×1)
    const A = new Float64Array(n * 4);
    const bx = new Float64Array(n);
    const by = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const p = pairs[i];
      A[i * 4]     = p.px;
      A[i * 4 + 1] = p.py;
      A[i * 4 + 2] = p.pz;
      A[i * 4 + 3] = 1.0;
      bx[i] = p.ix;
      by[i] = p.iy;
    }

    // Solve A * px = bx  and  A * py = by  via normal equations: (A^T A) x = A^T b
    const px = this._leastSquares4(A, bx, n);
    const py = this._leastSquares4(A, by, n);

    // Compute residual
    let res = 0;
    for (let i = 0; i < n; i++) {
      const p = pairs[i];
      const ex = px[0]*p.px + px[1]*p.py + px[2]*p.pz + px[3] - p.ix;
      const ey = py[0]*p.px + py[1]*p.py + py[2]*p.pz + py[3] - p.iy;
      res += ex*ex + ey*ey;
    }
    res = Math.sqrt(res / n);

    return { a: px[0], b: px[1], c: px[2], d: px[3],
             e: py[0], f: py[1], g: py[2], h: py[3], residual: res };
  }

  /**
   * Solve (A^T A) x = A^T b  for 4 unknowns via Gauss elimination.
   * A is n×4 (flat), b is n×1.
   */
  _leastSquares4(A, b, n) {
    // Build 4×4 normal matrix ATA and 4×1 ATb
    const ATA = new Float64Array(16);
    const ATb = new Float64Array(4);
    for (let i = 0; i < n; i++) {
      const a0 = A[i*4], a1 = A[i*4+1], a2 = A[i*4+2], a3 = A[i*4+3];
      const bi = b[i];
      ATA[0]  += a0*a0; ATA[1]  += a0*a1; ATA[2]  += a0*a2; ATA[3]  += a0*a3;
      ATA[4]  += a1*a0; ATA[5]  += a1*a1; ATA[6]  += a1*a2; ATA[7]  += a1*a3;
      ATA[8]  += a2*a0; ATA[9]  += a2*a1; ATA[10] += a2*a2; ATA[11] += a2*a3;
      ATA[12] += a3*a0; ATA[13] += a3*a1; ATA[14] += a3*a2; ATA[15] += a3*a3;
      ATb[0] += a0*bi; ATb[1] += a1*bi; ATb[2] += a2*bi; ATb[3] += a3*bi;
    }

    // Gauss elimination on [ATA | ATb]  (4×5 augmented)
    const M = [
      [ATA[0],  ATA[1],  ATA[2],  ATA[3],  ATb[0]],
      [ATA[4],  ATA[5],  ATA[6],  ATA[7],  ATb[1]],
      [ATA[8],  ATA[9],  ATA[10], ATA[11], ATb[2]],
      [ATA[12], ATA[13], ATA[14], ATA[15], ATb[3]],
    ];

    // Forward elimination with partial pivoting
    for (let col = 0; col < 4; col++) {
      let maxRow = col, maxVal = Math.abs(M[col][col]);
      for (let row = col + 1; row < 4; row++) {
        if (Math.abs(M[row][col]) > maxVal) {
          maxVal = Math.abs(M[row][col]);
          maxRow = row;
        }
      }
      if (maxRow !== col) { const tmp = M[col]; M[col] = M[maxRow]; M[maxRow] = tmp; }
      if (Math.abs(M[col][col]) < 1e-15) continue;

      for (let row = col + 1; row < 4; row++) {
        const factor = M[row][col] / M[col][col];
        for (let j = col; j < 5; j++) M[row][j] -= factor * M[col][j];
      }
    }

    // Back substitution
    const x = [0, 0, 0, 0];
    for (let i = 3; i >= 0; i--) {
      let sum = M[i][4];
      for (let j = i + 1; j < 4; j++) sum -= M[i][j] * x[j];
      x[i] = Math.abs(M[i][i]) > 1e-15 ? sum / M[i][i] : 0;
    }
    return x;
  }

  // -----------------------------------------------------------------------
  // Rasteriser — FLAME mesh faces
  // -----------------------------------------------------------------------

  /**
   * Rasterise all FLAME UV faces, sampling the source photo via projected
   * per-vertex image coordinates.
   */
  _rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH) {
    const out = outImageData.data;
    const nFaces = uvFaces.length / 3;

    // Per-pixel Z-buffer (depth) not needed since faces don't overlap in UV space.
    // We just rasterise every face and fill the output.

    for (let f = 0; f < nFaces; f++) {
      const vi0 = uvFaces[f * 3], vi1 = uvFaces[f * 3 + 1], vi2 = uvFaces[f * 3 + 2];

      // UV-space triangle (0-1)
      const u0 = uvCoords[vi0 * 2], v0 = uvCoords[vi0 * 2 + 1];
      const u1 = uvCoords[vi1 * 2], v1 = uvCoords[vi1 * 2 + 1];
      const u2 = uvCoords[vi2 * 2], v2 = uvCoords[vi2 * 2 + 1];

      // Image-space coordinates (normalised 0-1) for each UV vertex
      const ix0 = uvImgCoords[vi0 * 2], iy0 = uvImgCoords[vi0 * 2 + 1];
      const ix1 = uvImgCoords[vi1 * 2], iy1 = uvImgCoords[vi1 * 2 + 1];
      const ix2 = uvImgCoords[vi2 * 2], iy2 = uvImgCoords[vi2 * 2 + 1];

      // Skip faces whose projected image coords are entirely outside [0,1]
      const allIx = [ix0, ix1, ix2], allIy = [iy0, iy1, iy2];
      if (Math.min(...allIx) > 1.1 || Math.max(...allIx) < -0.1 ||
          Math.min(...allIy) > 1.1 || Math.max(...allIy) < -0.1) continue;

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
        const v = (py + 0.5) / size;
        for (let px = minPx; px <= maxPx; px++) {
          const u = (px + 0.5) / size;

          const w0 = ((v1 - v2) * (u - u2) + (u2 - u1) * (v - v2)) * invDenom;
          const w1 = ((v2 - v0) * (u - u2) + (u0 - u2) * (v - v2)) * invDenom;
          const w2 = 1.0 - w0 - w1;

          if (w0 < -0.001 || w1 < -0.001 || w2 < -0.001) continue;

          // Map to image space
          const imgX = w0 * ix0 + w1 * ix1 + w2 * ix2;
          const imgY = w0 * iy0 + w1 * iy1 + w2 * iy2;

          // Clamp to image bounds
          if (imgX < 0 || imgX > 1 || imgY < 0 || imgY > 1) continue;

          const outIdx = (py * size + px) * 4;
          const rgba = this._bilinearSample(srcData, srcW, srcH, imgX * srcW, imgY * srcH);
          out[outIdx]     = rgba[0];
          out[outIdx + 1] = rgba[1];
          out[outIdx + 2] = rgba[2];
          out[outIdx + 3] = 255;
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
