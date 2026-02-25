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
   * Warp the photo into FLAME UV space using piecewise affine transformation.
   */
  _warpTextureToUV(img, landmarks, mapping, meshGen) {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // 1. Draw source image onto a temp canvas for pixel sampling
    const maxSrc = 2048; // cap for performance
    const scale = Math.min(1, maxSrc / Math.max(img.naturalWidth, img.naturalHeight));
    const srcW = Math.round(img.naturalWidth * scale);
    const srcH = Math.round(img.naturalHeight * scale);
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    const srcCtx = srcCanvas.getContext('2d');
    srcCtx.drawImage(img, 0, 0, srcW, srcH);
    const srcImageData = srcCtx.getImageData(0, 0, srcW, srcH);

    // 2. Build control point pairs:  (imgX, imgY) normalised 0-1  ↔  (uvU, uvV) 0-1
    const controlPoints = this._buildControlPoints(landmarks, mapping, meshGen);
    console.log(`PhotoUploader: ${controlPoints.length} control points mapped`);

    if (controlPoints.length < 10) {
      console.warn('PhotoUploader: Too few control points — falling back to naive projection');
      return this._naiveTextureProjection(img);
    }

    // 3. Build arrays for Delaunay (UV space) and corresponding image-space coords
    const allUV = [];
    const allImg = [];
    for (const cp of controlPoints) {
      allUV.push({ x: cp.uvU, y: cp.uvV });
      allImg.push({ x: cp.imgX, y: cp.imgY });
    }

    // 4. Add border/corner points
    this._addBorderPoints(allUV, allImg);

    // 5. Delaunay triangulate the UV-space points
    const triangles = this._delaunayTriangulate(allUV);
    console.log(`PhotoUploader: Delaunay produced ${triangles.length} triangles`);

    // 6. Rasterise the warp
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeWarp(outImageData, size, allUV, allImg, triangles, srcImageData);

    // 7. Fill unmapped areas with skin colour
    this._fillUnmappedRegions(outImageData);
    ctx.putImageData(outImageData, 0, 0);

    console.log('PhotoUploader: ✅ Piecewise affine warp complete');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.92),
      width: size,
      height: size
    };
  }

  /**
   * Build 105 control point pairs mapping image landmarks → FLAME UV coords.
   */
  _buildControlPoints(landmarks, mapping, meshGen) {
    const uvCoords = meshGen.flameUVCoords;   // Float32Array  5118*2
    const uvFaces = meshGen.flameUVFaces;     // Uint32Array   9976*3
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;

    const points = [];
    const EPS = 1e-4;

    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;

      // Image-space landmark (normalised 0–1)
      const lm = landmarks[mpIdx];
      const imgX = lm.x;
      const imgY = lm.y;

      // FLAME UV position via barycentric interpolation
      const fIdx = faceIndices[i];
      const uvi0 = uvFaces[fIdx * 3 + 0];
      const uvi1 = uvFaces[fIdx * 3 + 1];
      const uvi2 = uvFaces[fIdx * 3 + 2];

      const u0 = uvCoords[uvi0 * 2],     v0 = uvCoords[uvi0 * 2 + 1];
      const u1 = uvCoords[uvi1 * 2],     v1 = uvCoords[uvi1 * 2 + 1];
      const u2 = uvCoords[uvi2 * 2],     v2 = uvCoords[uvi2 * 2 + 1];

      const bc = baryCoords[i];
      const uvU = u0 * bc[0] + u1 * bc[1] + u2 * bc[2];
      const uvV = v0 * bc[0] + v1 * bc[1] + v2 * bc[2];

      // Skip degenerate points
      if (isNaN(uvU) || isNaN(uvV) || isNaN(imgX) || isNaN(imgY)) continue;
      if (uvU < -EPS || uvU > 1 + EPS || uvV < -EPS || uvV > 1 + EPS) continue;

      points.push({ imgX, imgY, uvU, uvV });
    }

    // Deduplicate: remove points too close in UV space (< 0.003)
    const MIN_DIST_SQ = 0.003 * 0.003;
    const deduped = [];
    for (const p of points) {
      let tooClose = false;
      for (const q of deduped) {
        const du = p.uvU - q.uvU;
        const dv = p.uvV - q.uvV;
        if (du * du + dv * dv < MIN_DIST_SQ) { tooClose = true; break; }
      }
      if (!tooClose) deduped.push(p);
    }
    return deduped;
  }

  /**
   * Add border / corner points so the Delaunay triangulation covers the full
   * UV [0,1]² square.  For each added UV point, we estimate an image-space
   * position by extrapolating from the nearest existing control point.
   */
  _addBorderPoints(allUV, allImg) {
    const borderUVs = [
      { x: 0,    y: 0 },    { x: 0.5,  y: 0 },    { x: 1,    y: 0 },
      { x: 0,    y: 0.25 }, { x: 1,    y: 0.25 },
      { x: 0,    y: 0.5 },  { x: 1,    y: 0.5 },
      { x: 0,    y: 0.75 }, { x: 1,    y: 0.75 },
      { x: 0,    y: 1 },    { x: 0.5,  y: 1 },    { x: 1,    y: 1 },
      { x: 0.25, y: 0 },    { x: 0.75, y: 0 },
      { x: 0.25, y: 1 },    { x: 0.75, y: 1 },
    ];

    for (const bUV of borderUVs) {
      // Find nearest existing control point
      let bestDist = Infinity;
      let bestIdx = 0;
      for (let i = 0; i < allUV.length; i++) {
        const dx = allUV[i].x - bUV.x;
        const dy = allUV[i].y - bUV.y;
        const d = dx * dx + dy * dy;
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }

      // Extrapolate image position from nearest point
      const nearUV = allUV[bestIdx];
      const nearImg = allImg[bestIdx];
      const offsetU = bUV.x - nearUV.x;
      const offsetV = bUV.y - nearUV.y;

      allUV.push(bUV);
      allImg.push({
        x: Math.max(0, Math.min(1, nearImg.x + offsetU * 0.5)),
        y: Math.max(0, Math.min(1, nearImg.y + offsetV * 0.5)),
      });
    }
  }

  // -----------------------------------------------------------------------
  // Delaunay triangulation (Bowyer-Watson)
  // -----------------------------------------------------------------------

  /**
   * Bowyer-Watson incremental Delaunay triangulation.
   * @param {Array<{x:number, y:number}>} points
   * @returns {Array<[number, number, number]>} Triangle index triples
   */
  _delaunayTriangulate(points) {
    const n = points.length;
    if (n < 3) return [];

    // Super-triangle that encloses the [0,1]² square with margin
    const st0 = { x: -5, y: -5 };
    const st1 = { x: 15, y: -5 };
    const st2 = { x: 5,  y: 15 };
    const allPts = [...points, st0, st1, st2];
    const si0 = n, si1 = n + 1, si2 = n + 2;

    let tris = [{ a: si0, b: si1, c: si2 }];

    for (let i = 0; i < n; i++) {
      const p = allPts[i];
      const bad = [];

      for (let t = 0; t < tris.length; t++) {
        const tri = tris[t];
        if (this._inCircumcircle(p, allPts[tri.a], allPts[tri.b], allPts[tri.c])) {
          bad.push(t);
        }
      }

      // Build boundary polygon of the hole
      const edges = [];
      for (const ti of bad) {
        const tri = tris[ti];
        const te = [[tri.a, tri.b], [tri.b, tri.c], [tri.c, tri.a]];
        for (const e of te) {
          const shared = bad.some(oi =>
            oi !== ti && this._triHasEdge(tris[oi], e[0], e[1])
          );
          if (!shared) edges.push(e);
        }
      }

      // Remove bad triangles (reverse order to preserve indices)
      bad.sort((a, b) => b - a);
      for (const ti of bad) tris.splice(ti, 1);

      // Create new triangles from boundary edges to inserted point
      for (const [ea, eb] of edges) {
        tris.push({ a: i, b: ea, c: eb });
      }
    }

    // Remove triangles referencing super-triangle vertices
    tris = tris.filter(t => t.a < n && t.b < n && t.c < n);
    return tris.map(t => [t.a, t.b, t.c]);
  }

  _inCircumcircle(p, a, b, c) {
    // Ensure consistent winding (CCW)
    const cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let A = a, B = b, C = c;
    if (cross < 0) { B = c; C = b; } // swap to CCW

    const ax = A.x - p.x, ay = A.y - p.y;
    const bx = B.x - p.x, by = B.y - p.y;
    const cx = C.x - p.x, cy = C.y - p.y;
    const det = (ax * ax + ay * ay) * (bx * cy - cx * by)
              - (bx * bx + by * by) * (ax * cy - cx * ay)
              + (cx * cx + cy * cy) * (ax * by - bx * ay);
    return det > 0;
  }

  _triHasEdge(tri, ea, eb) {
    const v = [tri.a, tri.b, tri.c];
    return v.includes(ea) && v.includes(eb);
  }

  // -----------------------------------------------------------------------
  // Rasteriser
  // -----------------------------------------------------------------------

  /**
   * Rasterise the piecewise affine warp from image to UV space.
   */
  _rasterizeWarp(outImageData, size, uvPts, imgPts, triangles, srcImageData) {
    const out = outImageData.data;
    const srcW = srcImageData.width;
    const srcH = srcImageData.height;
    const src = srcImageData.data;

    // Pre-compute per-triangle data
    const triData = triangles.map(([i, j, k]) => {
      const td = {
        // UV-space triangle (0-1)
        u0: uvPts[i].x, v0: uvPts[i].y,
        u1: uvPts[j].x, v1: uvPts[j].y,
        u2: uvPts[k].x, v2: uvPts[k].y,
        // Image-space triangle (normalised 0-1)
        ix0: imgPts[i].x, iy0: imgPts[i].y,
        ix1: imgPts[j].x, iy1: imgPts[j].y,
        ix2: imgPts[k].x, iy2: imgPts[k].y,
      };
      // Barycentric denominator
      td.denom = (td.v1 - td.v2) * (td.u0 - td.u2) + (td.u2 - td.u1) * (td.v0 - td.v2);
      // Bounding box in UV space (as pixel coords)
      td.minPx = Math.max(0, Math.floor(Math.min(td.u0, td.u1, td.u2) * size));
      td.maxPx = Math.min(size - 1, Math.ceil(Math.max(td.u0, td.u1, td.u2) * size));
      td.minPy = Math.max(0, Math.floor((1 - Math.max(td.v0, td.v1, td.v2)) * size));
      td.maxPy = Math.min(size - 1, Math.ceil((1 - Math.min(td.v0, td.v1, td.v2)) * size));
      return td;
    });

    // Build a spatial grid for accelerated triangle lookup (32×32 cells)
    const GRID = 32;
    const grid = new Array(GRID * GRID);
    for (let c = 0; c < grid.length; c++) grid[c] = [];
    for (let ti = 0; ti < triData.length; ti++) {
      const td = triData[ti];
      const minU = Math.min(td.u0, td.u1, td.u2);
      const maxU = Math.max(td.u0, td.u1, td.u2);
      const minV = Math.min(td.v0, td.v1, td.v2);
      const maxV = Math.max(td.v0, td.v1, td.v2);
      const gx0 = Math.max(0, Math.floor(minU * GRID));
      const gx1 = Math.min(GRID - 1, Math.floor(maxU * GRID));
      const gy0 = Math.max(0, Math.floor(minV * GRID));
      const gy1 = Math.min(GRID - 1, Math.floor(maxV * GRID));
      for (let gy = gy0; gy <= gy1; gy++) {
        for (let gx = gx0; gx <= gx1; gx++) {
          grid[gy * GRID + gx].push(ti);
        }
      }
    }

    for (let py = 0; py < size; py++) {
      // No V-flip here: Three.js loadTexture sets flipY=true which handles
      // the canvas top-to-bottom → OpenGL bottom-to-top conversion.
      const v = (py + 0.5) / size;
      const gy = Math.min(GRID - 1, Math.max(0, Math.floor(v * GRID)));

      for (let px = 0; px < size; px++) {
        const u = (px + 0.5) / size;
        const outIdx = (py * size + px) * 4;
        const gx = Math.min(GRID - 1, Math.max(0, Math.floor(u * GRID)));

        // Check triangles in this grid cell
        const candidates = grid[gy * GRID + gx];
        let found = false;

        for (const ti of candidates) {
          const td = triData[ti];
          if (Math.abs(td.denom) < 1e-12) continue;

          const w0 = ((td.v1 - td.v2) * (u - td.u2) + (td.u2 - td.u1) * (v - td.v2)) / td.denom;
          const w1 = ((td.v2 - td.v0) * (u - td.u2) + (td.u0 - td.u2) * (v - td.v2)) / td.denom;
          const w2 = 1.0 - w0 - w1;

          if (w0 >= -0.001 && w1 >= -0.001 && w2 >= -0.001) {
            // Map to image space (normalised 0-1)
            const imgX = w0 * td.ix0 + w1 * td.ix1 + w2 * td.ix2;
            const imgY = w0 * td.iy0 + w1 * td.iy1 + w2 * td.iy2;

            // Bilinear sample
            const rgba = this._bilinearSample(src, srcW, srcH, imgX * srcW, imgY * srcH);
            out[outIdx]     = rgba[0];
            out[outIdx + 1] = rgba[1];
            out[outIdx + 2] = rgba[2];
            out[outIdx + 3] = 255;
            found = true;
            break;
          }
        }

        if (!found) {
          out[outIdx + 3] = 0; // sentinel: unmapped
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
