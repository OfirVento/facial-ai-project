/**
 * TextureProjector
 *
 * Handles all texture-projection logic extracted from PhotoUploader:
 *   - TPS-based UV-to-image mapping (single-view)
 *   - Multi-view projection pipeline with N-V confidence blending
 *   - Dense camera-based texture baking
 *   - Vertex projection & face-visibility computation
 *   - Diagnostic overlay / heatmap generators
 *   - Face-boundary masking & UV utilities
 *   - Mesh-face rasterisation with depth buffering
 *
 * Cross-module dependencies:
 *   - TexturePostProcess  (post-processing: delight, erode, blend, etc.)
 *   - ShapeFitter          (pose refinement & camera RMSE)
 *
 * @module TextureProjector
 */

export class TextureProjector {
  /**
   * @param {object} [options]
   * @param {import('./TexturePostProcess.js').TexturePostProcess} [options.postProcessor]
   * @param {import('./ShapeFitter.js').ShapeFitter} [options.shapeFitter]
   */
  constructor(options = {}) {
    /** @type {import('./TexturePostProcess.js').TexturePostProcess|null} */
    this._postProcessor = options.postProcessor ?? null;
    /** @type {import('./ShapeFitter.js').ShapeFitter|null} */
    this._shapeFitter = options.shapeFitter ?? null;
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
   *    This gives 105 (u,v) <-> (imgX, imgY) correspondences.
   *
   * 2. Fit two TPS functions:  UV -> image_x  and  UV -> image_y.
   *    TPS provides smooth, globally consistent interpolation through all
   *    105 control points, handling non-linear warps gracefully.
   *
   * 3. Evaluate TPS at ALL 5118 FLAME UV vertex positions to obtain
   *    per-vertex image coordinates.  This is the key difference from the
   *    Delaunay approach: every single UV vertex gets a smooth, high-quality
   *    image mapping -- not just the 105 control points.
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

    // --- 1. Build UV <-> image control points from 105 mapped landmarks ---
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
    console.log(`TextureProjector: ${N} UV↔image control points for TPS`);
    for (let i = 0; i < Math.min(5, N); i++) {
      console.log(`  [${i}] UV(${ctrlU[i].toFixed(4)}, ${ctrlV[i].toFixed(4)}) → Img(${ctrlIX[i].toFixed(4)}, ${ctrlIY[i].toFixed(4)})`);
    }

    if (N < 10) {
      console.warn('TextureProjector: Too few control points — falling back');
      return this._naiveTextureProjection(img);
    }

    // ===== DIAGNOSTIC: Draw debug overlays =====
    this._lastDebugData = {
      srcCanvas, srcW, srcH, img,
      ctrlU, ctrlV, ctrlIX, ctrlIY, N,
      landmarks, mpIndices, mapping
    };
    this._drawDebugOverlays(srcCanvas, srcW, srcH, ctrlU, ctrlV, ctrlIX, ctrlIY, N);

    // --- 2. Fit TPS: UV -> image_x and UV -> image_y ---
    console.log('TextureProjector: Fitting TPS interpolation…');
    const tpsX = this._fitTPS(ctrlU, ctrlV, ctrlIX);
    const tpsY = this._fitTPS(ctrlU, ctrlV, ctrlIY);
    console.log('TextureProjector: TPS fitted successfully');

    // ===== DIAGNOSTIC: TPS interpolation error at control points =====
    let maxErrX = 0, maxErrY = 0;
    for (let i = 0; i < N; i++) {
      const predX = this._evalTPS(tpsX, ctrlU[i], ctrlV[i]);
      const predY = this._evalTPS(tpsY, ctrlU[i], ctrlV[i]);
      maxErrX = Math.max(maxErrX, Math.abs(predX - ctrlIX[i]));
      maxErrY = Math.max(maxErrY, Math.abs(predY - ctrlIY[i]));
    }
    console.log(`TextureProjector DIAG: TPS max control-point error: X=${maxErrX.toFixed(6)}, Y=${maxErrY.toFixed(6)}`);

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
    console.log(`TextureProjector DIAG: TPS output range: X=[${minIX.toFixed(4)}, ${maxIX.toFixed(4)}], Y=[${minIY.toFixed(4)}, ${maxIY.toFixed(4)}]`);
    console.log(`TextureProjector: TPS evaluated at ${nUV} UV vertices`);

    // --- 4. Rasterise all 9976 FLAME UV faces ---
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH);

    // ===== DIAGNOSTIC: Count mapped vs unmapped pixels =====
    let mappedPixels = 0;
    const totalPixels = size * size;
    for (let i = 0; i < totalPixels; i++) {
      if (outImageData.data[i * 4 + 3] === 255) mappedPixels++;
    }
    console.log(`TextureProjector DIAG: Mapped ${mappedPixels}/${totalPixels} pixels (${(mappedPixels/totalPixels*100).toFixed(1)}%) before fill`);

    // ===== DIAGNOSTIC: Save pre-fill texture =====
    const preFillCanvas = document.createElement('canvas');
    preFillCanvas.width = size;
    preFillCanvas.height = size;
    preFillCanvas.getContext('2d').putImageData(outImageData, 0, 0);
    this._debugPreFillTexture = preFillCanvas.toDataURL('image/png');

    // --- 5. Fill unmapped areas ---
    this._fillUnmappedRegions(outImageData);
    ctx.putImageData(outImageData, 0, 0);

    console.log('TextureProjector: ✅ TPS + mesh texture projection complete');

    // ===== DIAGNOSTIC: Save final texture =====
    this._debugFinalTexture = canvas.toDataURL('image/png');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/png'),
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
   * - Debug B: 1024x1024 UV atlas with 105 UV control point dots
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

    // Draw UV points (V mapped directly: v=0->top, v=1->bottom of canvas)
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

    console.log('TextureProjector DIAG: Debug overlays generated (photo + UV atlas)');
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
    // Top-left corner of canvas -> (U~0, V~0 in canvas space)
    ctx.fillStyle = '#ff0000';
    ctx.fillText('TL: U=0, V=0', size * 0.25, 40);
    ctx.fillText('(should=chin-right)', size * 0.25, 70);
    // Top-right -> (U~1, V~0)
    ctx.fillStyle = '#00cc00';
    ctx.fillText('TR: U=1, V=0', size * 0.75, 40);
    ctx.fillText('(should=chin-left)', size * 0.75, 70);
    // Bottom-left -> (U~0, V~1)
    ctx.fillStyle = '#0066ff';
    ctx.fillText('BL: U=0, V=1', size * 0.25, size - 20);
    ctx.fillText('(should=scalp-right)', size * 0.25, size - 50);
    // Bottom-right -> (U~1, V~1)
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
  // Multi-View Projection Pipeline
  // -----------------------------------------------------------------------

  /**
   * Multi-view texture projection: accumulates UV projections from front +
   * left45 + right45 views with N-V confidence weighting.
   *
   * Each view is projected independently, then merged:
   * - Per-pixel confidence = face visibility (N-V) from that view's camera
   * - Overlapping regions: weighted average by confidence
   * - Non-overlapping regions: single view fills in
   * - Post-processing (erosion, softening, Laplacian blend) applied once
   */
  async _multiViewProjectTextureToUV(frontImg, frontLandmarks, mapping, meshGen, frontCamera, bridge, photos) {
    const size = 2048;
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;

    // Collect all available views
    const views = [];

    // Front view (already detected + fitted)
    views.push({
      label: 'front',
      img: frontImg,
      landmarks: frontLandmarks,
      camera: frontCamera,
    });

    // Side views: detect landmarks and estimate camera for each
    for (const angle of ['left45', 'right45']) {
      if (!photos[angle]) continue;

      try {
        const sideImg = new Image();
        sideImg.crossOrigin = 'anonymous';
        await new Promise((resolve, reject) => {
          sideImg.onload = resolve;
          sideImg.onerror = reject;
          sideImg.src = photos[angle].url;
        });

        const sideLandmarks = await bridge.detectFromImage(sideImg);
        if (!sideLandmarks || sideLandmarks.length < 468) {
          console.warn(`TextureProjector MULTI: No face detected in ${angle} photo, skipping`);
          continue;
        }

        // Estimate initial camera from the SAME fitted mesh (shape/expression from front)
        const initialCamera = this._shapeFitter._estimateCameraFromLandmarks(sideLandmarks, mapping, meshGen);
        if (!initialCamera) {
          console.warn(`TextureProjector MULTI: Camera estimation failed for ${angle}, skipping`);
          continue;
        }

        // Per-view pose refinement: keep beta fixed, refine R + scale/translation
        // via Levenberg-Marquardt to minimize reprojection error
        console.log(`TextureProjector MULTI: Refining ${angle} pose...`);
        const refinedCamera = this._shapeFitter._refineViewPose(sideLandmarks, mapping, meshGen, initialCamera);

        // Quality gate: reject side views with high reprojection error
        // (indicates the pose is unstable -- don't let it corrupt the texture)
        const finalRMSE = this._shapeFitter._computeCameraRMSE(sideLandmarks, mapping, meshGen, refinedCamera);
        if (finalRMSE > 0.05) {
          console.warn(`TextureProjector MULTI: ${angle} RMSE ${finalRMSE.toFixed(4)} > 0.05, skipping (pose unstable)`);
          continue;
        }

        console.log(`TextureProjector MULTI: ${angle} camera (refined): sx=${refinedCamera.sx.toFixed(4)}, sy=${refinedCamera.sy.toFixed(4)}, RMSE=${finalRMSE.toFixed(4)}`);
        views.push({
          label: angle,
          img: sideImg,
          landmarks: sideLandmarks,
          camera: refinedCamera,
        });
      } catch (err) {
        console.warn(`TextureProjector MULTI: Error processing ${angle}:`, err.message);
      }
    }

    console.log(`TextureProjector MULTI: Processing ${views.length} view(s): ${views.map(v => v.label).join(', ')}`);

    if (views.length < 2) {
      console.warn('TextureProjector MULTI: Only 1 view available, falling back to single-view');
      return null; // Let caller fall back to single-view pipeline
    }

    // --- Project each view to UV space ---
    const viewProjections = [];
    for (const view of views) {
      const proj = this._projectViewToUVRaw(view.img, view.landmarks, mapping, meshGen, view.camera, size);
      if (proj) {
        viewProjections.push({ ...view, ...proj });
        console.log(`TextureProjector MULTI: ${view.label} projected (${proj.mappedPixels} mapped pixels)`);
      }
    }

    if (viewProjections.length < 2) {
      console.warn('TextureProjector MULTI: Less than 2 successful projections, falling back');
      return null;
    }

    // --- Accumulate views with N-V confidence weighting ---
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const outImageData = ctx.createImageData(size, size);
    const od = outImageData.data;
    const N = size * size;

    // --- Front-priority accumulation ---
    // Front view is the "hero" -- it has the best alignment (shape/expression
    // were fitted to it). Side views should only fill where the front view
    // has poor or no coverage. This prevents forehead patch artifacts from
    // side view misalignment in the central face region.
    //
    // Strategy: front view gets a large weight boost (FRONT_BOOST).
    // Side views only contribute meaningfully where front confidence is low.
    const FRONT_BOOST = 4.0;  // Front view weight multiplier
    const SIDE_SUPPRESS_THRESHOLD = 0.5; // If front N-V > this, suppress side views

    const frontProj = viewProjections[0];
    const frontData = frontProj.imageData.data;
    const frontConf = frontProj.confidence;

    // First pass: write front view everywhere it has data
    let totalMapped = 0;
    for (let i = 0; i < N; i++) {
      if (frontData[i * 4 + 3] === 0) continue;
      const w = frontConf[i];
      if (w <= 0) continue;
      od[i * 4]     = frontData[i * 4];
      od[i * 4 + 1] = frontData[i * 4 + 1];
      od[i * 4 + 2] = frontData[i * 4 + 2];
      od[i * 4 + 3] = 255;
      totalMapped++;
    }

    // Second pass: blend in side views where front is weak or absent
    let sideContributions = 0;
    for (let v = 1; v < viewProjections.length; v++) {
      const sideProj = viewProjections[v];
      const sd = sideProj.imageData.data;
      const sc = sideProj.confidence;

      for (let i = 0; i < N; i++) {
        if (sd[i * 4 + 3] === 0) continue;
        const sideW = sc[i];
        if (sideW <= 0) continue;

        const frontW = frontConf[i] * FRONT_BOOST;

        if (frontData[i * 4 + 3] === 0 || frontConf[i] <= 0) {
          // Front has NO data here -- side view fills in entirely
          od[i * 4]     = sd[i * 4];
          od[i * 4 + 1] = sd[i * 4 + 1];
          od[i * 4 + 2] = sd[i * 4 + 2];
          od[i * 4 + 3] = 255;
          totalMapped++;
          sideContributions++;
        } else if (frontConf[i] < SIDE_SUPPRESS_THRESHOLD) {
          // Front has weak coverage -- blend side view in proportionally
          // Lower front confidence -> more side contribution
          const totalW = frontW + sideW;
          const frontBlend = frontW / totalW;
          const sideBlend = sideW / totalW;

          od[i * 4]     = Math.round(od[i * 4] * frontBlend + sd[i * 4] * sideBlend);
          od[i * 4 + 1] = Math.round(od[i * 4 + 1] * frontBlend + sd[i * 4 + 1] * sideBlend);
          od[i * 4 + 2] = Math.round(od[i * 4 + 2] * frontBlend + sd[i * 4 + 2] * sideBlend);
          sideContributions++;
        }
        // else: front has strong coverage -- ignore side view entirely
      }
    }
    console.log(`TextureProjector MULTI: Accumulated ${totalMapped}/${N} pixels, side views contributed to ${sideContributions} pixels`);

    // --- Color harmonization: match side views' brightness to front ---
    // Front view is "hero" -- side views may have different lighting
    this._harmonizeMultiViewColors(outImageData, viewProjections, size);

    // --- Store diagnostic data for overlays ---
    this._diagCameraParams = frontCamera;
    this._diagFaceVisibility = viewProjections[0].faceVisibility;
    this._diagProjectedCoords = viewProjections[0].projectedCoords;
    this._diagMeshGen = meshGen;
    this._lastDebugData = viewProjections[0].debugData;

    // --- Apply UV mask ---
    const uvMask = this._buildUVMask(size, uvCoords, uvFaces);
    this._applyUVMask(outImageData, size, uvMask);

    // --- Edge dilation ---
    const preDilationAlpha = new Uint8Array(N);
    for (let i = 0; i < N; i++) preDilationAlpha[i] = outImageData.data[i * 4 + 3] > 0 ? 1 : 0;
    this._postProcessor._dilateEdges(outImageData, size, 80);
    this._postProcessor._blurDilatedBand(outImageData, size, preDilationAlpha);

    // --- Eye socket feathering (use front landmarks) ---
    this._postProcessor._featherEyeSockets(outImageData, size, frontLandmarks, mapping, uvCoords, uvFaces);

    // --- Diagnostic coverage ---
    let mappedPixels = 0;
    for (let i = 0; i < N; i++) {
      if (outImageData.data[i * 4 + 3] > 0) mappedPixels++;
    }
    console.log(`TextureProjector MULTI: ${mappedPixels}/${N} pixels mapped (${(mappedPixels / N * 100).toFixed(1)}%) after dilation`);

    // --- Save raw UV texture debug ---
    {
      const rawCanvas = document.createElement('canvas');
      rawCanvas.width = size; rawCanvas.height = size;
      rawCanvas.getContext('2d').putImageData(
        new ImageData(new Uint8ClampedArray(outImageData.data), size, size), 0, 0);
      this._debugPhotoUV_raw = rawCanvas.toDataURL('image/png');
    }

    // --- Delighting ---
    const preAvg = this._postProcessor._computeChannelAverages(outImageData);
    this._postProcessor._delightTexture(outImageData, size);
    this._postProcessor._restoreColorBalance(outImageData, preAvg);

    // --- Alpha erosion + softening ---
    const preErosionAlpha = new Uint8Array(N);
    for (let i = 0; i < N; i++) preErosionAlpha[i] = outImageData.data[i * 4 + 3];
    this._diagPreErosionAlpha = preErosionAlpha;
    this._diagTextureSize = size;

    // Expert round 4: increase erosion, drastically shrink softening.
    // Wide softening (85-128px) was pulling in background -> glowing halo.
    // Move boundary inward with more erosion instead of wider blur.
    const ERODE_PX = Math.max(12, Math.round(size / 120));   // ~17px at 2048 (was ~12)
    this._postProcessor._erodeAlpha(outImageData, size, ERODE_PX);
    // Extra erosion in jawline zone (V < 0.25) and forehead/scalp (V > 0.60)
    const JAWLINE_ERODE = Math.max(10, Math.round(size / 140));  // ~15px at 2048
    this._postProcessor._erodeAlphaRegion(outImageData, size, JAWLINE_ERODE, 0.0, 0.25);
    const FOREHEAD_EXTRA_ERODE = Math.max(10, Math.round(size / 140));
    this._postProcessor._erodeAlphaRegion(outImageData, size, FOREHEAD_EXTRA_ERODE, 0.60, 1.0);

    // Expert round 4: reduce softening to 20-40px (was 85-128px)
    const renderMode = this._renderModeHint || 'hybrid';
    const softenRadius = Math.max(12, Math.round(size / 64));   // ~32px at 2048
    const foreheadSoftenRadius = Math.max(16, Math.round(size / 48));  // ~43px at 2048
    this._postProcessor._softenAlphaBoundaryRegional(outImageData, size, softenRadius, foreheadSoftenRadius, 0.60);

    // --- Save alpha coverage debug ---
    {
      const alphaCanvas = document.createElement('canvas');
      alphaCanvas.width = size; alphaCanvas.height = size;
      const alphaCtx = alphaCanvas.getContext('2d');
      const alphaData = alphaCtx.createImageData(size, size);
      for (let i = 0; i < N; i++) {
        const a = outImageData.data[i * 4 + 3];
        alphaData.data[i * 4] = a; alphaData.data[i * 4 + 1] = a;
        alphaData.data[i * 4 + 2] = a; alphaData.data[i * 4 + 3] = 255;
      }
      alphaCtx.putImageData(alphaData, 0, 0);
      this._debugAlphaCoverage = alphaCanvas.toDataURL('image/png');
    }

    // --- Laplacian blend with albedo ---
    const albedoLayer = this._postProcessor._prepareAlbedoLayer(outImageData, meshGen, size, frontLandmarks, mapping);
    {
      const albCanvas = document.createElement('canvas');
      albCanvas.width = size; albCanvas.height = size;
      albCanvas.getContext('2d').putImageData(albedoLayer, 0, 0);
      this._debugAlbedoTinted = albCanvas.toDataURL('image/png');
    }

    const blendedData = this._postProcessor._laplacianBlend(outImageData, albedoLayer, size, 5);
    // DISABLED per expert round 4: ring artifacts likely from these passes.
    // Reintroduce one-by-one only after clean baseline is confirmed.
    // this._postProcessor._boundaryRingMatch(blendedData, outImageData, albedoLayer, size);
    // this._postProcessor._bakeSubtleAO(blendedData, outImageData, meshGen, size);

    ctx.putImageData(blendedData, 0, 0);
    console.log('TextureProjector MULTI: Multi-view projection complete');
    this._debugFinalTexture = canvas.toDataURL('image/png');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/png'),
      width: size,
      height: size
    };
  }

  /**
   * Project a single view to UV space, returning raw imageData + per-pixel
   * confidence (N-V). Used by multi-view accumulation pipeline.
   *
   * Returns: { imageData, confidence, faceVisibility, projectedCoords, mappedPixels, debugData }
   */
  _projectViewToUVRaw(img, landmarks, mapping, meshGen, camera, size) {
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;

    // Source image
    const maxSrc = 2048;
    const sc = Math.min(1, maxSrc / Math.max(img.naturalWidth, img.naturalHeight));
    const srcW = Math.round(img.naturalWidth * sc);
    const srcH = Math.round(img.naturalHeight * sc);
    const srcCanvas = document.createElement('canvas');
    srcCanvas.width = srcW; srcCanvas.height = srcH;
    srcCanvas.getContext('2d').drawImage(img, 0, 0, srcW, srcH);
    const srcData = srcCanvas.getContext('2d').getImageData(0, 0, srcW, srcH).data;

    // Project vertices
    const { coords: projectedCoords, depths: vertexDepths } = this._projectAllVertices(meshGen, camera);

    // Face visibility (N-V)
    const faceVisibility = this._computeFaceVisibility(meshGen, camera);

    // Map to UV
    const uvImgCoords = this._mapProjectionToUV(projectedCoords, meshGen, faceVisibility);

    // Compute UV depths for z-buffering
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;

    // Build debug data (UV control points for front view)
    const ctrlU = [], ctrlV = [], ctrlIX = [], ctrlIY = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      ctrlIX.push(lm.x); ctrlIY.push(lm.y);
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const uvi0 = uvFaces[fi * 3], uvi1 = uvFaces[fi * 3 + 1], uvi2 = uvFaces[fi * 3 + 2];
      const u = bc[0] * uvCoords[uvi0 * 2] + bc[1] * uvCoords[uvi1 * 2] + bc[2] * uvCoords[uvi2 * 2];
      const v = bc[0] * uvCoords[uvi0 * 2 + 1] + bc[1] * uvCoords[uvi1 * 2 + 1] + bc[2] * uvCoords[uvi2 * 2 + 1];
      ctrlU.push(u); ctrlV.push(v);
    }

    // UV depths
    const nUVVerts = uvCoords.length / 2;
    const uvDepths = new Float32Array(nUVVerts);

    // Excluded faces + boundary mask
    const excludedFaces = this._buildInnerMouthMask(meshGen);
    const faceBoundaryMask = this._buildFaceBoundaryMask(landmarks, srcW, srcH);

    // Rasterize
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size; tempCanvas.height = size;
    const imageData = tempCanvas.getContext('2d').createImageData(size, size);
    const expectedWindingSign = Math.sign(camera.sx * camera.sy);
    this._rasterizeMeshFaces(imageData, size, uvCoords, uvFaces, uvImgCoords,
      srcData, srcW, srcH, faceVisibility, excludedFaces, faceBoundaryMask,
      uvDepths, expectedWindingSign);

    // Apply UV mask
    const uvMask = this._buildUVMask(size, uvCoords, uvFaces);
    this._applyUVMask(imageData, size, uvMask);

    // Build per-pixel confidence from face visibility
    // For each UV pixel, find which face it belongs to and use that face's N-V
    const confidence = new Float32Array(size * size);
    const nFaces = uvFaces.length / 3;

    // Rasterize face visibility into UV pixel confidence
    for (let f = 0; f < nFaces; f++) {
      const vis = faceVisibility[f];
      if (vis <= 0) continue;

      const ui0 = uvFaces[f * 3], ui1 = uvFaces[f * 3 + 1], ui2 = uvFaces[f * 3 + 2];
      const ux0 = uvCoords[ui0 * 2] * size, uy0 = uvCoords[ui0 * 2 + 1] * size;
      const ux1 = uvCoords[ui1 * 2] * size, uy1 = uvCoords[ui1 * 2 + 1] * size;
      const ux2 = uvCoords[ui2 * 2] * size, uy2 = uvCoords[ui2 * 2 + 1] * size;

      const minX = Math.max(0, Math.floor(Math.min(ux0, ux1, ux2)));
      const maxX = Math.min(size - 1, Math.ceil(Math.max(ux0, ux1, ux2)));
      const minY = Math.max(0, Math.floor(Math.min(uy0, uy1, uy2)));
      const maxY = Math.min(size - 1, Math.ceil(Math.max(uy0, uy1, uy2)));

      const area = (ux1 - ux0) * (uy2 - uy0) - (ux2 - ux0) * (uy1 - uy0);
      if (Math.abs(area) < 0.01) continue;
      const invArea = 1.0 / area;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const w0 = ((ux1 - x) * (uy2 - y) - (ux2 - x) * (uy1 - y)) * invArea;
          const w1 = ((ux2 - x) * (uy0 - y) - (ux0 - x) * (uy2 - y)) * invArea;
          const w2 = 1.0 - w0 - w1;
          if (w0 < -0.01 || w1 < -0.01 || w2 < -0.01) continue;

          const idx = y * size + x;
          // Only set confidence where we have actual pixel data
          if (imageData.data[idx * 4 + 3] > 0) {
            confidence[idx] = Math.max(confidence[idx], vis);
          }
        }
      }
    }

    // Count mapped pixels
    let mappedPixels = 0;
    for (let i = 0; i < size * size; i++) {
      if (imageData.data[i * 4 + 3] > 0) mappedPixels++;
    }

    return {
      imageData,
      confidence,
      faceVisibility,
      projectedCoords,
      mappedPixels,
      debugData: {
        srcCanvas, srcW, srcH, img,
        ctrlU, ctrlV, ctrlIX, ctrlIY, N: ctrlU.length,
        landmarks, mpIndices, mapping
      }
    };
  }

  /**
   * Color harmonization: match side views' mean brightness/color to front view
   * in their overlapping regions. This compensates for different lighting
   * conditions between photos.
   */
  _harmonizeMultiViewColors(outImageData, viewProjections, size) {
    if (viewProjections.length < 2) return;

    const frontProj = viewProjections[0];
    const N = size * size;

    for (let v = 1; v < viewProjections.length; v++) {
      const sideProj = viewProjections[v];
      const fd = frontProj.imageData.data;
      const sd = sideProj.imageData.data;

      // Find overlapping pixels (both views have data)
      let fR = 0, fG = 0, fB = 0, sR = 0, sG = 0, sB = 0, count = 0;
      for (let i = 0; i < N; i++) {
        if (fd[i * 4 + 3] > 0 && sd[i * 4 + 3] > 0) {
          fR += fd[i * 4]; fG += fd[i * 4 + 1]; fB += fd[i * 4 + 2];
          sR += sd[i * 4]; sG += sd[i * 4 + 1]; sB += sd[i * 4 + 2];
          count++;
        }
      }

      if (count < 100) {
        console.log(`TextureProjector MULTI: ${sideProj.label} — insufficient overlap (${count}px), skipping harmonization`);
        continue;
      }

      // Compute brightness ratios in overlap
      const ratioR = (fR / count) / Math.max(1, sR / count);
      const ratioG = (fG / count) / Math.max(1, sG / count);
      const ratioB = (fB / count) / Math.max(1, sB / count);

      // Clamp to prevent extreme corrections (max 40% shift)
      const clamp = (v) => Math.max(0.6, Math.min(1.4, v));
      const cR = clamp(ratioR), cG = clamp(ratioG), cB = clamp(ratioB);

      console.log(`TextureProjector MULTI: ${sideProj.label} harmonization: R=${cR.toFixed(3)}, G=${cG.toFixed(3)}, B=${cB.toFixed(3)} (${count} overlap pixels)`);

      // Apply correction to the accumulated output, weighted by how much
      // this side view contributed. We correct pixels where side view was
      // the dominant contributor (front confidence < side confidence).
      const od = outImageData.data;
      const fc = frontProj.confidence;
      const sc = sideProj.confidence;

      for (let i = 0; i < N; i++) {
        // Only correct pixels where side view contributed more than front
        if (sc[i] <= 0 || fc[i] >= sc[i]) continue;
        if (od[i * 4 + 3] === 0) continue;

        // Blend strength: proportional to how dominant side view is
        const totalW = fc[i] + sc[i];
        const sideWeight = totalW > 0 ? sc[i] / totalW : 0;
        const corrStr = sideWeight; // 0 = front dominant, 1 = side dominant

        const idx = i * 4;
        od[idx]     = Math.max(0, Math.min(255, Math.round(od[idx] * (1 + corrStr * (cR - 1)))));
        od[idx + 1] = Math.max(0, Math.min(255, Math.round(od[idx + 1] * (1 + corrStr * (cG - 1)))));
        od[idx + 2] = Math.max(0, Math.min(255, Math.round(od[idx + 2] * (1 + corrStr * (cB - 1)))));
      }
    }
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
  _denseProjectTextureToUV(img, landmarks, mapping, meshGen, fittedCamera = null) {
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

    // --- 1. Use fitted camera (single camera end-to-end) or estimate fresh ---
    let cameraParams;
    if (fittedCamera) {
      cameraParams = fittedCamera;
      console.log(`TextureProjector DENSE: Using FITTED camera (consistent with shape/expr fitting)`);
    } else {
      cameraParams = this._shapeFitter._estimateCameraFromLandmarks(landmarks, mapping, meshGen);
      if (!cameraParams) {
        console.error('TextureProjector: Camera estimation FAILED — cannot proceed');
        return null;
      }
      console.warn('TextureProjector DENSE: No fitted camera provided — estimated independently (may diverge!)');
    }
    console.log(`TextureProjector DENSE: camera sx=${cameraParams.sx.toFixed(4)}, sy=${cameraParams.sy.toFixed(4)}, tx=${cameraParams.tx.toFixed(4)}, ty=${cameraParams.ty.toFixed(4)}`);
    // P3 verification: confirm bake camera is identical to fit camera
    if (fittedCamera && cameraParams !== fittedCamera) {
      console.error('CAMERA MISMATCH: projection using different camera object than fitting!');
    }

    // --- 2. Project all position vertices to image space ---
    const { coords: projectedCoords, depths: vertexDepths } = this._projectAllVertices(meshGen, cameraParams);

    // --- 3. Compute per-face visibility (backface culling) ---
    const faceVisibility = this._computeFaceVisibility(meshGen, cameraParams);

    // Store diagnostic data for overlay rendering (P1 expert requirement)
    this._diagCameraParams = cameraParams;
    this._diagFaceVisibility = faceVisibility;
    this._diagProjectedCoords = projectedCoords;
    this._diagMeshGen = meshGen;

    // --- 3b. Build debug overlay: MediaPipe landmarks + projected FLAME outline ---
    this._buildDebugOverlay(img, landmarks, projectedCoords, meshGen, srcW, srcH);

    // --- 4. Map position-vertex projections to UV-vertex coords ---
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;
    const uvImgCoords = this._mapProjectionToUV(projectedCoords, meshGen, faceVisibility);

    // --- 4a-depth. Map position-vertex depths to UV-vertex depths ---
    const uvDepths = this._mapDepthsToUV(vertexDepths, meshGen);

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
    // Pass expected winding sign from camera sx*sy for triangle inversion rejection
    const expectedWindingSign = Math.sign(cameraParams.sx * cameraParams.sy);
    const outImageData = ctx.createImageData(size, size);
    this._rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH, faceVisibility, excludedFaces, faceBoundaryMask, uvDepths, expectedWindingSign);

    // --- 5a. Apply UV mask: clear pixels outside valid UV footprint ---
    this._applyUVMask(outImageData, size, uvMask);

    // --- 5a2. Snapshot pre-dilation alpha for blur band ---
    const preDilationAlpha = new Uint8Array(size * size);
    for (let i = 0; i < size * size; i++) {
      preDilationAlpha[i] = outImageData.data[i * 4 + 3] > 0 ? 1 : 0;
    }

    // --- 5a3. Edge dilation: expand mapped pixels into unmapped (alpha=0) regions ---
    // Phase 10b: 80 passes (was 40) -- pushes face-edge skin further into ear/neck UV areas.
    this._postProcessor._dilateEdges(outImageData, size, 80);

    // --- 5a4. Blur dilated band: smooth transition between original and dilated pixels ---
    this._postProcessor._blurDilatedBand(outImageData, size, preDilationAlpha);

    // --- 5a3. Eye socket feathering: smooth boundary at eye sockets ---
    this._postProcessor._featherEyeSockets(outImageData, size, landmarks, mapping, uvCoords, uvFaces);

    // --- Diagnostic: pixel coverage ---
    let mappedPixels = 0;
    const totalPixels = size * size;
    for (let i = 0; i < totalPixels; i++) {
      if (outImageData.data[i * 4 + 3] > 0) mappedPixels++;
    }
    console.log(`TextureProjector DENSE: Mapped ${mappedPixels}/${totalPixels} pixels (${(mappedPixels / totalPixels * 100).toFixed(1)}%) after dilation`);

    // --- Diagnostic: UV coverage mask (white=sampled, black=unmapped) ---
    // Used to diagnose forehead seam: is it no-coverage or hard alpha edge?
    const coverageCanvas = document.createElement('canvas');
    coverageCanvas.width = size;
    coverageCanvas.height = size;
    const coverageCtx = coverageCanvas.getContext('2d');
    const coverageData = coverageCtx.createImageData(size, size);
    for (let i = 0; i < size * size; i++) {
      const hasData = outImageData.data[i * 4 + 3] > 0 ? 255 : 0;
      coverageData.data[i * 4] = hasData;
      coverageData.data[i * 4 + 1] = hasData;
      coverageData.data[i * 4 + 2] = hasData;
      coverageData.data[i * 4 + 3] = 255;
    }
    coverageCtx.putImageData(coverageData, 0, 0);
    this._debugCoverageMask = coverageCanvas.toDataURL('image/png');
    console.log('TextureProjector DIAG: UV coverage mask saved to _debugCoverageMask');

    // --- DIAGNOSTIC: Save raw UV rasterization (before any processing) ---
    {
      const rawCanvas = document.createElement('canvas');
      rawCanvas.width = size; rawCanvas.height = size;
      rawCanvas.getContext('2d').putImageData(
        new ImageData(new Uint8ClampedArray(outImageData.data), size, size), 0, 0);
      this._debugPhotoUV_raw = rawCanvas.toDataURL('image/png');
    }

    // --- Phase 2: Delighting for PBR + HDRI ---
    // Removes baked photo shadows that would cause double-shadowing with 3D lights.
    // NOTE: _smoothTextureColors DISABLED per expert review (destroys pore detail -> wax look)
    const preAvg = this._postProcessor._computeChannelAverages(outImageData);
    this._postProcessor._delightTexture(outImageData, size);
    // this._postProcessor._smoothTextureColors(outImageData, size);  // Expert: disabled -- kills pores
    this._postProcessor._restoreColorBalance(outImageData, preAvg);

    // --- 5c-i. Expert: erode alpha inward before softening ---
    // Forces the blend band to live entirely in "safe skin pixels" and avoids
    // pulling in boundary contamination (hair, background, ear edge).

    // P1 diagnostic: snapshot coverage alpha BEFORE erosion for overlay comparison
    const preErosionAlpha = new Uint8Array(size * size);
    for (let i = 0; i < size * size; i++) preErosionAlpha[i] = outImageData.data[i * 4 + 3];
    this._diagPreErosionAlpha = preErosionAlpha;
    this._diagTextureSize = size;

    // Expert round 4: increase erosion, drastically shrink softening.
    const ERODE_PX = Math.max(12, Math.round(size / 120));   // ~17px at 2048
    this._postProcessor._erodeAlpha(outImageData, size, ERODE_PX);
    // Extra erosion in jawline (V < 0.25) and forehead/scalp (V > 0.60)
    const JAWLINE_ERODE = Math.max(10, Math.round(size / 140));
    this._postProcessor._erodeAlphaRegion(outImageData, size, JAWLINE_ERODE, 0.0, 0.25);
    const FOREHEAD_EXTRA_ERODE = Math.max(10, Math.round(size / 140));
    this._postProcessor._erodeAlphaRegion(outImageData, size, FOREHEAD_EXTRA_ERODE, 0.60, 1.0);

    // Expert round 4: reduce softening to 20-40px (was 85-128px)
    const softenRadius = Math.max(12, Math.round(size / 64));   // ~32px at 2048
    const foreheadSoftenRadius = Math.max(16, Math.round(size / 48));  // ~43px at 2048
    this._postProcessor._softenAlphaBoundaryRegional(outImageData, size, softenRadius, foreheadSoftenRadius, 0.60);

    // --- Save pre-fill texture for debug ---
    const preFillCanvas = document.createElement('canvas');
    preFillCanvas.width = size;
    preFillCanvas.height = size;
    preFillCanvas.getContext('2d').putImageData(outImageData, 0, 0);
    this._debugPreFillTexture = preFillCanvas.toDataURL('image/png');

    // --- DIAGNOSTIC: Save alpha coverage mask ---
    {
      const alphaCanvas = document.createElement('canvas');
      alphaCanvas.width = size; alphaCanvas.height = size;
      const alphaCtx = alphaCanvas.getContext('2d');
      const alphaData = alphaCtx.createImageData(size, size);
      for (let i = 0; i < size * size; i++) {
        const a = outImageData.data[i * 4 + 3];
        alphaData.data[i * 4] = a;
        alphaData.data[i * 4 + 1] = a;
        alphaData.data[i * 4 + 2] = a;
        alphaData.data[i * 4 + 3] = 255;
      }
      alphaCtx.putImageData(alphaData, 0, 0);
      this._debugAlphaCoverage = alphaCanvas.toDataURL('image/png');
    }

    // --- 6. Mask-normalized Laplacian pyramid blending ---
    // Build full-coverage albedo layer, then blend with photo using 5-level pyramid
    const albedoLayer = this._postProcessor._prepareAlbedoLayer(outImageData, meshGen, size, landmarks, mapping);

    // --- DIAGNOSTIC: Save albedo tinted layer ---
    {
      const albCanvas = document.createElement('canvas');
      albCanvas.width = size; albCanvas.height = size;
      albCanvas.getContext('2d').putImageData(albedoLayer, 0, 0);
      this._debugAlbedoTinted = albCanvas.toDataURL('image/png');
    }

    const blendedData = this._postProcessor._laplacianBlend(outImageData, albedoLayer, size, 5);

    // DISABLED per expert round 4: ring artifacts likely from these passes.
    // Reintroduce one-by-one only after clean baseline is confirmed.
    // this._postProcessor._boundaryRingMatch(blendedData, outImageData, albedoLayer, size);
    // this._postProcessor._bakeSubtleAO(blendedData, outImageData, meshGen, size);

    ctx.putImageData(blendedData, 0, 0);

    console.log('TextureProjector: Dense projection texture complete');
    this._debugFinalTexture = canvas.toDataURL('image/png');

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/png'),
      width: size,
      height: size
    };
  }

  // -----------------------------------------------------------------------
  // Vertex projection & face visibility
  // -----------------------------------------------------------------------

  /**
   * Project all 5023 FLAME position vertices into image space.
   * Uses anisotropic weak-perspective: imgX = sx * Xrot + tx, imgY = sy * Yrot + ty.
   * Also computes rotated Z-depth for each vertex (used by depth buffer in rasterizer).
   * @returns {{ coords: Float32Array, depths: Float32Array }}
   *   coords: 5023*2 array of [imgX, imgY] pairs (normalized 0-1)
   *   depths: 5023 array of rotated Z values (camera-space depth)
   */
  _projectAllVertices(meshGen, cam) {
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const nVerts = verts.length / 3;
    const coords = new Float32Array(nVerts * 2);
    const depths = new Float32Array(nVerts);
    const { R, sx, sy, tx, ty } = cam;

    for (let i = 0; i < nVerts; i++) {
      const x = verts[i * 3], y = verts[i * 3 + 1], z = verts[i * 3 + 2];
      // Rotate
      const xr = R[0] * x + R[1] * y + R[2] * z;
      const yr = R[3] * x + R[4] * y + R[5] * z;
      const zr = R[6] * x + R[7] * y + R[8] * z;
      // Anisotropic weak-perspective projection
      coords[i * 2] = sx * xr + tx;
      coords[i * 2 + 1] = sy * yr + ty;
      // Camera-space depth (closer = smaller z in camera coords)
      depths[i] = zr;
    }

    // Diagnostic: count in-bounds vertices
    let inBounds = 0;
    for (let i = 0; i < nVerts; i++) {
      const ix = coords[i * 2], iy = coords[i * 2 + 1];
      if (ix >= 0 && ix < 1 && iy >= 0 && iy < 1) inBounds++;
    }
    console.log(`TextureProjector DENSE: ${inBounds}/${nVerts} vertices project into image bounds`);

    return { coords, depths };
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
    const { R } = cam;

    // View direction in model space = R^T * [0, 0, 1]
    // (camera looks along +Z in camera space)
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

      // Expert N-V grazing-angle cull: reject N-V < 0.35
      // Grazing triangles sample a tiny area and stretch it across a large surface
      // -> "melted cheek/jaw". Let tinted FLAME albedo fill these regions instead.
      if (cosAngle > 0.45) {
        visibility[f] = 1.0;
        visibleCount++;
      } else if (cosAngle > 0.35) {
        // Narrow fade zone (0.35-0.45) for smooth transition
        visibility[f] = (cosAngle - 0.35) / 0.10;
        visibleCount++;
      } else {
        // Below 0.2 -- reject (grazing/back-facing, causes swirl artifacts)
        visibility[f] = 0;
      }
    }

    console.log(`TextureProjector DENSE: ${visibleCount}/${nFaces} faces visible`);
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
   * Build debug overlay: draw MediaPipe landmarks (green) + projected FLAME mesh outline (red)
   * on the source photo. Stored as data URL for visual inspection.
   */
  _buildDebugOverlay(img, landmarks, projectedCoords, meshGen, srcW, srcH) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = srcW;
      canvas.height = srcH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, srcW, srcH);

      // Draw MediaPipe landmarks (green dots)
      ctx.fillStyle = '#00ff00';
      for (let i = 0; i < Math.min(landmarks.length, 478); i++) {
        const lm = landmarks[i];
        if (isNaN(lm.x) || isNaN(lm.y)) continue;
        const px = lm.x * srcW, py = lm.y * srcH;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw projected FLAME mesh edges (red lines) -- sample face edges for outline
      const posFaces = meshGen.flameFaces;
      const nFaces = posFaces.length / 3;
      ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
      ctx.lineWidth = 0.5;
      const maxFacesToDraw = Math.min(nFaces, 2000); // limit for performance
      for (let f = 0; f < maxFacesToDraw; f++) {
        const i0 = posFaces[f * 3], i1 = posFaces[f * 3 + 1], i2 = posFaces[f * 3 + 2];
        const x0 = projectedCoords[i0 * 2] * srcW, y0 = projectedCoords[i0 * 2 + 1] * srcH;
        const x1 = projectedCoords[i1 * 2] * srcW, y1 = projectedCoords[i1 * 2 + 1] * srcH;
        const x2 = projectedCoords[i2 * 2] * srcW, y2 = projectedCoords[i2 * 2 + 1] * srcH;
        // Skip out-of-bounds
        if (x0 < -srcW || x0 > srcW * 2 || y0 < -srcH || y0 > srcH * 2) continue;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.closePath();
        ctx.stroke();
      }

      // Draw 105 mapped FLAME landmark correspondences (cyan = projected FLAME, green = MediaPipe)
      const mpIndices = meshGen._mediaPipeMapping?.landmark_indices;
      if (mpIndices) {
        ctx.fillStyle = '#00ffff';
        for (let i = 0; i < mpIndices.length; i++) {
          const mpIdx = mpIndices[i];
          if (mpIdx >= landmarks.length) continue;
          // This would need FLAME projected position at this landmark -- skip for now
        }
      }

      this._debugOverlayUrl = canvas.toDataURL('image/jpeg', 0.85);
      console.log('TextureProjector DIAG: Debug overlay saved to _debugOverlayUrl');
    } catch (err) {
      console.warn('TextureProjector DIAG: Debug overlay failed:', err.message);
    }
  }

  // -----------------------------------------------------------------------
  // P1 Diagnostic Overlay Generators (Expert Round 3)
  // -----------------------------------------------------------------------

  /**
   * Generate a UV-space coverage heatmap texture (data URL).
   * Green = full photo coverage, Yellow = partial (blend zone), Red = no coverage.
   * Uses pre-erosion alpha to show the raw projection footprint.
   */
  generateCoverageHeatmap() {
    const alpha = this._diagPreErosionAlpha;
    const size = this._diagTextureSize;
    if (!alpha || !size) return null;

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(size, size);
    const d = img.data;

    for (let i = 0; i < size * size; i++) {
      const a = alpha[i];
      const idx = i * 4;
      if (a > 200) {
        // Full coverage: green
        d[idx] = 0; d[idx + 1] = 200; d[idx + 2] = 0; d[idx + 3] = 180;
      } else if (a > 0) {
        // Partial coverage: yellow, opacity proportional
        d[idx] = 255; d[idx + 1] = 200; d[idx + 2] = 0; d[idx + 3] = 150;
      } else {
        // No coverage: red
        d[idx] = 200; d[idx + 1] = 0; d[idx + 2] = 0; d[idx + 3] = 120;
      }
    }

    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL('image/png');
  }

  /**
   * Generate a UV-space N-V (face visibility) heatmap texture.
   * Green = fully visible (N-V > 0.45), Yellow = fade zone, Red = rejected.
   * Rasterizes per-face visibility values into UV space.
   */
  generateNVHeatmap() {
    const visibility = this._diagFaceVisibility;
    const meshGen = this._diagMeshGen;
    const size = this._diagTextureSize;
    if (!visibility || !meshGen || !size) return null;

    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;
    const nFaces = uvFaces.length / 3;

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Rasterize each face with its visibility color
    for (let f = 0; f < nFaces; f++) {
      const vis = visibility[f];
      const ui0 = uvFaces[f * 3], ui1 = uvFaces[f * 3 + 1], ui2 = uvFaces[f * 3 + 2];
      const x0 = uvCoords[ui0 * 2] * size, y0 = uvCoords[ui0 * 2 + 1] * size;
      const x1 = uvCoords[ui1 * 2] * size, y1 = uvCoords[ui1 * 2 + 1] * size;
      const x2 = uvCoords[ui2 * 2] * size, y2 = uvCoords[ui2 * 2 + 1] * size;

      // Color by visibility
      let r, g, b;
      if (vis >= 1.0) {
        r = 0; g = 200; b = 0;   // full visible: green
      } else if (vis > 0) {
        r = 255; g = 200; b = 0;  // fade zone: yellow
      } else {
        r = 200; g = 0; b = 0;    // rejected: red
      }

      ctx.fillStyle = `rgba(${r},${g},${b},0.7)`;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.closePath();
      ctx.fill();
    }

    return canvas.toDataURL('image/png');
  }

  /**
   * Generate a UV-space landmark overlay texture.
   * Shows projected 3D landmark positions as colored dots in UV space.
   * Green = low reprojection error, Red = high error.
   */
  generateLandmarkOverlay() {
    const debug = this._lastDebugData;
    const size = this._diagTextureSize;
    if (!debug || !size) return null;

    const { ctrlU, ctrlV, N } = debug;
    if (!ctrlU || N < 1) return null;

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Draw landmark positions as dots in UV space
    for (let i = 0; i < N; i++) {
      const x = ctrlU[i] * size;
      const y = ctrlV[i] * size;

      // All landmarks in cyan for now; could color by reprojection error
      ctx.fillStyle = '#00ffff';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();

      // Add a white border for visibility
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    return canvas.toDataURL('image/png');
  }

  // -----------------------------------------------------------------------
  // Boundary & UV utilities
  // -----------------------------------------------------------------------

  /**
   * Build a binary mask on the source photo that marks pixels inside the face boundary.
   * Uses MediaPipe's face oval landmarks to create a polygon, then rasterises it.
   *
   * @param {Array} landmarks - MediaPipe 478 landmarks (normalised 0-1)
   * @param {number} srcW - source image width in pixels
   * @param {number} srcH - source image height in pixels
   * @returns {Uint8Array} srcW*srcH mask (1 = inside face, 0 = outside)
   *          Also stores gradient version in this._faceBoundaryGradient for blending.
   */
  _buildFaceBoundaryMask(landmarks, srcW, srcH) {
    // MediaPipe standard face oval contour (ordered: forehead -> jaw -> forehead)
    // NOTE: only use BOUNDARY landmarks -- interior points break scanline polygon fill
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

    // Compute face size for relative margin
    let cx = 0, cy = 0;
    for (let i = 0; i < polyX.length; i++) { cx += polyX[i]; cy += polyY[i]; }
    cx /= polyX.length;
    cy /= polyY.length;
    let maxDist = 0;
    for (let i = 0; i < polyX.length; i++) {
      const d = Math.sqrt((polyX[i] - cx) ** 2 + (polyY[i] - cy) ** 2);
      if (d > maxDist) maxDist = d;
    }

    // Face-relative margin: ERODE inward by 3% of face radius
    // Prevents sampling hair/background pixels near the face boundary.
    // Paired with increased dilation + blur band to fill the resulting gap smoothly.
    const MARGIN_PX = Math.max(8, Math.round(maxDist * 0.03));
    for (let i = 0; i < polyX.length; i++) {
      const dx = polyX[i] - cx;
      const dy = polyY[i] - cy;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0.01) {
        polyX[i] -= (dx / len) * MARGIN_PX;  // shrink inward (was += outward)
        polyY[i] -= (dy / len) * MARGIN_PX;
      }
    }

    // Extend forehead landmarks upward to prevent "cap/helmet line" seam.
    // Vertices above the centroid by >30% of face radius get pushed up by 6% of radius.
    // This keeps jaw/cheek erosion intact while extending scalp coverage.
    const FOREHEAD_EXTEND = Math.round(maxDist * 0.06);
    const FOREHEAD_THRESHOLD = cy - maxDist * 0.30; // above this Y = forehead region
    for (let i = 0; i < polyX.length; i++) {
      if (polyY[i] < FOREHEAD_THRESHOLD) {
        polyY[i] -= FOREHEAD_EXTEND; // push upward (Y decreases upward in image space)
      }
    }

    // Scanline polygon rasterisation (binary core mask)
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
    console.log(`Face boundary mask: ${count}/${srcW * srcH} pixels inside face (${(count / (srcW * srcH) * 100).toFixed(1)}%), margin=${MARGIN_PX}px`);

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
   * Map position-vertex camera depths to UV-vertex depths.
   * @param {Float32Array} vertexDepths - 5023 camera-space Z values
   * @param {Object} meshGen
   * @returns {Float32Array} 5118 UV-vertex depth values
   */
  _mapDepthsToUV(vertexDepths, meshGen) {
    const uvToPosMap = meshGen.uvToPosMap;
    const nUV = meshGen.flameUVVertexCount;
    const result = new Float32Array(nUV);
    for (let i = 0; i < nUV; i++) {
      result[i] = vertexDepths[uvToPosMap[i]];
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
    if (cleared > 0) console.log(`TextureProjector DENSE: UV mask cleared ${cleared} out-of-footprint pixels`);
  }

  // -----------------------------------------------------------------------
  // Thin-Plate Spline (TPS) interpolation
  // -----------------------------------------------------------------------

  /**
   * Fit a 2D Thin-Plate Spline through N control points.
   *
   * Solves the (N+3)x(N+3) linear system:
   *   [K + lambda*I  P] [w]   [f]
   *   [P'            0] [a] = [0]
   *
   * where K_ij = U(r_ij), U(r) = r^2*ln(r), P = [1, u, v] polynomial.
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

    // TPS radial basis: U(r) = r^2*ln(r); U(0) = 0
    function tpsU(r2) {
      return r2 > 1e-20 ? r2 * Math.log(Math.sqrt(r2)) : 0;
    }

    // Build augmented matrix [A | b] of size M x (M+1)
    const mat = [];
    for (let i = 0; i < M; i++) mat[i] = new Float64Array(M + 1);

    // Upper-left: K + lambda*I
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
   * Solve an n x n linear system (given as n x (n+1) augmented matrix)
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
  _rasterizeMeshFaces(outImageData, size, uvCoords, uvFaces, uvImgCoords, srcData, srcW, srcH, faceVisibility = null, excludedFaces = null, faceBoundaryMask = null, uvDepths = null, expectedWindingSign = 0) {
    const out = outImageData.data;
    const nFaces = uvFaces.length / 3;

    // --- Depth buffer: closest surface wins (prevents overlapping face bleed) ---
    // In FLAME coords, closer-to-camera = larger rotated Z (nose > ears)
    const depthBuffer = new Float32Array(size * size);
    depthBuffer.fill(-Infinity);
    let invertedSkipped = 0;
    let depthRejected = 0;

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

      // --- Triangle inversion check: reject degenerate + inverted triangles ---
      // With anisotropic weak-perspective, sy is often negative (Y-flip).
      // Expected winding sign = sign(sx*sy). If a triangle's projected signed area
      // doesn't match, it's inverted (inside-out) and causes swirl/smear artifacts.
      const signedArea2D = (ix1 - ix0) * (iy2 - iy0) - (ix2 - ix0) * (iy1 - iy0);
      // 1) Reject degenerate triangles (near-zero area)
      if (Math.abs(signedArea2D) < 1e-6) {
        invertedSkipped++;
        continue;
      }
      // 2) Reject inverted triangles: projected winding must match expected sign(sx*sy)
      if (expectedWindingSign !== 0 && Math.sign(signedArea2D) !== expectedWindingSign) {
        invertedSkipped++;
        continue;
      }

      // Skip faces whose image coords are entirely far outside image
      if (Math.min(ix0, ix1, ix2) > 1.5 || Math.max(ix0, ix1, ix2) < -0.5 ||
          Math.min(iy0, iy1, iy2) > 1.5 || Math.max(iy0, iy1, iy2) < -0.5) continue;

      // Per-vertex depths for depth buffer interpolation
      const d0 = uvDepths ? uvDepths[vi0] : 0;
      const d1 = uvDepths ? uvDepths[vi1] : 0;
      const d2 = uvDepths ? uvDepths[vi2] : 0;

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

          // --- Depth test: only write if this fragment is closer (larger zr = closer) ---
          if (uvDepths) {
            const pixelDepth = w0 * d0 + w1 * d1 + w2 * d2;
            const dbIdx = py * size + px;
            if (pixelDepth <= depthBuffer[dbIdx]) {
              depthRejected++;
              continue;
            }
            depthBuffer[dbIdx] = pixelDepth;
          }

          // Interpolate image coordinates via barycentric weights
          const imgX = w0 * ix0 + w1 * ix1 + w2 * ix2;
          const imgY = w0 * iy0 + w1 * iy1 + w2 * iy2;

          // Clamp to image bounds (extends edge pixels outward for better coverage)
          const clampX = Math.max(0, Math.min(imgX, 0.9999));
          const clampY = Math.max(0, Math.min(imgY, 0.9999));
          // Skip pixels far outside (>15% beyond edge) -- those are extrapolation artifacts
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

    console.log(`TextureProjector RASTER: ${invertedSkipped} faces skipped (triangle inversion), ${depthRejected} fragments rejected (depth buffer)`);
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
}
