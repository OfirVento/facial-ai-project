/**
 * TexturePostProcess
 *
 * Post-processing pipeline for rasterized UV textures. Handles all image-space
 * operations that refine the photo-sampled face texture before final compositing:
 *
 *   - Edge dilation & dilated-band blurring (seam reduction at mapped/unmapped boundary)
 *   - Eye socket feathering (smooth eye contour transitions in UV space)
 *   - Alpha erosion & region erosion (shrink coverage boundary to avoid contamination)
 *   - Alpha boundary softening (gradient fade at photo edges)
 *   - Delighting (remove photo shading to produce flat-lit albedo)
 *   - Channel-average color balance restoration
 *   - Color smoothing (reduce reddish patches and uneven tones)
 *   - Alpha boundary blur (feathered edge for blending with FLAME albedo)
 *   - FLAME albedo fill (color-corrected fill for unmapped regions)
 *   - Mask-normalized Laplacian pyramid blending (photo + albedo multi-scale blend)
 *   - Boundary ring color matching (seam correction at ears/temples)
 *   - Ambient occlusion bake (mesh-concavity-based depth cues)
 *   - Albedo layer preparation (robust color-corrected FLAME albedo for blending)
 *
 * All methods are stateless — data flows entirely through parameters.
 * Internal helper calls use `this._method()` patterns for pyramid operations.
 */
export class TexturePostProcess {

  constructor() {
    // Stateless — all data flows through method parameters.
    // The _renderModeHint may be set externally before calling _delightTexture.
    this._renderModeHint = null;
  }

  // ===================================================================
  // Edge Dilation & Dilated Band Blur
  // ===================================================================

  /**
   * Dilate mapped texture edges outward into unmapped (alpha=0) regions.
   * Copies nearest mapped pixel color outward, reducing hard seam between
   * photo texture and albedo fill.
   * @param {ImageData} imageData - texture image data (modified in place)
   * @param {number} size - texture width/height
   * @param {number} iterations - number of dilation passes (pixels of expansion)
   */
  _dilateEdges(imageData, size, iterations = 12) {
    const data = imageData.data;
    let totalDilated = 0;

    for (let iter = 0; iter < iterations; iter++) {
      // Find boundary pixels: alpha=0 that have at least one alpha>0 neighbor
      const toFill = [];
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = (y * size + x) * 4;
          if (data[idx + 3] > 0) continue; // already mapped

          // Check 4-connected neighbors for mapped pixels
          let rSum = 0, gSum = 0, bSum = 0, aSum = 0, count = 0;
          const offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]];
          for (const [dx, dy] of offsets) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
            const nIdx = (ny * size + nx) * 4;
            if (data[nIdx + 3] > 0) {
              rSum += data[nIdx];
              gSum += data[nIdx + 1];
              bSum += data[nIdx + 2];
              aSum += data[nIdx + 3];
              count++;
            }
          }

          if (count > 0) {
            toFill.push([idx, Math.round(rSum / count), Math.round(gSum / count), Math.round(bSum / count), Math.round(aSum / count)]);
          }
        }
      }

      // Apply all fills for this pass
      for (const [idx, r, g, b, a] of toFill) {
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = a;
      }

      totalDilated += toFill.length;
      if (toFill.length === 0) break; // no more boundary pixels
    }

    if (totalDilated > 0) {
      console.log(`PhotoUploader DENSE: Edge dilation filled ${totalDilated} pixels over ${iterations} passes`);
    }
  }

  /**
   * Smooth the transition between original photo-sampled pixels and dilated fill pixels.
   * Identifies the "band" where dilation occurred and applies a localized box blur
   * to prevent banding or abrupt tonal shifts at the boundary.
   * @param {ImageData} imageData - texture image data (modified in place)
   * @param {number} size - texture width/height
   * @param {Uint8Array} preDilationAlpha - snapshot of which pixels had alpha>0 before dilation
   */
  _blurDilatedBand(imageData, size, preDilationAlpha) {
    const data = imageData.data;
    const N = size * size;
    const BLUR_R = Math.max(3, Math.round(size / 350)); // ~6px at 2048

    // Build band mask: pixels that were filled by dilation (alpha=0 before, >0 after)
    // Also include a border of original-coverage pixels within BLUR_R of the band
    const bandMask = new Uint8Array(N); // 0=skip, 1=in band
    for (let i = 0; i < N; i++) {
      if (data[i * 4 + 3] > 0 && !preDilationAlpha[i]) bandMask[i] = 1;
    }

    // Expand band mask inward into original coverage by BLUR_R pixels
    // so the blur covers both sides of the boundary
    const expandedBand = new Uint8Array(bandMask);
    for (let pass = 0; pass < BLUR_R; pass++) {
      const toAdd = [];
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const idx = y * size + x;
          if (expandedBand[idx]) continue; // already in band
          if (data[idx * 4 + 3] === 0) continue; // unmapped
          // Check if any 4-connected neighbor is in band
          const offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]];
          for (const [dx, dy] of offsets) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
              if (expandedBand[ny * size + nx]) { toAdd.push(idx); break; }
            }
          }
        }
      }
      for (const idx of toAdd) expandedBand[idx] = 1;
    }

    // Count band pixels for diagnostics
    let bandCount = 0;
    for (let i = 0; i < N; i++) if (expandedBand[i]) bandCount++;
    if (bandCount === 0) return;

    // Apply box blur only to expanded band pixels
    const blurredR = new Float32Array(N);
    const blurredG = new Float32Array(N);
    const blurredB = new Float32Array(N);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const idx = y * size + x;
        if (!expandedBand[idx]) continue;

        let rSum = 0, gSum = 0, bSum = 0, count = 0;
        for (let dy = -BLUR_R; dy <= BLUR_R; dy++) {
          for (let dx = -BLUR_R; dx <= BLUR_R; dx++) {
            const nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
            const nIdx = (ny * size + nx) * 4;
            if (data[nIdx + 3] === 0) continue; // skip unmapped
            rSum += data[nIdx];
            gSum += data[nIdx + 1];
            bSum += data[nIdx + 2];
            count++;
          }
        }
        if (count > 0) {
          blurredR[idx] = rSum / count;
          blurredG[idx] = gSum / count;
          blurredB[idx] = bSum / count;
        }
      }
    }

    // Write blurred values back
    for (let i = 0; i < N; i++) {
      if (!expandedBand[i]) continue;
      const idx = i * 4;
      data[idx] = Math.round(blurredR[i]);
      data[idx + 1] = Math.round(blurredG[i]);
      data[idx + 2] = Math.round(blurredB[i]);
    }

    console.log(`PhotoUploader DENSE: Blur band smoothed ${bandCount} pixels (radius=${BLUR_R})`);
  }

  // ===================================================================
  // Eye Socket Feathering
  // ===================================================================

  /**
   * Feather eye socket boundaries in UV space.
   * Maps MediaPipe eye contour landmarks to UV, identifies nearby texture boundary pixels,
   * and applies a small Gaussian blur to smooth the transition.
   * @param {ImageData} imageData - texture image data (modified in place)
   * @param {number} size - texture width/height
   * @param {Array} landmarks - MediaPipe landmarks
   * @param {Object} mapping - FLAME-MediaPipe mapping
   * @param {Float32Array} uvCoords - UV coordinates
   * @param {Uint32Array} uvFaces - UV face indices
   */
  _featherEyeSockets(imageData, size, landmarks, mapping, uvCoords, uvFaces) {
    if (!landmarks || !mapping) return;

    const data = imageData.data;
    const RADIUS = 3; // Gaussian feather radius in pixels

    // MediaPipe eye contour landmarks
    const RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    const LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
    const eyeLandmarks = [...RIGHT_EYE, ...LEFT_EYE];

    // Map eye landmarks to UV space
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const eyeUVPoints = [];

    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (!eyeLandmarks.includes(mpIdx)) continue;
      if (mpIdx >= landmarks.length) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const uvi0 = uvFaces[fi * 3], uvi1 = uvFaces[fi * 3 + 1], uvi2 = uvFaces[fi * 3 + 2];
      const u = bc[0] * uvCoords[uvi0 * 2] + bc[1] * uvCoords[uvi1 * 2] + bc[2] * uvCoords[uvi2 * 2];
      const v = bc[0] * uvCoords[uvi0 * 2 + 1] + bc[1] * uvCoords[uvi1 * 2 + 1] + bc[2] * uvCoords[uvi2 * 2 + 1];
      eyeUVPoints.push([u * size, v * size]);
    }

    if (eyeUVPoints.length < 4) {
      console.log('Eye feathering: insufficient UV eye points, skipping');
      return;
    }

    // Build a mask of pixels within RADIUS*2 of any eye UV landmark
    const SEARCH_RADIUS = RADIUS * 3;
    const featherPixels = new Set();
    for (const [eu, ev] of eyeUVPoints) {
      const minX = Math.max(0, Math.floor(eu - SEARCH_RADIUS));
      const maxX = Math.min(size - 1, Math.ceil(eu + SEARCH_RADIUS));
      const minY = Math.max(0, Math.floor(ev - SEARCH_RADIUS));
      const maxY = Math.min(size - 1, Math.ceil(ev + SEARCH_RADIUS));
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          const dist = Math.sqrt((x - eu) ** 2 + (y - ev) ** 2);
          if (dist <= SEARCH_RADIUS) {
            featherPixels.add(y * size + x);
          }
        }
      }
    }

    // Apply Gaussian blur only to feather region pixels that are at mapped/unmapped boundary
    // (pixels where alpha transitions from >0 to 0 within RADIUS)
    const kernel = [];
    let kernelSum = 0;
    const sigma = RADIUS / 2;
    for (let dy = -RADIUS; dy <= RADIUS; dy++) {
      for (let dx = -RADIUS; dx <= RADIUS; dx++) {
        const w = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        kernel.push({ dx, dy, w });
        kernelSum += w;
      }
    }
    kernel.forEach(k => k.w /= kernelSum);

    // Snapshot original data for reading
    const orig = new Uint8ClampedArray(data);
    let feathered = 0;

    for (const pixIdx of featherPixels) {
      const x = pixIdx % size;
      const y = Math.floor(pixIdx / size);
      const idx = pixIdx * 4;
      if (orig[idx + 3] === 0) continue; // unmapped — skip

      // Check if this is near a boundary (has unmapped neighbor within RADIUS)
      let nearBoundary = false;
      for (let dy = -RADIUS; dy <= RADIUS && !nearBoundary; dy++) {
        for (let dx = -RADIUS; dx <= RADIUS && !nearBoundary; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
          if (orig[(ny * size + nx) * 4 + 3] === 0) nearBoundary = true;
        }
      }
      if (!nearBoundary) continue;

      // Apply Gaussian blur from original data
      let rAcc = 0, gAcc = 0, bAcc = 0, aAcc = 0, wAcc = 0;
      for (const { dx, dy, w } of kernel) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
        const nIdx = (ny * size + nx) * 4;
        if (orig[nIdx + 3] === 0) continue; // skip unmapped in blur
        rAcc += orig[nIdx] * w;
        gAcc += orig[nIdx + 1] * w;
        bAcc += orig[nIdx + 2] * w;
        aAcc += orig[nIdx + 3] * w;
        wAcc += w;
      }

      if (wAcc > 0) {
        data[idx] = Math.round(rAcc / wAcc);
        data[idx + 1] = Math.round(gAcc / wAcc);
        data[idx + 2] = Math.round(bAcc / wAcc);
        data[idx + 3] = Math.round(aAcc / wAcc);
        feathered++;
      }
    }

    console.log(`Eye feathering: ${feathered} pixels feathered near ${eyeUVPoints.length} eye UV points`);
  }

  // ===================================================================
  // Alpha Erosion, Region Erosion, Softening & Box Blur
  // ===================================================================

  /**
   * Erode alpha mask inward by `radius` pixels.
   * Any pixel whose alpha > 0 that is within `radius` of an alpha=0 pixel
   * gets zeroed out. This shrinks the photo coverage boundary inward so the
   * subsequent softening blur lives entirely in safe, well-mapped skin pixels
   * instead of pulling in hair/background/ear-edge contamination.
   *
   * Uses a distance-field approach: compute min distance to any alpha=0 pixel,
   * then zero out everything within `radius`.
   */
  _erodeAlpha(imageData, size, radius = 12) {
    const data = imageData.data;
    const N = size * size;

    // Extract binary mask: 1 = has alpha, 0 = transparent
    const mask = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      mask[i] = data[i * 4 + 3] > 0 ? 1 : 0;
    }

    // Two-pass approximate distance to nearest zero-alpha pixel
    // using a separable min-filter (erosion by square structuring element).
    // For each row, find min distance to a 0-pixel within `radius` columns.
    // Then for each column, find min across rows.
    // A pixel is eroded if it's within `radius` of a boundary.

    // Pass 1: For each pixel, determine if any neighbor within `radius`
    // in horizontal direction is transparent
    const hDist = new Int16Array(N); // horizontal distance to nearest 0
    for (let y = 0; y < size; y++) {
      const row = y * size;
      // Forward pass: distance from left
      let dist = radius + 1;
      for (let x = 0; x < size; x++) {
        if (mask[row + x] === 0) {
          dist = 0;
        } else {
          dist++;
        }
        hDist[row + x] = dist;
      }
      // Backward pass: distance from right
      dist = radius + 1;
      for (let x = size - 1; x >= 0; x--) {
        if (mask[row + x] === 0) {
          dist = 0;
        } else {
          dist++;
        }
        hDist[row + x] = Math.min(hDist[row + x], dist);
      }
    }

    // Pass 2: Vertical pass using Chebyshev distance (square erosion)
    // For each pixel, check if any pixel within `radius` rows has hDist <= radius
    const eroded = new Uint8Array(N); // 1 = should be eroded (zeroed)
    const vTemp = new Int16Array(size); // column buffer
    for (let x = 0; x < size; x++) {
      // Extract column of hDist
      for (let y = 0; y < size; y++) {
        vTemp[y] = hDist[y * size + x];
      }
      // For each pixel in this column, find min hDist within +-radius rows
      // Use sliding window min
      // Simple approach: for each y, scan +-radius
      for (let y = 0; y < size; y++) {
        if (mask[y * size + x] === 0) continue; // already transparent
        let minH = vTemp[y];
        const yStart = Math.max(0, y - radius);
        const yEnd = Math.min(size - 1, y + radius);
        for (let yy = yStart; yy <= yEnd; yy++) {
          if (vTemp[yy] < minH) minH = vTemp[yy];
          if (minH === 0) break; // can't get lower
        }
        // If the nearest transparent pixel (Chebyshev) is within radius, erode
        if (minH <= radius) {
          eroded[y * size + x] = 1;
        }
      }
    }

    // Apply erosion
    let erodedCount = 0;
    for (let i = 0; i < N; i++) {
      if (eroded[i] && mask[i]) {
        data[i * 4 + 3] = 0;
        erodedCount++;
      }
    }

    console.log(`PhotoUploader DENSE: Alpha erosion: ${erodedCount} pixels eroded (radius=${radius})`);
  }

  /**
   * Region-specific alpha erosion: erode only pixels within a UV V range.
   * Used to push the forehead/scalp boundary inward where the photo-to-bald
   * transition is most visible.
   * @param {ImageData} imageData
   * @param {number} size
   * @param {number} radius - erosion radius in pixels
   * @param {number} vMin - minimum V coordinate (0-1) for the region
   * @param {number} vMax - maximum V coordinate (0-1) for the region
   */
  _erodeAlphaRegion(imageData, size, radius, vMin, vMax) {
    const data = imageData.data;
    const N = size * size;
    const yMin = Math.floor(vMin * size);
    const yMax = Math.ceil(vMax * size);

    // Extract binary mask for full image (needed for distance computation)
    const mask = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      mask[i] = data[i * 4 + 3] > 0 ? 1 : 0;
    }

    // Horizontal distance pass (full image, needed for correctness at boundaries)
    const hDist = new Int16Array(N);
    for (let y = 0; y < size; y++) {
      const row = y * size;
      let dist = radius + 1;
      for (let x = 0; x < size; x++) {
        dist = mask[row + x] === 0 ? 0 : dist + 1;
        hDist[row + x] = dist;
      }
      dist = radius + 1;
      for (let x = size - 1; x >= 0; x--) {
        dist = mask[row + x] === 0 ? 0 : dist + 1;
        hDist[row + x] = Math.min(hDist[row + x], dist);
      }
    }

    // Vertical pass + erosion — only apply in the V region
    let erodedCount = 0;
    const vTemp = new Int16Array(size);
    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) vTemp[y] = hDist[y * size + x];
      for (let y = yMin; y < yMax && y < size; y++) {
        if (mask[y * size + x] === 0) continue;
        let minH = vTemp[y];
        const ys = Math.max(0, y - radius);
        const ye = Math.min(size - 1, y + radius);
        for (let yy = ys; yy <= ye; yy++) {
          if (vTemp[yy] < minH) minH = vTemp[yy];
          if (minH === 0) break;
        }
        if (minH <= radius) {
          data[(y * size + x) * 4 + 3] = 0;
          erodedCount++;
        }
      }
    }
    console.log(`PhotoUploader DENSE: Region erosion (V=${vMin}-${vMax}): ${erodedCount} pixels eroded (radius=${radius})`);
  }

  /**
   * Region-aware alpha softening: applies wider blur in forehead/scalp zone.
   * Uses two passes — standard radius for most of the face, wider radius
   * for the forehead region — then blends the results by V coordinate.
   */
  _softenAlphaBoundaryRegional(imageData, size, baseRadius, foreheadRadius, foreheadVThreshold) {
    const data = imageData.data;
    const N = size * size;

    // Extract alpha
    const alpha = new Float32Array(N);
    for (let i = 0; i < N; i++) alpha[i] = data[i * 4 + 3];

    // Blur with base radius
    const blurredBase = this._boxBlurAlpha(alpha, size, baseRadius);
    // Blur with forehead radius (wider)
    const blurredWide = this._boxBlurAlpha(alpha, size, foreheadRadius);

    // Blend: use wider blur in forehead region, base elsewhere, with smooth transition
    const transitionBand = 0.05; // 5% of UV space for smooth transition
    let softened = 0;
    for (let i = 0; i < N; i++) {
      const v = Math.floor(i / size) / size;
      // Smooth blend factor: 0 in face region, 1 in forehead region
      let foreheadBlend = 0;
      if (v > foreheadVThreshold + transitionBand) {
        foreheadBlend = 1.0;
      } else if (v > foreheadVThreshold - transitionBand) {
        foreheadBlend = (v - foreheadVThreshold + transitionBand) / (2 * transitionBand);
      }

      const blurred = blurredBase[i] * (1 - foreheadBlend) + blurredWide[i] * foreheadBlend;
      const newAlpha = Math.min(alpha[i], blurred);
      if (newAlpha < alpha[i]) softened++;
      data[i * 4 + 3] = Math.round(newAlpha);
    }
    console.log(`PhotoUploader DENSE: Regional alpha softening: ${softened} pixels softened (base=${baseRadius}, forehead=${foreheadRadius})`);
  }

  /**
   * Two-pass box blur on a single-channel Float32Array.
   * @returns {Float32Array} blurred result
   */
  _boxBlurAlpha(alpha, size, radius) {
    const N = size * size;
    const tmp = new Float32Array(N);
    // Horizontal pass
    for (let y = 0; y < size; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x < Math.min(radius, size); x++) { sum += alpha[y * size + x]; count++; }
      for (let x = 0; x < size; x++) {
        if (x + radius < size) { sum += alpha[y * size + x + radius]; count++; }
        if (x - radius - 1 >= 0) { sum -= alpha[y * size + x - radius - 1]; count--; }
        tmp[y * size + x] = sum / count;
      }
    }
    // Vertical pass
    const blurred = new Float32Array(N);
    for (let x = 0; x < size; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y < Math.min(radius, size); y++) { sum += tmp[y * size + x]; count++; }
      for (let y = 0; y < size; y++) {
        if (y + radius < size) { sum += tmp[(y + radius) * size + x]; count++; }
        if (y - radius - 1 >= 0) { sum -= tmp[(y - radius - 1) * size + x]; count--; }
        blurred[y * size + x] = sum / count;
      }
    }
    return blurred;
  }

  /**
   * Soften alpha boundaries for smooth blending transitions.
   * Applies a box blur to just the alpha channel, creating gradual
   * photo-to-albedo transitions at face edges, visibility fades, etc.
   * Only reduces alpha (never increases beyond original), preserving fully-mapped regions.
   */
  _softenAlphaBoundary(imageData, size, radius = 16) {
    const data = imageData.data;
    const N = size * size;

    // Extract alpha channel
    const alpha = new Float32Array(N);
    for (let i = 0; i < N; i++) alpha[i] = data[i * 4 + 3];

    // Two-pass box blur on alpha
    const tmp = new Float32Array(N);
    // Horizontal
    for (let y = 0; y < size; y++) {
      let sum = 0, count = 0;
      for (let x = 0; x < Math.min(radius, size); x++) { sum += alpha[y * size + x]; count++; }
      for (let x = 0; x < size; x++) {
        if (x + radius < size) { sum += alpha[y * size + x + radius]; count++; }
        if (x - radius - 1 >= 0) { sum -= alpha[y * size + x - radius - 1]; count--; }
        tmp[y * size + x] = sum / count;
      }
    }
    // Vertical
    const blurred = new Float32Array(N);
    for (let x = 0; x < size; x++) {
      let sum = 0, count = 0;
      for (let y = 0; y < Math.min(radius, size); y++) { sum += tmp[y * size + x]; count++; }
      for (let y = 0; y < size; y++) {
        if (y + radius < size) { sum += tmp[(y + radius) * size + x]; count++; }
        if (y - radius - 1 >= 0) { sum -= tmp[(y - radius - 1) * size + x]; count--; }
        blurred[y * size + x] = sum / count;
      }
    }

    // Apply: use min(original, blurred) so we only soften edges, never inflate
    let softened = 0;
    for (let i = 0; i < N; i++) {
      const newAlpha = Math.min(alpha[i], blurred[i]);
      if (newAlpha < alpha[i]) softened++;
      data[i * 4 + 3] = Math.round(newAlpha);
    }
    console.log(`PhotoUploader DENSE: Alpha softening: ${softened} pixels softened (radius=${radius})`);
  }

  // ===================================================================
  // Delighting, Channel Averages & Color Balance
  // ===================================================================

  /**
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
    // Scale by photo brightness: well-lit photos need less delighting (shadows already subtle).
    // Dark photos need more correction. Also adaptive per-pixel for under-eye preservation.
    // Mode-aware delighting: hybrid needs minimal correction (photo shadows ARE
    // the diffuse shading in emissive mode). PBR needs more to prevent double-shadowing.
    const renderMode = this._renderModeHint || 'hybrid';
    const MAX_STRENGTH = renderMode === 'hybrid' ? 0.04 : 0.12;
    const lumScale = Math.min(1.0, Math.max(0.2, (180 - avgLum) / 100));
    const BASE_STRENGTH = MAX_STRENGTH * lumScale;
    console.log(`PhotoUploader DENSE: Delight strength scaled: avgLum=${avgLum.toFixed(1)}, lumScale=${lumScale.toFixed(2)}, BASE_STRENGTH=${BASE_STRENGTH.toFixed(3)}`);
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) continue;

      const shade = blurred[i];
      if (shade < 5) continue; // Avoid division by near-zero

      // Adaptive strength: scale by local luminance — dark regions get less correction
      // Under-eye areas (~lum 0.3) get ~60% of base strength; bright cheeks (~lum 0.7) get full
      const localLum = lum[i] / 255;
      const strength = BASE_STRENGTH * Math.min(1.0, localLum / 0.5);

      // Correction factor: how much to brighten/darken to remove shading
      const correction = avgLum / shade;
      // Blend between original and delighted based on adaptive strength
      const factor = 1.0 + strength * (correction - 1.0);

      data[i * 4]     = Math.min(255, Math.max(0, Math.round(data[i * 4] * factor)));
      data[i * 4 + 1] = Math.min(255, Math.max(0, Math.round(data[i * 4 + 1] * factor)));
      data[i * 4 + 2] = Math.min(255, Math.max(0, Math.round(data[i * 4 + 2] * factor)));
    }

    console.log(`PhotoUploader DENSE: Delighting applied (avgLum=${avgLum.toFixed(1)}, radius=${blurRadius}, baseStrength=${BASE_STRENGTH.toFixed(3)})`);

    // --- 4. Brightness normalization: preserve source luminance ---
    // Adaptive target: keep the source photo's own luminance unless it's extreme.
    // Only clamp if very dark (<100) or very bright (>210).
    // This prevents darkening bright selfies or brightening moody photos.
    const TARGET_LUM = Math.min(210, Math.max(100, avgLum));
    let postLumSum = 0, postCount = 0;
    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] === 0) continue;
      postLumSum += 0.2126 * data[i * 4] + 0.7152 * data[i * 4 + 1] + 0.0722 * data[i * 4 + 2];
      postCount++;
    }
    if (postCount > 100) {
      const postAvg = postLumSum / postCount;
      // Tighter clamp: avoid over-brightening which shifts lighter skin tones
      const brightnessFactor = Math.min(1.15, Math.max(0.85, TARGET_LUM / postAvg));
      if (Math.abs(brightnessFactor - 1.0) > 0.03) {
        for (let i = 0; i < total; i++) {
          if (data[i * 4 + 3] === 0) continue;
          data[i * 4]     = Math.min(255, Math.round(data[i * 4] * brightnessFactor));
          data[i * 4 + 1] = Math.min(255, Math.round(data[i * 4 + 1] * brightnessFactor));
          data[i * 4 + 2] = Math.min(255, Math.round(data[i * 4 + 2] * brightnessFactor));
        }
        console.log(`PhotoUploader DENSE: Brightness normalized: avgLum ${postAvg.toFixed(1)} → ${(postAvg * brightnessFactor).toFixed(1)} (factor=${brightnessFactor.toFixed(3)})`);
      } else {
        console.log(`PhotoUploader DENSE: Brightness OK: avgLum=${postAvg.toFixed(1)}, no correction needed`);
      }
    }
  }

  /**
   * Compute per-channel average R, G, B of mapped pixels.
   * @returns {{ r: number, g: number, b: number, count: number }}
   */
  _computeChannelAverages(imageData) {
    const data = imageData.data;
    const N = data.length / 4;
    let rSum = 0, gSum = 0, bSum = 0, count = 0;
    for (let i = 0; i < N; i++) {
      if (data[i * 4 + 3] === 0) continue;
      rSum += data[i * 4];
      gSum += data[i * 4 + 1];
      bSum += data[i * 4 + 2];
      count++;
    }
    if (count === 0) return { r: 128, g: 128, b: 128, count: 0 };
    return { r: rSum / count, g: gSum / count, b: bSum / count, count };
  }

  /**
   * Restore the per-channel color balance to match a target (typically the source photo's
   * original averages). This corrects any color cast introduced by delight/smooth/brightness steps.
   * Uses per-channel scaling clamped to +-15% to avoid extreme corrections.
   */
  _restoreColorBalance(imageData, targetAvg) {
    if (targetAvg.count < 100) return;

    const currentAvg = this._computeChannelAverages(imageData);
    if (currentAvg.count < 100) return;

    // Compute per-channel correction factors
    const factorR = targetAvg.r / Math.max(1, currentAvg.r);
    const factorG = targetAvg.g / Math.max(1, currentAvg.g);
    const factorB = targetAvg.b / Math.max(1, currentAvg.b);

    // Clamp to +-20% to allow fuller correction of processing-induced color shifts
    const clamp = (f) => Math.max(0.80, Math.min(1.20, f));
    const fR = clamp(factorR), fG = clamp(factorG), fB = clamp(factorB);

    if (Math.abs(fR - 1) < 0.02 && Math.abs(fG - 1) < 0.02 && Math.abs(fB - 1) < 0.02) {
      console.log('PhotoUploader DENSE: Color balance OK, no correction needed');
      return;
    }

    const data = imageData.data;
    const N = data.length / 4;
    for (let i = 0; i < N; i++) {
      if (data[i * 4 + 3] === 0) continue;
      data[i * 4]     = Math.min(255, Math.round(data[i * 4] * fR));
      data[i * 4 + 1] = Math.min(255, Math.round(data[i * 4 + 1] * fG));
      data[i * 4 + 2] = Math.min(255, Math.round(data[i * 4 + 2] * fB));
    }

    console.log(`PhotoUploader DENSE: Color balance restored: R×${fR.toFixed(3)}, G×${fG.toFixed(3)}, B×${fB.toFixed(3)}`);
  }

  // ===================================================================
  // Color Smoothing & Alpha Boundary Blur (some expert-disabled)
  // ===================================================================

  /**
   * Gently smooth the texture colors to reduce reddish patches and uneven tones.
   * Blends each pixel 25% toward its local average (box-blurred version).
   * Only operates on mapped pixels (alpha > 0). Preserves detail while reducing extremes.
   */
  _smoothTextureColors(imageData, size) {
    const data = imageData.data;
    const N = size * size;
    const BLEND = 0.10;  // 10% blend toward local average (further reduced to preserve skin detail)
    const RADIUS = Math.max(4, Math.round(size / 256)); // ~8px at 2048

    // Extract RGB for mapped pixels
    const channels = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
    for (let i = 0; i < N; i++) {
      if (data[i * 4 + 3] === 0) continue;
      channels[0][i] = data[i * 4];
      channels[1][i] = data[i * 4 + 1];
      channels[2][i] = data[i * 4 + 2];
    }

    // Box blur each channel
    for (let ch = 0; ch < 3; ch++) {
      const src = channels[ch];
      const tmp = new Float32Array(N);
      // Horizontal pass
      for (let y = 0; y < size; y++) {
        let sum = 0, count = 0;
        for (let x = 0; x < Math.min(RADIUS, size); x++) {
          const idx = y * size + x;
          if (data[idx * 4 + 3] > 0) { sum += src[idx]; count++; }
        }
        for (let x = 0; x < size; x++) {
          if (x + RADIUS < size) {
            const idx = y * size + x + RADIUS;
            if (data[idx * 4 + 3] > 0) { sum += src[idx]; count++; }
          }
          if (x - RADIUS - 1 >= 0) {
            const idx = y * size + x - RADIUS - 1;
            if (data[idx * 4 + 3] > 0) { sum -= src[idx]; count--; }
          }
          tmp[y * size + x] = count > 0 ? sum / count : src[y * size + x];
        }
      }
      // Vertical pass
      const blurred = new Float32Array(N);
      for (let x = 0; x < size; x++) {
        let sum = 0, count = 0;
        for (let y = 0; y < Math.min(RADIUS, size); y++) {
          sum += tmp[y * size + x]; count++;
        }
        for (let y = 0; y < size; y++) {
          if (y + RADIUS < size) { sum += tmp[(y + RADIUS) * size + x]; count++; }
          if (y - RADIUS - 1 >= 0) { sum -= tmp[(y - RADIUS - 1) * size + x]; count--; }
          blurred[y * size + x] = sum / count;
        }
      }

      // Adaptive blend: darker pixels blend more strongly toward local average
      // This reduces dark circles/shadows while preserving well-lit areas
      for (let i = 0; i < N; i++) {
        if (data[i * 4 + 3] === 0) continue;
        // Compute pixel luminance
        const lum = 0.2126 * data[i * 4] + 0.7152 * data[i * 4 + 1] + 0.0722 * data[i * 4 + 2];
        // Darker pixels get stronger blending (up to 2x BLEND for very dark areas)
        const darkBoost = lum < 100 ? BLEND * (1 + (100 - lum) / 100) : BLEND;
        const adaptiveBlend = Math.min(0.5, darkBoost); // cap at 50%
        const smoothed = (1 - adaptiveBlend) * src[i] + adaptiveBlend * blurred[i];
        data[i * 4 + ch] = Math.round(Math.max(0, Math.min(255, smoothed)));
      }
    }

    console.log(`PhotoUploader DENSE: Color smoothing applied (blend=${BLEND}, radius=${RADIUS})`);
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

  // ===================================================================
  // FLAME Albedo Fill
  // ===================================================================

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
    // Expert fix: filter sampling to reject shadows, desaturated, and extreme pixels
    // (same logic as _prepareAlbedoLayer to prevent green/gray contamination)
    let photoR = 0, photoG = 0, photoB = 0;
    let albR = 0, albG = 0, albB = 0;
    let sampleCount = 0;

    for (let i = 0; i < total; i++) {
      if (data[i * 4 + 3] !== 255) continue; // only fully-mapped photo pixels

      // Expert fix: luminance filter — reject shadows and highlights
      const pr = data[i * 4], pg = data[i * 4 + 1], pb = data[i * 4 + 2];
      const pLum = pr * 0.2126 + pg * 0.7152 + pb * 0.0722;
      if (pLum < 60 || pLum > 220) continue;

      // Expert fix: saturation filter — reject near-gray (shadow contamination)
      const maxC = Math.max(pr, pg, pb);
      const minC = Math.min(pr, pg, pb);
      if (maxC > 10 && (maxC - minC) / maxC < 0.05) continue;

      const px = i % size;
      const py = Math.floor(i / size);
      const u = (px + 0.5) / size;
      const v = (py + 0.5) / size;

      // Sample albedo at same UV
      const ax0 = Math.max(0, Math.min(Math.floor(u * albedoSize), albedoSize - 1));
      const ay0 = Math.max(0, Math.min(Math.floor(v * albedoSize), albedoSize - 1));
      const ai = (ay0 * albedoSize + ax0) * 3;

      photoR += pr;
      photoG += pg;
      photoB += pb;
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

  // ===================================================================
  // Mask-Normalized Laplacian Pyramid Blending
  // ===================================================================

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

    // Horizontal pass: data -> temp
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

    // Vertical pass: temp -> out
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
    // Use larger blur radius for wider mask-normalized transition (relative to texture size)
    const maskBlurR = Math.max(8, Math.round(size / 64));  // ~32px at 2048
    const blurredMasked = this._gaussBlur(maskedPhoto, size, size, ch, maskBlurR);
    const blurredMask = this._gaussBlur(mask, size, size, 1, maskBlurR);
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
    // CRITICAL FIX: Sharpen mask at coarse levels to prevent FLAME albedo
    // color bleeding INTO the photo region through low-frequency bands.
    // At fine levels (l=0,1) use soft mask for smooth edges.
    // At coarse levels (l=2,3,4) sharpen mask to keep photo color dominant.
    const blendedLap = [];
    for (let l = 0; l < levels; l++) {
      const pL = photoLap[l];
      const aL = albedoLap[l];
      const mL = maskGauss[l];
      const blended = new Float32Array(pL.data.length);
      const pw = pL.w;
      // Sharpen mask more at coarser levels: pow(mask, 1) at l=0, pow(mask, 2) at l=2, etc.
      const sharpPow = l <= 1 ? 1.0 : (1.0 + (l - 1) * 0.8);
      for (let y = 0; y < pL.h; y++) {
        for (let x = 0; x < pw; x++) {
          let mi = Math.max(0, Math.min(1, mL.data[y * pw + x]));
          // Sharpen: push mask toward 0 or 1 at coarser levels
          mi = Math.pow(mi, 1.0 / sharpPow);
          for (let c = 0; c < ch; c++) {
            const idx = (y * pw + x) * ch + c;
            blended[idx] = mi * pL.data[idx] + (1 - mi) * aL.data[idx];
          }
        }
      }
      blendedLap.push({ data: blended, w: pw, h: pL.h });
    }

    // Blend coarsest Gaussian level — use sharper mask to prevent tint bleed
    const pCoarse = photoGauss[levels];
    const aCoarse = albedoGauss[levels];
    const mCoarse = maskGauss[levels];
    const blendedCoarse = new Float32Array(pCoarse.data.length);
    for (let y = 0; y < pCoarse.h; y++) {
      for (let x = 0; x < pCoarse.w; x++) {
        let t = Math.max(0, Math.min(1, mCoarse.data[y * pCoarse.w + x]));
        // At coarsest level, strongly prefer photo to prevent overall color shift
        t = Math.pow(t, 0.3); // Push any masked area toward photo
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

  // ===================================================================
  // Boundary Ring Match (expert-disabled)
  // ===================================================================

  /**
   * P3 Expert: boundary-ring color matching.
   * Finds the photo-to-albedo boundary in the blended result, samples a narrow
   * band on each side, computes a local correction ratio, and applies it to
   * albedo-dominant pixels near the boundary with radial falloff.
   * This reduces the visible color seam at ears/temples without full spatial tinting.
   *
   * @param {ImageData} blendedData - the Laplacian-blended result (modified in place)
   * @param {ImageData} photoData - original photo UV texture (with alpha = coverage)
   * @param {ImageData} albedoData - color-corrected albedo layer
   * @param {number} size - texture size
   */
  _boundaryRingMatch(blendedData, photoData, albedoData, size) {
    const bd = blendedData.data;
    const pd = photoData.data;
    const N = size * size;

    // Step 1: Find boundary pixels — where photo alpha transitions from high to low.
    // Use the photo's alpha to identify the blend boundary.
    const alpha = new Float32Array(N);
    for (let i = 0; i < N; i++) alpha[i] = pd[i * 4 + 3] / 255;

    // Compute distance to boundary (approximate): distance to nearest alpha transition
    // Boundary is where alpha is in the 0.2-0.8 range (the blend zone)
    const INNER_BAND = 20;  // pixels inward from boundary (photo-dominant side)
    const OUTER_BAND = 40;  // pixels outward from boundary (albedo-dominant side)
    const FALLOFF_DIST = 80; // how far the correction fades out

    // Identify boundary pixels (where alpha is mid-range after softening)
    // For the blended result, we detect boundary differently:
    // compare blended color to photo color — large difference = boundary/albedo zone
    // Simpler: use the photo alpha mask to identify inner vs outer ring.

    // Find pixels near the boundary using alpha gradient
    // Inner ring: alpha 0.6-0.95 (photo-dominant near boundary)
    // Outer ring: alpha 0.05-0.4 (albedo-dominant near boundary)
    let innerR = 0, innerG = 0, innerB = 0, innerCount = 0;
    let outerR = 0, outerG = 0, outerB = 0, outerCount = 0;

    for (let i = 0; i < N; i++) {
      const a = alpha[i];
      const r = bd[i * 4], g = bd[i * 4 + 1], b = bd[i * 4 + 2];
      // Skip very dark or very bright pixels (unreliable)
      const lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
      if (lum < 40 || lum > 240) continue;

      if (a >= 0.6 && a <= 0.95) {
        // Inner band: photo-dominant side of boundary
        innerR += r; innerG += g; innerB += b;
        innerCount++;
      } else if (a >= 0.05 && a <= 0.4) {
        // Outer band: albedo-dominant side of boundary
        outerR += r; outerG += g; outerB += b;
        outerCount++;
      }
    }

    if (innerCount < 50 || outerCount < 50) {
      console.log(`Boundary ring match: insufficient samples (inner=${innerCount}, outer=${outerCount}), skipping`);
      return;
    }

    // Compute mean colors for each ring
    innerR /= innerCount; innerG /= innerCount; innerB /= innerCount;
    outerR /= outerCount; outerG /= outerCount; outerB /= outerCount;

    // Correction ratios: what multiplier would make outer ring match inner ring?
    const corrR = innerR > 5 ? (innerR / Math.max(5, outerR)) : 1;
    const corrG = innerG > 5 ? (innerG / Math.max(5, outerG)) : 1;
    const corrB = innerB > 5 ? (innerB / Math.max(5, outerB)) : 1;

    // Clamp to prevent wild corrections (max 30% shift)
    const clamp = (v) => Math.max(0.7, Math.min(1.3, v));
    const cR = clamp(corrR), cG = clamp(corrG), cB = clamp(corrB);

    console.log(`Boundary ring match: inner RGB=(${innerR.toFixed(0)},${innerG.toFixed(0)},${innerB.toFixed(0)}), outer RGB=(${outerR.toFixed(0)},${outerG.toFixed(0)},${outerB.toFixed(0)})`);
    console.log(`Boundary ring match: correction R=${cR.toFixed(3)}, G=${cG.toFixed(3)}, B=${cB.toFixed(3)}`);

    // Apply correction with falloff based on alpha (how "albedo-dominant" each pixel is)
    // Pixels with alpha=0 get full correction, alpha=1 get no correction.
    // The falloff is: strength = (1 - alpha) * correctionStrength
    let correctedCount = 0;
    for (let i = 0; i < N; i++) {
      const a = alpha[i];
      if (a >= 0.95) continue; // Fully photo-mapped, no correction needed
      if (a <= 0) continue;    // Fully outside, apply full correction

      // Smooth falloff: full correction at a=0, zero at a=0.8
      const strength = Math.max(0, Math.min(1, (0.8 - a) / 0.8));
      if (strength < 0.01) continue;

      const idx = i * 4;
      const r = bd[idx], g = bd[idx + 1], b = bd[idx + 2];

      // Blend toward corrected color
      bd[idx]     = Math.max(0, Math.min(255, Math.round(r * (1 + strength * (cR - 1)))));
      bd[idx + 1] = Math.max(0, Math.min(255, Math.round(g * (1 + strength * (cG - 1)))));
      bd[idx + 2] = Math.max(0, Math.min(255, Math.round(b * (1 + strength * (cB - 1)))));
      correctedCount++;
    }

    console.log(`Boundary ring match: ${correctedCount} pixels color-corrected`);
  }

  // ===================================================================
  // AO Bake (expert-disabled)
  // ===================================================================

  /**
   * P4 Expert: bake subtle ambient occlusion into the texture.
   * Computes per-vertex AO from mesh concavity (vertex normal vs. average
   * neighbor normal divergence) and rasterizes it into UV space.
   * Applied strongest in albedo-dominant regions, subtle in photo regions.
   *
   * This adds depth cues (darker eye sockets, nostrils, under-chin) without
   * requiring custom shaders or increasing specular (which causes plastic wrap).
   */
  _bakeSubtleAO(blendedData, photoData, meshGen, size) {
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const uvCoords = meshGen.flameUVCoords;
    const uvFaces = meshGen.flameUVFaces;
    if (!posFaces || !verts || !uvCoords || !uvFaces) return;

    const nVerts = verts.length / 3;
    const nFaces = posFaces.length / 3;

    // Step 1: Compute per-vertex normals (area-weighted)
    const normals = new Float32Array(nVerts * 3);
    for (let f = 0; f < nFaces; f++) {
      const i0 = posFaces[f * 3], i1 = posFaces[f * 3 + 1], i2 = posFaces[f * 3 + 2];
      const x0 = verts[i0 * 3], y0 = verts[i0 * 3 + 1], z0 = verts[i0 * 3 + 2];
      const x1 = verts[i1 * 3], y1 = verts[i1 * 3 + 1], z1 = verts[i1 * 3 + 2];
      const x2 = verts[i2 * 3], y2 = verts[i2 * 3 + 1], z2 = verts[i2 * 3 + 2];
      const ex1 = x1 - x0, ey1 = y1 - y0, ez1 = z1 - z0;
      const ex2 = x2 - x0, ey2 = y2 - y0, ez2 = z2 - z0;
      const nx = ey1 * ez2 - ez1 * ey2;
      const ny = ez1 * ex2 - ex1 * ez2;
      const nz = ex1 * ey2 - ey1 * ex2;
      for (const vi of [i0, i1, i2]) {
        normals[vi * 3] += nx;
        normals[vi * 3 + 1] += ny;
        normals[vi * 3 + 2] += nz;
      }
    }
    // Normalize
    for (let i = 0; i < nVerts; i++) {
      const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
      const len = Math.sqrt(x * x + y * y + z * z);
      if (len > 1e-10) {
        normals[i * 3] /= len; normals[i * 3 + 1] /= len; normals[i * 3 + 2] /= len;
      }
    }

    // Step 2: Compute per-vertex AO from concavity
    // Build adjacency: for each vertex, collect neighbor vertex indices
    const neighbors = new Array(nVerts);
    for (let i = 0; i < nVerts; i++) neighbors[i] = new Set();
    for (let f = 0; f < nFaces; f++) {
      const i0 = posFaces[f * 3], i1 = posFaces[f * 3 + 1], i2 = posFaces[f * 3 + 2];
      neighbors[i0].add(i1); neighbors[i0].add(i2);
      neighbors[i1].add(i0); neighbors[i1].add(i2);
      neighbors[i2].add(i0); neighbors[i2].add(i1);
    }

    // AO = how much the vertex is "enclosed" by neighbors
    // Measure: average dot(normal, direction to neighbor)
    // Negative average = convex (exposed), positive average = concave (occluded)
    const vertexAO = new Float32Array(nVerts);
    for (let i = 0; i < nVerts; i++) {
      const nx = normals[i * 3], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];
      const px = verts[i * 3], py = verts[i * 3 + 1], pz = verts[i * 3 + 2];
      let sumDot = 0, count = 0;

      for (const j of neighbors[i]) {
        const dx = verts[j * 3] - px;
        const dy = verts[j * 3 + 1] - py;
        const dz = verts[j * 3 + 2] - pz;
        const dLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dLen < 1e-10) continue;
        // dot(normal, direction_to_neighbor): positive = neighbor is above normal = concave
        sumDot += (nx * dx + ny * dy + nz * dz) / dLen;
        count++;
      }

      if (count > 0) {
        // Map: negative avg (convex) -> AO near 1 (bright)
        //       positive avg (concave) -> AO < 1 (dark)
        const avgDot = sumDot / count;
        // Clamp and invert: avgDot typically -0.3 to +0.3
        vertexAO[i] = Math.max(0.4, Math.min(1.0, 1.0 - avgDot * 2.0));
      } else {
        vertexAO[i] = 1.0;
      }
    }

    // Step 3: Rasterize per-vertex AO into UV space
    const aoMap = new Float32Array(size * size).fill(1.0);
    const aoCount = new Uint8Array(size * size);
    const nUVFaces = uvFaces.length / 3;

    for (let f = 0; f < nUVFaces && f < nFaces; f++) {
      const ui0 = uvFaces[f * 3], ui1 = uvFaces[f * 3 + 1], ui2 = uvFaces[f * 3 + 2];
      const pi0 = posFaces[f * 3], pi1 = posFaces[f * 3 + 1], pi2 = posFaces[f * 3 + 2];

      const ux0 = uvCoords[ui0 * 2] * size, uy0 = uvCoords[ui0 * 2 + 1] * size;
      const ux1 = uvCoords[ui1 * 2] * size, uy1 = uvCoords[ui1 * 2 + 1] * size;
      const ux2 = uvCoords[ui2 * 2] * size, uy2 = uvCoords[ui2 * 2 + 1] * size;

      const ao0 = vertexAO[pi0], ao1 = vertexAO[pi1], ao2 = vertexAO[pi2];

      // Bounding box
      const minX = Math.max(0, Math.floor(Math.min(ux0, ux1, ux2)));
      const maxX = Math.min(size - 1, Math.ceil(Math.max(ux0, ux1, ux2)));
      const minY = Math.max(0, Math.floor(Math.min(uy0, uy1, uy2)));
      const maxY = Math.min(size - 1, Math.ceil(Math.max(uy0, uy1, uy2)));

      const area = (ux1 - ux0) * (uy2 - uy0) - (ux2 - ux0) * (uy1 - uy0);
      if (Math.abs(area) < 0.01) continue;
      const invArea = 1.0 / area;

      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          // Barycentric coordinates
          const w0 = ((ux1 - x) * (uy2 - y) - (ux2 - x) * (uy1 - y)) * invArea;
          const w1 = ((ux2 - x) * (uy0 - y) - (ux0 - x) * (uy2 - y)) * invArea;
          const w2 = 1.0 - w0 - w1;
          if (w0 < -0.01 || w1 < -0.01 || w2 < -0.01) continue;

          const idx = y * size + x;
          const ao = w0 * ao0 + w1 * ao1 + w2 * ao2;
          if (aoCount[idx] === 0) {
            aoMap[idx] = ao;
          } else {
            aoMap[idx] = (aoMap[idx] * aoCount[idx] + ao) / (aoCount[idx] + 1);
          }
          aoCount[idx]++;
        }
      }
    }

    // Step 4: Blur AO map slightly for smoother result
    const blurredAO = this._boxBlurAlpha(aoMap, size, Math.max(4, Math.round(size / 256)));

    // Step 5: Apply AO to blended texture
    // Strength is modulated by photo coverage: less AO in photo regions (they have
    // natural shadows), more AO in albedo regions (need depth cues).
    const AO_STRENGTH = 0.20;  // Expert: subtle (0.15-0.25 range)
    const bd = blendedData.data;
    const pd = photoData.data;
    let aoApplied = 0;

    for (let i = 0; i < size * size; i++) {
      const ao = blurredAO[i];
      if (ao >= 0.99) continue; // No occlusion, skip

      const photoAlpha = pd[i * 4 + 3] / 255;
      // In photo-dominant pixels, reduce AO effect (photo has its own shadows)
      const effectiveStrength = AO_STRENGTH * (1.0 - photoAlpha * 0.7);
      const factor = 1.0 - effectiveStrength * (1.0 - ao);

      const idx = i * 4;
      bd[idx]     = Math.round(bd[idx] * factor);
      bd[idx + 1] = Math.round(bd[idx + 1] * factor);
      bd[idx + 2] = Math.round(bd[idx + 2] * factor);
      aoApplied++;
    }

    console.log(`PhotoUploader: Baked AO applied to ${aoApplied} pixels (strength=${AO_STRENGTH})`);
  }

  // ===================================================================
  // Prepare Albedo Layer
  // ===================================================================

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
      // No albedo -> fill with average photo skin color
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

    // Compute robust color correction using safe, central skin landmarks only.
    // Expert fix: removed chin (152), jaw left (234), jaw right (454) --
    // these sample jaw shadow, hair edges, neck which contaminate with green/gray.
    // Safe landmarks: upper forehead, nose bridge/tip, upper cheeks near nose.

    // Build UV-space bounding boxes for sampling regions
    let sampleBoxes = null;
    if (landmarks && mapping) {
      const SAMPLE_MP = [10, 8, 6, 117, 346, 101, 330];
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

      // Expert fix: tighter luminance filter (was 20-240, too permissive)
      // Rejects dark shadows and blown highlights that carry color casts
      const pLum = pd[i * 4] * 0.2126 + pd[i * 4 + 1] * 0.7152 + pd[i * 4 + 2] * 0.0722;
      const aLum = albedo[ai] * 0.2126 + albedo[ai + 1] * 0.7152 + albedo[ai + 2] * 0.0722;
      if (pLum < 60 || pLum > 220 || aLum < 30 || aLum > 230) continue;

      // Expert fix: reject near-gray pixels (low saturation = shadow contamination)
      const pr = pd[i * 4], pg = pd[i * 4 + 1], pb = pd[i * 4 + 2];
      const maxC = Math.max(pr, pg, pb);
      const minC = Math.min(pr, pg, pb);
      const sat = maxC > 10 ? (maxC - minC) / maxC : 0;
      if (sat < 0.05) continue;  // Skip desaturated pixels (shadow/background)

      samples.push({
        pr, pg, pb,
        ar: albedo[ai], ag: albedo[ai + 1], ab: albedo[ai + 2],
        lum: pLum
      });
    }

    // Use MEDIAN for robust color correction (resistant to outliers/shadow/beard)
    if (samples.length < 10) {
      console.warn(`Albedo color correction: only ${samples.length} samples — using defaults`);
      const crR = 1, crG = 1, crB = 1;
      for (let i = 0; i < size * size; i++) {
        const u = (i % size) / size;
        const v = Math.floor(i / size) / size;
        const ax = Math.min(albedoRes - 1, Math.floor(u * albedoRes));
        const ay = Math.min(albedoRes - 1, Math.floor(v * albedoRes));
        const ai = (ay * albedoRes + ax) * 3;
        ad[i * 4] = albedo[ai]; ad[i * 4 + 1] = albedo[ai + 1]; ad[i * 4 + 2] = albedo[ai + 2]; ad[i * 4 + 3] = 255;
      }
      return albedoData;
    }

    // Compute per-channel ratios and take median
    const ratiosR = [], ratiosG = [], ratiosB = [];
    for (const s of samples) {
      if (s.ar > 10) ratiosR.push(s.pr / s.ar);
      if (s.ag > 10) ratiosG.push(s.pg / s.ag);
      if (s.ab > 10) ratiosB.push(s.pb / s.ab);
    }
    ratiosR.sort((a, b) => a - b);
    ratiosG.sort((a, b) => a - b);
    ratiosB.sort((a, b) => a - b);

    const median = arr => arr.length > 0 ? arr[Math.floor(arr.length / 2)] : 1;
    const crR = Math.max(0.3, Math.min(3.0, median(ratiosR)));
    const crG = Math.max(0.3, Math.min(3.0, median(ratiosG)));
    const crB = Math.max(0.3, Math.min(3.0, median(ratiosB)));
    console.log(`Albedo color correction (MEDIAN): R=${crR.toFixed(3)}, G=${crG.toFixed(3)}, B=${crB.toFixed(3)}, samples=${samples.length}`);

    // Expert diagnostic: check for sampling contamination via variance
    if (samples.length > 20) {
      const avgR = samples.reduce((s, x) => s + x.pr, 0) / samples.length;
      const avgG = samples.reduce((s, x) => s + x.pg, 0) / samples.length;
      const stdR = Math.sqrt(samples.reduce((s, x) => s + (x.pr - avgR) ** 2, 0) / samples.length);
      const stdG = Math.sqrt(samples.reduce((s, x) => s + (x.pg - avgG) ** 2, 0) / samples.length);
      console.log(`Albedo sampling DIAG: avgRGB=(${avgR.toFixed(0)},${avgG.toFixed(0)}), stdR=${stdR.toFixed(1)}, stdG=${stdG.toFixed(1)}`);
      if (stdR > 40 || stdG > 40) {
        console.warn('Albedo sampling WARNING: HIGH variance — possible contamination from shadow/hair/background');
      }
    }

    // Phase 10b diagnostic: compare photo vs corrected albedo luminance + per-channel RGB
    const avgPhotoR = samples.reduce((s, x) => s + x.pr, 0) / samples.length;
    const avgPhotoG = samples.reduce((s, x) => s + x.pg, 0) / samples.length;
    const avgPhotoB = samples.reduce((s, x) => s + x.pb, 0) / samples.length;
    const avgAlbedoR = samples.reduce((s, x) => s + x.ar, 0) / samples.length;
    const avgAlbedoG = samples.reduce((s, x) => s + x.ag, 0) / samples.length;
    const avgAlbedoB = samples.reduce((s, x) => s + x.ab, 0) / samples.length;
    const photoLum = avgPhotoR * 0.2126 + avgPhotoG * 0.7152 + avgPhotoB * 0.0722;
    const correctedLum = (avgAlbedoR * crR) * 0.2126 + (avgAlbedoG * crG) * 0.7152 + (avgAlbedoB * crB) * 0.0722;
    const lumRatio = correctedLum / Math.max(1, photoLum);
    console.log(`Albedo tinting DIAG: photo lum=${photoLum.toFixed(1)}, corrected albedo lum=${correctedLum.toFixed(1)}, ratio=${lumRatio.toFixed(2)}`);
    console.log(`Albedo tinting DIAG: photo RGB=(${avgPhotoR.toFixed(0)},${avgPhotoG.toFixed(0)},${avgPhotoB.toFixed(0)}), corrected albedo RGB=(${(avgAlbedoR*crR).toFixed(0)},${(avgAlbedoG*crG).toFixed(0)},${(avgAlbedoB*crB).toFixed(0)})`);
    if (Math.abs(lumRatio - 1.0) > 0.15) {
      console.warn(`Albedo tinting WARNING: luminance mismatch ${((lumRatio - 1.0) * 100).toFixed(0)}% — tinting may need adjustment`);
    }

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
  // Simple fill utility (fallback when FLAME albedo not available)
  // -----------------------------------------------------------------------

  /**
   * Fill unmapped (alpha=0) pixels with average skin color.
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
