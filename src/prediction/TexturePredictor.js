/**
 * TexturePredictor — Rule-based texture modifications per treatment.
 *
 * Applies post-treatment texture changes to simulate:
 *   - Filler: mild smoothing in treated UV region (skin stretches, wrinkles reduce)
 *   - Botox: selective smoothing in forehead/crow's feet UV zones
 *   - Laser/peels: histogram equalization (reduces pigmentation variation)
 *
 * Operates on the albedo texture (2D Canvas) in UV space.
 * Regions are mapped from 3D clinical zones to approximate UV rectangles.
 */

// ---------------------------------------------------------------------------
// UV region approximations for texture modifications
// ---------------------------------------------------------------------------

/**
 * Approximate UV bounding boxes for clinical regions.
 * FLAME UV layout: nose center ≈ (0.5, 0.5), lips below, forehead above.
 * Each entry: { uMin, vMin, uMax, vMax } — in [0, 1] UV space.
 *
 * These are rough approximations for texture effects. For precise
 * per-vertex mapping, use FlameMeshGenerator's getRegionVertices()
 * with the UV coordinate array.
 */
const UV_REGION_BOUNDS = {
  // Lips
  lip_upper: { uMin: 0.35, vMin: 0.56, uMax: 0.65, vMax: 0.62 },
  lip_lower: { uMin: 0.35, vMin: 0.62, uMax: 0.65, vMax: 0.70 },
  lips_all:  { uMin: 0.30, vMin: 0.54, uMax: 0.70, vMax: 0.72 },

  // Cheeks
  cheek_left:  { uMin: 0.05, vMin: 0.38, uMax: 0.30, vMax: 0.60 },
  cheek_right: { uMin: 0.70, vMin: 0.38, uMax: 0.95, vMax: 0.60 },

  // Forehead
  forehead: { uMin: 0.20, vMin: 0.05, uMax: 0.80, vMax: 0.25 },

  // Nose
  nose_bridge: { uMin: 0.40, vMin: 0.28, uMax: 0.60, vMax: 0.50 },
  nose_tip:    { uMin: 0.42, vMin: 0.46, uMax: 0.58, vMax: 0.56 },

  // Under-eye
  under_eye_left:  { uMin: 0.15, vMin: 0.30, uMax: 0.35, vMax: 0.40 },
  under_eye_right: { uMin: 0.65, vMin: 0.30, uMax: 0.85, vMax: 0.40 },

  // Jaw / chin
  jaw:  { uMin: 0.10, vMin: 0.65, uMax: 0.90, vMax: 0.85 },
  chin: { uMin: 0.35, vMin: 0.72, uMax: 0.65, vMax: 0.85 },

  // Nasolabial
  nasolabial_left:  { uMin: 0.25, vMin: 0.48, uMax: 0.40, vMax: 0.62 },
  nasolabial_right: { uMin: 0.60, vMin: 0.48, uMax: 0.75, vMax: 0.62 },
};

/**
 * Treatment → UV region mapping for texture effects.
 */
const TREATMENT_UV_MAP = {
  lip_filler_full:   ['lips_all'],
  lip_filler_upper:  ['lip_upper'],
  lip_filler_lower:  ['lip_lower'],
  lip_border:        ['lip_upper'],
  cheek_filler:      ['cheek_left', 'cheek_right'],
  cheek_hollow_filler: ['cheek_left', 'cheek_right'],
  jaw_filler:        ['jaw'],
  chin_filler:       ['chin'],
  nose_filler_bridge: ['nose_bridge'],
  nose_filler_tip:   ['nose_tip'],
  tear_trough_filler: ['under_eye_left', 'under_eye_right'],
  temple_filler:     ['forehead'],
  nasolabial_filler: ['nasolabial_left', 'nasolabial_right'],
  botox_forehead:    ['forehead'],
  botox_glabella:    ['forehead'],
  botox_brow_lift:   ['forehead'],
};

/**
 * Effect types per treatment category.
 */
const TREATMENT_EFFECTS = {
  filler:  { blur: 1.5, brighten: 3, smoothAlpha: 0.15 },
  botox:   { blur: 2.0, brighten: 0, smoothAlpha: 0.25 },
  laser:   { blur: 0.5, brighten: 5, equalize: 0.3 },
};

// ---------------------------------------------------------------------------
// TexturePredictor
// ---------------------------------------------------------------------------

export class TexturePredictor {
  /**
   * Apply treatment texture effect to an albedo canvas.
   *
   * @param {HTMLCanvasElement} albedoCanvas - The current albedo texture
   * @param {string} treatmentId - Treatment key from GeometryPredictor
   * @param {object} [options]
   * @param {number} [options.intensity=1.0] - Effect intensity (0-2)
   * @param {number} [options.volume=1.0] - Treatment volume (affects effect strength)
   * @returns {HTMLCanvasElement} Modified albedo canvas (same canvas, mutated)
   */
  static apply(albedoCanvas, treatmentId, options = {}) {
    const intensity = options.intensity ?? 1.0;
    const volume = options.volume ?? 1.0;

    const uvRegions = TREATMENT_UV_MAP[treatmentId];
    if (!uvRegions || uvRegions.length === 0) return albedoCanvas;

    const ctx = albedoCanvas.getContext('2d');
    const w = albedoCanvas.width;
    const h = albedoCanvas.height;

    // Determine effect type
    const isBotox = treatmentId.startsWith('botox_');
    const effectType = isBotox ? TREATMENT_EFFECTS.botox : TREATMENT_EFFECTS.filler;

    // Scale effect by volume (more product = more visible texture change)
    const volumeFactor = Math.min(2.0, volume * 0.5);
    const blurRadius = effectType.blur * intensity * volumeFactor;
    const brightenAmount = effectType.brighten * intensity * volumeFactor;
    const smoothAlpha = effectType.smoothAlpha * intensity * Math.min(1, volumeFactor);

    for (const regionKey of uvRegions) {
      const bounds = UV_REGION_BOUNDS[regionKey];
      if (!bounds) continue;

      // Convert UV bounds to pixel coordinates
      const px = Math.floor(bounds.uMin * w);
      const py = Math.floor(bounds.vMin * h);
      const pw = Math.ceil((bounds.uMax - bounds.uMin) * w);
      const ph = Math.ceil((bounds.vMax - bounds.vMin) * h);

      if (pw <= 0 || ph <= 0) continue;

      // Get region pixels
      const imageData = ctx.getImageData(px, py, pw, ph);

      // Apply blur (Gaussian approximation via box blur)
      if (blurRadius > 0.5) {
        TexturePredictor._boxBlur(imageData, Math.round(blurRadius));
      }

      // Apply brightness boost
      if (brightenAmount > 0) {
        TexturePredictor._brighten(imageData, brightenAmount);
      }

      // Blend back with original using smoothAlpha for subtlety
      if (smoothAlpha < 1.0) {
        const original = ctx.getImageData(px, py, pw, ph);
        TexturePredictor._blend(imageData, original, smoothAlpha);
      }

      ctx.putImageData(imageData, px, py);
    }

    return albedoCanvas;
  }

  // =========================================================================
  // Image processing helpers
  // =========================================================================

  /**
   * Simple box blur on ImageData (in-place).
   * Approximates Gaussian blur when applied 2-3 times.
   */
  static _boxBlur(imageData, radius) {
    if (radius < 1) return;
    const { width, height, data } = imageData;
    const copy = new Uint8ClampedArray(data);

    // Horizontal pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0, count = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < width) {
            const idx = (y * width + nx) * 4;
            r += copy[idx];
            g += copy[idx + 1];
            b += copy[idx + 2];
            count++;
          }
        }
        const idx = (y * width + x) * 4;
        data[idx] = r / count;
        data[idx + 1] = g / count;
        data[idx + 2] = b / count;
      }
    }

    // Vertical pass
    const copy2 = new Uint8ClampedArray(data);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          const ny = y + dy;
          if (ny >= 0 && ny < height) {
            const idx = (ny * width + x) * 4;
            r += copy2[idx];
            g += copy2[idx + 1];
            b += copy2[idx + 2];
            count++;
          }
        }
        const idx = (y * width + x) * 4;
        data[idx] = r / count;
        data[idx + 1] = g / count;
        data[idx + 2] = b / count;
      }
    }
  }

  /**
   * Brighten ImageData (in-place).
   * @param {ImageData} imageData
   * @param {number} amount - brightness increase (0-255 scale)
   */
  static _brighten(imageData, amount) {
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = Math.min(255, data[i] + amount);
      data[i + 1] = Math.min(255, data[i + 1] + amount);
      data[i + 2] = Math.min(255, data[i + 2] + amount);
    }
  }

  /**
   * Blend modified ImageData with original.
   * result = modified * alpha + original * (1 - alpha)
   */
  static _blend(modified, original, alpha) {
    const mData = modified.data;
    const oData = original.data;
    const oneMinusAlpha = 1 - alpha;

    for (let i = 0; i < mData.length; i += 4) {
      mData[i] = Math.round(mData[i] * alpha + oData[i] * oneMinusAlpha);
      mData[i + 1] = Math.round(mData[i + 1] * alpha + oData[i + 1] * oneMinusAlpha);
      mData[i + 2] = Math.round(mData[i + 2] * alpha + oData[i + 2] * oneMinusAlpha);
    }
  }
}

export { UV_REGION_BOUNDS, TREATMENT_UV_MAP };
