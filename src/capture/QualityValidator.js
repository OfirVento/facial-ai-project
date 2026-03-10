/**
 * QualityValidator — Post-capture image quality analysis.
 *
 * Analyzes captured photos for:
 *   - Blur / sharpness (Laplacian variance on luminance)
 *   - Lighting quality (brightness, exposure, directional bias)
 *   - Overall quality score (0-100)
 *
 * All processing runs in-browser using Canvas 2D API.
 * Images are downsampled to ≤640px wide for performance.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Laplacian variance below this → image is blurry */
const BLUR_VARIANCE_THRESHOLD = 100;

/** Mean luminance thresholds */
const MIN_BRIGHTNESS = 60;
const MAX_BRIGHTNESS = 220;

/** Left-right brightness difference threshold for directional bias */
const LIGHTING_BIAS_THRESHOLD = 30;

/** Below this overall score → recommend retake */
const MIN_ACCEPTABLE_SCORE = 40;

/** Max processing width (downsample for performance) */
const MAX_ANALYSIS_WIDTH = 640;

// ---------------------------------------------------------------------------
// QualityValidator
// ---------------------------------------------------------------------------

export class QualityValidator {
  /**
   * Analyze a captured image for quality.
   *
   * @param {HTMLCanvasElement|HTMLImageElement|ImageBitmap} source
   * @returns {Promise<{
   *   overallScore: number,
   *   blur: { score: number, variance: number, ok: boolean },
   *   lighting: { score: number, meanBrightness: number, leftBrightness: number,
   *               rightBrightness: number, directionalBias: number, ok: boolean },
   *   feedback: string[],
   *   acceptable: boolean
   * }>}
   */
  async analyze(source) {
    const imageData = this._toImageData(source);
    const feedback = [];

    // --- Blur analysis ---
    const blur = this._analyzeBlur(imageData);
    if (!blur.ok) {
      if (blur.variance < BLUR_VARIANCE_THRESHOLD * 0.3) {
        feedback.push('Image is very blurry — hold the camera steadier');
      } else {
        feedback.push('Image is slightly blurry — try to keep still');
      }
    }

    // --- Lighting analysis ---
    const lighting = this._analyzeLighting(imageData);
    if (!lighting.ok) {
      if (lighting.meanBrightness < MIN_BRIGHTNESS) {
        feedback.push('Image is too dark — find better lighting');
      } else if (lighting.meanBrightness > MAX_BRIGHTNESS) {
        feedback.push('Image is overexposed — reduce lighting or move from direct light');
      }
      if (lighting.directionalBias > LIGHTING_BIAS_THRESHOLD) {
        const brighter = lighting.leftBrightness > lighting.rightBrightness ? 'left' : 'right';
        feedback.push(`Uneven lighting — the ${brighter} side is brighter`);
      }
    }

    // --- Overall score ---
    const overallScore = Math.round(blur.score * 0.6 + lighting.score * 0.4);
    const acceptable = overallScore >= MIN_ACCEPTABLE_SCORE;

    if (!acceptable && feedback.length === 0) {
      feedback.push('Image quality is low — try capturing again');
    }

    return { overallScore, blur, lighting, feedback, acceptable };
  }

  // =========================================================================
  // Blur detection — Laplacian variance
  // =========================================================================

  /**
   * Detect blur via Laplacian variance on luminance channel.
   *
   * Algorithm:
   *   1. Convert to grayscale luminance
   *   2. Apply 3×3 Laplacian kernel: [0,1,0; 1,-4,1; 0,1,0]
   *   3. Compute variance of Laplacian response
   *   4. Higher variance = sharper image
   *
   * @param {ImageData} imageData
   * @returns {{ variance: number, score: number, ok: boolean }}
   */
  _analyzeBlur(imageData) {
    const { width, height, data } = imageData;

    // Step 1: Extract luminance channel
    const lum = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    // Step 2: Apply Laplacian kernel
    let sum = 0;
    let sumSq = 0;
    let count = 0;

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const lap = -4 * lum[idx]
          + lum[idx - 1] + lum[idx + 1]
          + lum[idx - width] + lum[idx + width];
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }

    // Step 3: Compute variance
    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);

    // Step 4: Map variance to 0-100 score
    // variance < 50 = very blurry (~0)
    // variance > 500 = very sharp (~100)
    const score = Math.min(100, Math.max(0, (variance - 50) / 4.5));

    return {
      variance: Math.round(variance * 10) / 10,
      score: Math.round(score),
      ok: variance >= BLUR_VARIANCE_THRESHOLD,
    };
  }

  // =========================================================================
  // Lighting analysis — histogram
  // =========================================================================

  /**
   * Analyze lighting quality via luminance histogram.
   *
   * Checks:
   *   - Mean brightness (too dark / too bright)
   *   - Directional bias (left vs right half brightness difference)
   *   - Clipping (too many pixels at 0 or 255)
   *
   * @param {ImageData} imageData
   * @returns {{ meanBrightness, leftBrightness, rightBrightness, directionalBias, score, ok }}
   */
  _analyzeLighting(imageData) {
    const { width, height, data } = imageData;
    const halfWidth = Math.floor(width / 2);

    let totalSum = 0;
    let leftSum = 0, rightSum = 0;
    let leftCount = 0, rightCount = 0;
    const histogram = new Uint32Array(256);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const lum = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
        histogram[lum]++;
        totalSum += lum;

        if (x < halfWidth) {
          leftSum += lum;
          leftCount++;
        } else {
          rightSum += lum;
          rightCount++;
        }
      }
    }

    const n = width * height;
    const meanBrightness = Math.round(totalSum / n);
    const leftBrightness = Math.round(leftSum / leftCount);
    const rightBrightness = Math.round(rightSum / rightCount);
    const directionalBias = Math.abs(leftBrightness - rightBrightness);

    // Check for clipping
    const darkClipping = histogram[0] / n;
    const brightClipping = histogram[255] / n;

    // Score components
    let score = 100;
    if (meanBrightness < MIN_BRIGHTNESS) {
      score -= (MIN_BRIGHTNESS - meanBrightness) * 1.5;
    }
    if (meanBrightness > MAX_BRIGHTNESS) {
      score -= (meanBrightness - MAX_BRIGHTNESS) * 2;
    }
    if (directionalBias > LIGHTING_BIAS_THRESHOLD) {
      score -= (directionalBias - LIGHTING_BIAS_THRESHOLD) * 1.0;
    }
    if (darkClipping > 0.05) {
      score -= darkClipping * 200;
    }
    if (brightClipping > 0.05) {
      score -= brightClipping * 200;
    }

    score = Math.max(0, Math.min(100, Math.round(score)));

    const ok = meanBrightness >= MIN_BRIGHTNESS &&
               meanBrightness <= MAX_BRIGHTNESS &&
               directionalBias < LIGHTING_BIAS_THRESHOLD;

    return {
      meanBrightness,
      leftBrightness,
      rightBrightness,
      directionalBias,
      score,
      ok,
    };
  }

  // =========================================================================
  // Image conversion
  // =========================================================================

  /**
   * Convert an image source to ImageData, downsampling if needed.
   * @param {HTMLCanvasElement|HTMLImageElement|ImageBitmap} source
   * @returns {ImageData}
   */
  _toImageData(source) {
    let srcWidth, srcHeight;

    if (source instanceof HTMLCanvasElement) {
      srcWidth = source.width;
      srcHeight = source.height;
    } else if (source instanceof HTMLImageElement) {
      srcWidth = source.naturalWidth || source.width;
      srcHeight = source.naturalHeight || source.height;
    } else if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
      srcWidth = source.width;
      srcHeight = source.height;
    } else {
      // Assume canvas-like
      srcWidth = source.width;
      srcHeight = source.height;
    }

    // Downsample for performance
    let w = srcWidth;
    let h = srcHeight;
    if (w > MAX_ANALYSIS_WIDTH) {
      const scale = MAX_ANALYSIS_WIDTH / w;
      w = MAX_ANALYSIS_WIDTH;
      h = Math.round(h * scale);
    }

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(source, 0, 0, w, h);

    return ctx.getImageData(0, 0, w, h);
  }
}

export { BLUR_VARIANCE_THRESHOLD, MIN_ACCEPTABLE_SCORE };
