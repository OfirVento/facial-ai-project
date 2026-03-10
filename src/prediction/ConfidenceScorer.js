/**
 * ConfidenceScorer — Per-prediction confidence assessment.
 *
 * Evaluates how reliable a treatment prediction is based on:
 *   - Treatment type complexity
 *   - Volume requested (higher volumes = less predictable)
 *   - Available face data quality (FaceDNA capture quality)
 *   - Combination complexity
 *
 * Confidence levels:
 *   High   (0.8+): Standard injectables with conservative volumes
 *   Medium (0.5-0.8): Larger volumes, combinations, less common areas
 *   Low    (<0.5): Surgical procedures, extreme volumes, poor data quality
 */

// ---------------------------------------------------------------------------
// Treatment confidence baselines
// ---------------------------------------------------------------------------

/**
 * Base confidence scores per treatment type.
 * These represent confidence when the treatment is applied at
 * default volume with good-quality capture data.
 */
const BASE_CONFIDENCE = {
  // High confidence — well-understood injectable effects
  lip_filler_full:     0.85,
  lip_filler_upper:    0.87,
  lip_filler_lower:    0.87,
  lip_border:          0.82,
  cheek_filler:        0.83,
  cheek_hollow_filler: 0.80,
  nasolabial_filler:   0.82,

  // Medium-high confidence
  jaw_filler:          0.78,
  chin_filler:         0.77,
  tear_trough_filler:  0.75,
  temple_filler:       0.73,
  botox_forehead:      0.70,
  botox_glabella:      0.72,
  botox_brow_lift:     0.65,

  // Lower confidence — harder to predict accurately
  nose_filler_bridge:  0.60,
  nose_filler_tip:     0.55,

  // Combination — inherently less predictable
  liquid_facelift:     0.55,
};

// ---------------------------------------------------------------------------
// ConfidenceScorer
// ---------------------------------------------------------------------------

export class ConfidenceScorer {
  /**
   * Score the confidence of a treatment prediction.
   *
   * @param {string} treatmentId - Treatment key from GeometryPredictor
   * @param {object} [params]
   * @param {number} [params.volume] - Requested volume (ml or units)
   * @param {number} [params.maxVolume] - Treatment's max safe volume
   * @param {number} [params.captureQuality] - FaceDNA capture quality (0-1)
   * @param {boolean} [params.hasMultiView] - Whether multiple capture angles exist
   * @returns {{
   *   score: number,          // 0-1 overall confidence
   *   level: string,          // 'high' | 'medium' | 'low'
   *   factors: Object,        // Individual factor scores
   *   explanation: string     // Human-readable explanation
   * }}
   */
  static score(treatmentId, params = {}) {
    const baseScore = BASE_CONFIDENCE[treatmentId] ?? 0.50;

    // Factor 1: Volume ratio (higher ratio = less predictable)
    const volume = params.volume ?? 1.0;
    const maxVolume = params.maxVolume ?? 5.0;
    const volumeRatio = volume / maxVolume;
    // Score decreases as we approach max volume
    const volumeFactor = 1.0 - (volumeRatio * 0.3);

    // Factor 2: Capture quality
    const captureQuality = params.captureQuality ?? 1.0;
    const qualityFactor = 0.6 + (captureQuality * 0.4); // floor at 0.6

    // Factor 3: Multi-view bonus
    const multiViewFactor = params.hasMultiView ? 1.05 : 0.95;

    // Combine factors
    const finalScore = Math.max(0, Math.min(1,
      baseScore * volumeFactor * qualityFactor * multiViewFactor
    ));

    // Classify level
    let level, explanation;
    if (finalScore >= 0.8) {
      level = 'high';
      explanation = 'This prediction is highly reliable for this treatment type and parameters.';
    } else if (finalScore >= 0.5) {
      level = 'medium';
      const reasons = [];
      if (volumeRatio > 0.6) reasons.push('higher than typical volume');
      if (captureQuality < 0.7) reasons.push('lower capture quality');
      if (baseScore < 0.7) reasons.push('complex treatment type');
      explanation = reasons.length > 0 ?
        `Moderate confidence due to: ${reasons.join(', ')}.` :
        'Moderate confidence — results may vary from prediction.';
    } else {
      level = 'low';
      const reasons = [];
      if (baseScore < 0.6) reasons.push('treatment type is hard to predict accurately');
      if (volumeRatio > 0.8) reasons.push('volume near maximum limit');
      if (captureQuality < 0.5) reasons.push('poor capture quality');
      explanation = `Lower confidence: ${reasons.join('; ')}. Actual results may differ significantly.`;
    }

    return {
      score: Math.round(finalScore * 100) / 100,
      level,
      factors: {
        base: Math.round(baseScore * 100) / 100,
        volume: Math.round(volumeFactor * 100) / 100,
        quality: Math.round(qualityFactor * 100) / 100,
        multiView: Math.round(multiViewFactor * 100) / 100,
      },
      explanation,
    };
  }

  /**
   * Get a color for UI display based on confidence level.
   * @param {string} level - 'high' | 'medium' | 'low'
   * @returns {string} CSS color variable name
   */
  static colorForLevel(level) {
    switch (level) {
      case 'high': return 'var(--success)';
      case 'medium': return 'var(--warning)';
      case 'low': return 'var(--danger)';
      default: return 'var(--text-muted)';
    }
  }
}

export { BASE_CONFIDENCE };
