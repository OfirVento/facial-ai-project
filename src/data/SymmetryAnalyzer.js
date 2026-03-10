/**
 * SymmetryAnalyzer — Compare left/right region vertex geometry for facial symmetry.
 *
 * Uses the 52 clinical zones from FlameMeshGenerator, identifies L/R pairs,
 * mirrors left-side vertices across the sagittal plane (x=0 in FLAME space),
 * and computes per-region and overall symmetry scores.
 *
 * Scores: 1.0 = perfectly symmetric, 0.0 = maximum asymmetry.
 */

/**
 * Left/Right region pairs for symmetry comparison.
 * Each entry: [leftRegion, rightRegion, displayName]
 */
const SYMMETRY_PAIRS = [
  // Forehead
  ['forehead_left',          'forehead_right',          'Forehead'],

  // Brows
  ['brow_left',              'brow_right',              'Brow'],
  ['brow_inner_left',        'brow_inner_right',        'Inner Brow'],

  // Eyes
  ['eye_left_upper',         'eye_right_upper',         'Upper Eyelid'],
  ['eye_left_lower',         'eye_right_lower',         'Lower Eyelid'],
  ['eye_left_corner_inner',  'eye_right_corner_inner',  'Inner Canthus'],
  ['eye_left_corner_outer',  'eye_right_corner_outer',  'Outer Canthus'],

  // Under-eye
  ['under_eye_left',         'under_eye_right',         'Under-Eye'],
  ['tear_trough_left',       'tear_trough_right',       'Tear Trough'],

  // Nose
  ['nose_tip_left',          'nose_tip_right',          'Nose Tip'],
  ['nostril_left',           'nostril_right',            'Nostril'],

  // Cheeks
  ['cheek_left',             'cheek_right',              'Cheek'],
  ['cheekbone_left',         'cheekbone_right',          'Cheekbone'],
  ['cheek_hollow_left',      'cheek_hollow_right',       'Cheek Hollow'],

  // Nasolabial
  ['nasolabial_left',        'nasolabial_right',         'Nasolabial Fold'],

  // Lips
  ['lip_upper_left',         'lip_upper_right',          'Upper Lip'],
  ['lip_lower_left',         'lip_lower_right',          'Lower Lip'],
  ['lip_corner_left',        'lip_corner_right',         'Lip Corner'],

  // Chin
  ['chin_left',              'chin_right',               'Chin'],

  // Jaw
  ['jaw_left',               'jaw_right',                'Jaw'],
  ['jawline_left',           'jawline_right',            'Jawline'],

  // Temples
  ['temple_left',            'temple_right',             'Temple'],

  // Ears
  ['ear_left',               'ear_right',                'Ear'],
];

/**
 * Clinical significance thresholds (in metres).
 * FLAME vertex units are metres, so 1mm = 0.001.
 */
const ASYMMETRY_THRESHOLDS = {
  negligible: 0.0005,  // < 0.5mm — virtually undetectable
  mild:       0.001,   // < 1.0mm — subtle, typically not noticeable
  moderate:   0.002,   // < 2.0mm — noticeable on close inspection
  significant: 0.004,  // < 4.0mm — clearly visible
  // > 4mm = severe
};

export class SymmetryAnalyzer {
  /**
   * Analyze facial symmetry across all L/R region pairs.
   *
   * @param {Object} meshGen - FlameMeshGenerator instance with loaded mesh
   * @returns {Object} symmetry analysis result
   *   {
   *     overall: number (0-1),
   *     overallRMSE_mm: number,
   *     perRegion: { [displayName]: { score, rmse_mm, severity, leftCount, rightCount } },
   *     summary: string
   *   }
   */
  static analyze(meshGen) {
    if (!meshGen || !meshGen._flameCurrentVertices) {
      console.warn('SymmetryAnalyzer: No mesh data available');
      return null;
    }

    const vertices = meshGen._flameCurrentVertices;
    const perRegion = {};
    let totalWeightedScore = 0;
    let totalWeight = 0;

    for (const [leftReg, rightReg, displayName] of SYMMETRY_PAIRS) {
      const result = SymmetryAnalyzer._compareRegionPair(
        meshGen, leftReg, rightReg, vertices
      );

      if (result) {
        perRegion[displayName] = result;

        // Weight by number of vertices (larger regions matter more)
        const weight = result.leftCount + result.rightCount;
        totalWeightedScore += result.score * weight;
        totalWeight += weight;
      }
    }

    const overall = totalWeight > 0 ? totalWeightedScore / totalWeight : 1.0;

    // Compute overall RMSE from all region RMSEs
    let totalRMSEWeighted = 0;
    for (const [, val] of Object.entries(perRegion)) {
      const w = val.leftCount + val.rightCount;
      totalRMSEWeighted += val.rmse_mm * w;
    }
    const overallRMSE_mm = totalWeight > 0 ? totalRMSEWeighted / totalWeight : 0;

    const result = {
      overall: Math.round(overall * 1000) / 1000,
      overallRMSE_mm: Math.round(overallRMSE_mm * 100) / 100,
      overallSeverity: SymmetryAnalyzer._classifySeverity(overallRMSE_mm / 1000),
      perRegion,
      pairCount: Object.keys(perRegion).length,
    };

    console.log(`SymmetryAnalyzer: Overall symmetry = ${result.overall.toFixed(3)} ` +
      `(${result.overallSeverity}, RMSE = ${result.overallRMSE_mm.toFixed(2)} mm, ` +
      `${result.pairCount} pairs)`);

    return result;
  }

  // =========================================================================
  // Core comparison: L/R region pair
  // =========================================================================

  /**
   * Compare a single L/R region pair using centroid-based symmetry.
   *
   * Strategy:
   * 1. Compute centroid of left and right regions.
   * 2. Mirror left centroid across sagittal plane (negate X).
   * 3. Compute the distance between mirrored-left and right centroids.
   * 4. Also compare region "shape" by looking at vertex spread (StdDev).
   *
   * For vertex-level comparison when regions have same vertex count:
   * - Sort both regions' vertices by their Y coordinate (superior→inferior).
   * - Mirror left vertices' X coordinates.
   * - Compute per-vertex distance RMSE.
   *
   * @returns {Object|null} { score, rmse_mm, severity, leftCount, rightCount }
   */
  static _compareRegionPair(meshGen, leftReg, rightReg, vertices) {
    let leftIndices, rightIndices;
    try {
      leftIndices = meshGen.getRegionVertices(leftReg);
      rightIndices = meshGen.getRegionVertices(rightReg);
    } catch {
      return null;
    }

    if (!leftIndices?.length || !rightIndices?.length) return null;

    // -- Centroid comparison --
    const leftCentroid = SymmetryAnalyzer._centroid(leftIndices, vertices);
    const rightCentroid = SymmetryAnalyzer._centroid(rightIndices, vertices);

    // Mirror left centroid across sagittal plane (negate X)
    const mirroredLeft = [-leftCentroid[0], leftCentroid[1], leftCentroid[2]];

    // Distance between mirrored left and right centroids
    const centroidDist = SymmetryAnalyzer._dist3D(mirroredLeft, rightCentroid);

    // -- Shape comparison (spread/volume) --
    const leftSpread = SymmetryAnalyzer._regionSpread(leftIndices, vertices, leftCentroid);
    const rightSpread = SymmetryAnalyzer._regionSpread(rightIndices, vertices, rightCentroid);
    const spreadDiff = Math.abs(leftSpread - rightSpread);

    // Combined RMSE: centroid distance + shape difference
    const combinedRMSE = Math.sqrt(centroidDist * centroidDist + spreadDiff * spreadDiff);

    // -- Vertex-level comparison (when counts match) --
    let vertexRMSE = null;
    if (leftIndices.length === rightIndices.length && leftIndices.length >= 3) {
      vertexRMSE = SymmetryAnalyzer._vertexLevelRMSE(leftIndices, rightIndices, vertices);
    }

    // Use vertex-level RMSE when available (more accurate), else centroid-based
    const finalRMSE = vertexRMSE !== null ? vertexRMSE : combinedRMSE;

    // Convert RMSE to a 0-1 score
    // Score = 1.0 for perfect symmetry, decays with RMSE
    // Using sigmoid-like mapping: score = 1 / (1 + (rmse / threshold)^2)
    const threshold = 0.002; // 2mm — moderate asymmetry reference
    const score = 1.0 / (1.0 + (finalRMSE / threshold) * (finalRMSE / threshold));

    return {
      score: Math.round(score * 1000) / 1000,
      rmse_mm: Math.round(finalRMSE * 1000 * 100) / 100, // metres → mm, 2 decimal places
      severity: SymmetryAnalyzer._classifySeverity(finalRMSE),
      leftCount: leftIndices.length,
      rightCount: rightIndices.length,
    };
  }

  // =========================================================================
  // Vertex-level RMSE (when L/R regions have equal vertex counts)
  // =========================================================================

  /**
   * Sort vertices by Y-coordinate and compute per-vertex mirrored RMSE.
   * This gives a more detailed shape comparison than centroid-only.
   */
  static _vertexLevelRMSE(leftIndices, rightIndices, vertices) {
    // Sort both sides by Y (superior → inferior), then by Z (anterior → posterior)
    const sortByYZ = (indices) => {
      return [...indices].sort((a, b) => {
        const dy = vertices[a * 3 + 1] - vertices[b * 3 + 1];
        if (Math.abs(dy) > 1e-6) return dy;
        return vertices[a * 3 + 2] - vertices[b * 3 + 2];
      });
    };

    const sortedLeft = sortByYZ(leftIndices);
    const sortedRight = sortByYZ(rightIndices);

    let sumSqDist = 0;
    const n = sortedLeft.length;

    for (let i = 0; i < n; i++) {
      const li = sortedLeft[i];
      const ri = sortedRight[i];

      // Mirror left X (negate)
      const lx = -vertices[li * 3];
      const ly = vertices[li * 3 + 1];
      const lz = vertices[li * 3 + 2];

      const rx = vertices[ri * 3];
      const ry = vertices[ri * 3 + 1];
      const rz = vertices[ri * 3 + 2];

      const dx = lx - rx;
      const dy = ly - ry;
      const dz = lz - rz;

      sumSqDist += dx * dx + dy * dy + dz * dz;
    }

    return Math.sqrt(sumSqDist / n);
  }

  // =========================================================================
  // Geometry helpers
  // =========================================================================

  static _centroid(indices, vertices) {
    let cx = 0, cy = 0, cz = 0;
    for (const vi of indices) {
      cx += vertices[vi * 3];
      cy += vertices[vi * 3 + 1];
      cz += vertices[vi * 3 + 2];
    }
    const n = indices.length;
    return [cx / n, cy / n, cz / n];
  }

  /**
   * Compute average distance of vertices from centroid (spread/size).
   */
  static _regionSpread(indices, vertices, centroid) {
    let totalDist = 0;
    for (const vi of indices) {
      const dx = vertices[vi * 3] - centroid[0];
      const dy = vertices[vi * 3 + 1] - centroid[1];
      const dz = vertices[vi * 3 + 2] - centroid[2];
      totalDist += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return totalDist / indices.length;
  }

  static _dist3D(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Classify asymmetry severity based on RMSE in metres.
   * @param {number} rmse - RMSE in metres
   * @returns {string} severity level
   */
  static _classifySeverity(rmse) {
    if (rmse < ASYMMETRY_THRESHOLDS.negligible) return 'negligible';
    if (rmse < ASYMMETRY_THRESHOLDS.mild)       return 'mild';
    if (rmse < ASYMMETRY_THRESHOLDS.moderate)    return 'moderate';
    if (rmse < ASYMMETRY_THRESHOLDS.significant) return 'significant';
    return 'severe';
  }

  // =========================================================================
  // Report formatting
  // =========================================================================

  /**
   * Generate a formatted symmetry report.
   * @param {Object} analysis - Output of analyze()
   * @returns {string}
   */
  static formatReport(analysis) {
    if (!analysis) return 'No symmetry data available.';

    const lines = [
      '=== Facial Symmetry Analysis ===\n',
      `Overall Symmetry Score: ${(analysis.overall * 100).toFixed(1)}%`,
      `Overall RMSE: ${analysis.overallRMSE_mm.toFixed(2)} mm (${analysis.overallSeverity})`,
      `Regions Analyzed: ${analysis.pairCount}\n`,
      '--- Per-Region Symmetry ---',
    ];

    // Sort by score ascending (most asymmetric first)
    const sorted = Object.entries(analysis.perRegion)
      .sort((a, b) => a[1].score - b[1].score);

    for (const [name, data] of sorted) {
      const pct = (data.score * 100).toFixed(1);
      const bar = SymmetryAnalyzer._scoreBar(data.score);
      lines.push(`  ${bar} ${pct}%  ${name} (${data.rmse_mm.toFixed(2)} mm, ${data.severity})`);
    }

    return lines.join('\n');
  }

  /**
   * Simple ASCII bar for symmetry score visualization.
   */
  static _scoreBar(score) {
    const filled = Math.round(score * 10);
    return '[' + '#'.repeat(filled) + '.'.repeat(10 - filled) + ']';
  }

  /**
   * Get the top N most asymmetric regions for clinical focus.
   * @param {Object} analysis - Output of analyze()
   * @param {number} n - Number of regions to return
   * @returns {Array<{name, score, rmse_mm, severity}>}
   */
  static getMostAsymmetric(analysis, n = 5) {
    if (!analysis?.perRegion) return [];

    return Object.entries(analysis.perRegion)
      .sort((a, b) => a[1].score - b[1].score)
      .slice(0, n)
      .map(([name, data]) => ({
        name,
        score: data.score,
        rmse_mm: data.rmse_mm,
        severity: data.severity,
      }));
  }
}
