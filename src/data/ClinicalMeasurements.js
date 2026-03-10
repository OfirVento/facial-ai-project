/**
 * ClinicalMeasurements — Derive real-world mm measurements from FLAME 3D vertices.
 *
 * Uses the 52 clinical zones defined in FlameMeshGenerator to compute
 * standard facial proportions (distances, ratios, angles) used in
 * aesthetic and clinical facial analysis.
 *
 * All distances are in millimetres (FLAME vertices are in metres).
 */

const METRES_TO_MM = 1000;

/**
 * Left/Right region pairs for distance computations.
 * Each entry: [leftRegion, rightRegion, measurementName, clinicalDescription]
 */
const BILATERAL_DISTANCES = [
  ['cheekbone_left',       'cheekbone_right',       'bizygomaticWidth',     'Bizygomatic width (face width at cheekbones)'],
  ['jaw_left',             'jaw_right',             'bigonialWidth',        'Bigonial width (jaw width)'],
  ['jawline_left',         'jawline_right',         'jawlineWidth',         'Jawline border width'],
  ['temple_left',          'temple_right',          'bitemporalWidth',      'Bitemporal width'],
  ['cheek_hollow_left',    'cheek_hollow_right',    'buccalWidth',          'Buccal width (cheek hollow)'],
  ['eye_left_corner_inner','eye_right_corner_inner', 'intercanthalDist',    'Intercanthal distance'],
  ['eye_left_corner_outer','eye_right_corner_outer', 'outercanthalDist',    'Outer canthal distance'],
  ['nostril_left',         'nostril_right',          'noseWidth',            'Nasal width (alar width)'],
  ['lip_corner_left',      'lip_corner_right',       'mouthWidth',          'Mouth width (oral fissure)'],
  ['ear_left',             'ear_right',              'earToEarDist',        'Ear to ear distance'],
];

/**
 * Vertical / midline distances.
 * Each entry: [regionA, regionB, measurementName, clinicalDescription]
 */
const VERTICAL_DISTANCES = [
  ['forehead_center', 'chin_center',     'faceHeight',         'Total face height (trichion to menton)'],
  ['forehead_center', 'nose_bridge_upper','upperFaceHeight',   'Upper face height'],
  ['nose_bridge_upper','nose_tip',        'noseLength',        'Nose length (nasion to pronasale)'],
  ['nose_tip',         'lip_upper_center','nasalTipToLip',     'Nose tip to upper lip'],
  ['lip_upper_center', 'lip_lower_center','lipHeight',         'Total lip height (vermilion)'],
  ['lip_upper_center', 'chin_center',     'lowerFaceHeight',   'Lower face height (lip to menton)'],
  ['chin_center',      'chin',            'chinProjection',    'Chin projection depth'],
  ['brow_left',        'eye_left_upper',  'leftBrowHeight',    'Left brow to eyelid'],
  ['brow_right',       'eye_right_upper', 'rightBrowHeight',   'Right brow to eyelid'],
];

/**
 * Unilateral distances (single-side measurements).
 */
const UNILATERAL_DISTANCES = [
  ['eye_left_corner_inner',  'eye_left_corner_outer',  'leftPalpebralFissure',  'Left palpebral fissure width'],
  ['eye_right_corner_inner', 'eye_right_corner_outer', 'rightPalpebralFissure', 'Right palpebral fissure width'],
  ['lip_upper_left',         'lip_upper_right',        'cupidsBowWidth',        "Cupid's bow width"],
  ['nose_tip_left',          'nose_tip_right',          'noseTipWidth',         'Nose tip width'],
  ['nose_bridge_upper',      'nose_bridge_lower',       'nasalBridgeLength',    'Nasal bridge length'],
];

export class ClinicalMeasurements {
  /**
   * Compute all clinical measurements from the current mesh state.
   *
   * @param {Object} meshGen - FlameMeshGenerator instance with loaded mesh
   * @param {Object} [mapping] - Optional FLAME-MediaPipe mapping (unused for now)
   * @returns {Object} measurements — { name: { value: mm, description: string } }
   */
  static compute(meshGen, mapping = null) {
    if (!meshGen || !meshGen._flameCurrentVertices) {
      console.warn('ClinicalMeasurements: No mesh data available');
      return null;
    }

    const vertices = meshGen._flameCurrentVertices;
    const measurements = {};

    // -----------------------------------------------------------------------
    // 1. Bilateral distances (left-to-right centroid distances)
    // -----------------------------------------------------------------------
    for (const [leftReg, rightReg, name, desc] of BILATERAL_DISTANCES) {
      const leftCentroid = ClinicalMeasurements._regionCentroid(meshGen, leftReg, vertices);
      const rightCentroid = ClinicalMeasurements._regionCentroid(meshGen, rightReg, vertices);

      if (leftCentroid && rightCentroid) {
        measurements[name] = {
          value: ClinicalMeasurements._dist3D(leftCentroid, rightCentroid) * METRES_TO_MM,
          description: desc,
          unit: 'mm',
        };
      }
    }

    // -----------------------------------------------------------------------
    // 2. Vertical / midline distances
    // -----------------------------------------------------------------------
    for (const [regA, regB, name, desc] of VERTICAL_DISTANCES) {
      const centA = ClinicalMeasurements._regionCentroid(meshGen, regA, vertices);
      const centB = ClinicalMeasurements._regionCentroid(meshGen, regB, vertices);

      if (centA && centB) {
        measurements[name] = {
          value: ClinicalMeasurements._dist3D(centA, centB) * METRES_TO_MM,
          description: desc,
          unit: 'mm',
        };
      }
    }

    // -----------------------------------------------------------------------
    // 3. Unilateral distances
    // -----------------------------------------------------------------------
    for (const [regA, regB, name, desc] of UNILATERAL_DISTANCES) {
      const centA = ClinicalMeasurements._regionCentroid(meshGen, regA, vertices);
      const centB = ClinicalMeasurements._regionCentroid(meshGen, regB, vertices);

      if (centA && centB) {
        measurements[name] = {
          value: ClinicalMeasurements._dist3D(centA, centB) * METRES_TO_MM,
          description: desc,
          unit: 'mm',
        };
      }
    }

    // -----------------------------------------------------------------------
    // 4. Derived ratios (golden ratio, facial thirds, etc.)
    // -----------------------------------------------------------------------
    const ratios = ClinicalMeasurements._computeRatios(measurements);
    measurements._ratios = ratios;

    // -----------------------------------------------------------------------
    // 5. Facial profile angles
    // -----------------------------------------------------------------------
    const angles = ClinicalMeasurements._computeAngles(meshGen, vertices);
    if (angles) {
      measurements._angles = angles;
    }

    const count = Object.keys(measurements).filter(k => !k.startsWith('_')).length;
    console.log(`ClinicalMeasurements: Computed ${count} measurements, ` +
      `${Object.keys(ratios).length} ratios`);

    return measurements;
  }

  // =========================================================================
  // Derived ratios
  // =========================================================================

  static _computeRatios(measurements) {
    const ratios = {};
    const v = (name) => measurements[name]?.value;

    // Facial index (height / width × 100)
    if (v('faceHeight') && v('bizygomaticWidth')) {
      ratios.facialIndex = {
        value: (v('faceHeight') / v('bizygomaticWidth')) * 100,
        description: 'Facial index (height/width × 100). <85 = broad, 85-90 = average, >90 = long',
      };
    }

    // Jaw-to-cheekbone ratio
    if (v('bigonialWidth') && v('bizygomaticWidth')) {
      ratios.jawCheekRatio = {
        value: v('bigonialWidth') / v('bizygomaticWidth'),
        description: 'Jaw-to-cheekbone ratio. Ideal ≈ 0.75-0.80',
      };
    }

    // Nasal index (nose width / nose length × 100)
    if (v('noseWidth') && v('noseLength')) {
      ratios.nasalIndex = {
        value: (v('noseWidth') / v('noseLength')) * 100,
        description: 'Nasal index (width/length × 100). <70 leptorrhine, 70-85 mesorrhine, >85 platyrrhine',
      };
    }

    // Intercanthal-to-eye ratio
    if (v('intercanthalDist') && v('leftPalpebralFissure')) {
      ratios.intercanthalRatio = {
        value: v('intercanthalDist') / v('leftPalpebralFissure'),
        description: 'Intercanthal to palpebral fissure ratio. Ideal ≈ 1.0 (rule of fifths)',
      };
    }

    // Lip-to-face ratio
    if (v('mouthWidth') && v('bizygomaticWidth')) {
      ratios.lipFaceRatio = {
        value: v('mouthWidth') / v('bizygomaticWidth'),
        description: 'Mouth width to face width ratio. Ideal ≈ 0.38-0.42',
      };
    }

    // Nose-to-mouth ratio
    if (v('noseWidth') && v('mouthWidth')) {
      ratios.noseMouthRatio = {
        value: v('noseWidth') / v('mouthWidth'),
        description: 'Nose width to mouth width ratio. Ideal ≈ 0.60-0.70',
      };
    }

    // Upper-to-lower face ratio (facial thirds)
    if (v('upperFaceHeight') && v('lowerFaceHeight')) {
      ratios.upperLowerRatio = {
        value: v('upperFaceHeight') / v('lowerFaceHeight'),
        description: 'Upper to lower face ratio. Ideal ≈ 1.0 (equal thirds)',
      };
    }

    // Lip height ratio (lip height / lower face height)
    if (v('lipHeight') && v('lowerFaceHeight')) {
      ratios.lipLowerFaceRatio = {
        value: v('lipHeight') / v('lowerFaceHeight'),
        description: 'Lip height to lower face height ratio',
      };
    }

    return ratios;
  }

  // =========================================================================
  // Profile angles (lateral view analysis)
  // =========================================================================

  static _computeAngles(meshGen, vertices) {
    const angles = {};

    // Nasofrontal angle (forehead → nasion → nose tip, in YZ sagittal plane)
    const forehead = ClinicalMeasurements._regionCentroid(meshGen, 'forehead_center', vertices);
    const nasion = ClinicalMeasurements._regionCentroid(meshGen, 'nose_bridge_upper', vertices);
    const pronasale = ClinicalMeasurements._regionCentroid(meshGen, 'nose_tip', vertices);
    const subnas = ClinicalMeasurements._regionCentroid(meshGen, 'lip_upper_center', vertices);
    const menton = ClinicalMeasurements._regionCentroid(meshGen, 'chin_center', vertices);

    if (forehead && nasion && pronasale) {
      angles.nasofrontalAngle = {
        value: ClinicalMeasurements._angle3P(forehead, nasion, pronasale),
        description: 'Nasofrontal angle (forehead-nasion-tip). Normal: 115-130°',
        unit: 'degrees',
      };
    }

    // Nasolabial angle (nose tip → subnasale → upper lip)
    if (pronasale && subnas) {
      // Use the nose dorsum direction → lip direction
      const noseDorsum = ClinicalMeasurements._regionCentroid(meshGen, 'nose_bridge_lower', vertices);
      if (noseDorsum) {
        angles.nasolabialAngle = {
          value: ClinicalMeasurements._angle3P(noseDorsum, pronasale, subnas),
          description: 'Nasolabial angle. Normal: 90-110° (female higher, male lower)',
          unit: 'degrees',
        };
      }
    }

    // Mentolabial angle (lower lip → chin fold → chin point)
    const lowerLip = ClinicalMeasurements._regionCentroid(meshGen, 'lip_lower_center', vertices);
    if (lowerLip && menton) {
      const chinPt = ClinicalMeasurements._regionCentroid(meshGen, 'chin', vertices);
      if (chinPt) {
        angles.mentolabialAngle = {
          value: ClinicalMeasurements._angle3P(lowerLip, chinPt, menton),
          description: 'Mentolabial angle. Normal: 120-140°',
          unit: 'degrees',
        };
      }
    }

    return Object.keys(angles).length > 0 ? angles : null;
  }

  // =========================================================================
  // Geometry helpers
  // =========================================================================

  /**
   * Compute the centroid of a named region's vertices.
   * @param {Object} meshGen - FlameMeshGenerator
   * @param {string} regionName - One of the 52 clinical zones
   * @param {Float32Array} vertices - Current vertex positions (flat, x/y/z interleaved)
   * @returns {number[]|null} [x, y, z] centroid in metres, or null if region is empty
   */
  static _regionCentroid(meshGen, regionName, vertices) {
    let indices;
    try {
      indices = meshGen.getRegionVertices(regionName);
    } catch {
      return null;
    }
    if (!indices || indices.length === 0) return null;

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
   * Euclidean distance between two 3D points.
   */
  static _dist3D(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Angle at point B formed by vectors BA and BC, in degrees.
   * @param {number[]} a - Point A [x, y, z]
   * @param {number[]} b - Point B (vertex of angle) [x, y, z]
   * @param {number[]} c - Point C [x, y, z]
   * @returns {number} angle in degrees
   */
  static _angle3P(a, b, c) {
    const ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];

    const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    const magBA = Math.sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
    const magBC = Math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

    if (magBA < 1e-12 || magBC < 1e-12) return 0;

    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.acos(cosAngle) * (180 / Math.PI);
  }

  /**
   * Get a flat summary object for display (just names and values).
   * @param {Object} measurements - Output of compute()
   * @returns {Object} { name: value_in_mm, ... }
   */
  static getSummaryValues(measurements) {
    if (!measurements) return {};
    const summary = {};
    for (const [key, val] of Object.entries(measurements)) {
      if (key.startsWith('_')) continue; // skip _ratios, _angles
      if (val && typeof val.value === 'number') {
        summary[key] = Math.round(val.value * 10) / 10; // 1 decimal place mm
      }
    }
    return summary;
  }

  /**
   * Get a formatted report string for clinical display.
   * @param {Object} measurements - Output of compute()
   * @returns {string}
   */
  static formatReport(measurements) {
    if (!measurements) return 'No measurements available.';

    const lines = ['=== Clinical Facial Measurements ===\n'];

    // Main measurements
    for (const [key, val] of Object.entries(measurements)) {
      if (key.startsWith('_')) continue;
      if (val && typeof val.value === 'number') {
        lines.push(`${val.description}: ${val.value.toFixed(1)} ${val.unit || 'mm'}`);
      }
    }

    // Ratios
    if (measurements._ratios) {
      lines.push('\n--- Facial Ratios ---');
      for (const [, val] of Object.entries(measurements._ratios)) {
        lines.push(`${val.description}: ${val.value.toFixed(3)}`);
      }
    }

    // Angles
    if (measurements._angles) {
      lines.push('\n--- Profile Angles ---');
      for (const [, val] of Object.entries(measurements._angles)) {
        lines.push(`${val.description}: ${val.value.toFixed(1)}°`);
      }
    }

    return lines.join('\n');
  }
}
