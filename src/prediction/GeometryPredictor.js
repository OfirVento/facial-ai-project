/**
 * GeometryPredictor — Maps real treatment parameters to morph state.
 *
 * Provides calibrated treatment-to-morph mapping for common aesthetic procedures:
 *   - Injectable fillers (lips, cheeks, jaw, chin, temples, tear troughs)
 *   - Botox (forehead, crow's feet, brow lift)
 *   - Non-surgical rhinoplasty
 *
 * Each treatment maps (volume in ml, injection sites) → morphState object
 * compatible with FaceRenderer.applyMorphState().
 *
 * Calibration values are based on typical anatomical ranges. A 1ml lip filler
 * syringe produces approximately 0.08 inflate units in the morph system.
 */

// ---------------------------------------------------------------------------
// Treatment Definitions
// ---------------------------------------------------------------------------

/**
 * TREATMENT_CATALOG — comprehensive list of treatments with morph calibration.
 *
 * Each treatment entry:
 *   label:           Human-readable name
 *   category:        Treatment category for UI grouping
 *   primaryRegions:  Regions receiving the main deformation
 *   spreadRegions:   Adjacent regions receiving attenuated effect
 *   spreadFactor:    How much of the primary effect bleeds to spread regions (0-1)
 *   inflatePerML:    Morph system inflate units per 1ml of product
 *   maxVolume:       Maximum safe volume in ml (clinical cap)
 *   defaultVolume:   Default volume for preview
 *   description:     Clinical description for UI
 *   notes:           Additional clinical notes
 */
const TREATMENT_CATALOG = {
  // =========================================================================
  // LIP FILLERS
  // =========================================================================
  lip_filler_full: {
    label: 'Lip Filler (Full)',
    category: 'Lips',
    primaryRegions: [
      'lip_upper_center', 'lip_upper_left', 'lip_upper_right',
      'lip_lower_center', 'lip_lower_left', 'lip_lower_right',
    ],
    spreadRegions: ['lip_corner_left', 'lip_corner_right'],
    spreadFactor: 0.25,
    inflatePerML: 0.08,
    maxVolume: 2.0,
    defaultVolume: 1.0,
    description: 'Full lip augmentation (upper + lower)',
    notes: 'Typical: 0.5-1.5ml. Upper lip gets ~40%, lower lip ~60%.',
  },

  lip_filler_upper: {
    label: 'Upper Lip Filler',
    category: 'Lips',
    primaryRegions: ['lip_upper_center', 'lip_upper_left', 'lip_upper_right'],
    spreadRegions: ['lip_corner_left', 'lip_corner_right'],
    spreadFactor: 0.15,
    inflatePerML: 0.10,
    maxVolume: 1.0,
    defaultVolume: 0.5,
    description: 'Upper lip enhancement only',
    notes: 'Focus on cupid\'s bow definition and volume.',
  },

  lip_filler_lower: {
    label: 'Lower Lip Filler',
    category: 'Lips',
    primaryRegions: ['lip_lower_center', 'lip_lower_left', 'lip_lower_right'],
    spreadRegions: ['lip_corner_left', 'lip_corner_right'],
    spreadFactor: 0.15,
    inflatePerML: 0.10,
    maxVolume: 1.0,
    defaultVolume: 0.5,
    description: 'Lower lip enhancement only',
    notes: 'Adds fullness to lower vermilion.',
  },

  lip_border: {
    label: 'Lip Border Definition',
    category: 'Lips',
    primaryRegions: ['lip_upper_center'],
    spreadRegions: ['lip_upper_left', 'lip_upper_right'],
    spreadFactor: 0.4,
    inflatePerML: 0.05,
    maxVolume: 0.5,
    defaultVolume: 0.3,
    description: 'Lip border / vermilion border enhancement',
    notes: 'Subtle definition of the lip line.',
  },

  // =========================================================================
  // CHEEK & MIDFACE FILLERS
  // =========================================================================
  cheek_filler: {
    label: 'Cheek Filler',
    category: 'Cheeks',
    primaryRegions: ['cheekbone_left', 'cheekbone_right'],
    spreadRegions: ['cheek_left', 'cheek_right', 'under_eye_left', 'under_eye_right'],
    spreadFactor: 0.2,
    inflatePerML: 0.06,
    maxVolume: 4.0,
    defaultVolume: 2.0,
    description: 'Cheekbone augmentation / midface volumizing',
    notes: 'Typical: 1-2ml per side. Placed on malar eminence.',
  },

  cheek_hollow_filler: {
    label: 'Cheek Hollow Filler',
    category: 'Cheeks',
    primaryRegions: ['cheek_hollow_left', 'cheek_hollow_right'],
    spreadRegions: ['cheek_left', 'cheek_right', 'nasolabial_left', 'nasolabial_right'],
    spreadFactor: 0.25,
    inflatePerML: 0.07,
    maxVolume: 3.0,
    defaultVolume: 1.5,
    description: 'Fill sunken cheek hollows',
    notes: 'Restores buccal fat volume. Often combined with cheek filler.',
  },

  // =========================================================================
  // JAW & CHIN FILLERS
  // =========================================================================
  jaw_filler: {
    label: 'Jawline Filler',
    category: 'Lower Face',
    primaryRegions: ['jawline_left', 'jawline_right'],
    spreadRegions: ['jaw_left', 'jaw_right'],
    spreadFactor: 0.3,
    inflatePerML: 0.05,
    maxVolume: 4.0,
    defaultVolume: 2.0,
    description: 'Jawline definition and contouring',
    notes: 'Typical: 1-2ml per side. Creates sharper jaw angle.',
  },

  chin_filler: {
    label: 'Chin Filler',
    category: 'Lower Face',
    primaryRegions: ['chin_center', 'chin'],
    spreadRegions: ['chin_left', 'chin_right'],
    spreadFactor: 0.3,
    inflatePerML: 0.07,
    maxVolume: 2.0,
    defaultVolume: 1.0,
    description: 'Chin augmentation / projection',
    notes: 'Adds forward projection and length to chin.',
  },

  // =========================================================================
  // NOSE (NON-SURGICAL RHINOPLASTY)
  // =========================================================================
  nose_filler_bridge: {
    label: 'Nose Bridge Filler',
    category: 'Nose',
    primaryRegions: ['nose_bridge', 'nose_bridge_upper', 'nose_bridge_lower'],
    spreadRegions: ['nose_dorsum'],
    spreadFactor: 0.3,
    inflatePerML: 0.12,
    maxVolume: 1.0,
    defaultVolume: 0.3,
    description: 'Non-surgical rhinoplasty — bridge smoothing',
    notes: 'Smooths dorsal humps, straightens bridge. Very precise: 0.1-0.5ml.',
  },

  nose_filler_tip: {
    label: 'Nose Tip Refinement',
    category: 'Nose',
    primaryRegions: ['nose_tip', 'nose_tip_left', 'nose_tip_right'],
    spreadRegions: [],
    spreadFactor: 0,
    inflatePerML: 0.15,
    maxVolume: 0.5,
    defaultVolume: 0.2,
    description: 'Non-surgical nose tip refinement',
    notes: 'Subtle tip projection and shaping. Typically 0.1-0.3ml.',
  },

  // =========================================================================
  // UNDER-EYE / TEAR TROUGH
  // =========================================================================
  tear_trough_filler: {
    label: 'Tear Trough Filler',
    category: 'Eyes',
    primaryRegions: ['tear_trough_left', 'tear_trough_right'],
    spreadRegions: ['under_eye_left', 'under_eye_right'],
    spreadFactor: 0.3,
    inflatePerML: 0.12,
    maxVolume: 1.0,
    defaultVolume: 0.5,
    description: 'Under-eye hollow correction',
    notes: 'Addresses dark circles and hollowing. Typically 0.2-0.5ml per side.',
  },

  // =========================================================================
  // TEMPLE FILLER
  // =========================================================================
  temple_filler: {
    label: 'Temple Filler',
    category: 'Upper Face',
    primaryRegions: ['temple_left', 'temple_right'],
    spreadRegions: ['forehead_left', 'forehead_right'],
    spreadFactor: 0.15,
    inflatePerML: 0.05,
    maxVolume: 4.0,
    defaultVolume: 2.0,
    description: 'Temple volume restoration',
    notes: 'Corrects temporal hollowing. Typically 1-2ml per side.',
  },

  // =========================================================================
  // NASOLABIAL FOLD FILLER
  // =========================================================================
  nasolabial_filler: {
    label: 'Nasolabial Fold Filler',
    category: 'Cheeks',
    primaryRegions: ['nasolabial_left', 'nasolabial_right'],
    spreadRegions: ['cheek_left', 'cheek_right', 'lip_corner_left', 'lip_corner_right'],
    spreadFactor: 0.2,
    inflatePerML: 0.08,
    maxVolume: 2.0,
    defaultVolume: 1.0,
    description: 'Soften nasolabial folds (smile lines)',
    notes: 'Fills the fold between nose and mouth. Typically 0.5-1ml per side.',
  },

  // =========================================================================
  // BOTOX / NEUROTOXIN
  // =========================================================================
  botox_forehead: {
    label: 'Botox — Forehead',
    category: 'Upper Face',
    primaryRegions: ['forehead_center', 'forehead_left', 'forehead_right'],
    spreadRegions: [],
    spreadFactor: 0,
    inflatePerML: -0.02, // negative = slight deflation (muscle relaxation = smoothing)
    maxVolume: 30, // units, not ml
    defaultVolume: 15,
    description: 'Forehead line smoothing (neurotoxin)',
    notes: 'Typical: 10-20 units. Effect: reduced muscle movement, smoother skin.',
    unit: 'units',
  },

  botox_glabella: {
    label: 'Botox — Glabella (11s)',
    category: 'Upper Face',
    primaryRegions: ['brow_inner_left', 'brow_inner_right', 'forehead_center'],
    spreadRegions: ['brow_left', 'brow_right'],
    spreadFactor: 0.1,
    inflatePerML: -0.03,
    maxVolume: 25,
    defaultVolume: 15,
    description: 'Glabellar lines (frown lines / "11s")',
    notes: 'Typical: 15-25 units. Relaxes corrugator muscles.',
    unit: 'units',
  },

  botox_brow_lift: {
    label: 'Botox — Brow Lift',
    category: 'Eyes',
    primaryRegions: ['brow_left', 'brow_right'],
    spreadRegions: ['forehead_left', 'forehead_right'],
    spreadFactor: 0.15,
    inflatePerML: 0.015, // slight upward lift effect
    maxVolume: 10,
    defaultVolume: 5,
    description: 'Chemical brow lift via neurotoxin',
    notes: 'Typical: 2-5 units per side at tail of brow.',
    unit: 'units',
    translateY: 0.003, // slight upward translation per unit
  },

  // =========================================================================
  // COMBINATION PACKAGES
  // =========================================================================
  liquid_facelift: {
    label: 'Liquid Facelift',
    category: 'Combination',
    isCombo: true,
    components: [
      { treatment: 'cheek_filler', volume: 2.0 },
      { treatment: 'nasolabial_filler', volume: 1.0 },
      { treatment: 'jaw_filler', volume: 1.5 },
      { treatment: 'chin_filler', volume: 0.5 },
    ],
    description: 'Full face rejuvenation without surgery',
    notes: 'Combined approach for comprehensive facial balancing.',
  },
};

// ---------------------------------------------------------------------------
// GeometryPredictor
// ---------------------------------------------------------------------------

export class GeometryPredictor {
  /**
   * Get the full treatment catalog.
   * @returns {Object} treatment definitions keyed by ID
   */
  static get treatments() {
    return TREATMENT_CATALOG;
  }

  /**
   * Get treatments grouped by category for UI display.
   * @returns {Object} { category: [{ id, label, description, ... }] }
   */
  static getByCategory() {
    const cats = {};
    for (const [id, t] of Object.entries(TREATMENT_CATALOG)) {
      const cat = t.category || 'Other';
      if (!cats[cat]) cats[cat] = [];
      cats[cat].push({ id, ...t });
    }
    return cats;
  }

  /**
   * Predict the morph state for a given treatment and parameters.
   *
   * @param {string} treatmentId - Key from TREATMENT_CATALOG
   * @param {object} params
   * @param {number} params.volume - Amount of product (ml or units)
   * @param {number} [params.intensity=1.0] - Overall intensity multiplier (0-2)
   * @returns {{
   *   morphState: Object,   // { regionName: { inflate, translateX, translateY, translateZ } }
   *   treatment: Object,    // Treatment definition
   *   actualVolume: number, // Clamped volume
   *   unit: string,         // 'ml' or 'units'
   * }|null}
   */
  static predict(treatmentId, params = {}) {
    const treatment = TREATMENT_CATALOG[treatmentId];
    if (!treatment) {
      console.warn(`GeometryPredictor: Unknown treatment "${treatmentId}"`);
      return null;
    }

    // Handle combo treatments
    if (treatment.isCombo) {
      return GeometryPredictor._predictCombo(treatment, params);
    }

    const volume = Math.max(0, Math.min(params.volume ?? treatment.defaultVolume, treatment.maxVolume));
    const intensity = params.intensity ?? 1.0;
    const morphState = {};

    // Primary regions: full inflate effect
    const inflateValue = volume * treatment.inflatePerML * intensity;
    for (const region of treatment.primaryRegions) {
      morphState[region] = { inflate: inflateValue };

      // Add translation if defined
      if (treatment.translateY) {
        morphState[region].translateY = volume * treatment.translateY * intensity;
      }
    }

    // Spread regions: attenuated effect
    if (treatment.spreadRegions && treatment.spreadFactor > 0) {
      const spreadInflate = inflateValue * treatment.spreadFactor;
      for (const region of treatment.spreadRegions) {
        if (morphState[region]) {
          // Region is both primary and spread — add to existing
          morphState[region].inflate += spreadInflate;
        } else {
          morphState[region] = { inflate: spreadInflate };
        }
      }
    }

    return {
      morphState,
      treatment,
      actualVolume: volume,
      unit: treatment.unit || 'ml',
    };
  }

  /**
   * Predict combined treatment (multiple procedures applied together).
   */
  static _predictCombo(comboTreatment, params = {}) {
    const intensity = params.intensity ?? 1.0;
    const mergedMorphState = {};

    for (const component of comboTreatment.components) {
      const subResult = GeometryPredictor.predict(component.treatment, {
        volume: component.volume,
        intensity,
      });
      if (subResult) {
        // Merge morph states (additive)
        for (const [region, regionParams] of Object.entries(subResult.morphState)) {
          if (!mergedMorphState[region]) {
            mergedMorphState[region] = { ...regionParams };
          } else {
            mergedMorphState[region].inflate =
              (mergedMorphState[region].inflate || 0) + (regionParams.inflate || 0);
            if (regionParams.translateY) {
              mergedMorphState[region].translateY =
                (mergedMorphState[region].translateY || 0) + regionParams.translateY;
            }
          }
        }
      }
    }

    return {
      morphState: mergedMorphState,
      treatment: comboTreatment,
      actualVolume: comboTreatment.components.reduce((s, c) => s + c.volume, 0),
      unit: 'ml',
    };
  }

  /**
   * Get a list of treatment IDs and labels for UI dropdown.
   * @returns {Array<{id: string, label: string, category: string, defaultVolume: number, maxVolume: number, unit: string}>}
   */
  static getOptions() {
    return Object.entries(TREATMENT_CATALOG).map(([id, t]) => ({
      id,
      label: t.label,
      category: t.category,
      defaultVolume: t.defaultVolume || 1.0,
      maxVolume: t.maxVolume || 5.0,
      unit: t.unit || 'ml',
      description: t.description,
    }));
  }
}

export { TREATMENT_CATALOG };
