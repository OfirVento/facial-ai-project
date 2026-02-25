/**
 * MedicalAdvisorAgent.js
 * Medical knowledge advisor with a comprehensive procedure database.
 * Provides procedure recommendations, feasibility analysis, and region-based lookups.
 *
 * DISCLAIMER: This is an educational/visualization tool only and does NOT
 * constitute medical advice. Always consult a board-certified physician.
 */

import { CLINICAL_ZONES } from './NLUAgent.js';

// ---------------------------------------------------------------------------
// Procedure types enum
// ---------------------------------------------------------------------------
export const PROCEDURE_TYPE = {
  INJECTABLE: 'injectable',
  SURGICAL:   'surgical',
  TREATMENT:  'treatment',
};

// ---------------------------------------------------------------------------
// Invasiveness scale: 1 (minimal) to 10 (major surgery)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Complete procedure database  (30+ entries)
// ---------------------------------------------------------------------------
export const PROCEDURE_DATABASE = [
  // ─────────────────────────── FILLERS ───────────────────────────
  {
    id: 'lip_filler',
    name: 'Lip Filler (Hyaluronic Acid)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Hyaluronic acid dermal filler injected into the lips to add volume, improve symmetry, and define the lip border.',
    recoveryTime: '1-3 days swelling, 1-2 weeks full settle',
    permanence: 'Temporary (6-12 months)',
    costRange: { min: 500, max: 1500, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right', 'lip_corner_left', 'lip_corner_right', 'cupids_bow'],
    maxAchievableChange: 0.35,
    alternatives: ['lip_lift', 'lip_implant'],
  },
  {
    id: 'cheek_filler',
    name: 'Cheek Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Dermal filler (HA or calcium hydroxylapatite) injected into the mid-face to restore volume and contour the cheekbones.',
    recoveryTime: '1-2 days swelling, 2 weeks full settle',
    permanence: 'Temporary (12-18 months)',
    costRange: { min: 800, max: 2500, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['cheek_left', 'cheek_right', 'cheekbone_left', 'cheekbone_right'],
    maxAchievableChange: 0.30,
    alternatives: ['cheek_implant', 'fat_transfer_cheek'],
  },
  {
    id: 'jaw_filler',
    name: 'Jawline Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Dermal filler placed along the jawline and jaw angle to create a sharper, more defined jawline contour.',
    recoveryTime: '1-3 days swelling, 2 weeks full settle',
    permanence: 'Temporary (12-18 months)',
    costRange: { min: 1000, max: 3000, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
    maxAchievableChange: 0.30,
    alternatives: ['chin_implant', 'jaw_surgery'],
  },
  {
    id: 'chin_filler',
    name: 'Chin Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Filler injected into the chin to add projection, length, or width for improved facial balance.',
    recoveryTime: '1-2 days swelling, 2 weeks full settle',
    permanence: 'Temporary (12-18 months)',
    costRange: { min: 700, max: 2000, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['chin_center', 'chin_left', 'chin_right'],
    maxAchievableChange: 0.25,
    alternatives: ['chin_implant', 'genioplasty'],
  },
  {
    id: 'temple_filler',
    name: 'Temple Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Dermal filler injected into the temporal hollows to restore youthful convexity and fullness.',
    recoveryTime: '1-2 days, minimal downtime',
    permanence: 'Temporary (12-24 months)',
    costRange: { min: 600, max: 1800, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['temple_left', 'temple_right'],
    maxAchievableChange: 0.25,
    alternatives: ['fat_transfer_temple'],
  },
  {
    id: 'nasolabial_filler',
    name: 'Nasolabial Fold Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Filler injected along the nasolabial folds (smile lines) to soften their appearance.',
    recoveryTime: '1-2 days swelling',
    permanence: 'Temporary (6-12 months)',
    costRange: { min: 500, max: 1500, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['nasolabial_left', 'nasolabial_right'],
    maxAchievableChange: 0.30,
    alternatives: ['facelift', 'thread_lift'],
  },
  {
    id: 'marionette_filler',
    name: 'Marionette Line Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Filler placed at the oral commissures and marionette lines to reduce the appearance of downturned mouth corners.',
    recoveryTime: '1-2 days swelling',
    permanence: 'Temporary (6-12 months)',
    costRange: { min: 500, max: 1500, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['marionette_left', 'marionette_right', 'lip_corner_left', 'lip_corner_right'],
    maxAchievableChange: 0.25,
    alternatives: ['facelift', 'thread_lift'],
  },
  {
    id: 'tear_trough_filler',
    name: 'Tear Trough / Under-Eye Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Gentle filler placement under the eyes to reduce hollowness, dark circles, and tear trough depressions.',
    recoveryTime: '3-7 days (bruising common)',
    permanence: 'Temporary (9-12 months)',
    costRange: { min: 600, max: 1800, currency: 'USD' },
    invasiveness: 3,
    relatedRegions: ['lower_eyelid_left', 'lower_eyelid_right'],
    maxAchievableChange: 0.20,
    alternatives: ['lower_blepharoplasty', 'prp_under_eye'],
  },
  {
    id: 'nose_filler',
    name: 'Non-Surgical Rhinoplasty (Nose Filler)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Filler injected into the nose to smooth bumps, lift the tip, or improve symmetry without surgery.',
    recoveryTime: '1-2 days, minimal swelling',
    permanence: 'Temporary (6-12 months)',
    costRange: { min: 600, max: 1500, currency: 'USD' },
    invasiveness: 3,
    relatedRegions: ['nasal_bridge', 'nasal_dorsum', 'nasal_tip'],
    maxAchievableChange: 0.15,
    alternatives: ['rhinoplasty'],
  },
  {
    id: 'forehead_filler',
    name: 'Forehead Filler',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Filler injected into the forehead to create a smoother, rounder contour and reduce concavity.',
    recoveryTime: '1-3 days swelling',
    permanence: 'Temporary (12-18 months)',
    costRange: { min: 800, max: 2000, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['forehead_center', 'forehead_left', 'forehead_right'],
    maxAchievableChange: 0.20,
    alternatives: ['fat_transfer_forehead'],
  },

  // ─────────────────────────── BOTOX ───────────────────────────
  {
    id: 'botox_forehead',
    name: 'Botox - Forehead Lines',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected into the frontalis muscle to reduce horizontal forehead lines.',
    recoveryTime: 'None (may have small bumps for hours)',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 200, max: 600, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['forehead_center', 'forehead_left', 'forehead_right'],
    maxAchievableChange: 0.10,
    alternatives: ['laser_resurfacing', 'microneedling'],
  },
  {
    id: 'botox_glabella',
    name: 'Botox - Glabella (Frown Lines / 11 Lines)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected into the corrugator and procerus muscles to smooth vertical frown lines between the brows.',
    recoveryTime: 'None',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 200, max: 500, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['glabella'],
    maxAchievableChange: 0.10,
    alternatives: ['filler_glabella'],
  },
  {
    id: 'botox_crow_feet',
    name: "Botox - Crow's Feet",
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected around the orbicularis oculi to reduce lateral canthal lines (crow\'s feet).',
    recoveryTime: 'None',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 200, max: 500, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['crow_feet_left', 'crow_feet_right'],
    maxAchievableChange: 0.10,
    alternatives: ['laser_resurfacing', 'chemical_peel'],
  },
  {
    id: 'botox_masseter',
    name: 'Botox - Masseter (Jaw Slimming)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected into the masseter muscle to slim the lower face and reduce jaw width.',
    recoveryTime: 'None (results develop over 4-6 weeks)',
    permanence: 'Temporary (4-6 months)',
    costRange: { min: 400, max: 1200, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['jaw_angle_left', 'jaw_angle_right'],
    maxAchievableChange: 0.25,
    alternatives: ['jaw_surgery', 'buccal_fat_removal'],
  },
  {
    id: 'botox_brow_lift',
    name: 'Botox Brow Lift',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Strategic botulinum toxin placement to create a subtle brow elevation by relaxing the depressor muscles.',
    recoveryTime: 'None',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 200, max: 500, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['brow_left', 'brow_right', 'glabella'],
    maxAchievableChange: 0.10,
    alternatives: ['brow_lift', 'thread_lift_brow'],
  },
  {
    id: 'botox_dao',
    name: 'Botox - DAO (Mouth Corner Lift)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected into the depressor anguli oris (DAO) to lift down-turned mouth corners.',
    recoveryTime: 'None',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 150, max: 400, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['lip_corner_left', 'lip_corner_right', 'marionette_left', 'marionette_right'],
    maxAchievableChange: 0.10,
    alternatives: ['marionette_filler', 'lip_filler'],
  },
  {
    id: 'botox_platysmal',
    name: 'Botox - Platysmal Bands (Nefertiti Lift)',
    type: PROCEDURE_TYPE.INJECTABLE,
    description: 'Botulinum toxin injected into the platysma muscle bands to reduce neck banding and create a tighter jawline.',
    recoveryTime: 'None',
    permanence: 'Temporary (3-4 months)',
    costRange: { min: 300, max: 800, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['neck_left', 'neck_right', 'jawline_left', 'jawline_right'],
    maxAchievableChange: 0.15,
    alternatives: ['neck_lift', 'ultherapy'],
  },

  // ─────────────────────────── SURGERIES ───────────────────────────
  {
    id: 'rhinoplasty',
    name: 'Rhinoplasty (Nose Job)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical reshaping of the nose including bridge reduction, tip refinement, nostril narrowing, and overall size adjustment.',
    recoveryTime: '1-2 weeks initial, 6-12 months full healing with residual swelling',
    permanence: 'Permanent',
    costRange: { min: 5000, max: 15000, currency: 'USD' },
    invasiveness: 7,
    relatedRegions: ['nasal_bridge', 'nasal_dorsum', 'nasal_tip', 'nasal_ala_left', 'nasal_ala_right', 'nostril_left', 'nostril_right'],
    maxAchievableChange: 0.50,
    alternatives: ['nose_filler'],
  },
  {
    id: 'blepharoplasty',
    name: 'Blepharoplasty (Eyelid Surgery)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical removal of excess skin, muscle, and fat from the upper and/or lower eyelids to rejuvenate the eye area.',
    recoveryTime: '1-2 weeks swelling, 3-6 months full settle',
    permanence: 'Long-lasting (10+ years)',
    costRange: { min: 3000, max: 8000, currency: 'USD' },
    invasiveness: 5,
    relatedRegions: ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
    maxAchievableChange: 0.35,
    alternatives: ['tear_trough_filler', 'laser_resurfacing'],
  },
  {
    id: 'facelift',
    name: 'Facelift (Rhytidectomy)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical lifting and tightening of the face and neck tissues including SMAS layer repositioning for comprehensive rejuvenation.',
    recoveryTime: '2-4 weeks initial, 3-6 months full healing',
    permanence: 'Long-lasting (5-10 years)',
    costRange: { min: 8000, max: 25000, currency: 'USD' },
    invasiveness: 8,
    relatedRegions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'nasolabial_left', 'nasolabial_right', 'marionette_left', 'marionette_right', 'neck_left', 'neck_right'],
    maxAchievableChange: 0.45,
    alternatives: ['thread_lift', 'ultherapy', 'rf_microneedling'],
  },
  {
    id: 'brow_lift',
    name: 'Brow Lift (Forehead Lift)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical elevation of the brow position and smoothing of forehead wrinkles through endoscopic or open approach.',
    recoveryTime: '1-2 weeks swelling, 3-6 months full settle',
    permanence: 'Long-lasting (5-10 years)',
    costRange: { min: 4000, max: 10000, currency: 'USD' },
    invasiveness: 6,
    relatedRegions: ['brow_left', 'brow_right', 'forehead_center', 'forehead_left', 'forehead_right', 'glabella'],
    maxAchievableChange: 0.30,
    alternatives: ['botox_brow_lift', 'thread_lift_brow'],
  },
  {
    id: 'neck_lift',
    name: 'Neck Lift (Platysmaplasty)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical tightening of the neck muscles, removal of excess skin, and liposuction for a more defined neck and jawline.',
    recoveryTime: '2-3 weeks initial, 3-6 months full healing',
    permanence: 'Long-lasting (5-10 years)',
    costRange: { min: 5000, max: 15000, currency: 'USD' },
    invasiveness: 7,
    relatedRegions: ['neck_left', 'neck_right', 'under_chin', 'jawline_left', 'jawline_right'],
    maxAchievableChange: 0.40,
    alternatives: ['kybella', 'ultherapy', 'botox_platysmal'],
  },
  {
    id: 'chin_implant',
    name: 'Chin Implant (Mentoplasty)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical placement of a silicone implant to augment chin projection and shape for improved facial balance.',
    recoveryTime: '1-2 weeks swelling, 6-8 weeks full settle',
    permanence: 'Permanent (implant)',
    costRange: { min: 3000, max: 8000, currency: 'USD' },
    invasiveness: 5,
    relatedRegions: ['chin_center', 'chin_left', 'chin_right'],
    maxAchievableChange: 0.40,
    alternatives: ['chin_filler'],
  },
  {
    id: 'cheek_implant',
    name: 'Cheek Implant (Malar Augmentation)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical placement of implants over the cheekbones to enhance mid-face projection and volume.',
    recoveryTime: '1-2 weeks swelling, 6-8 weeks full settle',
    permanence: 'Permanent (implant)',
    costRange: { min: 4000, max: 10000, currency: 'USD' },
    invasiveness: 6,
    relatedRegions: ['cheekbone_left', 'cheekbone_right', 'cheek_left', 'cheek_right'],
    maxAchievableChange: 0.40,
    alternatives: ['cheek_filler'],
  },
  {
    id: 'otoplasty',
    name: 'Otoplasty (Ear Pinning)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical reshaping of the ear cartilage to pin back protruding ears or correct asymmetry.',
    recoveryTime: '1-2 weeks bandaged, 6 weeks full heal',
    permanence: 'Permanent',
    costRange: { min: 3000, max: 8000, currency: 'USD' },
    invasiveness: 4,
    relatedRegions: ['ear_left', 'ear_right'],
    maxAchievableChange: 0.50,
    alternatives: [],
  },
  {
    id: 'lip_lift',
    name: 'Lip Lift (Bullhorn)',
    type: PROCEDURE_TYPE.SURGICAL,
    description: 'Surgical shortening of the philtrum (space between nose and lip) to elevate the upper lip and show more vermilion.',
    recoveryTime: '1-2 weeks, 3-6 months scar maturation',
    permanence: 'Permanent',
    costRange: { min: 3000, max: 6000, currency: 'USD' },
    invasiveness: 4,
    relatedRegions: ['philtrum', 'upper_lip_center', 'upper_lip_left', 'upper_lip_right'],
    maxAchievableChange: 0.30,
    alternatives: ['lip_filler'],
  },

  // ─────────────────────────── TREATMENTS ───────────────────────────
  {
    id: 'microneedling',
    name: 'Microneedling',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Controlled micro-injury to the skin using fine needles to stimulate collagen production and improve skin texture.',
    recoveryTime: '1-3 days redness',
    permanence: 'Semi-permanent (requires series)',
    costRange: { min: 200, max: 700, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['forehead_center', 'forehead_left', 'forehead_right', 'cheek_left', 'cheek_right', 'nasolabial_left', 'nasolabial_right', 'perioral'],
    maxAchievableChange: 0.10,
    alternatives: ['rf_microneedling', 'laser_resurfacing', 'chemical_peel'],
  },
  {
    id: 'laser_resurfacing',
    name: 'Laser Resurfacing (CO2 / Erbium)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Ablative or non-ablative laser treatment to remove damaged skin layers, stimulate collagen, and improve texture and tone.',
    recoveryTime: '5-14 days (ablative), 1-3 days (non-ablative)',
    permanence: 'Long-lasting with maintenance',
    costRange: { min: 1000, max: 5000, currency: 'USD' },
    invasiveness: 4,
    relatedRegions: ['forehead_center', 'forehead_left', 'forehead_right', 'cheek_left', 'cheek_right', 'perioral', 'crow_feet_left', 'crow_feet_right'],
    maxAchievableChange: 0.15,
    alternatives: ['chemical_peel', 'microneedling', 'rf_microneedling'],
  },
  {
    id: 'chemical_peel',
    name: 'Chemical Peel',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Chemical solution applied to the skin to exfoliate and regenerate new skin, improving texture, tone, and fine lines.',
    recoveryTime: '3-7 days peeling (medium), 1-2 days (superficial)',
    permanence: 'Temporary (requires series)',
    costRange: { min: 150, max: 800, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['forehead_center', 'cheek_left', 'cheek_right', 'perioral', 'nasolabial_left', 'nasolabial_right'],
    maxAchievableChange: 0.08,
    alternatives: ['microneedling', 'laser_resurfacing'],
  },
  {
    id: 'ultherapy',
    name: 'Ultherapy (Micro-Focused Ultrasound)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Non-invasive ultrasound energy delivered to deep tissue layers to lift and tighten the skin on the face and neck.',
    recoveryTime: 'None to minimal (may have mild swelling)',
    permanence: 'Results develop over 2-3 months, last 1-2 years',
    costRange: { min: 2000, max: 5000, currency: 'USD' },
    invasiveness: 3,
    relatedRegions: ['brow_left', 'brow_right', 'cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'neck_left', 'neck_right'],
    maxAchievableChange: 0.15,
    alternatives: ['rf_microneedling', 'thread_lift', 'facelift'],
  },
  {
    id: 'rf_microneedling',
    name: 'RF Microneedling (Morpheus8 / Vivace)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Combination of microneedling with radiofrequency energy to tighten skin, reduce fat, and stimulate deep collagen.',
    recoveryTime: '2-5 days redness and swelling',
    permanence: 'Semi-permanent (requires series, lasts 1-2 years)',
    costRange: { min: 500, max: 2000, currency: 'USD' },
    invasiveness: 3,
    relatedRegions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'nasolabial_left', 'nasolabial_right', 'neck_left', 'neck_right', 'under_chin'],
    maxAchievableChange: 0.15,
    alternatives: ['ultherapy', 'microneedling', 'laser_resurfacing'],
  },
  {
    id: 'ipl',
    name: 'IPL (Intense Pulsed Light)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Broad-spectrum light treatment targeting pigmentation, redness, and vascular lesions for overall skin tone improvement.',
    recoveryTime: '1-3 days mild redness',
    permanence: 'Temporary (requires maintenance)',
    costRange: { min: 300, max: 800, currency: 'USD' },
    invasiveness: 1,
    relatedRegions: ['forehead_center', 'forehead_left', 'forehead_right', 'cheek_left', 'cheek_right'],
    maxAchievableChange: 0.05,
    alternatives: ['laser_resurfacing', 'chemical_peel'],
  },
  {
    id: 'thread_lift',
    name: 'Thread Lift (PDO Threads)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Dissolvable threads inserted under the skin to provide a lifting effect for sagging areas of the mid-face, jawline, or neck.',
    recoveryTime: '3-7 days swelling, 2-4 weeks full settle',
    permanence: 'Semi-permanent (1-2 years)',
    costRange: { min: 1500, max: 5000, currency: 'USD' },
    invasiveness: 4,
    relatedRegions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'nasolabial_left', 'nasolabial_right', 'brow_left', 'brow_right', 'neck_left', 'neck_right'],
    maxAchievableChange: 0.20,
    alternatives: ['facelift', 'ultherapy', 'rf_microneedling'],
  },
  {
    id: 'prp',
    name: 'PRP (Platelet-Rich Plasma) Therapy',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Concentrated platelets from the patient\'s own blood injected into the skin to promote healing, collagen production, and rejuvenation.',
    recoveryTime: '1-2 days mild redness',
    permanence: 'Temporary (requires series, 3-6 month intervals)',
    costRange: { min: 500, max: 1500, currency: 'USD' },
    invasiveness: 2,
    relatedRegions: ['forehead_center', 'cheek_left', 'cheek_right', 'lower_eyelid_left', 'lower_eyelid_right', 'perioral'],
    maxAchievableChange: 0.08,
    alternatives: ['microneedling', 'laser_resurfacing'],
  },
  {
    id: 'kybella',
    name: 'Kybella (Deoxycholic Acid)',
    type: PROCEDURE_TYPE.TREATMENT,
    description: 'Injectable deoxycholic acid that permanently destroys fat cells under the chin to reduce submental fullness (double chin).',
    recoveryTime: '3-7 days significant swelling per session, 2-4 sessions typical',
    permanence: 'Permanent (fat cell destruction)',
    costRange: { min: 1200, max: 3000, currency: 'USD' },
    invasiveness: 3,
    relatedRegions: ['under_chin', 'jawline_left', 'jawline_right'],
    maxAchievableChange: 0.30,
    alternatives: ['neck_lift', 'coolsculpting_chin'],
  },
];

// ---------------------------------------------------------------------------
// Medical disclaimer text
// ---------------------------------------------------------------------------
const MEDICAL_DISCLAIMER = `DISCLAIMER: This analysis is generated by an AI visualization tool for educational and consultation planning purposes only. It does NOT constitute medical advice, diagnosis, or treatment recommendations. Results shown are approximate visualizations and may not reflect actual surgical or procedural outcomes. Always consult with a board-certified plastic surgeon, dermatologist, or qualified medical professional before undergoing any procedure. Individual results vary based on anatomy, skin quality, healing response, and many other factors.`;

// ---------------------------------------------------------------------------
// MedicalAdvisorAgent class
// ---------------------------------------------------------------------------
export class MedicalAdvisorAgent {
  constructor() {
    this.procedures = PROCEDURE_DATABASE;
    this.disclaimer = MEDICAL_DISCLAIMER;

    // Build region-to-procedure index for fast lookup
    this._regionIndex = new Map();
    for (const proc of this.procedures) {
      for (const region of proc.relatedRegions) {
        if (!this._regionIndex.has(region)) {
          this._regionIndex.set(region, []);
        }
        this._regionIndex.get(region).push(proc);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Analyze current morph changes and recommend procedures.
   * @param {Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>} currentChanges
   *   The morph state keyed by region name.
   * @param {string[]} regions  Array of affected region names.
   * @returns {{ recommendations: Array<{procedure: object, feasibility: number, rationale: string}>,
   *             warnings: string[],
   *             disclaimer: string }}
   */
  analyze(currentChanges, regions) {
    const recommendations = [];
    const warnings = [];
    const seenProcedures = new Set();

    // Compute magnitude for each affected region
    const regionMagnitudes = {};
    for (const region of regions) {
      const change = currentChanges[region];
      if (!change) continue;

      const d = change.displacement || { x: 0, y: 0, z: 0 };
      const magnitude = Math.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2) + Math.abs(change.inflate || 0);
      regionMagnitudes[region] = magnitude;
    }

    // Find applicable procedures
    for (const region of regions) {
      const magnitude = regionMagnitudes[region] || 0;
      const applicableProcs = this._regionIndex.get(region) || [];

      for (const proc of applicableProcs) {
        if (seenProcedures.has(proc.id)) continue;
        seenProcedures.add(proc.id);

        // Calculate feasibility score (0-1)
        const feasibility = this._calculateFeasibility(proc, regionMagnitudes, regions);

        // Generate rationale
        const rationale = this._generateRationale(proc, magnitude, regions);

        recommendations.push({
          procedure: {
            id: proc.id,
            name: proc.name,
            type: proc.type,
            description: proc.description,
            recoveryTime: proc.recoveryTime,
            permanence: proc.permanence,
            costRange: proc.costRange,
            invasiveness: proc.invasiveness,
            maxAchievableChange: proc.maxAchievableChange,
            alternatives: proc.alternatives,
          },
          feasibility,
          rationale,
        });

        // Warn about high-magnitude changes that exceed procedure limits
        if (magnitude > proc.maxAchievableChange) {
          warnings.push(
            `The requested change for ${region} (${(magnitude * 100).toFixed(0)}%) may exceed what ${proc.name} can achieve alone (max ~${(proc.maxAchievableChange * 100).toFixed(0)}%). A combination approach or surgical option may be needed.`
          );
        }
      }
    }

    // Sort by feasibility descending
    recommendations.sort((a, b) => b.feasibility - a.feasibility);

    // Group by type for clarity
    const grouped = {
      injectable: recommendations.filter(r => r.procedure.type === PROCEDURE_TYPE.INJECTABLE),
      surgical: recommendations.filter(r => r.procedure.type === PROCEDURE_TYPE.SURGICAL),
      treatment: recommendations.filter(r => r.procedure.type === PROCEDURE_TYPE.TREATMENT),
    };

    return {
      recommendations,
      grouped,
      warnings,
      disclaimer: this.disclaimer,
    };
  }

  /**
   * Get all applicable procedures for a specific region.
   * @param {string} regionName  Clinical zone name.
   * @returns {{ procedures: Array<object>, disclaimer: string }}
   */
  getProcedureByRegion(regionName) {
    const procs = this._regionIndex.get(regionName) || [];

    return {
      procedures: procs.map(p => ({
        id: p.id,
        name: p.name,
        type: p.type,
        description: p.description,
        recoveryTime: p.recoveryTime,
        permanence: p.permanence,
        costRange: p.costRange,
        invasiveness: p.invasiveness,
        maxAchievableChange: p.maxAchievableChange,
        alternatives: p.alternatives,
      })),
      region: regionName,
      disclaimer: this.disclaimer,
    };
  }

  /**
   * Get a single procedure by ID.
   * @param {string} procedureId
   * @returns {object|null}
   */
  getProcedureById(procedureId) {
    return this.procedures.find(p => p.id === procedureId) || null;
  }

  /**
   * Get all procedures.
   * @returns {object[]}
   */
  getAllProcedures() {
    return this.procedures.map(p => ({
      id: p.id,
      name: p.name,
      type: p.type,
      invasiveness: p.invasiveness,
      costRange: p.costRange,
      permanence: p.permanence,
    }));
  }

  /**
   * Search procedures by keyword.
   * @param {string} query
   * @returns {object[]}
   */
  searchProcedures(query) {
    const q = query.toLowerCase();
    return this.procedures.filter(p =>
      p.name.toLowerCase().includes(q) ||
      p.description.toLowerCase().includes(q) ||
      p.id.includes(q)
    );
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  /** @private */
  _calculateFeasibility(procedure, regionMagnitudes, affectedRegions) {
    // How many of the procedure's related regions overlap with affected regions?
    const overlapping = procedure.relatedRegions.filter(r => affectedRegions.includes(r));
    const regionCoverage = overlapping.length / Math.max(procedure.relatedRegions.length, 1);

    // How much of the requested change can this procedure achieve?
    let magnitudeMatch = 1.0;
    for (const region of overlapping) {
      const requested = regionMagnitudes[region] || 0;
      if (requested > 0) {
        const achievable = procedure.maxAchievableChange;
        magnitudeMatch = Math.min(magnitudeMatch, Math.min(achievable / requested, 1.0));
      }
    }

    // Prefer less invasive options (lower invasiveness = higher score)
    const invasivenessBonus = (10 - procedure.invasiveness) / 10;

    // Composite score
    const score = (regionCoverage * 0.4) + (magnitudeMatch * 0.4) + (invasivenessBonus * 0.2);
    return Math.round(score * 100) / 100;
  }

  /** @private */
  _generateRationale(procedure, magnitude, regions) {
    const overlapping = procedure.relatedRegions.filter(r => regions.includes(r));
    const pct = (magnitude * 100).toFixed(0);

    if (procedure.type === PROCEDURE_TYPE.INJECTABLE) {
      return `${procedure.name} is a minimally invasive option (invasiveness: ${procedure.invasiveness}/10) ` +
             `that addresses ${overlapping.length} of the targeted region(s). ` +
             `It can achieve up to ~${(procedure.maxAchievableChange * 100).toFixed(0)}% change, ` +
             `with ${procedure.recoveryTime} recovery. ` +
             `Cost range: $${procedure.costRange.min}-$${procedure.costRange.max}.`;
    }

    if (procedure.type === PROCEDURE_TYPE.SURGICAL) {
      return `${procedure.name} is a surgical option (invasiveness: ${procedure.invasiveness}/10) ` +
             `that can achieve more dramatic and permanent results up to ~${(procedure.maxAchievableChange * 100).toFixed(0)}% change ` +
             `across ${overlapping.length} region(s). ` +
             `Recovery: ${procedure.recoveryTime}. ` +
             `Cost range: $${procedure.costRange.min}-$${procedure.costRange.max}.`;
    }

    return `${procedure.name} is a non-surgical treatment (invasiveness: ${procedure.invasiveness}/10) ` +
           `offering gradual improvement up to ~${(procedure.maxAchievableChange * 100).toFixed(0)}% change. ` +
           `Recovery: ${procedure.recoveryTime}. ` +
           `Cost range: $${procedure.costRange.min}-$${procedure.costRange.max}.`;
  }
}

export default MedicalAdvisorAgent;
