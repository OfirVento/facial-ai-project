/**
 * NLUAgent.js
 * Enhanced Natural Language Understanding agent that maps conversational
 * commands to FLAME mesh morph parameters across 52 clinical zones.
 *
 * Output format:
 *   { regions: Record<string, {displacement: {x,y,z}, inflate: number}>,
 *     explanation: string,
 *     confidence: number }
 */

// ---------------------------------------------------------------------------
// 52 clinical zone names aligned with the FLAME topology
// ---------------------------------------------------------------------------
export const CLINICAL_ZONES = [
  'forehead_center', 'forehead_left', 'forehead_right',
  'glabella',
  'brow_left', 'brow_right',
  'temple_left', 'temple_right',
  'upper_eyelid_left', 'upper_eyelid_right',
  'lower_eyelid_left', 'lower_eyelid_right',
  'crow_feet_left', 'crow_feet_right',
  'nasal_bridge', 'nasal_dorsum', 'nasal_tip',
  'nostril_left', 'nostril_right',
  'nasal_ala_left', 'nasal_ala_right',
  'cheek_left', 'cheek_right',
  'cheekbone_left', 'cheekbone_right',
  'nasolabial_left', 'nasolabial_right',
  'upper_lip_center', 'upper_lip_left', 'upper_lip_right',
  'lower_lip_center', 'lower_lip_left', 'lower_lip_right',
  'lip_corner_left', 'lip_corner_right',
  'cupids_bow',
  'philtrum',
  'marionette_left', 'marionette_right',
  'chin_center', 'chin_left', 'chin_right',
  'jawline_left', 'jawline_right',
  'jaw_angle_left', 'jaw_angle_right',
  'under_chin',
  'neck_left', 'neck_right',
  'ear_left', 'ear_right',
  'perioral'
];

// ---------------------------------------------------------------------------
// Intensity vocabulary  →  numeric multiplier  (0-1 range)
// ---------------------------------------------------------------------------
const INTENSITY_MAP = {
  // very small
  tiny:        0.05,
  barely:      0.05,
  'just a touch': 0.05,
  'a hair':    0.05,
  'a smidge':  0.05,
  imperceptible: 0.05,

  // small
  subtle:      0.10,
  slightly:    0.10,
  'a tad':     0.10,
  mildly:      0.10,
  faintly:     0.10,
  gently:      0.10,
  softly:      0.10,
  delicately:  0.10,

  // low-medium
  'a little':  0.15,
  'a bit':     0.15,
  'a touch':   0.15,
  somewhat:    0.15,
  'a little bit': 0.15,
  slightly:    0.15,

  // medium
  moderate:    0.25,
  moderately:  0.25,
  medium:      0.25,
  noticeably:  0.25,
  fairly:      0.25,
  reasonably:  0.25,

  // medium-high
  noticeable:  0.35,
  considerable: 0.35,
  notably:     0.35,
  meaningfully: 0.35,
  clearly:     0.35,
  visibly:     0.35,
  definitely:  0.35,

  // high
  'a lot':     0.45,
  much:        0.45,
  substantially: 0.45,
  greatly:     0.45,
  considerably: 0.45,
  really:      0.45,
  quite:       0.45,
  pretty:      0.45,

  // very high
  significant: 0.55,
  significantly: 0.55,
  strongly:    0.55,
  major:       0.55,
  very:        0.55,
  intensely:   0.55,

  // dramatic
  dramatic:    0.65,
  dramatically: 0.65,
  drastic:     0.65,
  drastically: 0.65,
  'a ton':     0.65,
  huge:        0.65,
  massive:     0.65,
  seriously:   0.65,

  // extreme
  extreme:     0.80,
  extremely:   0.80,
  'super':     0.80,
  ultra:       0.80,
  radical:     0.80,
  radically:   0.80,
  immensely:   0.80,
  enormously:  0.80,

  // maximum
  maximum:     1.00,
  max:         1.00,
  'all the way': 1.00,
  'as much as possible': 1.00,
  '100%':      1.00,
  total:       1.00,
  completely:  1.00,
  fully:       1.00,
  absolute:    1.00,
};

// ---------------------------------------------------------------------------
// Direction vocabulary  →  axis sign modifiers
// ---------------------------------------------------------------------------
const DIRECTION_MAP = {
  // x-axis (lateral)
  thinner:   { axis: 'x', sign: -1 },
  narrower:  { axis: 'x', sign: -1 },
  slimmer:   { axis: 'x', sign: -1 },
  wider:     { axis: 'x', sign:  1 },
  broader:   { axis: 'x', sign:  1 },
  'spread out': { axis: 'x', sign: 1 },
  flatter:   { axis: 'x', sign: -1 },

  // y-axis (vertical)
  higher:    { axis: 'y', sign:  1 },
  lifted:    { axis: 'y', sign:  1 },
  raised:    { axis: 'y', sign:  1 },
  'lift up': { axis: 'y', sign:  1 },
  elevated:  { axis: 'y', sign:  1 },
  lower:     { axis: 'y', sign: -1 },
  lowered:   { axis: 'y', sign: -1 },
  dropped:   { axis: 'y', sign: -1 },
  'bring down': { axis: 'y', sign: -1 },
  longer:    { axis: 'y', sign:  1 },
  shorter:   { axis: 'y', sign: -1 },

  // z-axis (depth / projection)
  forward:   { axis: 'z', sign:  1 },
  'more projected': { axis: 'z', sign: 1 },
  'project more': { axis: 'z', sign: 1 },
  projected: { axis: 'z', sign:  1 },
  'pushed out': { axis: 'z', sign: 1 },
  'stick out': { axis: 'z', sign: 1 },
  backward:  { axis: 'z', sign: -1 },
  recessed:  { axis: 'z', sign: -1 },
  'pushed back': { axis: 'z', sign: -1 },
  flattened: { axis: 'z', sign: -1 },

  // size (inflate)
  bigger:    { axis: 'inflate', sign:  1 },
  larger:    { axis: 'inflate', sign:  1 },
  fuller:    { axis: 'inflate', sign:  1 },
  plumper:   { axis: 'inflate', sign:  1 },
  'more volume': { axis: 'inflate', sign: 1 },
  volumize:  { axis: 'inflate', sign:  1 },
  'puffed up': { axis: 'inflate', sign: 1 },
  augmented: { axis: 'inflate', sign:  1 },
  enhanced:  { axis: 'inflate', sign:  1 },
  smaller:   { axis: 'inflate', sign: -1 },
  reduced:   { axis: 'inflate', sign: -1 },
  'less volume': { axis: 'inflate', sign: -1 },
  diminished: { axis: 'inflate', sign: -1 },
  deflated:  { axis: 'inflate', sign: -1 },
  thinner_volume: { axis: 'inflate', sign: -1 },
};

// ---------------------------------------------------------------------------
// Relative command vocabulary
// ---------------------------------------------------------------------------
const RELATIVE_COMMANDS = {
  'more of that':  { type: 'scale', factor: 1.5 },
  'more':          { type: 'scale', factor: 1.3 },
  'a bit more':    { type: 'scale', factor: 1.15 },
  'a little more': { type: 'scale', factor: 1.15 },
  'slightly more': { type: 'scale', factor: 1.10 },
  'way more':      { type: 'scale', factor: 2.0 },
  'much more':     { type: 'scale', factor: 1.8 },
  'double that':   { type: 'scale', factor: 2.0 },
  'triple that':   { type: 'scale', factor: 3.0 },
  'less':          { type: 'scale', factor: 0.7 },
  'a bit less':    { type: 'scale', factor: 0.85 },
  'a little less': { type: 'scale', factor: 0.85 },
  'way less':      { type: 'scale', factor: 0.4 },
  'much less':     { type: 'scale', factor: 0.5 },
  'half of that':  { type: 'scale', factor: 0.5 },
  'half as much':  { type: 'scale', factor: 0.5 },
  'undo':          { type: 'undo' },
  'undo that':     { type: 'undo' },
  'revert':        { type: 'undo' },
  'take that back': { type: 'undo' },
  'go back':       { type: 'undo' },
  'never mind':    { type: 'undo' },
  'reset':         { type: 'reset' },
  'start over':    { type: 'reset' },
  'clear all':     { type: 'reset' },
  'remove everything': { type: 'reset' },
};

// ---------------------------------------------------------------------------
// 200+ phrase  →  region + direction + default intensity mappings
// ---------------------------------------------------------------------------
const PHRASE_MAP = [
  // ─── LIPS ───
  { patterns: ['bigger lips', 'fuller lips', 'plump lips', 'plumper lips', 'augment lips', 'enhance lips', 'pump up lips', 'volumize lips', 'lip augmentation'],
    regions: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['thinner lips', 'smaller lips', 'reduce lips', 'less lip volume', 'deflate lips', 'slim lips'],
    regions: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['bigger upper lip', 'fuller upper lip', 'plump upper lip', 'enhance upper lip', 'upper lip filler', 'volumize upper lip'],
    regions: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'cupids_bow'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['bigger lower lip', 'fuller lower lip', 'plump lower lip', 'enhance lower lip', 'lower lip filler'],
    regions: ['lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['wider lips', 'broader lips', 'stretch lips', 'widen lips'],
    regions: ['lip_corner_left', 'lip_corner_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['narrower lips', 'thinner lips width', 'slim lip width'],
    regions: ['lip_corner_left', 'lip_corner_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['lip lift', 'lift upper lip', 'shorten upper lip', 'raise lip', 'raise upper lip'],
    regions: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'philtrum'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },

  { patterns: ['cupids bow', 'enhance cupids bow', 'define cupids bow', 'sharper cupids bow', "cupid's bow"],
    regions: ['cupids_bow'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['lip corners up', 'lift lip corners', 'raise lip corners', 'turn up lip corners', 'smile corners'],
    regions: ['lip_corner_left', 'lip_corner_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },

  { patterns: ['lip corners down', 'lower lip corners', 'drop lip corners', 'turn down lip corners', 'sad mouth'],
    regions: ['lip_corner_left', 'lip_corner_right'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.15 },

  { patterns: ['project lips', 'lips forward', 'push lips out', 'pout lips', 'lip projection'],
    regions: ['upper_lip_center', 'lower_lip_center'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.20 },

  // ─── NOSE ───
  { patterns: ['thinner nose', 'slim nose', 'narrow nose', 'narrower nose', 'reduce nose width', 'skinny nose', 'slim down nose'],
    regions: ['nasal_ala_left', 'nasal_ala_right', 'nostril_left', 'nostril_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['wider nose', 'broaden nose', 'widen nose', 'broader nose'],
    regions: ['nasal_ala_left', 'nasal_ala_right', 'nostril_left', 'nostril_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['smaller nose', 'reduce nose', 'shrink nose', 'nose reduction', 'less nose'],
    regions: ['nasal_tip', 'nasal_dorsum', 'nasal_bridge', 'nasal_ala_left', 'nasal_ala_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['bigger nose', 'enlarge nose', 'larger nose'],
    regions: ['nasal_tip', 'nasal_dorsum', 'nasal_bridge', 'nasal_ala_left', 'nasal_ala_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['nose up', 'upturned nose', 'tip up', 'rotate nose up', 'upturn nose tip', 'lift nose tip', 'raise nose tip'],
    regions: ['nasal_tip'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['nose down', 'lower nose tip', 'drop nose tip', 'downturned nose', 'droopy nose'],
    regions: ['nasal_tip'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['project nose', 'nose forward', 'nose projection', 'more nose projection', 'nose bridge higher'],
    regions: ['nasal_bridge', 'nasal_dorsum'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['flatten nose bridge', 'reduce nose bridge', 'lower nose bridge', 'nose bridge down'],
    regions: ['nasal_bridge', 'nasal_dorsum'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['refine nose tip', 'smaller nose tip', 'pinch nose tip', 'define nose tip', 'sharper nose tip', 'pointy nose'],
    regions: ['nasal_tip'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['bulbous nose tip', 'rounder nose tip', 'bigger nose tip', 'wider nose tip'],
    regions: ['nasal_tip'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['smaller nostrils', 'reduce nostrils', 'narrower nostrils', 'pinch nostrils'],
    regions: ['nostril_left', 'nostril_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['flare nostrils', 'wider nostrils', 'bigger nostrils', 'expand nostrils'],
    regions: ['nostril_left', 'nostril_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['rhinoplasty', 'nose job', 'nose surgery', 'nose reshaping'],
    regions: ['nasal_tip', 'nasal_dorsum', 'nasal_bridge', 'nasal_ala_left', 'nasal_ala_right', 'nostril_left', 'nostril_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['remove nose bump', 'smooth nose bridge', 'straight nose', 'straighten nose', 'remove dorsal hump', 'shave nose bump'],
    regions: ['nasal_dorsum'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.25 },

  // ─── CHEEKS ───
  { patterns: ['fuller cheeks', 'bigger cheeks', 'plump cheeks', 'cheek filler', 'volumize cheeks', 'cheek augmentation', 'cheek volume'],
    regions: ['cheek_left', 'cheek_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['slimmer cheeks', 'reduce cheeks', 'thinner cheeks', 'slim cheeks', 'less cheek volume', 'cheek reduction'],
    regions: ['cheek_left', 'cheek_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['higher cheekbones', 'lift cheekbones', 'raise cheekbones', 'cheekbone lift', 'more cheekbone definition', 'cheekbone enhancement'],
    regions: ['cheekbone_left', 'cheekbone_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['wider cheekbones', 'broader cheekbones', 'widen cheekbones', 'cheekbone width'],
    regions: ['cheekbone_left', 'cheekbone_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['project cheekbones', 'cheekbones forward', 'more cheekbone projection', 'prominent cheekbones', 'cheekbone projection'],
    regions: ['cheekbone_left', 'cheekbone_right'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['hollow cheeks', 'sculpt cheeks', 'cheek sculpting', 'buccal fat removal', 'contour cheeks', 'cheek contour'],
    regions: ['cheek_left', 'cheek_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.30 },

  // ─── JAW / CHIN ───
  { patterns: ['wider jaw', 'broader jaw', 'square jaw', 'widen jaw', 'masculine jaw', 'jaw augmentation', 'stronger jaw'],
    regions: ['jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['slimmer jaw', 'narrower jaw', 'slim jaw', 'reduce jaw', 'v-line jaw', 'jaw slimming', 'less jaw width', 'softer jaw'],
    regions: ['jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['define jawline', 'sharper jawline', 'jawline definition', 'sculpt jawline', 'jawline contouring', 'chiseled jaw'],
    regions: ['jawline_left', 'jawline_right'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['jaw angle filler', 'jaw angle augmentation', 'define jaw angle', 'sharper jaw angle', 'square jaw angle'],
    regions: ['jaw_angle_left', 'jaw_angle_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['bigger chin', 'chin augmentation', 'chin implant', 'project chin', 'chin forward', 'stronger chin', 'more chin', 'enhance chin'],
    regions: ['chin_center', 'chin_left', 'chin_right'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['smaller chin', 'reduce chin', 'recede chin', 'chin reduction', 'less chin', 'chin back'],
    regions: ['chin_center', 'chin_left', 'chin_right'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['wider chin', 'broaden chin', 'chin width'],
    regions: ['chin_left', 'chin_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['narrower chin', 'slim chin', 'pointy chin', 'v-shaped chin'],
    regions: ['chin_left', 'chin_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['longer chin', 'lengthen chin', 'elongate chin', 'extend chin'],
    regions: ['chin_center'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['shorter chin', 'shorten chin', 'reduce chin height'],
    regions: ['chin_center'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['double chin', 'reduce double chin', 'remove double chin', 'slim under chin', 'tighten under chin', 'submental fat'],
    regions: ['under_chin'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.35 },

  // ─── FOREHEAD / BROW ───
  { patterns: ['lift brows', 'raise brows', 'brow lift', 'higher brows', 'raise eyebrows', 'lift eyebrows', 'brow elevation'],
    regions: ['brow_left', 'brow_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['lower brows', 'drop brows', 'lower eyebrows', 'heavier brows'],
    regions: ['brow_left', 'brow_right'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['lift left brow', 'raise left brow', 'left brow higher', 'left brow lift'],
    regions: ['brow_left'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['lift right brow', 'raise right brow', 'right brow higher', 'right brow lift'],
    regions: ['brow_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['smooth forehead', 'forehead botox', 'reduce forehead lines', 'flatten forehead', 'forehead wrinkles'],
    regions: ['forehead_center', 'forehead_left', 'forehead_right'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.10 },

  { patterns: ['rounder forehead', 'fuller forehead', 'forehead augmentation', 'forehead filler', 'convex forehead'],
    regions: ['forehead_center', 'forehead_left', 'forehead_right'],
    direction: { axis: 'z', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['reduce glabella', 'smooth glabella', 'glabella botox', 'remove frown lines', 'eleven lines', 'frown lines'],
    regions: ['glabella'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.15 },

  // ─── EYES ───
  { patterns: ['open eyes', 'bigger eyes', 'wider eyes', 'enlarge eyes', 'more open eyes', 'eye opening'],
    regions: ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },

  { patterns: ['hooded eyes', 'heavier eyelids', 'droopy eyelids', 'more hooded'],
    regions: ['upper_eyelid_left', 'upper_eyelid_right'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.15 },

  { patterns: ['upper eyelid lift', 'lift upper eyelids', 'blepharoplasty upper', 'eyelid surgery', 'upper bleph'],
    regions: ['upper_eyelid_left', 'upper_eyelid_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['lower eyelid lift', 'lower eyelid tightening', 'lower bleph', 'blepharoplasty lower', 'under eye tightening'],
    regions: ['lower_eyelid_left', 'lower_eyelid_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },

  { patterns: ['reduce crow feet', 'smooth crow feet', "crow's feet botox", 'crow feet botox', 'lateral eye lines'],
    regions: ['crow_feet_left', 'crow_feet_right'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.15 },

  { patterns: ['under eye filler', 'tear trough filler', 'reduce under eye hollows', 'fill under eyes', 'under eye bags', 'tear trough', 'dark circles filler'],
    regions: ['lower_eyelid_left', 'lower_eyelid_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.20 },

  // ─── TEMPLES ───
  { patterns: ['temple filler', 'fill temples', 'temple augmentation', 'fuller temples', 'volumize temples', 'temple volume'],
    regions: ['temple_left', 'temple_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['reduce temples', 'slimmer temples', 'temple reduction'],
    regions: ['temple_left', 'temple_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.20 },

  // ─── NASOLABIAL / MARIONETTE ───
  { patterns: ['reduce nasolabial folds', 'nasolabial filler', 'smile lines', 'soften nasolabial', 'fill nasolabial', 'laugh lines', 'nasolabial fold filler'],
    regions: ['nasolabial_left', 'nasolabial_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  { patterns: ['reduce marionette lines', 'marionette filler', 'fill marionette', 'soften marionette', 'marionette lines filler'],
    regions: ['marionette_left', 'marionette_right'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.25 },

  // ─── NECK / EARS ───
  { patterns: ['tighten neck', 'neck lift', 'reduce neck bands', 'neck contouring', 'slim neck', 'platysmal bands'],
    regions: ['neck_left', 'neck_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.25 },

  { patterns: ['pin ears', 'reduce ear projection', 'otoplasty', 'ear pinning', 'flatten ears', 'ears back', 'smaller ears'],
    regions: ['ear_left', 'ear_right'],
    direction: { axis: 'z', sign: -1 }, defaultIntensity: 0.30 },

  // ─── PERIORAL ───
  { patterns: ['reduce perioral lines', 'smoker lines', 'lip lines', 'perioral rejuvenation', 'perioral wrinkles'],
    regions: ['perioral'],
    direction: { axis: 'inflate', sign: 1 }, defaultIntensity: 0.15 },

  // ─── FACELIFT / GLOBAL ───
  { patterns: ['facelift', 'face lift', 'tighten face', 'face tightening', 'lift face', 'facial rejuvenation', 'tighten everything'],
    regions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'nasolabial_left', 'nasolabial_right', 'marionette_left', 'marionette_right', 'neck_left', 'neck_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.20 },

  { patterns: ['slim face', 'thinner face', 'reduce face width', 'narrow face', 'face slimming', 'slimmer face'],
    regions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
    direction: { axis: 'x', sign: -1 }, defaultIntensity: 0.20 },

  { patterns: ['wider face', 'broader face', 'widen face', 'rounder face'],
    regions: ['cheek_left', 'cheek_right', 'jawline_left', 'jawline_right'],
    direction: { axis: 'x', sign: 1 }, defaultIntensity: 0.20 },

  // ─── MASSETER ───
  { patterns: ['reduce masseter', 'masseter botox', 'slim masseter', 'masseter reduction', 'jawline slim', 'reduce jaw muscle'],
    regions: ['jaw_angle_left', 'jaw_angle_right'],
    direction: { axis: 'inflate', sign: -1 }, defaultIntensity: 0.30 },

  // ─── DAO (depressor anguli oris) ───
  { patterns: ['dao botox', 'reduce mouth frown', 'lift mouth corners', 'depressor anguli oris'],
    regions: ['lip_corner_left', 'lip_corner_right', 'marionette_left', 'marionette_right'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },

  // ─── PHILTRUM ───
  { patterns: ['shorten philtrum', 'reduce philtrum', 'philtrum reduction', 'shorter philtrum'],
    regions: ['philtrum'],
    direction: { axis: 'y', sign: -1 }, defaultIntensity: 0.15 },

  { patterns: ['longer philtrum', 'lengthen philtrum', 'elongate philtrum'],
    regions: ['philtrum'],
    direction: { axis: 'y', sign: 1 }, defaultIntensity: 0.15 },
];

// ---------------------------------------------------------------------------
// Laterality patterns  – detect left / right specificity
// ---------------------------------------------------------------------------
const LATERALITY_PATTERNS = [
  { pattern: /\b(?:only|just)\s+(?:the\s+)?(?:left|right)\b/i, capture: true },
  { pattern: /\b(?:left|right)\s+side\b/i, capture: true },
  { pattern: /\bleft\s+([\w\s]+)/i, side: 'left' },
  { pattern: /\bright\s+([\w\s]+)/i, side: 'right' },
];

// ---------------------------------------------------------------------------
// NLUAgent class
// ---------------------------------------------------------------------------
export class NLUAgent {
  constructor() {
    /** @type {Array<{regions: Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>, explanation:string, confidence:number}>} */
    this.history = [];

    /** Last parsed result for relative commands */
    this.lastResult = null;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Parse a natural language command into morph parameters.
   * @param {string} input  Raw user text
   * @returns {{ regions: Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>,
   *             explanation: string,
   *             confidence: number,
   *             action?: string }}
   */
  parse(input) {
    const text = input.trim().toLowerCase();

    // 1. Check for relative / meta commands first
    const relativeResult = this._parseRelativeCommand(text);
    if (relativeResult) return relativeResult;

    // 2. Try phrase-map matching
    const phraseResult = this._parsePhraseMap(text);
    if (phraseResult && phraseResult.confidence > 0) {
      this.history.push(phraseResult);
      this.lastResult = phraseResult;
      return phraseResult;
    }

    // 3. Try compositional parsing (region + direction + intensity)
    const compositionalResult = this._parseCompositional(text);
    if (compositionalResult && compositionalResult.confidence > 0) {
      this.history.push(compositionalResult);
      this.lastResult = compositionalResult;
      return compositionalResult;
    }

    // 4. Nothing recognized
    return {
      regions: {},
      explanation: `Could not understand: "${input}". Try something like "make my lips fuller" or "raise my cheekbones a little".`,
      confidence: 0,
    };
  }

  /**
   * Return all 52 clinical zone names.
   * @returns {string[]}
   */
  getZones() {
    return [...CLINICAL_ZONES];
  }

  /**
   * Return intensity map for UI sliders or debugging.
   * @returns {Record<string, number>}
   */
  getIntensityMap() {
    return { ...INTENSITY_MAP };
  }

  // -------------------------------------------------------------------------
  // Internal: Relative / meta command parsing
  // -------------------------------------------------------------------------

  /** @private */
  _parseRelativeCommand(text) {
    // Sort by longest key first so "a bit more" matches before "more"
    const sortedKeys = Object.keys(RELATIVE_COMMANDS).sort((a, b) => b.length - a.length);

    for (const key of sortedKeys) {
      if (text.includes(key)) {
        const cmd = RELATIVE_COMMANDS[key];

        if (cmd.type === 'undo') {
          const undone = this.history.pop();
          this.lastResult = this.history[this.history.length - 1] || null;
          return {
            regions: undone ? this._invertRegions(undone.regions) : {},
            explanation: undone ? `Undid last change: ${undone.explanation}` : 'Nothing to undo.',
            confidence: undone ? 0.95 : 0.5,
            action: 'undo',
          };
        }

        if (cmd.type === 'reset') {
          this.history = [];
          this.lastResult = null;
          return {
            regions: {},
            explanation: 'Reset all changes.',
            confidence: 1.0,
            action: 'reset',
          };
        }

        if (cmd.type === 'scale' && this.lastResult) {
          const scaledRegions = this._scaleRegions(this.lastResult.regions, cmd.factor);
          const result = {
            regions: scaledRegions,
            explanation: `Adjusted previous change by factor ${cmd.factor}: ${this.lastResult.explanation}`,
            confidence: 0.90,
            action: 'scale',
          };
          this.history.push(result);
          this.lastResult = result;
          return result;
        }

        if (cmd.type === 'scale' && !this.lastResult) {
          return {
            regions: {},
            explanation: 'No previous change to adjust. Please make a specific request first.',
            confidence: 0.3,
          };
        }
      }
    }
    return null;
  }

  // -------------------------------------------------------------------------
  // Internal: Phrase-map matching
  // -------------------------------------------------------------------------

  /** @private */
  _parsePhraseMap(text) {
    let bestMatch = null;
    let bestScore = 0;

    for (const entry of PHRASE_MAP) {
      for (const pattern of entry.patterns) {
        const score = this._fuzzyScore(text, pattern);
        if (score > bestScore) {
          bestScore = score;
          bestMatch = entry;
        }
      }
    }

    if (!bestMatch || bestScore < 0.35) return null;

    // Parse intensity from text
    const intensity = this._parseIntensity(text) ?? bestMatch.defaultIntensity;

    // Apply laterality filtering
    const filteredRegions = this._applyLaterality(text, bestMatch.regions);

    // Build output
    const regions = {};
    for (const regionName of filteredRegions) {
      const dir = bestMatch.direction;
      if (dir.axis === 'inflate') {
        regions[regionName] = {
          displacement: { x: 0, y: 0, z: 0 },
          inflate: intensity * dir.sign,
        };
      } else {
        const displacement = { x: 0, y: 0, z: 0 };
        displacement[dir.axis] = intensity * dir.sign;
        regions[regionName] = {
          displacement,
          inflate: 0,
        };
      }
    }

    const matchedPattern = bestMatch.patterns.find(p => this._fuzzyScore(text, p) === bestScore) || bestMatch.patterns[0];
    return {
      regions,
      explanation: `Matched "${matchedPattern}" — applying ${(intensity * 100).toFixed(0)}% ${bestMatch.direction.sign > 0 ? 'positive' : 'negative'} change to ${filteredRegions.join(', ')}.`,
      confidence: Math.min(bestScore, 0.98),
    };
  }

  // -------------------------------------------------------------------------
  // Internal: Compositional parsing (region + direction + intensity)
  // -------------------------------------------------------------------------

  /** @private */
  _parseCompositional(text) {
    // Attempt to find a region name in the text
    const detectedRegions = this._detectRegions(text);
    if (detectedRegions.length === 0) return null;

    // Attempt to find a direction
    const direction = this._parseDirection(text);
    if (!direction) return null;

    // Parse intensity
    const intensity = this._parseIntensity(text) ?? 0.25;

    // Apply laterality
    const filteredRegions = this._applyLaterality(text, detectedRegions);

    const regions = {};
    for (const regionName of filteredRegions) {
      if (direction.axis === 'inflate') {
        regions[regionName] = {
          displacement: { x: 0, y: 0, z: 0 },
          inflate: intensity * direction.sign,
        };
      } else {
        const displacement = { x: 0, y: 0, z: 0 };
        displacement[direction.axis] = intensity * direction.sign;
        regions[regionName] = {
          displacement,
          inflate: 0,
        };
      }
    }

    return {
      regions,
      explanation: `Compositional parse: ${direction.axis} ${direction.sign > 0 ? '+' : '-'}${(intensity * 100).toFixed(0)}% on ${filteredRegions.join(', ')}.`,
      confidence: 0.65,
    };
  }

  // -------------------------------------------------------------------------
  // Internal: helpers
  // -------------------------------------------------------------------------

  /** Fuzzy matching score between user text and a pattern phrase. */
  _fuzzyScore(text, pattern) {
    // Exact inclusion
    if (text.includes(pattern)) return 0.95;

    // Word-overlap score
    const patternWords = pattern.split(/\s+/);
    const textWords = text.split(/\s+/);
    let matchCount = 0;
    for (const pw of patternWords) {
      if (textWords.some(tw => tw === pw || tw.startsWith(pw) || pw.startsWith(tw))) {
        matchCount++;
      }
    }
    return matchCount / patternWords.length;
  }

  /** Extract intensity multiplier from text. Returns null if none found. */
  _parseIntensity(text) {
    // Sort by longest key first for greedy matching
    const sorted = Object.entries(INTENSITY_MAP).sort((a, b) => b[0].length - a[0].length);
    for (const [phrase, value] of sorted) {
      if (text.includes(phrase)) return value;
    }

    // Try percentage pattern: "30%", "50 percent"
    const pctMatch = text.match(/(\d{1,3})\s*(?:%|percent)/);
    if (pctMatch) {
      return Math.min(parseInt(pctMatch[1], 10) / 100, 1.0);
    }

    return null;
  }

  /** Extract direction from text. */
  _parseDirection(text) {
    const sorted = Object.entries(DIRECTION_MAP).sort((a, b) => b[0].length - a[0].length);
    for (const [phrase, dir] of sorted) {
      if (text.includes(phrase)) return dir;
    }
    return null;
  }

  /** Detect clinical zone references in text. */
  _detectRegions(text) {
    const found = [];
    // Build friendlier aliases
    const aliases = {
      lip: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
      lips: ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
      'upper lip': ['upper_lip_center', 'upper_lip_left', 'upper_lip_right'],
      'lower lip': ['lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
      nose: ['nasal_tip', 'nasal_dorsum', 'nasal_bridge', 'nasal_ala_left', 'nasal_ala_right'],
      'nose tip': ['nasal_tip'],
      'nose bridge': ['nasal_bridge', 'nasal_dorsum'],
      nostril: ['nostril_left', 'nostril_right'],
      nostrils: ['nostril_left', 'nostril_right'],
      cheek: ['cheek_left', 'cheek_right'],
      cheeks: ['cheek_left', 'cheek_right'],
      cheekbone: ['cheekbone_left', 'cheekbone_right'],
      cheekbones: ['cheekbone_left', 'cheekbone_right'],
      jaw: ['jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
      jawline: ['jawline_left', 'jawline_right'],
      chin: ['chin_center', 'chin_left', 'chin_right'],
      brow: ['brow_left', 'brow_right'],
      brows: ['brow_left', 'brow_right'],
      eyebrow: ['brow_left', 'brow_right'],
      eyebrows: ['brow_left', 'brow_right'],
      forehead: ['forehead_center', 'forehead_left', 'forehead_right'],
      temple: ['temple_left', 'temple_right'],
      temples: ['temple_left', 'temple_right'],
      eyelid: ['upper_eyelid_left', 'upper_eyelid_right'],
      eyelids: ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
      'upper eyelid': ['upper_eyelid_left', 'upper_eyelid_right'],
      'lower eyelid': ['lower_eyelid_left', 'lower_eyelid_right'],
      eyes: ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
      ear: ['ear_left', 'ear_right'],
      ears: ['ear_left', 'ear_right'],
      neck: ['neck_left', 'neck_right'],
      'under chin': ['under_chin'],
      glabella: ['glabella'],
      philtrum: ['philtrum'],
      'lip corner': ['lip_corner_left', 'lip_corner_right'],
      'lip corners': ['lip_corner_left', 'lip_corner_right'],
      'nasolabial': ['nasolabial_left', 'nasolabial_right'],
      'marionette': ['marionette_left', 'marionette_right'],
      'crow feet': ['crow_feet_left', 'crow_feet_right'],
      "crow's feet": ['crow_feet_left', 'crow_feet_right'],
    };

    // Sort by longest key first
    const sortedAliases = Object.entries(aliases).sort((a, b) => b[0].length - a[0].length);
    for (const [alias, zones] of sortedAliases) {
      if (text.includes(alias)) {
        found.push(...zones);
        break; // take the first (longest) match
      }
    }

    // Also try direct clinical zone names (underscores replaced with spaces)
    if (found.length === 0) {
      for (const zone of CLINICAL_ZONES) {
        const friendly = zone.replace(/_/g, ' ');
        if (text.includes(friendly)) {
          found.push(zone);
        }
      }
    }

    return [...new Set(found)];
  }

  /** Filter regions by laterality (left/right) if specified in text. */
  _applyLaterality(text, regions) {
    const hasLeft = /\b(?:only|just)\s+(?:the\s+)?left\b|\bleft\s+side\b/i.test(text);
    const hasRight = /\b(?:only|just)\s+(?:the\s+)?right\b|\bright\s+side\b/i.test(text);

    // Also detect more specific: "left nostril", "right cheekbone", etc.
    const leftMention = /\bleft\b/i.test(text);
    const rightMention = /\bright\b/i.test(text);

    if (hasLeft || (leftMention && !rightMention)) {
      const filtered = regions.filter(r => r.includes('left') || r.includes('center') || (!r.includes('right') && !r.includes('left')));
      // But exclude center if there are left-specific versions
      return filtered.length > 0 ? filtered : regions;
    }

    if (hasRight || (rightMention && !leftMention)) {
      const filtered = regions.filter(r => r.includes('right') || r.includes('center') || (!r.includes('right') && !r.includes('left')));
      return filtered.length > 0 ? filtered : regions;
    }

    return regions;
  }

  /** Invert all displacements and inflate values (for undo). */
  _invertRegions(regions) {
    const inverted = {};
    for (const [name, data] of Object.entries(regions)) {
      inverted[name] = {
        displacement: {
          x: -data.displacement.x,
          y: -data.displacement.y,
          z: -data.displacement.z,
        },
        inflate: -data.inflate,
      };
    }
    return inverted;
  }

  /** Scale all displacements and inflate values. */
  _scaleRegions(regions, factor) {
    const scaled = {};
    for (const [name, data] of Object.entries(regions)) {
      scaled[name] = {
        displacement: {
          x: data.displacement.x * factor,
          y: data.displacement.y * factor,
          z: data.displacement.z * factor,
        },
        inflate: data.inflate * factor,
      };
    }
    return scaled;
  }
}

export default NLUAgent;
