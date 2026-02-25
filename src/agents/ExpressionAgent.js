/**
 * ExpressionAgent.js
 * Expression animation agent that defines 12+ facial expressions as
 * vertex-region displacement maps and provides animation sequencing.
 *
 * Each expression is defined as a set of clinical-zone displacements
 * relative to the neutral FLAME mesh. Talking uses an 8-frame mouth
 * viseme sequence for speech animation.
 */

// ---------------------------------------------------------------------------
// Expression definitions
// Each expression maps region names to { displacement: {x,y,z}, inflate }
// Values are normalized (-1 to 1 range); the renderer scales by amplitude.
// ---------------------------------------------------------------------------

const EXPRESSION_LIBRARY = {
  // ─── SMILE ───
  smile: {
    name: 'Smile',
    duration: 600,  // ms to reach full expression
    holdDuration: 0, // 0 = hold until released
    regions: {
      lip_corner_left:   { displacement: { x: 0.12, y: 0.15, z: 0.05 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: -0.12, y: 0.15, z: 0.05 }, inflate: 0 },
      upper_lip_center:  { displacement: { x: 0, y: 0.03, z: 0.02 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0, y: -0.02, z: 0.01 }, inflate: 0 },
      cheek_left:        { displacement: { x: 0.05, y: 0.08, z: 0.06 }, inflate: 0.04 },
      cheek_right:       { displacement: { x: -0.05, y: 0.08, z: 0.06 }, inflate: 0.04 },
      nasolabial_left:   { displacement: { x: 0.02, y: 0.03, z: 0.03 }, inflate: 0.02 },
      nasolabial_right:  { displacement: { x: -0.02, y: 0.03, z: 0.03 }, inflate: 0.02 },
      lower_eyelid_left: { displacement: { x: 0, y: 0.03, z: 0 }, inflate: 0 },
      lower_eyelid_right:{ displacement: { x: 0, y: 0.03, z: 0 }, inflate: 0 },
      crow_feet_left:    { displacement: { x: 0.01, y: 0.01, z: -0.01 }, inflate: 0 },
      crow_feet_right:   { displacement: { x: -0.01, y: 0.01, z: -0.01 }, inflate: 0 },
    },
  },

  // ─── BIG SMILE ───
  big_smile: {
    name: 'Big Smile',
    duration: 500,
    holdDuration: 0,
    regions: {
      lip_corner_left:   { displacement: { x: 0.18, y: 0.22, z: 0.08 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: -0.18, y: 0.22, z: 0.08 }, inflate: 0 },
      upper_lip_center:  { displacement: { x: 0, y: 0.06, z: 0.04 }, inflate: 0 },
      upper_lip_left:    { displacement: { x: 0.04, y: 0.05, z: 0.03 }, inflate: 0 },
      upper_lip_right:   { displacement: { x: -0.04, y: 0.05, z: 0.03 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0, y: -0.06, z: 0.02 }, inflate: 0 },
      lower_lip_left:    { displacement: { x: 0.02, y: -0.04, z: 0.02 }, inflate: 0 },
      lower_lip_right:   { displacement: { x: -0.02, y: -0.04, z: 0.02 }, inflate: 0 },
      cheek_left:        { displacement: { x: 0.08, y: 0.14, z: 0.10 }, inflate: 0.08 },
      cheek_right:       { displacement: { x: -0.08, y: 0.14, z: 0.10 }, inflate: 0.08 },
      cheekbone_left:    { displacement: { x: 0.02, y: 0.04, z: 0.03 }, inflate: 0 },
      cheekbone_right:   { displacement: { x: -0.02, y: 0.04, z: 0.03 }, inflate: 0 },
      nasolabial_left:   { displacement: { x: 0.04, y: 0.05, z: 0.05 }, inflate: 0.04 },
      nasolabial_right:  { displacement: { x: -0.04, y: 0.05, z: 0.05 }, inflate: 0.04 },
      lower_eyelid_left: { displacement: { x: 0, y: 0.06, z: 0 }, inflate: 0 },
      lower_eyelid_right:{ displacement: { x: 0, y: 0.06, z: 0 }, inflate: 0 },
      crow_feet_left:    { displacement: { x: 0.02, y: 0.02, z: -0.02 }, inflate: 0 },
      crow_feet_right:   { displacement: { x: -0.02, y: 0.02, z: -0.02 }, inflate: 0 },
      upper_eyelid_left: { displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
      upper_eyelid_right:{ displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
    },
  },

  // ─── SUBTLE SMILE ───
  subtle_smile: {
    name: 'Subtle Smile',
    duration: 800,
    holdDuration: 0,
    regions: {
      lip_corner_left:   { displacement: { x: 0.05, y: 0.06, z: 0.02 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: -0.05, y: 0.06, z: 0.02 }, inflate: 0 },
      cheek_left:        { displacement: { x: 0.02, y: 0.03, z: 0.02 }, inflate: 0.01 },
      cheek_right:       { displacement: { x: -0.02, y: 0.03, z: 0.02 }, inflate: 0.01 },
      lower_eyelid_left: { displacement: { x: 0, y: 0.01, z: 0 }, inflate: 0 },
      lower_eyelid_right:{ displacement: { x: 0, y: 0.01, z: 0 }, inflate: 0 },
    },
  },

  // ─── FROWN ───
  frown: {
    name: 'Frown',
    duration: 600,
    holdDuration: 0,
    regions: {
      lip_corner_left:   { displacement: { x: -0.03, y: -0.10, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: 0.03, y: -0.10, z: 0 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0, y: -0.03, z: 0.02 }, inflate: 0 },
      brow_left:         { displacement: { x: 0.02, y: -0.06, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: -0.02, y: -0.06, z: 0 }, inflate: 0 },
      glabella:          { displacement: { x: 0, y: -0.03, z: 0.03 }, inflate: 0.02 },
      marionette_left:   { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      marionette_right:  { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      cheek_left:        { displacement: { x: 0, y: -0.02, z: 0 }, inflate: -0.01 },
      cheek_right:       { displacement: { x: 0, y: -0.02, z: 0 }, inflate: -0.01 },
    },
  },

  // ─── SURPRISE ───
  surprise: {
    name: 'Surprise',
    duration: 300,
    holdDuration: 0,
    regions: {
      brow_left:         { displacement: { x: 0, y: 0.15, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: 0, y: 0.15, z: 0 }, inflate: 0 },
      forehead_center:   { displacement: { x: 0, y: 0.03, z: 0 }, inflate: 0 },
      forehead_left:     { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      forehead_right:    { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      upper_eyelid_left: { displacement: { x: 0, y: 0.08, z: 0 }, inflate: 0 },
      upper_eyelid_right:{ displacement: { x: 0, y: 0.08, z: 0 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0, y: -0.12, z: 0 }, inflate: 0 },
      lower_lip_left:    { displacement: { x: 0, y: -0.08, z: 0 }, inflate: 0 },
      lower_lip_right:   { displacement: { x: 0, y: -0.08, z: 0 }, inflate: 0 },
      upper_lip_center:  { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      chin_center:       { displacement: { x: 0, y: -0.06, z: 0 }, inflate: 0 },
    },
  },

  // ─── THINKING ───
  thinking: {
    name: 'Thinking',
    duration: 800,
    holdDuration: 0,
    regions: {
      brow_left:         { displacement: { x: 0, y: 0.04, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: -0.02, y: -0.03, z: 0 }, inflate: 0 },
      glabella:          { displacement: { x: 0, y: -0.01, z: 0.02 }, inflate: 0.01 },
      upper_eyelid_left: { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      upper_eyelid_right:{ displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
      lip_corner_left:   { displacement: { x: -0.03, y: -0.02, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: 0.04, y: 0.01, z: 0 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0.02, y: 0.02, z: 0.03 }, inflate: 0 },
    },
  },

  // ─── TALKING (8-frame sequence) ───
  talking: {
    name: 'Talking',
    duration: 120,  // ms per frame
    holdDuration: 0,
    isSequence: true,
    loop: true,
    frames: [
      // Frame 0: Neutral / rest
      {
        upper_lip_center: { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
      },
      // Frame 1: Slight open  (viseme: M/B/P)
      {
        upper_lip_center: { displacement: { x: 0, y: 0.01, z: 0.02 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.01, z: 0.02 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: -0.01, y: 0, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: 0.01, y: 0, z: 0 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.01, z: 0 }, inflate: 0 },
      },
      // Frame 2: Medium open (viseme: AH)
      {
        upper_lip_center: { displacement: { x: 0, y: 0.03, z: 0 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.08, z: 0 }, inflate: 0 },
        lower_lip_left:   { displacement: { x: 0, y: -0.06, z: 0 }, inflate: 0 },
        lower_lip_right:  { displacement: { x: 0, y: -0.06, z: 0 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: 0.02, y: -0.02, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: -0.02, y: -0.02, z: 0 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.06, z: 0 }, inflate: 0 },
      },
      // Frame 3: Wide open (viseme: AA)
      {
        upper_lip_center: { displacement: { x: 0, y: 0.04, z: 0 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.12, z: 0 }, inflate: 0 },
        lower_lip_left:   { displacement: { x: 0, y: -0.09, z: 0 }, inflate: 0 },
        lower_lip_right:  { displacement: { x: 0, y: -0.09, z: 0 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: 0.04, y: -0.03, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: -0.04, y: -0.03, z: 0 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.10, z: 0 }, inflate: 0 },
        cheek_left:       { displacement: { x: 0.01, y: -0.01, z: 0 }, inflate: -0.01 },
        cheek_right:      { displacement: { x: -0.01, y: -0.01, z: 0 }, inflate: -0.01 },
      },
      // Frame 4: OO shape (viseme: OO/UW)
      {
        upper_lip_center: { displacement: { x: 0, y: 0.02, z: 0.06 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.04, z: 0.06 }, inflate: 0 },
        upper_lip_left:   { displacement: { x: -0.02, y: 0.01, z: 0.04 }, inflate: 0 },
        upper_lip_right:  { displacement: { x: 0.02, y: 0.01, z: 0.04 }, inflate: 0 },
        lower_lip_left:   { displacement: { x: -0.02, y: -0.02, z: 0.04 }, inflate: 0 },
        lower_lip_right:  { displacement: { x: 0.02, y: -0.02, z: 0.04 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: -0.04, y: 0, z: 0.02 }, inflate: 0 },
        lip_corner_right: { displacement: { x: 0.04, y: 0, z: 0.02 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      },
      // Frame 5: EE shape (viseme: IY)
      {
        upper_lip_center: { displacement: { x: 0, y: 0.01, z: -0.01 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.02, z: -0.01 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: 0.06, y: 0.04, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: -0.06, y: 0.04, z: 0 }, inflate: 0 },
        cheek_left:       { displacement: { x: 0.02, y: 0.02, z: 0.02 }, inflate: 0.01 },
        cheek_right:      { displacement: { x: -0.02, y: 0.02, z: 0.02 }, inflate: 0.01 },
        chin_center:      { displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
      },
      // Frame 6: F/V shape
      {
        upper_lip_center: { displacement: { x: 0, y: -0.01, z: 0 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: 0.02, z: 0.04 }, inflate: 0 },
        lower_lip_left:   { displacement: { x: 0, y: 0.01, z: 0.03 }, inflate: 0 },
        lower_lip_right:  { displacement: { x: 0, y: 0.01, z: 0.03 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.01, z: 0 }, inflate: 0 },
      },
      // Frame 7: TH / L shape
      {
        upper_lip_center: { displacement: { x: 0, y: 0.01, z: 0 }, inflate: 0 },
        lower_lip_center: { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
        lip_corner_left:  { displacement: { x: 0.01, y: 0, z: 0 }, inflate: 0 },
        lip_corner_right: { displacement: { x: -0.01, y: 0, z: 0 }, inflate: 0 },
        chin_center:      { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      },
    ],
  },

  // ─── RESTING ───
  resting: {
    name: 'Resting',
    duration: 1000,
    holdDuration: 0,
    regions: {
      // Very minimal displacement – almost neutral but with slight natural asymmetry
      lip_corner_left:   { displacement: { x: 0, y: -0.01, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: 0, y: -0.005, z: 0 }, inflate: 0 },
      brow_left:         { displacement: { x: 0, y: -0.005, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
      upper_eyelid_left: { displacement: { x: 0, y: -0.01, z: 0 }, inflate: 0 },
      upper_eyelid_right:{ displacement: { x: 0, y: -0.008, z: 0 }, inflate: 0 },
    },
  },

  // ─── LAUGH ───
  laugh: {
    name: 'Laugh',
    duration: 400,
    holdDuration: 0,
    regions: {
      lip_corner_left:    { displacement: { x: 0.15, y: 0.20, z: 0.06 }, inflate: 0 },
      lip_corner_right:   { displacement: { x: -0.15, y: 0.20, z: 0.06 }, inflate: 0 },
      upper_lip_center:   { displacement: { x: 0, y: 0.05, z: 0.03 }, inflate: 0 },
      lower_lip_center:   { displacement: { x: 0, y: -0.10, z: 0 }, inflate: 0 },
      lower_lip_left:     { displacement: { x: 0, y: -0.07, z: 0 }, inflate: 0 },
      lower_lip_right:    { displacement: { x: 0, y: -0.07, z: 0 }, inflate: 0 },
      cheek_left:         { displacement: { x: 0.08, y: 0.12, z: 0.10 }, inflate: 0.08 },
      cheek_right:        { displacement: { x: -0.08, y: 0.12, z: 0.10 }, inflate: 0.08 },
      cheekbone_left:     { displacement: { x: 0.02, y: 0.04, z: 0.02 }, inflate: 0 },
      cheekbone_right:    { displacement: { x: -0.02, y: 0.04, z: 0.02 }, inflate: 0 },
      nasolabial_left:    { displacement: { x: 0.05, y: 0.06, z: 0.06 }, inflate: 0.05 },
      nasolabial_right:   { displacement: { x: -0.05, y: 0.06, z: 0.06 }, inflate: 0.05 },
      lower_eyelid_left:  { displacement: { x: 0, y: 0.06, z: 0 }, inflate: 0 },
      lower_eyelid_right: { displacement: { x: 0, y: 0.06, z: 0 }, inflate: 0 },
      upper_eyelid_left:  { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      upper_eyelid_right: { displacement: { x: 0, y: -0.03, z: 0 }, inflate: 0 },
      crow_feet_left:     { displacement: { x: 0.02, y: 0.02, z: -0.02 }, inflate: 0 },
      crow_feet_right:    { displacement: { x: -0.02, y: 0.02, z: -0.02 }, inflate: 0 },
      chin_center:        { displacement: { x: 0, y: -0.08, z: 0 }, inflate: 0 },
    },
  },

  // ─── CONCERNED ───
  concerned: {
    name: 'Concerned',
    duration: 700,
    holdDuration: 0,
    regions: {
      brow_left:         { displacement: { x: 0.02, y: 0.06, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: -0.02, y: 0.06, z: 0 }, inflate: 0 },
      glabella:          { displacement: { x: 0, y: -0.02, z: 0.02 }, inflate: 0.02 },
      forehead_center:   { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      lip_corner_left:   { displacement: { x: -0.02, y: -0.05, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: 0.02, y: -0.05, z: 0 }, inflate: 0 },
      lower_lip_center:  { displacement: { x: 0, y: -0.02, z: 0.02 }, inflate: 0 },
      upper_eyelid_left: { displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
      upper_eyelid_right:{ displacement: { x: 0, y: 0.02, z: 0 }, inflate: 0 },
    },
  },

  // ─── DISGUST ───
  disgust: {
    name: 'Disgust',
    duration: 500,
    holdDuration: 0,
    regions: {
      upper_lip_center:  { displacement: { x: 0, y: 0.06, z: 0.04 }, inflate: 0 },
      upper_lip_left:    { displacement: { x: 0.02, y: 0.05, z: 0.03 }, inflate: 0 },
      upper_lip_right:   { displacement: { x: -0.02, y: 0.05, z: 0.03 }, inflate: 0 },
      nasal_ala_left:    { displacement: { x: 0.03, y: 0.04, z: 0.02 }, inflate: 0.02 },
      nasal_ala_right:   { displacement: { x: -0.03, y: 0.04, z: 0.02 }, inflate: 0.02 },
      nasal_tip:         { displacement: { x: 0, y: 0.02, z: 0.02 }, inflate: 0 },
      brow_left:         { displacement: { x: 0, y: -0.04, z: 0 }, inflate: 0 },
      brow_right:        { displacement: { x: 0, y: -0.04, z: 0 }, inflate: 0 },
      glabella:          { displacement: { x: 0, y: -0.02, z: 0.02 }, inflate: 0.02 },
      lower_lip_center:  { displacement: { x: 0, y: -0.04, z: 0.02 }, inflate: 0 },
      lip_corner_left:   { displacement: { x: -0.03, y: -0.06, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: 0.03, y: -0.06, z: 0 }, inflate: 0 },
      chin_center:       { displacement: { x: 0, y: 0.02, z: 0.04 }, inflate: 0.02 },
      nasolabial_left:   { displacement: { x: 0.03, y: 0.04, z: 0.04 }, inflate: 0.03 },
      nasolabial_right:  { displacement: { x: -0.03, y: 0.04, z: 0.04 }, inflate: 0.03 },
    },
  },

  // ─── WINK LEFT ───
  wink_left: {
    name: 'Wink Left',
    duration: 200,
    holdDuration: 400,
    regions: {
      upper_eyelid_left: { displacement: { x: 0, y: -0.10, z: 0 }, inflate: 0 },
      lower_eyelid_left: { displacement: { x: 0, y: 0.04, z: 0 }, inflate: 0 },
      cheek_left:        { displacement: { x: 0.02, y: 0.04, z: 0.02 }, inflate: 0.02 },
      crow_feet_left:    { displacement: { x: 0.01, y: 0.01, z: -0.01 }, inflate: 0 },
      brow_left:         { displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
      lip_corner_left:   { displacement: { x: 0.03, y: 0.03, z: 0 }, inflate: 0 },
    },
  },

  // ─── WINK RIGHT ───
  wink_right: {
    name: 'Wink Right',
    duration: 200,
    holdDuration: 400,
    regions: {
      upper_eyelid_right:{ displacement: { x: 0, y: -0.10, z: 0 }, inflate: 0 },
      lower_eyelid_right:{ displacement: { x: 0, y: 0.04, z: 0 }, inflate: 0 },
      cheek_right:       { displacement: { x: -0.02, y: 0.04, z: 0.02 }, inflate: 0.02 },
      crow_feet_right:   { displacement: { x: -0.01, y: 0.01, z: -0.01 }, inflate: 0 },
      brow_right:        { displacement: { x: 0, y: -0.02, z: 0 }, inflate: 0 },
      lip_corner_right:  { displacement: { x: -0.03, y: 0.03, z: 0 }, inflate: 0 },
    },
  },
};

// ---------------------------------------------------------------------------
// Text → expression detection keywords
// ---------------------------------------------------------------------------
const EXPRESSION_KEYWORDS = {
  smile:         ['smile', 'happy', 'grin', 'smiling', 'pleased', 'cheerful', 'glad'],
  big_smile:     ['big smile', 'wide smile', 'huge smile', 'beaming', 'ear to ear', 'grinning', 'toothy smile', 'ecstatic'],
  subtle_smile:  ['subtle smile', 'slight smile', 'half smile', 'mona lisa', 'barely smiling', 'hint of a smile', 'smirk'],
  frown:         ['frown', 'sad', 'unhappy', 'upset', 'disappointed', 'frowning', 'down', 'sorrowful'],
  surprise:      ['surprise', 'surprised', 'shocked', 'astonished', 'amazed', 'wow', 'startled', 'gasp'],
  thinking:      ['thinking', 'thoughtful', 'contemplating', 'pondering', 'considering', 'hmm', 'puzzled', 'curious'],
  talking:       ['talking', 'speaking', 'talk', 'speak', 'say', 'saying', 'conversation', 'mouth moving', 'animate mouth'],
  resting:       ['resting', 'rest', 'neutral', 'relaxed', 'calm', 'at ease', 'normal', 'default', 'natural'],
  laugh:         ['laugh', 'laughing', 'lol', 'hilarious', 'cracking up', 'ha ha', 'haha', 'belly laugh', 'giggle'],
  concerned:     ['concerned', 'worried', 'anxious', 'nervous', 'uneasy', 'apprehensive', 'troubled'],
  disgust:       ['disgust', 'disgusted', 'grossed out', 'repulsed', 'yuck', 'ew', 'ugh', 'revolted'],
  wink_left:     ['wink left', 'left wink', 'wink with left eye'],
  wink_right:    ['wink right', 'right wink', 'wink with right eye', 'wink'],
};

// ---------------------------------------------------------------------------
// ExpressionAgent class
// ---------------------------------------------------------------------------
export class ExpressionAgent {
  constructor() {
    this.expressions = EXPRESSION_LIBRARY;
    this.currentExpression = null;
    this.animationTimer = null;
    this.currentFrame = 0;
    this._onFrameCallback = null;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Detect an expression from text and return full animation data.
   * @param {string} text  Natural language input.
   * @returns {{ expression: string|null,
   *             name: string,
   *             data: object,
   *             animationParams: { duration: number, holdDuration: number, isSequence: boolean, frameCount: number },
   *             confidence: number } | null}
   */
  animate(text) {
    const lower = text.toLowerCase().trim();

    // Find best matching expression
    let bestKey = null;
    let bestScore = 0;

    for (const [key, keywords] of Object.entries(EXPRESSION_KEYWORDS)) {
      for (const kw of keywords) {
        if (lower.includes(kw)) {
          // Prefer longer keyword matches
          const score = kw.length / lower.length + (kw.split(' ').length * 0.1);
          if (score > bestScore) {
            bestScore = score;
            bestKey = key;
          }
        }
      }
    }

    if (!bestKey) return null;

    const expr = this.expressions[bestKey];
    this.currentExpression = bestKey;
    this.currentFrame = 0;

    const isSequence = !!expr.isSequence;
    const frameCount = isSequence ? expr.frames.length : 1;

    return {
      expression: bestKey,
      name: expr.name,
      data: isSequence ? expr.frames : expr.regions,
      animationParams: {
        duration: expr.duration,
        holdDuration: expr.holdDuration,
        isSequence,
        loop: expr.loop || false,
        frameCount,
      },
      confidence: Math.min(bestScore + 0.5, 0.98),
    };
  }

  /**
   * Get data for a specific expression by key.
   * @param {string} expressionKey
   * @returns {object|null}
   */
  getExpression(expressionKey) {
    const expr = this.expressions[expressionKey];
    if (!expr) return null;

    return {
      expression: expressionKey,
      name: expr.name,
      data: expr.isSequence ? expr.frames : expr.regions,
      animationParams: {
        duration: expr.duration,
        holdDuration: expr.holdDuration,
        isSequence: !!expr.isSequence,
        loop: expr.loop || false,
        frameCount: expr.isSequence ? expr.frames.length : 1,
      },
    };
  }

  /**
   * List all available expressions.
   * @returns {Array<{key: string, name: string, isSequence: boolean}>}
   */
  getAvailableExpressions() {
    return Object.entries(this.expressions).map(([key, expr]) => ({
      key,
      name: expr.name,
      isSequence: !!expr.isSequence,
      duration: expr.duration,
    }));
  }

  /**
   * Start playing a sequence expression (e.g., talking).
   * Calls onFrame with the current frame data at each step.
   * @param {string} expressionKey
   * @param {(frameData: object, frameIndex: number) => void} onFrame
   * @returns {{ stop: () => void }}
   */
  playSequence(expressionKey, onFrame) {
    const expr = this.expressions[expressionKey];
    if (!expr || !expr.isSequence) {
      return { stop: () => {} };
    }

    this.stopSequence();
    this.currentExpression = expressionKey;
    this.currentFrame = 0;
    this._onFrameCallback = onFrame;

    const frames = expr.frames;
    const frameDuration = expr.duration; // ms per frame

    const tick = () => {
      if (this._onFrameCallback) {
        this._onFrameCallback(frames[this.currentFrame], this.currentFrame);
      }
      this.currentFrame = (this.currentFrame + 1) % frames.length;
    };

    // Fire first frame immediately
    tick();

    this.animationTimer = setInterval(tick, frameDuration);

    return {
      stop: () => this.stopSequence(),
    };
  }

  /**
   * Stop any running sequence animation.
   */
  stopSequence() {
    if (this.animationTimer) {
      clearInterval(this.animationTimer);
      this.animationTimer = null;
    }
    this._onFrameCallback = null;
    this.currentFrame = 0;
  }

  /**
   * Interpolate between neutral and a target expression.
   * Useful for smooth transitions.
   * @param {object} targetRegions  Region displacement map.
   * @param {number} t  Interpolation factor (0 = neutral, 1 = full expression).
   * @returns {object}  Interpolated region map.
   */
  interpolate(targetRegions, t) {
    const clamped = Math.max(0, Math.min(1, t));
    const result = {};

    for (const [region, data] of Object.entries(targetRegions)) {
      result[region] = {
        displacement: {
          x: data.displacement.x * clamped,
          y: data.displacement.y * clamped,
          z: data.displacement.z * clamped,
        },
        inflate: data.inflate * clamped,
      };
    }

    return result;
  }

  /**
   * Blend two expression region maps together.
   * @param {object} exprA  First expression regions.
   * @param {object} exprB  Second expression regions.
   * @param {number} blendFactor  0 = all A, 1 = all B.
   * @returns {object}  Blended regions.
   */
  blend(exprA, exprB, blendFactor) {
    const t = Math.max(0, Math.min(1, blendFactor));
    const allKeys = new Set([...Object.keys(exprA), ...Object.keys(exprB)]);
    const result = {};

    for (const key of allKeys) {
      const a = exprA[key] || { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 };
      const b = exprB[key] || { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 };

      result[key] = {
        displacement: {
          x: a.displacement.x * (1 - t) + b.displacement.x * t,
          y: a.displacement.y * (1 - t) + b.displacement.y * t,
          z: a.displacement.z * (1 - t) + b.displacement.z * t,
        },
        inflate: a.inflate * (1 - t) + b.inflate * t,
      };
    }

    return result;
  }
}

export default ExpressionAgent;
