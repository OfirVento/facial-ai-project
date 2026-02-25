/**
 * FlameMeshGenerator.js
 *
 * Generates a FLAME-compatible face mesh programmatically with proper facial
 * topology, 52 named clinical sub-region vertex groups, UV coordinates, normals,
 * and quad/tri topology following FLAME's structure.
 *
 * The mesh is built by creating a UV-sphere base and then applying a series of
 * analytic deformation fields that carve out eye sockets, raise the nose bridge,
 * shape lips with a cupid's bow, define the chin/jaw, etc.
 *
 * Additionally supports loading real FLAME model data from pre-converted web
 * files (binary PCA bases, face indices, region maps). When loaded, shape and
 * expression parameters can be applied via PCA deformation, and DECA
 * reconstruction outputs can drive the mesh directly.  The analytic UV-sphere
 * approach is kept as the default fallback when FLAME data is not available.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Latitude rings (poles included). More rings = more vertices. */
const DEFAULT_RINGS = 52;
/** Longitude segments per ring. */
const DEFAULT_SEGMENTS = 52;
/** Base head radius before deformation (in scene units, ~8.5 cm). */
const BASE_RADIUS = 0.085;

/** FLAME model constants */
const FLAME_VERTEX_COUNT = 5023;
const FLAME_FACE_COUNT = 9976;
const FLAME_SHAPE_COMPONENTS = 50;
const FLAME_EXPRESSION_COMPONENTS = 50;

/**
 * Descriptive names for the first 20 FLAME shape PCA components.
 * Beyond these, names fall back to a generic "Shape Component N".
 */
const SHAPE_PARAM_NAMES = [
  'Overall Head Size',
  'Head Width',
  'Head Height',
  'Jaw Width',
  'Jaw Length',
  'Forehead Height',
  'Brow Ridge Depth',
  'Nose Length',
  'Nose Width',
  'Nose Bridge Height',
  'Cheekbone Prominence',
  'Chin Protrusion',
  'Eye Socket Depth',
  'Upper Face Width',
  'Mid-Face Length',
  'Lower Face Width',
  'Lip Thickness',
  'Philtrum Length',
  'Ear Size',
  'Face Taper',
];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Compute a smooth falloff value in [0, 1].
 * Uses a cosine-based hermite when the sample is within the given radius of
 * the centre; returns 0 outside.
 */
function smoothFalloff(px, py, cx, cy, radius) {
  const dx = px - cx;
  const dy = py - cy;
  const d = Math.sqrt(dx * dx + dy * dy);
  if (d >= radius) return 0;
  const t = d / radius;
  return 0.5 * (1 + Math.cos(Math.PI * t)); // hermite-ish bell
}

/**
 * Elliptical falloff.  Returns [0,1] based on distance inside the ellipse
 * defined by (cx,cy) with semi-axes (rx,ry).
 */
function ellipseFalloff(px, py, cx, cy, rx, ry) {
  const dx = (px - cx) / rx;
  const dy = (py - cy) / ry;
  const d2 = dx * dx + dy * dy;
  if (d2 >= 1) return 0;
  const d = Math.sqrt(d2);
  return 0.5 * (1 + Math.cos(Math.PI * d));
}

/**
 * Clamp a value between lo and hi.
 */
function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Linear interpolation.
 */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// Region metadata definition
// ---------------------------------------------------------------------------

/**
 * Each entry describes a named clinical zone.
 * `parent`  - name of the parent region (or null for root).
 * `clinical`- short clinical descriptor for UI / tooltips.
 */
const REGION_META_DEFS = {
  full_face:               { label: 'Full Face',                  parent: null,              clinical: 'Entire facial region' },

  // -- Forehead --
  forehead:                { label: 'Forehead',                   parent: 'full_face',       clinical: 'Frontal bone region' },
  forehead_left:           { label: 'Forehead Left',              parent: 'forehead',        clinical: 'Left frontal region' },
  forehead_right:          { label: 'Forehead Right',             parent: 'forehead',        clinical: 'Right frontal region' },
  forehead_center:         { label: 'Forehead Center',            parent: 'forehead',        clinical: 'Glabella and mid-frontal region' },

  // -- Brows --
  brow_left:               { label: 'Left Brow',                  parent: 'forehead',        clinical: 'Left supraorbital ridge' },
  brow_right:              { label: 'Right Brow',                 parent: 'forehead',        clinical: 'Right supraorbital ridge' },
  brow_inner_left:         { label: 'Left Inner Brow',            parent: 'brow_left',       clinical: 'Left medial brow' },
  brow_inner_right:        { label: 'Right Inner Brow',           parent: 'brow_right',      clinical: 'Right medial brow' },

  // -- Eyes --
  eye_left_upper:          { label: 'Left Upper Eyelid',          parent: 'full_face',       clinical: 'Left upper palpebral region' },
  eye_left_lower:          { label: 'Left Lower Eyelid',          parent: 'full_face',       clinical: 'Left lower palpebral region' },
  eye_right_upper:         { label: 'Right Upper Eyelid',         parent: 'full_face',       clinical: 'Right upper palpebral region' },
  eye_right_lower:         { label: 'Right Lower Eyelid',         parent: 'full_face',       clinical: 'Right lower palpebral region' },
  eye_left_corner_inner:   { label: 'Left Inner Canthus',         parent: 'eye_left_upper',  clinical: 'Left medial canthus' },
  eye_left_corner_outer:   { label: 'Left Outer Canthus',         parent: 'eye_left_upper',  clinical: 'Left lateral canthus' },
  eye_right_corner_inner:  { label: 'Right Inner Canthus',        parent: 'eye_right_upper', clinical: 'Right medial canthus' },
  eye_right_corner_outer:  { label: 'Right Outer Canthus',        parent: 'eye_right_upper', clinical: 'Right lateral canthus' },

  // -- Under-eye --
  under_eye_left:          { label: 'Left Under-Eye',             parent: 'full_face',       clinical: 'Left infraorbital region' },
  under_eye_right:         { label: 'Right Under-Eye',            parent: 'full_face',       clinical: 'Right infraorbital region' },
  tear_trough_left:        { label: 'Left Tear Trough',           parent: 'under_eye_left',  clinical: 'Left nasojugal groove' },
  tear_trough_right:       { label: 'Right Tear Trough',          parent: 'under_eye_right', clinical: 'Right nasojugal groove' },

  // -- Nose --
  nose_bridge:             { label: 'Nose Bridge',                parent: 'full_face',       clinical: 'Nasal dorsum' },
  nose_bridge_upper:       { label: 'Upper Nose Bridge',          parent: 'nose_bridge',     clinical: 'Upper nasal dorsum (nasion area)' },
  nose_bridge_lower:       { label: 'Lower Nose Bridge',          parent: 'nose_bridge',     clinical: 'Lower nasal dorsum (rhinion area)' },
  nose_tip:                { label: 'Nose Tip',                   parent: 'full_face',       clinical: 'Nasal apex / pronasale' },
  nose_tip_left:           { label: 'Nose Tip Left',              parent: 'nose_tip',        clinical: 'Left dome of nasal tip' },
  nose_tip_right:          { label: 'Nose Tip Right',             parent: 'nose_tip',        clinical: 'Right dome of nasal tip' },
  nostril_left:            { label: 'Left Nostril',               parent: 'full_face',       clinical: 'Left alar region' },
  nostril_right:           { label: 'Right Nostril',              parent: 'full_face',       clinical: 'Right alar region' },
  nose_dorsum:             { label: 'Nose Dorsum',                parent: 'nose_bridge',     clinical: 'Full nasal dorsum length' },

  // -- Cheeks --
  cheek_left:              { label: 'Left Cheek',                 parent: 'full_face',       clinical: 'Left buccal region' },
  cheek_right:             { label: 'Right Cheek',                parent: 'full_face',       clinical: 'Right buccal region' },
  cheekbone_left:          { label: 'Left Cheekbone',             parent: 'cheek_left',      clinical: 'Left zygomatic prominence' },
  cheekbone_right:         { label: 'Right Cheekbone',            parent: 'cheek_right',     clinical: 'Right zygomatic prominence' },
  cheek_hollow_left:       { label: 'Left Cheek Hollow',          parent: 'cheek_left',      clinical: 'Left buccal fat pad region' },
  cheek_hollow_right:      { label: 'Right Cheek Hollow',         parent: 'cheek_right',     clinical: 'Right buccal fat pad region' },

  // -- Nasolabial --
  nasolabial_left:         { label: 'Left Nasolabial Fold',       parent: 'full_face',       clinical: 'Left nasolabial sulcus' },
  nasolabial_right:        { label: 'Right Nasolabial Fold',      parent: 'full_face',       clinical: 'Right nasolabial sulcus' },

  // -- Lips --
  lip_upper:               { label: 'Upper Lip',                  parent: 'full_face',       clinical: 'Upper vermilion' },
  lip_upper_left:          { label: 'Upper Lip Left',             parent: 'lip_upper',       clinical: 'Left upper vermilion' },
  lip_upper_right:         { label: 'Upper Lip Right',            parent: 'lip_upper',       clinical: 'Right upper vermilion' },
  lip_upper_center:        { label: 'Cupid\'s Bow',              parent: 'lip_upper',       clinical: 'Philtral columns / cupid\'s bow' },
  lip_lower:               { label: 'Lower Lip',                  parent: 'full_face',       clinical: 'Lower vermilion' },
  lip_lower_left:          { label: 'Lower Lip Left',             parent: 'lip_lower',       clinical: 'Left lower vermilion' },
  lip_lower_right:         { label: 'Lower Lip Right',            parent: 'lip_lower',       clinical: 'Right lower vermilion' },
  lip_lower_center:        { label: 'Lower Lip Center',           parent: 'lip_lower',       clinical: 'Central lower vermilion' },
  lip_corner_left:         { label: 'Left Lip Corner',            parent: 'full_face',       clinical: 'Left oral commissure' },
  lip_corner_right:        { label: 'Right Lip Corner',           parent: 'full_face',       clinical: 'Right oral commissure' },

  // -- Chin --
  chin:                    { label: 'Chin',                       parent: 'full_face',       clinical: 'Mental region' },
  chin_center:             { label: 'Chin Center',                parent: 'chin',            clinical: 'Pogonion / mental protuberance' },
  chin_left:               { label: 'Chin Left',                  parent: 'chin',            clinical: 'Left parasymphyseal region' },
  chin_right:              { label: 'Chin Right',                 parent: 'chin',            clinical: 'Right parasymphyseal region' },

  // -- Jaw --
  jaw_left:                { label: 'Left Jaw',                   parent: 'full_face',       clinical: 'Left mandibular body' },
  jaw_right:               { label: 'Right Jaw',                  parent: 'full_face',       clinical: 'Right mandibular body' },
  jawline_left:            { label: 'Left Jawline',               parent: 'jaw_left',        clinical: 'Left mandibular border' },
  jawline_right:           { label: 'Right Jawline',              parent: 'jaw_right',       clinical: 'Right mandibular border' },

  // -- Temples --
  temple_left:             { label: 'Left Temple',                parent: 'full_face',       clinical: 'Left temporal fossa' },
  temple_right:            { label: 'Right Temple',               parent: 'full_face',       clinical: 'Right temporal fossa' },

  // -- Ears --
  ear_left:                { label: 'Left Ear',                   parent: 'full_face',       clinical: 'Left auricle' },
  ear_right:               { label: 'Right Ear',                  parent: 'full_face',       clinical: 'Right auricle' },

  // -- Neck --
  neck:                    { label: 'Neck',                       parent: null,              clinical: 'Cervical region' },
};

// ---------------------------------------------------------------------------
// Mesh generation helpers
// ---------------------------------------------------------------------------

/**
 * Build a UV sphere and return raw typed arrays.
 *
 * The sphere has its pole axis along +Y, front face along +Z.
 * Returns { positions, uvs, indices, ringCount, segCount }
 *
 * Total vertex count = (rings + 1) * (segments + 1).
 * (We duplicate the seam column for proper UV wrapping.)
 */
function buildBaseSphere(rings, segments, radius) {
  const vertCount = (rings + 1) * (segments + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);

  let vi = 0;
  let ui = 0;

  for (let r = 0; r <= rings; r++) {
    const v = r / rings;                // 0 (top pole) .. 1 (bottom pole)
    const phi = v * Math.PI;            // polar angle
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let s = 0; s <= segments; s++) {
      const u = s / segments;           // 0 .. 1 around
      const theta = u * 2 * Math.PI;   // azimuthal angle

      const x = radius * sinPhi * Math.sin(theta);
      const y = radius * cosPhi;
      const z = radius * sinPhi * Math.cos(theta);

      positions[vi++] = x;
      positions[vi++] = y;
      positions[vi++] = z;

      uvs[ui++] = u;
      uvs[ui++] = 1 - v; // flip V so forehead is UV-top
    }
  }

  // Build quad indices (two triangles per quad), CCW winding
  const quads = rings * segments;
  const indices = new Uint32Array(quads * 6);
  let ii = 0;
  const stride = segments + 1;

  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * stride + s;
      const b = a + stride;
      const c = b + 1;
      const d = a + 1;

      // Triangle 1
      indices[ii++] = a;
      indices[ii++] = b;
      indices[ii++] = d;

      // Triangle 2
      indices[ii++] = d;
      indices[ii++] = b;
      indices[ii++] = c;
    }
  }

  return { positions, uvs, indices, ringCount: rings, segCount: segments, vertCount, stride };
}

// ---------------------------------------------------------------------------
// Face deformation fields
// ---------------------------------------------------------------------------

/**
 * Convert a vertex index back to its (ring, segment) on the UV sphere.
 */
function indexToRS(idx, stride) {
  const r = Math.floor(idx / stride);
  const s = idx % stride;
  return { r, s };
}

/**
 * Parametric (u, v) for a vertex index.  u in [0,1] azimuthal, v in [0,1] polar.
 */
function indexToUV(idx, rings, segments, stride) {
  const { r, s } = indexToRS(idx, stride);
  return { u: s / segments, v: r / rings };
}

/**
 * Apply all face-shaping deformations in-place on the positions array.
 *
 * Coordinate system:
 *   +X  = viewer's left  (subject's right)
 *   +Y  = up
 *   +Z  = toward viewer  (front of face)
 *
 * Strategy: We iterate every vertex once, compute its (u,v) on the sphere,
 * then apply additive displacement in (x,y,z) driven by smooth spatial
 * falloff functions.  This is conceptually similar to blend-shape sculpting.
 */
function applyFaceDeformations(positions, rings, segments, stride, radius) {
  const vertCount = positions.length / 3;

  for (let i = 0; i < vertCount; i++) {
    const { u, v } = indexToUV(i, rings, segments, stride);

    let x = positions[i * 3];
    let y = positions[i * 3 + 1];
    let z = positions[i * 3 + 2];

    // -- Normalised direction from centre --
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    const nx0 = x / len;
    const ny0 = y / len;
    const nz0 = z / len;

    // ---------------------------------------------------------------
    // 1.  Scale sphere into an ovoid head shape (taller, narrower at jaw)
    // ---------------------------------------------------------------
    {
      const yNorm = y / radius;  // -1 .. +1
      // Widen at cheekbone level, narrow at chin
      const widthFactor = 1.0
        + 0.08 * smoothFalloff(0, yNorm, 0, 0.15, 0.5)   // cheekbone width
        - 0.18 * smoothFalloff(0, yNorm, 0, -0.7, 0.45)   // chin narrow
        - 0.06 * smoothFalloff(0, yNorm, 0, 0.85, 0.3);   // cranium slight narrow

      x *= widthFactor;

      // Elongate vertically
      y *= 1.15;

      // Flatten back of head slightly, push front forward
      if (nz0 < 0) {
        z *= 0.88; // flatten back
      } else {
        z *= 1.0 + 0.06 * nz0; // slight front protrusion
      }
    }

    // ---------------------------------------------------------------
    // 2.  Forehead — dome shape
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const ff = smoothFalloff(0, yNorm, 0, 0.75, 0.35);
      const frontBias = clamp(nz0 * 2, 0, 1);
      z += ff * frontBias * radius * 0.14;
      y += ff * radius * 0.04;
    }

    // ---------------------------------------------------------------
    // 3.  Eye sockets — inward depressions
    // ---------------------------------------------------------------
    {
      // Left eye (subject left = +X)
      const eyeLX = 0.31;
      const eyeY  = 0.22;
      const ff_l = ellipseFalloff(nx0, ny0 * (radius * 1.15 / radius), eyeLX, eyeY, 0.14, 0.08);
      z -= ff_l * radius * 0.22;
      y -= ff_l * radius * 0.02;

      // Right eye
      const ff_r = ellipseFalloff(nx0, ny0 * (radius * 1.15 / radius), -eyeLX, eyeY, 0.14, 0.08);
      z -= ff_r * radius * 0.22;
      y -= ff_r * radius * 0.02;
    }

    // ---------------------------------------------------------------
    // 4.  Brow ridge — slight protrusion above eyes
    // ---------------------------------------------------------------
    {
      const browY = 0.35;
      const ff_l = ellipseFalloff(nx0, ny0, 0.25, browY, 0.22, 0.06);
      const ff_r = ellipseFalloff(nx0, ny0, -0.25, browY, 0.22, 0.06);
      const ff = Math.max(ff_l, ff_r);
      const frontBias = clamp(nz0 * 2.5, 0, 1);
      z += ff * frontBias * radius * 0.10;
    }

    // ---------------------------------------------------------------
    // 5.  Nose bridge — elongated ridge down the centre
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      // Bridge runs from between eyes to just above lip
      const bridgeFF = ellipseFalloff(nx0, yNorm, 0, 0.10, 0.07, 0.22);
      const frontBias = clamp(nz0 * 3, 0, 1);
      z += bridgeFF * frontBias * radius * 0.30;

      // Widen slightly at bottom (nasal bone -> cartilage)
      const lowerNose = smoothFalloff(0, yNorm, 0, -0.03, 0.10);
      const sideSpread = Math.abs(nx0) * 0.6;
      z += lowerNose * frontBias * sideSpread * radius * 0.08;
    }

    // ---------------------------------------------------------------
    // 6.  Nose tip — rounded protrusion
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const tipFF = smoothFalloff(nx0, yNorm, 0, -0.05, 0.09);
      const frontBias = clamp(nz0 * 3, 0, 1);
      z += tipFF * frontBias * radius * 0.28;
      y -= tipFF * radius * 0.02; // droop slightly
    }

    // ---------------------------------------------------------------
    // 7.  Nostrils — small lateral bulges
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const nlL = smoothFalloff(nx0, yNorm, 0.08, -0.10, 0.06);
      const nlR = smoothFalloff(nx0, yNorm, -0.08, -0.10, 0.06);
      const frontBias = clamp(nz0 * 2.5, 0, 1);
      z += (nlL + nlR) * frontBias * radius * 0.10;
      // Push nostrils out laterally
      x += nlL * radius * 0.04;
      x -= nlR * radius * 0.04;
    }

    // ---------------------------------------------------------------
    // 8.  Cheekbones — lateral protrusion
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const cbL = ellipseFalloff(nx0, yNorm, 0.55, 0.08, 0.25, 0.14);
      const cbR = ellipseFalloff(nx0, yNorm, -0.55, 0.08, 0.25, 0.14);
      x += cbL * radius * 0.08;
      x -= cbR * radius * 0.08;
      z += (cbL + cbR) * radius * 0.04;
    }

    // ---------------------------------------------------------------
    // 9.  Cheek hollows — slight inward depression below cheekbone
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const chL = ellipseFalloff(nx0, yNorm, 0.42, -0.12, 0.18, 0.10);
      const chR = ellipseFalloff(nx0, yNorm, -0.42, -0.12, 0.18, 0.10);
      z -= (chL + chR) * radius * 0.04;
      x -= chL * radius * 0.03;
      x += chR * radius * 0.03;
    }

    // ---------------------------------------------------------------
    // 10. Lips — protruding ridge with cupid's bow
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const frontBias = clamp(nz0 * 3, 0, 1);

      // Upper lip
      const ulFF = ellipseFalloff(nx0, yNorm, 0, -0.20, 0.16, 0.04);
      z += ulFF * frontBias * radius * 0.14;

      // Cupid's bow — two peaks
      const cbL = smoothFalloff(nx0, yNorm, 0.04, -0.185, 0.03);
      const cbR = smoothFalloff(nx0, yNorm, -0.04, -0.185, 0.03);
      z += (cbL + cbR) * frontBias * radius * 0.06;
      y -= (cbL + cbR) * radius * 0.005;

      // Philtrum dip (between the two cupid bow peaks)
      const philtrumFF = smoothFalloff(nx0, yNorm, 0, -0.185, 0.02);
      z -= philtrumFF * frontBias * radius * 0.03;

      // Lower lip — slightly fuller
      const llFF = ellipseFalloff(nx0, yNorm, 0, -0.27, 0.14, 0.04);
      z += llFF * frontBias * radius * 0.12;

      // Lip separation groove
      const sepFF = ellipseFalloff(nx0, yNorm, 0, -0.235, 0.13, 0.015);
      z -= sepFF * frontBias * radius * 0.06;
    }

    // ---------------------------------------------------------------
    // 11. Chin — forward protrusion, rounded
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const chinFF = smoothFalloff(nx0, yNorm, 0, -0.52, 0.18);
      const frontBias = clamp(nz0 * 2.5, 0, 1);
      z += chinFF * frontBias * radius * 0.12;
      y -= chinFF * radius * 0.03;
    }

    // ---------------------------------------------------------------
    // 12. Jaw angle — slight lateral bulge at mandibular angle
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const jawL = ellipseFalloff(nx0, yNorm, 0.55, -0.42, 0.15, 0.12);
      const jawR = ellipseFalloff(nx0, yNorm, -0.55, -0.42, 0.15, 0.12);
      x += jawL * radius * 0.06;
      x -= jawR * radius * 0.06;
    }

    // ---------------------------------------------------------------
    // 13. Temple concavity
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const tL = ellipseFalloff(nx0, yNorm, 0.62, 0.38, 0.15, 0.14);
      const tR = ellipseFalloff(nx0, yNorm, -0.62, 0.38, 0.15, 0.14);
      z -= (tL + tR) * radius * 0.04;
      x -= tL * radius * 0.04;
      x += tR * radius * 0.04;
    }

    // ---------------------------------------------------------------
    // 14. Nasolabial folds — subtle crease from nose wing to mouth corner
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const nlL = ellipseFalloff(nx0, yNorm, 0.16, -0.15, 0.04, 0.14);
      const nlR = ellipseFalloff(nx0, yNorm, -0.16, -0.15, 0.04, 0.14);
      const frontBias = clamp(nz0 * 2.5, 0, 1);
      z -= (nlL + nlR) * frontBias * radius * 0.03;
    }

    // ---------------------------------------------------------------
    // 15. Ear placeholders — small lateral bumps
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const earL = ellipseFalloff(nx0, yNorm, 0.92, 0.18, 0.12, 0.18);
      const earR = ellipseFalloff(nx0, yNorm, -0.92, 0.18, 0.12, 0.18);
      x += earL * radius * 0.14;
      x -= earR * radius * 0.14;
      z -= (earL + earR) * radius * 0.03; // push back
    }

    // ---------------------------------------------------------------
    // 16. Under-eye / tear-trough depression
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      const frontBias = clamp(nz0 * 2.5, 0, 1);
      const ueL = ellipseFalloff(nx0, yNorm, 0.22, 0.10, 0.12, 0.05);
      const ueR = ellipseFalloff(nx0, yNorm, -0.22, 0.10, 0.12, 0.05);
      z -= (ueL + ueR) * frontBias * radius * 0.04;
    }

    // ---------------------------------------------------------------
    // 17. Neck — taper below jaw
    // ---------------------------------------------------------------
    {
      const yNorm = y / (radius * 1.15);
      if (yNorm < -0.65) {
        const t = clamp((yNorm - (-0.65)) / (-1.0 - (-0.65)), 0, 1);
        const taper = lerp(1.0, 0.72, t);
        x *= taper;
        z *= taper;
      }
    }

    positions[i * 3]     = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
  }
}

// ---------------------------------------------------------------------------
// Region classification
// ---------------------------------------------------------------------------

/**
 * Classify each vertex into zero-or-more named regions.
 *
 * Classification is based on the *deformed* vertex position in (nx, ny)
 * normalised direction space (front-facing hemisphere).
 *
 * Returns Record<string, number[]>.
 */
function classifyVertices(positions, rings, segments, stride, radius) {
  const vertCount = positions.length / 3;

  /** @type {Record<string, number[]>} */
  const regions = {};
  for (const name of Object.keys(REGION_META_DEFS)) {
    regions[name] = [];
  }

  for (let i = 0; i < vertCount; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    const nx = x / len;
    const ny = y / len;
    const nz = z / len;
    const yNorm = y / (radius * 1.15); // normalised height

    // We only classify front-facing vertices for most facial zones
    const isFront = nz > -0.1;
    const isFrontStrong = nz > 0.15;

    // ---- full_face: everything above neck that is front-facing ----
    if (isFront && yNorm > -0.65) {
      regions.full_face.push(i);
    }

    // ---- Forehead ----
    if (isFrontStrong && yNorm > 0.38 && yNorm < 0.92) {
      regions.forehead.push(i);

      if (nx > 0.08) regions.forehead_left.push(i);
      else if (nx < -0.08) regions.forehead_right.push(i);
      else regions.forehead_center.push(i);
    }

    // ---- Brows ----
    if (isFrontStrong && yNorm > 0.28 && yNorm < 0.42) {
      if (nx > 0.08 && nx < 0.50) {
        regions.brow_left.push(i);
        if (nx < 0.22) regions.brow_inner_left.push(i);
      }
      if (nx < -0.08 && nx > -0.50) {
        regions.brow_right.push(i);
        if (nx > -0.22) regions.brow_inner_right.push(i);
      }
    }

    // ---- Eyes ----
    const eyeLX = 0.31;
    const eyeRX = -0.31;
    const eyeY = 0.22;

    // Left eye
    {
      const edx = nx - eyeLX;
      const edy = (yNorm - eyeY);
      const ed = Math.sqrt(edx * edx + (edy * edy) * 4); // squash vertically

      if (isFrontStrong && ed < 0.18) {
        if (edy > 0) regions.eye_left_upper.push(i);
        else regions.eye_left_lower.push(i);

        if (edx < -0.06 && Math.abs(edy) < 0.06) regions.eye_left_corner_inner.push(i);
        if (edx > 0.06 && Math.abs(edy) < 0.06) regions.eye_left_corner_outer.push(i);
      }
    }

    // Right eye
    {
      const edx = nx - eyeRX;
      const edy = (yNorm - eyeY);
      const ed = Math.sqrt(edx * edx + (edy * edy) * 4);

      if (isFrontStrong && ed < 0.18) {
        if (edy > 0) regions.eye_right_upper.push(i);
        else regions.eye_right_lower.push(i);

        if (edx > 0.06 && Math.abs(edy) < 0.06) regions.eye_right_corner_inner.push(i);
        if (edx < -0.06 && Math.abs(edy) < 0.06) regions.eye_right_corner_outer.push(i);
      }
    }

    // ---- Under-eye ----
    if (isFrontStrong && yNorm > 0.05 && yNorm < 0.20) {
      if (nx > 0.10 && nx < 0.48) {
        regions.under_eye_left.push(i);
        if (nx < 0.28 && yNorm > 0.10) regions.tear_trough_left.push(i);
      }
      if (nx < -0.10 && nx > -0.48) {
        regions.under_eye_right.push(i);
        if (nx > -0.28 && yNorm > 0.10) regions.tear_trough_right.push(i);
      }
    }

    // ---- Nose bridge ----
    if (isFrontStrong && Math.abs(nx) < 0.10 && yNorm > -0.05 && yNorm < 0.30) {
      regions.nose_bridge.push(i);
      regions.nose_dorsum.push(i);
      if (yNorm > 0.15) regions.nose_bridge_upper.push(i);
      else regions.nose_bridge_lower.push(i);
    }

    // ---- Nose tip ----
    if (isFrontStrong && Math.abs(nx) < 0.12 && yNorm > -0.12 && yNorm < 0.02) {
      regions.nose_tip.push(i);
      if (nx > 0.02) regions.nose_tip_left.push(i);
      if (nx < -0.02) regions.nose_tip_right.push(i);
    }

    // ---- Nostrils ----
    if (isFrontStrong && yNorm > -0.15 && yNorm < -0.02) {
      if (nx > 0.05 && nx < 0.18) regions.nostril_left.push(i);
      if (nx < -0.05 && nx > -0.18) regions.nostril_right.push(i);
    }

    // ---- Cheeks ----
    if (isFront && yNorm > -0.30 && yNorm < 0.15) {
      if (nx > 0.22) {
        regions.cheek_left.push(i);
        if (yNorm > -0.05) regions.cheekbone_left.push(i);
        else regions.cheek_hollow_left.push(i);
      }
      if (nx < -0.22) {
        regions.cheek_right.push(i);
        if (yNorm > -0.05) regions.cheekbone_right.push(i);
        else regions.cheek_hollow_right.push(i);
      }
    }

    // ---- Nasolabial folds ----
    if (isFrontStrong && yNorm > -0.30 && yNorm < -0.02) {
      if (nx > 0.10 && nx < 0.22) regions.nasolabial_left.push(i);
      if (nx < -0.10 && nx > -0.22) regions.nasolabial_right.push(i);
    }

    // ---- Lips ----
    const lipY = -0.235;
    const lipHalfH = 0.065;

    if (isFrontStrong && Math.abs(nx) < 0.22 && yNorm > lipY - lipHalfH && yNorm < lipY + lipHalfH) {
      // Upper lip
      if (yNorm > lipY) {
        regions.lip_upper.push(i);
        if (nx > 0.04) regions.lip_upper_left.push(i);
        else if (nx < -0.04) regions.lip_upper_right.push(i);
        else regions.lip_upper_center.push(i);
      }
      // Lower lip
      if (yNorm <= lipY) {
        regions.lip_lower.push(i);
        if (nx > 0.04) regions.lip_lower_left.push(i);
        else if (nx < -0.04) regions.lip_lower_right.push(i);
        else regions.lip_lower_center.push(i);
      }
      // Lip corners
      if (Math.abs(yNorm - lipY) < 0.03) {
        if (nx > 0.14) regions.lip_corner_left.push(i);
        if (nx < -0.14) regions.lip_corner_right.push(i);
      }
    }

    // ---- Chin ----
    if (isFrontStrong && yNorm > -0.62 && yNorm < -0.35 && Math.abs(nx) < 0.35) {
      regions.chin.push(i);
      if (Math.abs(nx) < 0.10) regions.chin_center.push(i);
      else if (nx > 0) regions.chin_left.push(i);
      else regions.chin_right.push(i);
    }

    // ---- Jaw ----
    if (isFront && yNorm > -0.58 && yNorm < -0.20) {
      if (nx > 0.30) {
        regions.jaw_left.push(i);
        if (yNorm < -0.35) regions.jawline_left.push(i);
      }
      if (nx < -0.30) {
        regions.jaw_right.push(i);
        if (yNorm < -0.35) regions.jawline_right.push(i);
      }
    }

    // ---- Temples ----
    if (isFront && yNorm > 0.20 && yNorm < 0.55) {
      if (nx > 0.45) regions.temple_left.push(i);
      if (nx < -0.45) regions.temple_right.push(i);
    }

    // ---- Ears ----
    if (yNorm > -0.05 && yNorm < 0.40) {
      if (nx > 0.80) regions.ear_left.push(i);
      if (nx < -0.80) regions.ear_right.push(i);
    }

    // ---- Neck ----
    if (yNorm < -0.62) {
      regions.neck.push(i);
    }
  }

  return regions;
}

// ---------------------------------------------------------------------------
// Normal computation
// ---------------------------------------------------------------------------

/**
 * Compute smooth vertex normals from indexed triangle geometry.
 * Returns a Float32Array of normals (same length as positions).
 */
function computeNormals(positions, indices) {
  const normals = new Float32Array(positions.length);

  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const pC = new THREE.Vector3();
  const ab = new THREE.Vector3();
  const ac = new THREE.Vector3();
  const faceNormal = new THREE.Vector3();

  for (let t = 0; t < indices.length; t += 3) {
    const ia = indices[t];
    const ib = indices[t + 1];
    const ic = indices[t + 2];

    pA.set(positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2]);
    pB.set(positions[ib * 3], positions[ib * 3 + 1], positions[ib * 3 + 2]);
    pC.set(positions[ic * 3], positions[ic * 3 + 1], positions[ic * 3 + 2]);

    ab.subVectors(pB, pA);
    ac.subVectors(pC, pA);
    faceNormal.crossVectors(ab, ac);
    // Don't normalise — area weighting is desirable

    normals[ia * 3]     += faceNormal.x;
    normals[ia * 3 + 1] += faceNormal.y;
    normals[ia * 3 + 2] += faceNormal.z;

    normals[ib * 3]     += faceNormal.x;
    normals[ib * 3 + 1] += faceNormal.y;
    normals[ib * 3 + 2] += faceNormal.z;

    normals[ic * 3]     += faceNormal.x;
    normals[ic * 3 + 1] += faceNormal.y;
    normals[ic * 3 + 2] += faceNormal.z;
  }

  // Normalise each vertex normal
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i];
    const ny = normals[i + 1];
    const nz = normals[i + 2];
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
    normals[i]     = nx / len;
    normals[i + 1] = ny / len;
    normals[i + 2] = nz / len;
  }

  return normals;
}

// ---------------------------------------------------------------------------
// FLAME UV re-mapping
// ---------------------------------------------------------------------------

/**
 * Remap the default spherical UVs to a FLAME-style flat face UV layout.
 *
 * Standard FLAME UV layout places the face unwrapped roughly flat in the
 * centre of the UV square, with the back of the head filling the remaining
 * area.  We approximate this by projecting front-facing vertices onto an
 * orthographic XY plane, and wrapping back-facing vertices around the
 * periphery.
 */
function remapUVs(positions, uvs, radius) {
  const vertCount = positions.length / 3;

  for (let i = 0; i < vertCount; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const len = Math.sqrt(x * x + y * y + z * z) || 1;
    const nz = z / len;

    if (nz > 0) {
      // Front-facing: orthographic projection, centred in UV space
      // Map x ∈ [-radius, radius] -> u ∈ [0.15, 0.85]
      // Map y ∈ [-radius*1.15, radius*1.15] -> v ∈ [0.10, 0.90]
      const uNew = 0.15 + 0.70 * clamp((x / (radius * 1.1) + 1) * 0.5, 0, 1);
      const vNew = 0.10 + 0.80 * clamp((y / (radius * 1.3) + 1) * 0.5, 0, 1);
      uvs[i * 2]     = uNew;
      uvs[i * 2 + 1] = vNew;
    }
    // else: keep original spherical UV for back of head
  }
}

// ---------------------------------------------------------------------------
// FLAME data helpers
// ---------------------------------------------------------------------------

/**
 * Generate simple spherical UVs for FLAME topology vertices.
 * Projects vertex positions onto a UV sphere for texture mapping when
 * the FLAME model does not include its own UV coordinates.
 *
 * @param {Float32Array} positions - Vertex positions (vertexCount * 3).
 * @returns {Float32Array} UV coordinates (vertexCount * 2).
 */
function generateFlameUVs(positions) {
  const vertCount = positions.length / 3;
  const uvs = new Float32Array(vertCount * 2);

  for (let i = 0; i < vertCount; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    const len = Math.sqrt(x * x + y * y + z * z) || 1;

    // Spherical projection
    const nx = x / len;
    const ny = y / len;
    const nz = z / len;

    // u from atan2(x, z), v from asin(y)
    const u = 0.5 + Math.atan2(nx, nz) / (2 * Math.PI);
    const v = 0.5 + Math.asin(clamp(ny, -1, 1)) / Math.PI;

    uvs[i * 2]     = u;
    uvs[i * 2 + 1] = v;
  }

  return uvs;
}

/**
 * Fetch a binary file and return it as an ArrayBuffer.
 * @param {string} url
 * @returns {Promise<ArrayBuffer>}
 */
async function fetchBinary(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

/**
 * Fetch a JSON file and return the parsed object.
 * @param {string} url
 * @returns {Promise<any>}
 */
async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// FlameMeshGenerator class
// ---------------------------------------------------------------------------

export class FlameMeshGenerator {
  /**
   * @param {object} [options]
   * @param {number} [options.rings=52]     - Latitude subdivisions.
   * @param {number} [options.segments=52]  - Longitude subdivisions.
   * @param {number} [options.radius=0.085] - Base sphere radius (metres).
   */
  constructor(options = {}) {
    /** @private */ this._rings    = options.rings    ?? DEFAULT_RINGS;
    /** @private */ this._segments = options.segments ?? DEFAULT_SEGMENTS;
    /** @private */ this._radius   = options.radius   ?? BASE_RADIUS;

    /** @private @type {THREE.BufferGeometry | null} */
    this._geometry = null;
    /** @private @type {Record<string, number[]> | null} */
    this._regions = null;
    /** @private @type {Record<string, {label:string, parent:string|null, clinical:string}> | null} */
    this._regionMeta = null;

    // ----- FLAME model data (populated by loadFLAME) -----

    /** @private Whether real FLAME data has been loaded. */
    this._flameLoaded = false;
    /** @private FLAME template metadata from JSON. */
    this._flameMeta = null;
    /** @private Float32Array — original template vertex positions (vertexCount * 3). */
    this._flameTemplateVertices = null;
    /** @private Float32Array — shape PCA basis (vertexCount * 3 * shapeComponents). */
    this._flameShapeBasis = null;
    /** @private Float32Array — expression PCA basis (vertexCount * 3 * expressionComponents). */
    this._flameExpressionBasis = null;
    /** @private Uint32Array — face indices (faceCount * 3). */
    this._flameFaces = null;
    /** @private Region vertex index map from flame_regions.json. */
    this._flameRegions = null;

    /** @private Float32Array — current shaped vertex positions (after shape params applied). */
    this._flameShapedVertices = null;
    /** @private Float32Array — current final vertex positions (after shape + expression). */
    this._flameCurrentVertices = null;
    /** @private Current shape parameters. */
    this._currentShapeParams = null;
    /** @private Current expression parameters. */
    this._currentExpressionParams = null;
  }

  // -----------------------------------------------------------------------
  // Public API — Analytic generation (original)
  // -----------------------------------------------------------------------

  /**
   * Generate the FLAME-compatible face mesh.
   *
   * @returns {{
   *   geometry: THREE.BufferGeometry,
   *   regions: Record<string, number[]>,
   *   regionMeta: Record<string, {label: string, parent: string|null, clinical: string}>
   * }}
   */
  generate() {
    const { _rings: rings, _segments: segments, _radius: radius } = this;

    // 1. Build UV sphere base
    const { positions, uvs, indices, stride, vertCount } = buildBaseSphere(rings, segments, radius);

    // 2. Apply face deformations
    applyFaceDeformations(positions, rings, segments, stride, radius);

    // 3. Remap UVs to FLAME-style layout
    remapUVs(positions, uvs, radius);

    // 4. Compute smooth normals
    const normals = computeNormals(positions, indices);

    // 5. Build THREE.BufferGeometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('normal',   new THREE.BufferAttribute(normals, 3));
    geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    // 6. Classify vertices into named regions
    const regions = classifyVertices(positions, rings, segments, stride, radius);

    // 7. Deep-copy region metadata
    const regionMeta = {};
    for (const [name, def] of Object.entries(REGION_META_DEFS)) {
      regionMeta[name] = { ...def };
    }

    // Cache for later queries
    this._geometry   = geometry;
    this._regions    = regions;
    this._regionMeta = regionMeta;

    return { geometry, regions, regionMeta };
  }

  // -----------------------------------------------------------------------
  // Public API — FLAME model loading
  // -----------------------------------------------------------------------

  /**
   * Load pre-converted FLAME model data from binary/JSON web files.
   *
   * Fetches the template mesh, PCA shape/expression bases, face indices,
   * and region vertex map.  Constructs a Three.js BufferGeometry from the
   * real FLAME topology and returns the same { geometry, regions, regionMeta }
   * shape as `generate()`.
   *
   * If any fetch fails, falls back to the analytic `generate()` method and
   * logs a warning to the console.
   *
   * @param {string} [basePath='/models/flame/web'] - URL path to the directory
   *   containing the pre-converted FLAME web files.
   * @returns {Promise<{
   *   geometry: THREE.BufferGeometry,
   *   regions: Record<string, number[]>,
   *   regionMeta: Record<string, {label: string, parent: string|null, clinical: string}>
   * }>}
   */
  async loadFLAME(basePath = '/models/flame/web') {
    // Normalise path: strip trailing slash
    const base = basePath.replace(/\/+$/, '');

    try {
      // Fetch all files in parallel
      const [
        templateMeta,
        verticesBuf,
        shapeBasisBuf,
        expressionBasisBuf,
        facesBuf,
        regionsData,
      ] = await Promise.all([
        fetchJSON(`${base}/flame_template.json`),
        fetchBinary(`${base}/flame_template_vertices.bin`),
        fetchBinary(`${base}/flame_shape_basis.bin`),
        fetchBinary(`${base}/flame_expression_basis.bin`),
        fetchBinary(`${base}/flame_faces.bin`),
        fetchJSON(`${base}/flame_regions.json`),
      ]);

      // Validate template metadata (support both snake_case and camelCase field names)
      const vertexCount = templateMeta.vertex_count ?? templateMeta.vertexCount ?? FLAME_VERTEX_COUNT;
      const faceCount = templateMeta.face_count ?? templateMeta.faceCount ?? FLAME_FACE_COUNT;
      const shapeComponents = templateMeta.shape_param_count ?? templateMeta.shapeComponents ?? FLAME_SHAPE_COMPONENTS;
      const expressionComponents = templateMeta.expression_param_count ?? templateMeta.expressionComponents ?? FLAME_EXPRESSION_COMPONENTS;

      // Wrap ArrayBuffers into typed arrays
      const templateVertices = new Float32Array(verticesBuf);
      const shapeBasis = new Float32Array(shapeBasisBuf);
      const expressionBasis = new Float32Array(expressionBasisBuf);
      const faces = new Uint32Array(facesBuf);

      // Validate sizes
      const expectedVertSize = vertexCount * 3;
      if (templateVertices.length !== expectedVertSize) {
        throw new Error(
          `FLAME vertex data size mismatch: expected ${expectedVertSize} floats, got ${templateVertices.length}`
        );
      }
      const expectedShapeSize = vertexCount * 3 * shapeComponents;
      if (shapeBasis.length !== expectedShapeSize) {
        throw new Error(
          `FLAME shape basis size mismatch: expected ${expectedShapeSize} floats, got ${shapeBasis.length}`
        );
      }
      const expectedExprSize = vertexCount * 3 * expressionComponents;
      if (expressionBasis.length !== expectedExprSize) {
        throw new Error(
          `FLAME expression basis size mismatch: expected ${expectedExprSize} floats, got ${expressionBasis.length}`
        );
      }
      const expectedFaceSize = faceCount * 3;
      if (faces.length !== expectedFaceSize) {
        throw new Error(
          `FLAME face index size mismatch: expected ${expectedFaceSize} uints, got ${faces.length}`
        );
      }

      // After the main Promise.all, try loading texture data (non-critical, can fail gracefully)
      let uvCoordsBuf = null;
      let uvFacesBuf = null;
      let albedoDiffuseBuf = null;
      let albedoSpecularBuf = null;

      try {
        [uvCoordsBuf, uvFacesBuf] = await Promise.all([
          fetchBinary(`${base}/flame_uv.bin`),
          fetchBinary(`${base}/flame_uv_faces.bin`),
        ]);
        console.log('FlameMeshGenerator: UV mapping loaded');
      } catch (e) {
        console.warn('FlameMeshGenerator: UV mapping not available, using spherical UVs');
      }

      try {
        [albedoDiffuseBuf, albedoSpecularBuf] = await Promise.all([
          fetchBinary(`${base}/flame_albedo_diffuse.bin`),
          fetchBinary(`${base}/flame_albedo_specular.bin`),
        ]);
        console.log('FlameMeshGenerator: Albedo textures loaded');
      } catch (e) {
        console.warn('FlameMeshGenerator: Albedo textures not available');
      }

      // Store FLAME data internally
      this._flameMeta = {
        vertexCount,
        faceCount,
        shapeComponents,
        expressionComponents,
        metadata: templateMeta.metadata ?? {},
      };
      this._flameTemplateVertices = templateVertices;
      this._flameShapeBasis = shapeBasis;
      this._flameExpressionBasis = expressionBasis;
      this._flameFaces = faces;
      this._flameRegions = regionsData;

      // Store UV mapping data if available
      if (uvCoordsBuf && uvFacesBuf) {
        this._flameUVCoords = new Float32Array(uvCoordsBuf);
        this._flameUVFaces = new Uint32Array(uvFacesBuf);
        // Build position-to-UV vertex mapping
        const uvVertexCount = this._flameUVCoords.length / 2;
        this._flameUVVertexCount = uvVertexCount;
        // Map: for each UV vertex, which position vertex it corresponds to
        this._uvToPosMap = new Uint32Array(uvVertexCount);
        for (let f = 0; f < faceCount; f++) {
          for (let c = 0; c < 3; c++) {
            const posIdx = faces[f * 3 + c];
            const uvIdx = this._flameUVFaces[f * 3 + c];
            this._uvToPosMap[uvIdx] = posIdx;
          }
        }
      }

      // Store albedo texture data
      if (albedoDiffuseBuf) {
        this._flameAlbedoDiffuse = new Uint8Array(albedoDiffuseBuf);
      }
      if (albedoSpecularBuf) {
        this._flameAlbedoSpecular = new Uint8Array(albedoSpecularBuf);
      }
      this._flameTemplateMeta = templateMeta;

      // Create working copies of vertex data
      this._flameShapedVertices = new Float32Array(templateVertices);
      this._flameCurrentVertices = new Float32Array(templateVertices);

      // Reset current parameters
      this._currentShapeParams = new Float32Array(shapeComponents);
      this._currentExpressionParams = new Float32Array(expressionComponents);

      // Build the Three.js geometry from FLAME data
      const result = this._buildFlameGeometry();

      this._flameLoaded = true;

      return result;

    } catch (err) {
      console.warn(
        'FlameMeshGenerator: Failed to load FLAME model data, falling back to analytic generation.',
        err
      );
      return this.generate();
    }
  }

  // -----------------------------------------------------------------------
  // Public API — FLAME parameter application
  // -----------------------------------------------------------------------

  /**
   * Apply FLAME shape parameters via PCA deformation.
   *
   * Computes: v_shaped = v_template + shapedirs * params
   *
   * Updates the Three.js geometry positions in-place and stores the current
   * shape parameters for later use (e.g. when expression params are applied
   * on top).
   *
   * @param {number[]} params - Array of floats (up to 50 values, matching
   *   the first N shape PCA components).  Values beyond the number of
   *   available components are ignored.
   */
  applyShapeParams(params) {
    if (!this._flameLoaded) {
      throw new Error('FlameMeshGenerator: call loadFLAME() before applying shape parameters.');
    }

    const { vertexCount, shapeComponents } = this._flameMeta;
    const templateVerts = this._flameTemplateVertices;
    const shapeBasis = this._flameShapeBasis;
    const shapedVerts = this._flameShapedVertices;

    // Determine number of components to apply
    const numParams = Math.min(params.length, shapeComponents);

    // Store current params (zero-padded)
    this._currentShapeParams = new Float32Array(shapeComponents);
    for (let p = 0; p < numParams; p++) {
      this._currentShapeParams[p] = params[p];
    }

    // v_shaped = v_template + shapedirs * params
    // shapeBasis is laid out as: [vertex0_x_comp0, vertex0_x_comp1, ..., vertex0_y_comp0, ...]
    // Actually stored as row-major: basis[v*3*C + coord*C + comp]
    // where v = vertex index, coord = 0/1/2 (x/y/z), comp = component index
    // Equivalently: basis[(v*3 + coord) * C + comp]
    const V3 = vertexCount * 3;

    // Start from template
    shapedVerts.set(templateVerts);

    // Add shape deformation
    for (let i = 0; i < V3; i++) {
      let delta = 0;
      const basisOffset = i * shapeComponents;
      for (let p = 0; p < numParams; p++) {
        delta += shapeBasis[basisOffset + p] * params[p];
      }
      shapedVerts[i] += delta;
    }

    // Also re-apply expression params if they exist
    const currentVerts = this._flameCurrentVertices;
    currentVerts.set(shapedVerts);

    if (this._currentExpressionParams) {
      const exprBasis = this._flameExpressionBasis;
      const { expressionComponents } = this._flameMeta;
      const numExpr = Math.min(this._currentExpressionParams.length, expressionComponents);
      let hasNonZero = false;
      for (let p = 0; p < numExpr; p++) {
        if (this._currentExpressionParams[p] !== 0) { hasNonZero = true; break; }
      }
      if (hasNonZero) {
        for (let i = 0; i < V3; i++) {
          let delta = 0;
          const basisOffset = i * expressionComponents;
          for (let p = 0; p < numExpr; p++) {
            delta += exprBasis[basisOffset + p] * this._currentExpressionParams[p];
          }
          currentVerts[i] += delta;
        }
      }
    }

    // Update Three.js geometry in-place
    this._updateGeometryPositions(currentVerts);
  }

  /**
   * Apply FLAME expression parameters via PCA deformation.
   *
   * Computes: v_expressed = v_shaped + exprdirs * params
   *
   * The deformation is applied on top of the current shaped vertices
   * (after applyShapeParams), not on top of the raw template.  Updates
   * geometry positions in-place.
   *
   * @param {number[]} params - Array of floats (up to 50 values, matching
   *   the first N expression PCA components).
   */
  applyExpressionParams(params) {
    if (!this._flameLoaded) {
      throw new Error('FlameMeshGenerator: call loadFLAME() before applying expression parameters.');
    }

    const { vertexCount, expressionComponents } = this._flameMeta;
    const shapedVerts = this._flameShapedVertices;
    const exprBasis = this._flameExpressionBasis;
    const currentVerts = this._flameCurrentVertices;

    // Determine number of components to apply
    const numParams = Math.min(params.length, expressionComponents);

    // Store current params (zero-padded)
    this._currentExpressionParams = new Float32Array(expressionComponents);
    for (let p = 0; p < numParams; p++) {
      this._currentExpressionParams[p] = params[p];
    }

    // v_expressed = v_shaped + exprdirs * params
    const V3 = vertexCount * 3;

    // Start from shaped vertices
    currentVerts.set(shapedVerts);

    // Add expression deformation
    for (let i = 0; i < V3; i++) {
      let delta = 0;
      const basisOffset = i * expressionComponents;
      for (let p = 0; p < numParams; p++) {
        delta += exprBasis[basisOffset + p] * params[p];
      }
      currentVerts[i] += delta;
    }

    // Update Three.js geometry in-place
    this._updateGeometryPositions(currentVerts);
  }

  /**
   * Apply DECA reconstruction output to the FLAME mesh.
   *
   * DECA provides shape parameters (identity), expression parameters,
   * and optionally pose parameters.  This method applies shape then
   * expression in sequence.
   *
   * @param {object} decaParams
   * @param {number[]} decaParams.shape_params - Shape PCA coefficients.
   * @param {number[]} decaParams.expression_params - Expression PCA coefficients.
   * @param {number[]} [decaParams.pose_params] - Pose parameters (currently unused;
   *   reserved for future jaw rotation / global pose support).
   * @returns {THREE.BufferGeometry} The updated geometry.
   */
  setFromDECA(decaParams) {
    if (!this._flameLoaded) {
      throw new Error('FlameMeshGenerator: call loadFLAME() before applying DECA parameters.');
    }

    const { shape_params, expression_params } = decaParams;

    // Apply shape first, then expression
    if (shape_params && shape_params.length > 0) {
      this.applyShapeParams(shape_params);
    }
    if (expression_params && expression_params.length > 0) {
      this.applyExpressionParams(expression_params);
    }

    // pose_params reserved for future use (jaw rotation, global pose)
    // TODO: Implement pose parameter support when needed

    return this._geometry;
  }

  /**
   * Get the valid range and human-readable name for a shape parameter index.
   *
   * Useful for building UI sliders that let users explore shape space.
   * The range is a reasonable default for FLAME PCA components; actual
   * values can exceed these bounds but results may become unrealistic.
   *
   * @param {number} index - Shape parameter index (0-49).
   * @returns {{ min: number, max: number, name: string }}
   */
  getShapeParamRange(index) {
    const maxIndex = this._flameMeta
      ? this._flameMeta.shapeComponents - 1
      : FLAME_SHAPE_COMPONENTS - 1;

    const clampedIndex = clamp(Math.floor(index), 0, maxIndex);

    // PCA components have decreasing variance; earlier components need wider range.
    // First few components (overall size, width, height) need ~[-3, 3].
    // Later components have smaller effect and [-2, 2] is usually sufficient.
    const min = clampedIndex < 10 ? -3.0 : -2.0;
    const max = clampedIndex < 10 ?  3.0 :  2.0;

    const name = clampedIndex < SHAPE_PARAM_NAMES.length
      ? SHAPE_PARAM_NAMES[clampedIndex]
      : `Shape Component ${clampedIndex}`;

    return { min, max, name };
  }

  // -----------------------------------------------------------------------
  // Public API — Region queries (original)
  // -----------------------------------------------------------------------

  /**
   * Get the vertex indices for a named region.
   *
   * @param {string} regionName - One of the 52 clinical zone names.
   * @returns {number[]} Array of vertex indices (empty if region unknown).
   */
  getRegionVertices(regionName) {
    if (!this._regions) {
      throw new Error('FlameMeshGenerator: call generate() or loadFLAME() before querying regions.');
    }
    return this._regions[regionName] ?? [];
  }

  /**
   * Get all direct child region names for a given region.
   *
   * @param {string} regionName - Parent region name.
   * @returns {string[]} Array of child region names.
   */
  getRegionChildren(regionName) {
    if (!this._regionMeta) {
      throw new Error('FlameMeshGenerator: call generate() or loadFLAME() before querying regions.');
    }
    const children = [];
    for (const [name, meta] of Object.entries(this._regionMeta)) {
      if (meta.parent === regionName) {
        children.push(name);
      }
    }
    return children;
  }

  // -----------------------------------------------------------------------
  // Convenience getters
  // -----------------------------------------------------------------------

  /** @returns {THREE.BufferGeometry | null} The generated geometry, or null if not yet generated. */
  get geometry() {
    return this._geometry;
  }

  /** @returns {Record<string, number[]> | null} All region vertex-index arrays. */
  get regions() {
    return this._regions;
  }

  /** @returns {Record<string, {label:string, parent:string|null, clinical:string}> | null} Region metadata. */
  get regionMeta() {
    return this._regionMeta;
  }

  /** @returns {number} Total vertex count of the generated mesh. */
  get vertexCount() {
    if (!this._geometry) return 0;
    return this._geometry.getAttribute('position').count;
  }

  /** @returns {boolean} True if real FLAME model data has been loaded via loadFLAME(). */
  get isFlameLoaded() {
    return this._flameLoaded;
  }

  /** @returns {Uint8Array|null} Raw diffuse albedo texture data (512x512 RGB uint8) */
  get albedoDiffuseData() {
    return this._flameAlbedoDiffuse ?? null;
  }

  /** @returns {Uint8Array|null} Raw specular albedo texture data (512x512 RGB uint8) */
  get albedoSpecularData() {
    return this._flameAlbedoSpecular ?? null;
  }

  /** @returns {boolean} Whether albedo textures are loaded */
  get hasAlbedo() {
    return !!this._flameAlbedoDiffuse;
  }

  /** @returns {object|null} Template metadata from flame_template.json */
  get flameMeta() {
    return this._flameTemplateMeta ?? null;
  }

  /** @returns {Float32Array|null} FLAME UV coordinates (5118×2 floats, u/v pairs in 0-1). */
  get flameUVCoords() {
    return this._flameUVCoords ?? null;
  }

  /** @returns {Uint32Array|null} FLAME UV face indices (9976×3, into UV vertex space). */
  get flameUVFaces() {
    return this._flameUVFaces ?? null;
  }

  /** @returns {Uint32Array|null} FLAME 3D face indices (9976×3, into position vertex space). */
  get flameFaces() {
    return this._flameFaces ?? null;
  }

  /** @returns {Uint32Array|null} Map from UV vertex index → position vertex index. */
  get uvToPosMap() {
    return this._uvToPosMap ?? null;
  }

  /** @returns {number} Number of UV vertices (5118 when UV data is loaded). */
  get flameUVVertexCount() {
    return this._flameUVVertexCount ?? 0;
  }

  /** @returns {string[]} All 52 region names. */
  static get regionNames() {
    return Object.keys(REGION_META_DEFS);
  }

  // -----------------------------------------------------------------------
  // Private — FLAME geometry construction
  // -----------------------------------------------------------------------

  /**
   * Build a Three.js BufferGeometry from loaded FLAME model data.
   *
   * Creates geometry with position, normal, UV, and index attributes.
   * The face mesh is centred at origin, facing +Z, and scaled to match
   * our scene conventions (~0.085 unit radius).
   *
   * @private
   * @returns {{
   *   geometry: THREE.BufferGeometry,
   *   regions: Record<string, number[]>,
   *   regionMeta: Record<string, {label: string, parent: string|null, clinical: string}>
   * }}
   */
  _buildFlameGeometry() {
    const { vertexCount, faceCount } = this._flameMeta;
    const templateVerts = this._flameCurrentVertices;
    const faces = this._flameFaces;

    // -- Compute bounding box to determine scale --
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (let i = 0; i < vertexCount; i++) {
      const x = templateVerts[i * 3];
      const y = templateVerts[i * 3 + 1];
      const z = templateVerts[i * 3 + 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    const extentX = maxX - minX;
    const extentY = maxY - minY;
    const extentZ = maxZ - minZ;
    const maxExtent = Math.max(extentX, extentY, extentZ);

    // FLAME uses metres; our scene expects ~0.085 radius (0.17 diameter).
    // Scale the FLAME mesh so that its largest extent maps to ~0.17 units.
    const targetDiameter = this._radius * 2;
    const scaleFactor = maxExtent > 0 ? targetDiameter / maxExtent : 1;

    // Centre of bounding box
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    // Create scaled positions array (centred at origin, facing +Z)
    const positions = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) {
      positions[i * 3]     = (templateVerts[i * 3]     - cx) * scaleFactor;
      positions[i * 3 + 1] = (templateVerts[i * 3 + 1] - cy) * scaleFactor;
      positions[i * 3 + 2] = (templateVerts[i * 3 + 2] - cz) * scaleFactor;
    }

    // -- Build Three.js geometry --
    const geometry = new THREE.BufferGeometry();

    if (this._flameUVCoords && this._flameUVFaces) {
      // Use proper UV topology (5118 vertices to match UV mapping)
      const uvVertexCount = this._flameUVVertexCount;
      const uvCoords = this._flameUVCoords;
      const uvFaces = this._flameUVFaces;

      // Create UV-expanded position array
      const uvPositions = new Float32Array(uvVertexCount * 3);
      for (let i = 0; i < uvVertexCount; i++) {
        const posIdx = this._uvToPosMap[i];
        uvPositions[i * 3]     = positions[posIdx * 3];
        uvPositions[i * 3 + 1] = positions[posIdx * 3 + 1];
        uvPositions[i * 3 + 2] = positions[posIdx * 3 + 2];
      }

      // Compute normals for UV-expanded geometry
      const uvNormals = computeNormals(uvPositions, uvFaces);

      geometry.setAttribute('position', new THREE.BufferAttribute(uvPositions, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(uvNormals, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(uvCoords, 2));
      geometry.setIndex(new THREE.BufferAttribute(uvFaces, 1));
    } else {
      // Fallback: spherical UVs with original topology
      const normals = computeNormals(positions, faces);
      const uvs = generateFlameUVs(positions);
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
      geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    }

    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    // -- Build regions from flame_regions.json --
    const regions = this._buildFlameRegions();

    // -- Deep-copy region metadata --
    const regionMeta = {};
    for (const [name, def] of Object.entries(REGION_META_DEFS)) {
      regionMeta[name] = { ...def };
    }

    // Cache results
    this._geometry   = geometry;
    this._regions    = regions;
    this._regionMeta = regionMeta;

    // Store scale and offset for later position updates
    this._flameScale = scaleFactor;
    this._flameCenterX = cx;
    this._flameCenterY = cy;
    this._flameCenterZ = cz;

    return { geometry, regions, regionMeta };
  }

  /**
   * Build region vertex index map from the loaded flame_regions.json data.
   *
   * The JSON file maps region names to arrays of vertex indices.  We use the
   * same 52 clinical zone names as REGION_META_DEFS.  Any region in
   * REGION_META_DEFS that is missing from the JSON will get an empty array.
   * Any extra regions in the JSON that are not in REGION_META_DEFS are
   * silently ignored.
   *
   * @private
   * @returns {Record<string, number[]>}
   */
  _buildFlameRegions() {
    const regions = {};

    // Initialise all expected regions to empty arrays
    for (const name of Object.keys(REGION_META_DEFS)) {
      regions[name] = [];
    }

    // Overlay with data from flame_regions.json
    if (this._flameRegions) {
      const zonesData = this._flameRegions.zones ?? this._flameRegions;
      for (const [name, data] of Object.entries(zonesData)) {
        if (name in regions) {
          // Handle both direct arrays and {vertex_indices: [...]} format
          if (Array.isArray(data)) {
            regions[name] = data.slice();
          } else if (data && Array.isArray(data.vertex_indices)) {
            regions[name] = data.vertex_indices.slice();
          }
        }
      }
    }

    // If using UV topology, remap region indices to UV vertex space
    if (this._uvToPosMap) {
      // Build reverse map: position idx -> list of UV indices
      const posToUvMap = {};
      for (let uvIdx = 0; uvIdx < this._flameUVVertexCount; uvIdx++) {
        const posIdx = this._uvToPosMap[uvIdx];
        if (!posToUvMap[posIdx]) posToUvMap[posIdx] = [];
        posToUvMap[posIdx].push(uvIdx);
      }

      // Remap each region
      for (const name of Object.keys(regions)) {
        const posIndices = regions[name];
        const uvIndices = [];
        for (const posIdx of posIndices) {
          const mapped = posToUvMap[posIdx];
          if (mapped) uvIndices.push(...mapped);
        }
        regions[name] = uvIndices;
      }
    }

    return regions;
  }

  /**
   * Update the Three.js geometry positions in-place from a flat vertex array.
   *
   * Applies the stored scale and centering transform so the mesh stays at
   * the origin with the correct scene-space size.  Also recomputes normals.
   *
   * @private
   * @param {Float32Array} vertices - Raw FLAME-space vertex positions (vertexCount * 3).
   */
  _updateGeometryPositions(vertices) {
    if (!this._geometry) return;

    const posAttr = this._geometry.getAttribute('position');
    const positions = posAttr.array;
    const vertexCount = this._flameMeta.vertexCount;

    const s = this._flameScale;
    const cx = this._flameCenterX;
    const cy = this._flameCenterY;
    const cz = this._flameCenterZ;

    for (let i = 0; i < vertexCount; i++) {
      positions[i * 3]     = (vertices[i * 3]     - cx) * s;
      positions[i * 3 + 1] = (vertices[i * 3 + 1] - cy) * s;
      positions[i * 3 + 2] = (vertices[i * 3 + 2] - cz) * s;
    }

    // If using UV topology, remap positions from PCA space (5023) to UV space (5118)
    if (this._uvToPosMap && this._flameUVFaces) {
      const uvVertexCount = this._flameUVVertexCount;
      // The geometry was built with UV topology, so posAttr has uvVertexCount entries
      // Map each UV vertex to its position vertex
      for (let i = 0; i < uvVertexCount; i++) {
        const posIdx = this._uvToPosMap[i];
        positions[i * 3]     = (vertices[posIdx * 3]     - cx) * s;
        positions[i * 3 + 1] = (vertices[posIdx * 3 + 1] - cy) * s;
        positions[i * 3 + 2] = (vertices[posIdx * 3 + 2] - cz) * s;
      }
    }

    posAttr.needsUpdate = true;

    // Recompute normals (use UV faces if available, otherwise position faces)
    const faces = this._flameUVFaces ?? this._flameFaces;
    const newNormals = computeNormals(positions, faces);
    const normalAttr = this._geometry.getAttribute('normal');
    normalAttr.array.set(newNormals);
    normalAttr.needsUpdate = true;

    // Recompute bounds
    this._geometry.computeBoundingSphere();
    this._geometry.computeBoundingBox();
  }
}

export default FlameMeshGenerator;
