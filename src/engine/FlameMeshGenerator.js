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
  }

  // -----------------------------------------------------------------------
  // Public API
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

  /**
   * Get the vertex indices for a named region.
   *
   * @param {string} regionName - One of the 52 clinical zone names.
   * @returns {number[]} Array of vertex indices (empty if region unknown).
   */
  getRegionVertices(regionName) {
    if (!this._regions) {
      throw new Error('FlameMeshGenerator: call generate() before querying regions.');
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
      throw new Error('FlameMeshGenerator: call generate() before querying regions.');
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

  /** @returns {string[]} All 52 region names. */
  static get regionNames() {
    return Object.keys(REGION_META_DEFS);
  }
}

export default FlameMeshGenerator;
