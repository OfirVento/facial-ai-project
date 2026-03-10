/**
 * DepthEstimator — Face-to-camera distance estimation from MediaPipe landmarks.
 *
 * Uses inter-pupillary distance (IPD) in normalized image coordinates
 * as a proxy for face distance. Average human IPD ≈ 63mm; when the
 * measured IPD in the image is larger the face is closer, when smaller
 * the face is farther.
 *
 * Provides simple "move closer / farther" guidance for the CaptureGuide.
 */

import { LANDMARK_DEFINITIONS, computeInterPupillaryDistance } from '../engine/MediaPipeBridge.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Target IPD in normalized [0,1] image coordinates for ideal capture distance */
const IDEAL_IPD = 0.18;

/** Acceptable IPD tolerance (±) around the ideal */
const IPD_TOLERANCE = 0.04;

/** Average human inter-pupillary distance in mm (reference only) */
const AVERAGE_IPD_MM = 63;

/**
 * Key landmark indices for mean z-depth computation.
 * These are stable, prominent landmarks that give a reliable depth signal.
 */
const DEPTH_LANDMARKS = [
  LANDMARK_DEFINITIONS.nose_tip,        // 1
  LANDMARK_DEFINITIONS.chin_tip,        // 152
  LANDMARK_DEFINITIONS.forehead_center, // 10
  LANDMARK_DEFINITIONS.right_eye_outer, // 33
  LANDMARK_DEFINITIONS.left_eye_outer,  // 263
];

// ---------------------------------------------------------------------------
// DepthEstimator
// ---------------------------------------------------------------------------

export class DepthEstimator {
  /**
   * Estimate face-to-camera distance and provide guidance.
   *
   * @param {Array<{x:number, y:number, z:number}>} landmarks - 478 MediaPipe landmarks
   * @returns {{
   *   ipdNormalized: number,
   *   relativeDistance: number,
   *   feedback: string|null,
   *   isInRange: boolean,
   *   zDepthMean: number
   * }}
   */
  estimate(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return {
        ipdNormalized: 0,
        relativeDistance: 1,
        feedback: null,
        isInRange: false,
        zDepthMean: 0,
      };
    }

    // Compute IPD in normalized image coordinates
    const ipd = computeInterPupillaryDistance(landmarks);

    // Relative distance: >1 means too far, <1 means too close
    const relativeDistance = IDEAL_IPD / (ipd || 0.01);

    // Determine if in range
    const isInRange = ipd >= (IDEAL_IPD - IPD_TOLERANCE) && ipd <= (IDEAL_IPD + IPD_TOLERANCE);

    // Generate feedback
    let feedback = null;
    if (ipd < IDEAL_IPD - IPD_TOLERANCE) {
      if (ipd < IDEAL_IPD * 0.5) {
        feedback = 'Move much closer to camera';
      } else {
        feedback = 'Move a little closer';
      }
    } else if (ipd > IDEAL_IPD + IPD_TOLERANCE) {
      if (ipd > IDEAL_IPD * 1.5) {
        feedback = 'Move farther from camera';
      } else {
        feedback = 'Move a little farther';
      }
    }

    // Compute mean z-depth for supplementary signal
    const zDepthMean = this._meanZDepth(landmarks);

    return {
      ipdNormalized: Math.round(ipd * 1000) / 1000,
      relativeDistance: Math.round(relativeDistance * 100) / 100,
      feedback,
      isInRange,
      zDepthMean: Math.round(zDepthMean * 1000) / 1000,
    };
  }

  // =========================================================================
  // Private
  // =========================================================================

  /**
   * Compute mean z-depth across key face landmarks.
   * @param {Array<{x,y,z}>} landmarks
   * @returns {number} average z-depth
   */
  _meanZDepth(landmarks) {
    let sum = 0;
    let count = 0;

    for (const idx of DEPTH_LANDMARKS) {
      if (idx < landmarks.length && landmarks[idx]) {
        sum += landmarks[idx].z || 0;
        count++;
      }
    }

    return count > 0 ? sum / count : 0;
  }
}

export { IDEAL_IPD, IPD_TOLERANCE };
