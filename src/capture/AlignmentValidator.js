/**
 * AlignmentValidator — Real-time head pose validation for guided capture.
 *
 * Validates head alignment against per-view constraints using MediaPipe
 * 478-landmark data and head pose (pitch/yaw/roll in radians).
 *
 * Provides continuous feedback: directional hints, centering guidance,
 * face-size requirements, and eye-openness checks.
 */

import { LANDMARK_DEFINITIONS } from '../engine/MediaPipeBridge.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Degrees → radians helper */
const DEG = Math.PI / 180;

/**
 * Per-view head pose constraints (all values in radians).
 *
 * MediaPipeBridge.landmarksToHeadPose() returns:
 *   yaw:   positive = subject faces right, negative = faces left
 *   pitch: radians, relative
 *   roll:  radians from horizontal eye line
 */
const VIEW_CONSTRAINTS = {
  front: {
    pitch: { min: -15 * DEG, max: 15 * DEG },
    yaw:   { min: -10 * DEG, max: 10 * DEG },
    roll:  { min: -5 * DEG,  max: 5 * DEG },
    label: 'Front',
  },
  left45: {
    pitch: { min: -15 * DEG, max: 15 * DEG },
    yaw:   { min: -55 * DEG, max: -35 * DEG },
    roll:  { min: -8 * DEG,  max: 8 * DEG },
    label: 'Left 45°',
  },
  right45: {
    pitch: { min: -15 * DEG, max: 15 * DEG },
    yaw:   { min: 35 * DEG,  max: 55 * DEG },
    roll:  { min: -8 * DEG,  max: 8 * DEG },
    label: 'Right 45°',
  },
};

/** Face centering: nose_tip must be within this fraction of frame center */
const CENTER_TOLERANCE = 0.20; // nose within 30%-70% of frame (center 40%)

/** Minimum fraction of frame height that face should fill */
const MIN_FACE_HEIGHT_RATIO = 0.30;

/** Minimum eye aspect ratio to consider eyes "open" */
const MIN_EYE_ASPECT_RATIO = 0.015;

/** Number of consecutive aligned frames before auto-capture */
const REQUIRED_STABLE_FRAMES = 10; // ~0.33s at 30fps

// ---------------------------------------------------------------------------
// AlignmentValidator
// ---------------------------------------------------------------------------

export class AlignmentValidator {
  constructor() {
    this._stableFrameCount = 0;
  }

  /**
   * Validate alignment for a specific view type.
   *
   * @param {string} viewType - 'front' | 'left45' | 'right45'
   * @param {Array<{x:number, y:number, z:number}>} landmarks - 478 normalized landmarks
   * @param {{pitch:number, yaw:number, roll:number}} headPose - radians
   * @param {{width:number, height:number}} frameSize - video dimensions in pixels
   * @returns {{
   *   isAligned: boolean,
   *   feedback: string[],
   *   confidence: number,
   *   checks: { pose: boolean, centering: boolean, faceSize: boolean, eyesOpen: boolean },
   *   stableFrames: number,
   *   readyToCapture: boolean
   * }}
   */
  validate(viewType, landmarks, headPose, frameSize) {
    if (!landmarks || landmarks.length < 468 || !headPose) {
      this._stableFrameCount = 0;
      return {
        isAligned: false,
        feedback: ['No face detected'],
        confidence: 0,
        checks: { pose: false, centering: false, faceSize: false, eyesOpen: false },
        stableFrames: 0,
        readyToCapture: false,
      };
    }

    const constraints = VIEW_CONSTRAINTS[viewType] || VIEW_CONSTRAINTS.front;
    const feedback = [];

    // --- 1. Head pose check ---
    const poseResult = this._checkPose(viewType, headPose, constraints);
    if (!poseResult.ok) feedback.push(...poseResult.feedback);

    // --- 2. Face centering check ---
    const centerResult = this._checkCentering(landmarks);
    if (!centerResult.ok) feedback.push(centerResult.feedback);

    // --- 3. Face size check ---
    const sizeResult = this._checkFaceSize(landmarks);
    if (!sizeResult.ok) feedback.push(sizeResult.feedback);

    // --- 4. Eyes open check ---
    const eyesResult = this._checkEyesOpen(landmarks);
    if (!eyesResult.ok) feedback.push(eyesResult.feedback);

    // --- Aggregate ---
    const isAligned = poseResult.ok && centerResult.ok && sizeResult.ok && eyesResult.ok;

    if (isAligned) {
      this._stableFrameCount++;
    } else {
      this._stableFrameCount = 0;
    }

    // Overall confidence: weighted average of individual check confidences
    const confidence = (
      poseResult.confidence * 0.50 +
      (centerResult.ok ? 1.0 : centerResult.confidence) * 0.20 +
      (sizeResult.ok ? 1.0 : sizeResult.confidence) * 0.15 +
      (eyesResult.ok ? 1.0 : 0.0) * 0.15
    );

    const readyToCapture = isAligned && this._stableFrameCount >= REQUIRED_STABLE_FRAMES;

    if (isAligned && !readyToCapture) {
      const remaining = REQUIRED_STABLE_FRAMES - this._stableFrameCount;
      feedback.push(`Hold still... (${remaining} frames)`);
    } else if (readyToCapture) {
      feedback.push('Ready to capture!');
    }

    return {
      isAligned,
      feedback,
      confidence: Math.max(0, Math.min(1, confidence)),
      checks: {
        pose: poseResult.ok,
        centering: centerResult.ok,
        faceSize: sizeResult.ok,
        eyesOpen: eyesResult.ok,
      },
      stableFrames: this._stableFrameCount,
      readyToCapture,
    };
  }

  /**
   * Reset the stable frame counter (e.g., when switching views).
   */
  reset() {
    this._stableFrameCount = 0;
  }

  // =========================================================================
  // Private checks
  // =========================================================================

  /**
   * Check head pose angles against view constraints.
   * @returns {{ ok: boolean, feedback: string[], confidence: number }}
   */
  _checkPose(viewType, headPose, constraints) {
    const feedback = [];
    let totalError = 0;
    let maxPossibleError = 0;

    // --- Yaw ---
    const yawMid = (constraints.yaw.min + constraints.yaw.max) / 2;
    const yawRange = (constraints.yaw.max - constraints.yaw.min) / 2;
    const yawError = Math.abs(headPose.yaw - yawMid);
    if (headPose.yaw < constraints.yaw.min) {
      const dirHint = viewType === 'front' ? 'Turn head slightly right' :
        viewType === 'left45' ? 'Turn head more to your left' : 'Turn head less to your right';
      feedback.push(dirHint);
    } else if (headPose.yaw > constraints.yaw.max) {
      const dirHint = viewType === 'front' ? 'Turn head slightly left' :
        viewType === 'right45' ? 'Turn head more to your right' : 'Turn head less to your left';
      feedback.push(dirHint);
    }
    totalError += Math.max(0, yawError - yawRange);
    maxPossibleError += 45 * DEG;

    // --- Pitch ---
    const pitchMid = (constraints.pitch.min + constraints.pitch.max) / 2;
    const pitchRange = (constraints.pitch.max - constraints.pitch.min) / 2;
    const pitchError = Math.abs(headPose.pitch - pitchMid);
    if (headPose.pitch < constraints.pitch.min) {
      feedback.push('Tilt chin up a little');
    } else if (headPose.pitch > constraints.pitch.max) {
      feedback.push('Tilt chin down a little');
    }
    totalError += Math.max(0, pitchError - pitchRange);
    maxPossibleError += 30 * DEG;

    // --- Roll ---
    const rollMid = (constraints.roll.min + constraints.roll.max) / 2;
    const rollRange = (constraints.roll.max - constraints.roll.min) / 2;
    const rollError = Math.abs(headPose.roll - rollMid);
    if (headPose.roll < constraints.roll.min) {
      feedback.push('Straighten your head (tilt right)');
    } else if (headPose.roll > constraints.roll.max) {
      feedback.push('Straighten your head (tilt left)');
    }
    totalError += Math.max(0, rollError - rollRange);
    maxPossibleError += 15 * DEG;

    const ok = feedback.length === 0;
    const confidence = 1.0 - Math.min(1, totalError / maxPossibleError);

    return { ok, feedback, confidence };
  }

  /**
   * Check if nose tip is centered in the frame.
   * Landmarks are in normalized [0,1] coordinates.
   */
  _checkCentering(landmarks) {
    const noseTip = landmarks[LANDMARK_DEFINITIONS.nose_tip];
    if (!noseTip) return { ok: false, feedback: 'Cannot detect face position', confidence: 0 };

    const xOff = Math.abs(noseTip.x - 0.5);
    const yOff = Math.abs(noseTip.y - 0.5);

    const ok = xOff <= CENTER_TOLERANCE && yOff <= CENTER_TOLERANCE;
    const maxOff = Math.max(xOff, yOff);
    const confidence = 1.0 - Math.min(1, maxOff / 0.5);

    let feedback = '';
    if (!ok) {
      const parts = [];
      if (noseTip.x < 0.5 - CENTER_TOLERANCE) parts.push('right');
      else if (noseTip.x > 0.5 + CENTER_TOLERANCE) parts.push('left');
      if (noseTip.y < 0.5 - CENTER_TOLERANCE) parts.push('down');
      else if (noseTip.y > 0.5 + CENTER_TOLERANCE) parts.push('up');
      feedback = `Move face ${parts.join(' and ')} to center`;
    }

    return { ok, feedback, confidence };
  }

  /**
   * Check if face landmarks span at least MIN_FACE_HEIGHT_RATIO of the frame.
   */
  _checkFaceSize(landmarks) {
    const forehead = landmarks[LANDMARK_DEFINITIONS.forehead_center];
    const chin = landmarks[LANDMARK_DEFINITIONS.chin_tip];
    if (!forehead || !chin) return { ok: false, feedback: 'Cannot measure face size', confidence: 0 };

    const heightRatio = Math.abs(chin.y - forehead.y);
    const ok = heightRatio >= MIN_FACE_HEIGHT_RATIO;
    const confidence = Math.min(1, heightRatio / MIN_FACE_HEIGHT_RATIO);

    let feedback = '';
    if (!ok) {
      feedback = heightRatio < MIN_FACE_HEIGHT_RATIO * 0.5 ?
        'Move much closer to camera' : 'Move a little closer to camera';
    }

    return { ok, feedback, confidence, ratio: heightRatio };
  }

  /**
   * Check if both eyes are open using Eye Aspect Ratio.
   * EAR = vertical distance / horizontal distance for each eye.
   */
  _checkEyesOpen(landmarks) {
    // Right eye
    const rUpper = landmarks[LANDMARK_DEFINITIONS.right_eye_upper];
    const rLower = landmarks[LANDMARK_DEFINITIONS.right_eye_lower];
    const rInner = landmarks[LANDMARK_DEFINITIONS.right_eye_inner];
    const rOuter = landmarks[LANDMARK_DEFINITIONS.right_eye_outer];

    // Left eye
    const lUpper = landmarks[LANDMARK_DEFINITIONS.left_eye_upper];
    const lLower = landmarks[LANDMARK_DEFINITIONS.left_eye_lower];
    const lInner = landmarks[LANDMARK_DEFINITIONS.left_eye_inner];
    const lOuter = landmarks[LANDMARK_DEFINITIONS.left_eye_outer];

    if (!rUpper || !rLower || !lUpper || !lLower) {
      return { ok: true, feedback: '' }; // can't check, assume ok
    }

    const rightEAR = Math.abs(rUpper.y - rLower.y);
    const leftEAR = Math.abs(lUpper.y - lLower.y);

    const ok = rightEAR >= MIN_EYE_ASPECT_RATIO && leftEAR >= MIN_EYE_ASPECT_RATIO;
    const feedback = ok ? '' : 'Please open your eyes';

    return { ok, feedback };
  }
}

export { VIEW_CONSTRAINTS, REQUIRED_STABLE_FRAMES };
