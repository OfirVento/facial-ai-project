/**
 * ShapeFitter — FLAME shape/expression fitting and camera estimation.
 *
 * Extracts 3D face shape and expression parameters from 2D MediaPipe landmarks
 * by fitting FLAME PCA basis vectors via alternating linear least-squares.
 *
 * Provides:
 *  - Weak-perspective camera estimation (anisotropic sx/sy + translation)
 *  - Rotation estimation from 2D-3D correspondences (Procrustes)
 *  - Per-view pose refinement via Levenberg-Marquardt
 *  - FLAME shape (beta) fitting from landmarks
 *  - FLAME expression (epsilon) fitting from landmarks
 *  - Focal length / camera distance estimation from inter-pupillary distance
 *
 * All methods are stateless — data flows through parameters.
 */
export class ShapeFitter {
  constructor() {
    // Stateless: all data flows through method parameters.
  }

  // -----------------------------------------------------------------------
  // Linear algebra utility
  // -----------------------------------------------------------------------

  /**
   * Solve A·x = b for symmetric positive-definite A via Cholesky decomposition.
   * @param {Float64Array} A - n×n SPD matrix (row-major, overwritten with L)
   * @param {Float64Array} b - n×1 right-hand side
   * @param {number} n      - dimension
   * @returns {Float64Array} n×1 solution vector x
   */
  _solveCholesky(A, b, n) {
    // Cholesky: A = L * L^T  (in-place, lower triangle)
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += A[j * n + k] * A[j * n + k];
      const diag = A[j * n + j] - sum;
      if (diag <= 0) {
        // Not SPD — add small ridge and retry
        console.warn('Cholesky: matrix not SPD, adding ridge');
        for (let i = 0; i < n; i++) A[i * n + i] += 1e-6;
        return this._solveCholesky(A, b, n);
      }
      A[j * n + j] = Math.sqrt(diag);
      for (let i = j + 1; i < n; i++) {
        let s = 0;
        for (let k = 0; k < j; k++) s += A[i * n + k] * A[j * n + k];
        A[i * n + j] = (A[i * n + j] - s) / A[j * n + j];
      }
    }

    // Forward substitution: L · y = b
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < i; k++) s += A[i * n + k] * y[k];
      y[i] = (b[i] - s) / A[i * n + i];
    }

    // Back substitution: L^T · x = y
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = 0;
      for (let k = i + 1; k < n; k++) s += A[k * n + i] * x[k];
      x[i] = (y[i] - s) / A[i * n + i];
    }

    return x;
  }

  // -----------------------------------------------------------------------
  // Rotation & Euler angle utilities
  // -----------------------------------------------------------------------

  /**
   * Euler angles (pitch, yaw, roll) to 3x3 rotation matrix (row-major).
   * Convention: R = Ry(yaw) * Rx(pitch) * Rz(roll)
   * @returns {Float64Array} 9-element row-major rotation matrix
   */
  _eulerToRotationMatrix(pitch, yaw, roll) {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cr = Math.cos(roll), sr = Math.sin(roll);

    return new Float64Array([
      cy * cr + sy * sp * sr,   -cy * sr + sy * sp * cr,  sy * cp,
      cp * sr,                   cp * cr,                  -sp,
      -sy * cr + cy * sp * sr,   sy * sr + cy * sp * cr,   cy * cp,
    ]);
  }

  /**
   * Estimate rotation matrix from 2D-3D correspondences using Procrustes alignment.
   * Aligns the centred 3D X,Y to the centred 2D positions, then computes R.
   * @returns {Float64Array} 3x3 rotation matrix (row-major)
   */
  _estimateRotationFromLandmarks(pts3d, pts2d) {
    const N = pts3d.length;

    // Centre both point sets
    let cx3 = 0, cy3 = 0, cz3 = 0, cx2 = 0, cy2 = 0;
    for (let i = 0; i < N; i++) {
      cx3 += pts3d[i][0]; cy3 += pts3d[i][1]; cz3 += pts3d[i][2];
      cx2 += pts2d[i][0]; cy2 += pts2d[i][1];
    }
    cx3 /= N; cy3 /= N; cz3 /= N; cx2 /= N; cy2 /= N;

    // Build cross-covariance matrix H (2x3) between centred 2D and centred 3D
    // We only have 2D observations, so we fit the X,Y projection
    // Compute an approximate yaw from the z-depth variation
    let H00 = 0, H01 = 0, H02 = 0;
    let H10 = 0, H11 = 0, H12 = 0;
    for (let i = 0; i < N; i++) {
      const dx2 = pts2d[i][0] - cx2;
      const dy2 = pts2d[i][1] - cy2;
      const dx3 = pts3d[i][0] - cx3;
      const dy3 = pts3d[i][1] - cy3;
      const dz3 = pts3d[i][2] - cz3;
      H00 += dx2 * dx3; H01 += dx2 * dy3; H02 += dx2 * dz3;
      H10 += dy2 * dx3; H11 += dy2 * dy3; H12 += dy2 * dz3;
    }

    // Estimate yaw from the ratio of z-correlation to x-correlation
    const yaw = Math.atan2(H02, H00 + 1e-10);
    // Estimate pitch from the ratio of z-correlation to y-correlation
    const pitch = Math.atan2(-H12, -(H11 + 1e-10));
    // Estimate roll from the skew between x and y
    const roll = Math.atan2(H10, H00 + 1e-10);

    // Clamp to reasonable ranges (wider yaw for 45° side views)
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const yawC = clamp(yaw, -1.2, 1.2);     // ~69° — covers 45° side views
    const pitchC = clamp(pitch, -0.6, 0.6);
    const rollC = clamp(roll, -0.4, 0.4);

    return this._eulerToRotationMatrix(pitchC, yawC, rollC);
  }

  // -----------------------------------------------------------------------
  // Camera estimation
  // -----------------------------------------------------------------------

  /**
   * Estimate focal length and camera-to-face distance from inter-pupillary distance.
   *
   * Uses MediaPipe iris landmarks (468, 473) to measure pixel IPD, then
   * combines with FLAME mesh IPD to estimate the camera distance (Z_face).
   * Assumes a typical smartphone focal length (~26-32mm equiv).
   *
   * @param {Array} landmarks - MediaPipe 478 landmarks (normalised 0-1)
   * @param {number} imgW - image width in pixels
   * @param {number} imgH - image height in pixels
   * @param {object} meshGen - FlameMeshGenerator instance
   * @returns {{ fxNorm: number, fyNorm: number, Z_face: number }}
   */
  _estimateFocalLength(landmarks, imgW, imgH, meshGen) {
    // 1. Measure IPD from MediaPipe iris landmarks (in normalised coords)
    let lmL = landmarks[473]; // left iris center
    let lmR = landmarks[468]; // right iris center

    // Fallback to outer eye corners if iris not available
    if (!lmL || isNaN(lmL.x) || !lmR || isNaN(lmR.x)) {
      lmL = landmarks[263]; // left eye outer
      lmR = landmarks[33];  // right eye outer
    }
    if (!lmL || isNaN(lmL.x) || !lmR || isNaN(lmR.x)) {
      console.warn('Focal length: No eye landmarks, using default Z_face=0.5');
      return { fxNorm: 1.0, fyNorm: -1.0, Z_face: 0.5 };
    }

    const ipdPixX = (lmL.x - lmR.x) * imgW;
    const ipdPixY = (lmL.y - lmR.y) * imgH;
    const ipdPixels = Math.sqrt(ipdPixX * ipdPixX + ipdPixY * ipdPixY);

    if (ipdPixels < 5) {
      console.warn('Focal length: IPD too small, using default Z_face=0.5');
      return { fxNorm: 1.0, fyNorm: -1.0, Z_face: 0.5 };
    }

    // 2. Measure FLAME mesh IPD from eye region centroids
    const regions = meshGen._flameRegions;
    const verts = meshGen._flameShapedVertices ?? meshGen._flameCurrentVertices ?? meshGen._flameTemplateVertices;
    let ipdFlame = 0.063; // default anatomical IPD (63mm)

    if (regions && verts) {
      const leftEye = regions['eye_left_upper'] || regions['eye_left_lower'];
      const rightEye = regions['eye_right_upper'] || regions['eye_right_lower'];
      if (leftEye && rightEye) {
        let lx = 0, ly = 0, lz = 0, ln = 0;
        let rx = 0, ry = 0, rz = 0, rn = 0;
        for (const vi of leftEye) {
          lx += verts[vi * 3]; ly += verts[vi * 3 + 1]; lz += verts[vi * 3 + 2]; ln++;
        }
        for (const vi of rightEye) {
          rx += verts[vi * 3]; ry += verts[vi * 3 + 1]; rz += verts[vi * 3 + 2]; rn++;
        }
        if (ln > 0 && rn > 0) {
          const dx = lx / ln - rx / rn;
          const dy = ly / ln - ry / rn;
          const dz = lz / ln - rz / rn;
          ipdFlame = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
    }

    // 3. Estimate focal length and distance
    // Assume fx_pixels ≈ max(imgW, imgH) (typical smartphone ~26-32mm equiv)
    const fxPixels = Math.max(imgW, imgH);
    const Z_face = Math.max(0.15, Math.min(1.5, fxPixels * ipdFlame / ipdPixels));

    // Normalised focal length (in [0,1] image coordinate space)
    const fxNorm = fxPixels / imgW;
    const fyNorm = -(fxPixels / imgH); // negative for Y-flip (image Y is downward)

    console.log(`Focal length: IPD_px=${ipdPixels.toFixed(1)}, IPD_3d=${(ipdFlame * 1000).toFixed(1)}mm, ` +
      `fx=${fxPixels}px, Z_face=${Z_face.toFixed(3)}m`);

    return { fxNorm, fyNorm, Z_face };
  }

  /**
   * Estimate anisotropic weak-perspective camera for shape/expression fitting.
   *
   * Solves for separate sx, sy, tx, ty (4 unknowns via two independent 2x2 systems):
   *   imgX = sx * (R·p).x + tx
   *   imgY = sy * (R·p).y + ty
   *
   * Anisotropic scales let the camera absorb proportion differences without
   * forcing shape/expression to distort the mesh.
   */
  _estimateShapeFitCamera(lmk3D, pts2D) {
    const N = lmk3D.length;

    // Estimate rotation via Procrustes
    const R = this._estimateRotationFromLandmarks(lmk3D, pts2D);

    // Rotate all 3D points
    const xr = new Float64Array(N);
    const yr = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const [x, y, z] = lmk3D[i];
      xr[i] = R[0] * x + R[1] * y + R[2] * z;
      yr[i] = R[3] * x + R[4] * y + R[5] * z;
    }

    // Solve X: [xr_i 1] [sx; tx] = [obsX_i]  (2x2 system)
    let Axx = 0, Ax1 = 0, bx0 = 0, bx1 = 0;
    for (let i = 0; i < N; i++) {
      Axx += xr[i] * xr[i];
      Ax1 += xr[i];
      bx0 += xr[i] * pts2D[i][0];
      bx1 += pts2D[i][0];
    }
    const detX = Axx * N - Ax1 * Ax1;
    if (Math.abs(detX) < 1e-15) return null;
    const sx = (bx0 * N - bx1 * Ax1) / detX;
    const tx = (Axx * bx1 - Ax1 * bx0) / detX;

    // Solve Y: [yr_i 1] [sy; ty] = [obsY_i]  (2x2 system)
    let Ayy = 0, Ay1 = 0, by0 = 0, by1 = 0;
    for (let i = 0; i < N; i++) {
      Ayy += yr[i] * yr[i];
      Ay1 += yr[i];
      by0 += yr[i] * pts2D[i][1];
      by1 += pts2D[i][1];
    }
    const detY = Ayy * N - Ay1 * Ay1;
    if (Math.abs(detY) < 1e-15) return null;
    const sy = (by0 * N - by1 * Ay1) / detY;
    const ty = (Ayy * by1 - Ay1 * by0) / detY;

    if (!isFinite(sx) || !isFinite(sy) || Math.abs(sx) < 1e-6) return null;

    // Soft aspect ratio constraint: prevent sx/sy from diverging more than 15%
    // from isotropic. This forces the SHAPE to absorb width/height differences
    // rather than the camera absorbing everything.
    const absRatio = Math.abs(sx / sy);
    const MAX_ANISO = 1.15; // allow up to 15% anisotropy
    if (absRatio > MAX_ANISO || absRatio < 1 / MAX_ANISO) {
      const isoScale = Math.sqrt(Math.abs(sx * sy)); // geometric mean
      const sxSign = Math.sign(sx), sySign = Math.sign(sy);
      const clampedSx = sxSign * Math.min(isoScale * MAX_ANISO, Math.abs(sx));
      const clampedSy = sySign * Math.min(isoScale * MAX_ANISO, Math.abs(sy));
      // Blend: 60% clamped, 40% original (soft constraint)
      const finalSx = 0.6 * clampedSx + 0.4 * sx;
      const finalSy = 0.6 * clampedSy + 0.4 * sy;
      return { R, sx: finalSx, sy: finalSy, tx, ty };
    }

    return { R, sx, sy, tx, ty };
  }

  /**
   * Estimate camera parameters from 105 2D-3D landmark correspondences.
   *
   * Uses weak-perspective fit:
   *   imgX ≈ s * Xrot + tx
   *   imgY ≈ s * Yrot + ty
   *
   * where Xrot,Yrot are rotated 3D coords (rotation from MediaPipe head pose).
   *
   * @returns {{ R: Float64Array, fx: number, fy: number, cx: number, cy: number,
   *             tx: number, ty: number, tz: number }} | null
   */
  _estimateCameraFromLandmarks(landmarks, mapping, meshGen) {
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;

    // --- Extract 105 3D-2D correspondences ---
    const pts3d = [];
    const pts2d = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      const x = bc[0] * verts[pi0 * 3] + bc[1] * verts[pi1 * 3] + bc[2] * verts[pi2 * 3];
      const y = bc[0] * verts[pi0 * 3 + 1] + bc[1] * verts[pi1 * 3 + 1] + bc[2] * verts[pi2 * 3 + 1];
      const z = bc[0] * verts[pi0 * 3 + 2] + bc[1] * verts[pi1 * 3 + 2] + bc[2] * verts[pi2 * 3 + 2];

      pts3d.push([x, y, z]);
      pts2d.push([lm.x, lm.y]);
    }

    const N = pts3d.length;
    if (N < 10) return null;
    console.log(`ShapeFitter DENSE: ${N} 2D-3D correspondences for camera estimation`);

    // --- Get rotation from MediaPipe head pose ---
    const R = this._estimateRotationFromLandmarks(pts3d, pts2d);

    // --- Rotate all 3D points ---
    const xr = new Float64Array(N);
    const yr = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const [x, y, z] = pts3d[i];
      xr[i] = R[0] * x + R[1] * y + R[2] * z;
      yr[i] = R[3] * x + R[4] * y + R[5] * z;
    }

    // --- Weak-perspective least squares with separate X/Y scales ---
    // imgX = sx * xr + tx  (2x2 system)
    // imgY = sy * yr + ty  (2x2 system, separate — handles Y-flip naturally)

    // Solve X: [xr_i  1] [sx] = [imgX_i]
    let Axx = 0, Ax1 = 0, A11x = 0, bx0 = 0, bx1 = 0;
    for (let i = 0; i < N; i++) {
      Axx += xr[i] * xr[i];
      Ax1 += xr[i];
      bx0 += xr[i] * pts2d[i][0];
      bx1 += pts2d[i][0];
    }
    A11x = N;
    const detX = Axx * A11x - Ax1 * Ax1;
    if (Math.abs(detX) < 1e-15) return null;
    const sx = (bx0 * A11x - bx1 * Ax1) / detX;
    const tx = (Axx * bx1 - Ax1 * bx0) / detX;

    // Solve Y: [yr_i  1] [sy] = [imgY_i]
    let Ayy = 0, Ay1 = 0, A11y = 0, by0 = 0, by1 = 0;
    for (let i = 0; i < N; i++) {
      Ayy += yr[i] * yr[i];
      Ay1 += yr[i];
      by0 += yr[i] * pts2d[i][1];
      by1 += pts2d[i][1];
    }
    A11y = N;
    const detY = Ayy * A11y - Ay1 * Ay1;
    if (Math.abs(detY) < 1e-15) return null;
    const sy = (by0 * A11y - by1 * Ay1) / detY;
    const ty = (Ayy * by1 - Ay1 * by0) / detY;

    if (!isFinite(sx) || !isFinite(sy) || Math.abs(sx) < 1e-6) return null;

    console.log(`ShapeFitter DENSE: Anisotropic weak-persp: sx=${sx.toFixed(4)}, sy=${sy.toFixed(4)}, tx=${tx.toFixed(4)}, ty=${ty.toFixed(4)}`);

    // --- Validate: reprojection error using weak perspective model ---
    let totalErr = 0;
    for (let i = 0; i < N; i++) {
      const predX = sx * xr[i] + tx;
      const predY = sy * yr[i] + ty;
      const ex = predX - pts2d[i][0];
      const ey = predY - pts2d[i][1];
      totalErr += Math.sqrt(ex * ex + ey * ey);
    }
    const meanErr = totalErr / N;
    console.log(`ShapeFitter DENSE: Mean reprojection error = ${meanErr.toFixed(6)} (${N} points)`);

    if (meanErr > 0.1) {
      console.warn('ShapeFitter DENSE: Reprojection error too high, falling back');
      return null;
    }

    return { R, sx, sy, tx, ty };
  }

  // -----------------------------------------------------------------------
  // Pose refinement
  // -----------------------------------------------------------------------

  /**
   * Per-view pose refinement via Levenberg-Marquardt.
   * Keeps mesh shape (beta) fixed; refines rotation (pitch/yaw/roll) with
   * scale+translation solved linearly in the inner loop.
   *
   * @param {Array} landmarks  - MediaPipe landmarks for this view
   * @param {Object} mapping   - FLAME<->MediaPipe mapping
   * @param {Object} meshGen   - MeshGenerator with current fitted vertices
   * @param {Object} initialCamera - Initial camera estimate {R, sx, sy, tx, ty}
   * @param {number} maxIters  - Maximum LM iterations (default 20)
   * @returns {Object} Refined {R, sx, sy, tx, ty} or initialCamera if refinement fails
   */
  _refineViewPose(landmarks, mapping, meshGen, initialCamera, maxIters = 20) {
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;

    // Build 3D-2D correspondences (same as _estimateCameraFromLandmarks)
    const pts3d = [];
    const pts2d = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      const x = bc[0] * verts[pi0 * 3] + bc[1] * verts[pi1 * 3] + bc[2] * verts[pi2 * 3];
      const y = bc[0] * verts[pi0 * 3 + 1] + bc[1] * verts[pi1 * 3 + 1] + bc[2] * verts[pi2 * 3 + 1];
      const z = bc[0] * verts[pi0 * 3 + 2] + bc[1] * verts[pi1 * 3 + 2] + bc[2] * verts[pi2 * 3 + 2];

      pts3d.push([x, y, z]);
      pts2d.push([lm.x, lm.y]);
    }

    const N = pts3d.length;
    if (N < 10) return initialCamera;

    // Extract initial Euler angles from R
    // R = Ry(yaw) * Rx(pitch) * Rz(roll)
    // R[5] = -sin(pitch), R[2] = sin(yaw)*cos(pitch), R[8] = cos(yaw)*cos(pitch)
    // R[3] = cos(pitch)*sin(roll), R[4] = cos(pitch)*cos(roll)
    const R0 = initialCamera.R;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    let pitch = Math.asin(-clamp(R0[5], -1, 1));
    let yaw   = Math.atan2(R0[2], R0[8]);
    let roll  = Math.atan2(R0[3], R0[4]);

    // Inner function: given Euler angles, solve linear system for sx/sy/tx/ty,
    // return residuals and total squared error
    const computeAtAngles = (p, yw, rl) => {
      const R = this._eulerToRotationMatrix(p, yw, rl);
      const xr = new Float64Array(N);
      const yr = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const [x, y, z] = pts3d[i];
        xr[i] = R[0] * x + R[1] * y + R[2] * z;
        yr[i] = R[3] * x + R[4] * y + R[5] * z;
      }

      // Solve X: imgX = sx * xr + tx
      let Axx = 0, Ax1 = 0, bx0 = 0, bx1 = 0;
      for (let i = 0; i < N; i++) {
        Axx += xr[i] * xr[i]; Ax1 += xr[i];
        bx0 += xr[i] * pts2d[i][0]; bx1 += pts2d[i][0];
      }
      const detX = Axx * N - Ax1 * Ax1;
      if (Math.abs(detX) < 1e-15) return null;
      const sx = (bx0 * N - bx1 * Ax1) / detX;
      const tx = (Axx * bx1 - Ax1 * bx0) / detX;

      // Solve Y: imgY = sy * yr + ty
      let Ayy = 0, Ay1 = 0, by0 = 0, by1 = 0;
      for (let i = 0; i < N; i++) {
        Ayy += yr[i] * yr[i]; Ay1 += yr[i];
        by0 += yr[i] * pts2d[i][1]; by1 += pts2d[i][1];
      }
      const detY = Ayy * N - Ay1 * Ay1;
      if (Math.abs(detY) < 1e-15) return null;
      const sy = (by0 * N - by1 * Ay1) / detY;
      const ty = (Ayy * by1 - Ay1 * by0) / detY;

      // Residuals (2N vector)
      const residuals = new Float64Array(2 * N);
      let totalErr = 0;
      for (let i = 0; i < N; i++) {
        const ex = sx * xr[i] + tx - pts2d[i][0];
        const ey = sy * yr[i] + ty - pts2d[i][1];
        residuals[2 * i] = ex;
        residuals[2 * i + 1] = ey;
        totalErr += ex * ex + ey * ey;
      }

      return { R, sx, sy, tx, ty, residuals, error: totalErr };
    };

    let current = computeAtAngles(pitch, yaw, roll);
    if (!current) return initialCamera;

    const initialRMSE = Math.sqrt(current.error / N);
    console.log(`ShapeFitter REFINE: Initial RMSE = ${initialRMSE.toFixed(6)} (${N} pts)`);

    const eps = 1e-5;      // Numerical differentiation step
    let lambda = 1e-3;     // LM damping factor

    for (let iter = 0; iter < maxIters; iter++) {
      // Numerical Jacobian of residuals wrt [pitch, yaw, roll] — (2N × 3)
      const angles = [pitch, yaw, roll];
      const J = new Float64Array(2 * N * 3);

      for (let j = 0; j < 3; j++) {
        const aPlus = [...angles]; aPlus[j] += eps;
        const aMinus = [...angles]; aMinus[j] -= eps;

        const rPlus = computeAtAngles(aPlus[0], aPlus[1], aPlus[2]);
        const rMinus = computeAtAngles(aMinus[0], aMinus[1], aMinus[2]);
        if (!rPlus || !rMinus) continue;

        for (let i = 0; i < 2 * N; i++) {
          J[i * 3 + j] = (rPlus.residuals[i] - rMinus.residuals[i]) / (2 * eps);
        }
      }

      // Normal equations: (J^T J + lambda diag(J^T J)) delta = -J^T r
      const JtJ = new Float64Array(9);
      const Jtr = new Float64Array(3);
      for (let i = 0; i < 2 * N; i++) {
        for (let a = 0; a < 3; a++) {
          Jtr[a] -= J[i * 3 + a] * current.residuals[i];
          for (let b = 0; b < 3; b++) {
            JtJ[a * 3 + b] += J[i * 3 + a] * J[i * 3 + b];
          }
        }
      }

      // LM damping on diagonal
      JtJ[0] *= (1 + lambda); JtJ[4] *= (1 + lambda); JtJ[8] *= (1 + lambda);
      // Ensure positive diagonal
      JtJ[0] = Math.max(JtJ[0], 1e-12);
      JtJ[4] = Math.max(JtJ[4], 1e-12);
      JtJ[8] = Math.max(JtJ[8], 1e-12);

      // Solve 3x3 system via Cholesky
      const delta = this._solveCholesky(JtJ.slice(), Jtr.slice(), 3);

      const newPitch = pitch + delta[0];
      const newYaw   = yaw   + delta[1];
      const newRoll  = roll  + delta[2];

      const candidate = computeAtAngles(newPitch, newYaw, newRoll);
      if (!candidate) { lambda *= 10; continue; }

      if (candidate.error < current.error) {
        pitch = newPitch;
        yaw = newYaw;
        roll = newRoll;
        current = candidate;
        lambda = Math.max(lambda * 0.5, 1e-8);

        // Convergence: step smaller than 1e-7 radians
        const stepNorm = Math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
        if (stepNorm < 1e-7) break;
      } else {
        lambda *= 10;
        if (lambda > 1e8) break;
      }
    }

    const finalRMSE = Math.sqrt(current.error / N);
    console.log(`ShapeFitter REFINE: Refined RMSE = ${finalRMSE.toFixed(6)} (improvement: ${((1 - finalRMSE / initialRMSE) * 100).toFixed(1)}%)`);
    console.log(`ShapeFitter REFINE: Camera: sx=${current.sx.toFixed(4)}, sy=${current.sy.toFixed(4)}, tx=${current.tx.toFixed(4)}, ty=${current.ty.toFixed(4)}`);

    return { R: current.R, sx: current.sx, sy: current.sy, tx: current.tx, ty: current.ty };
  }

  /**
   * Compute RMSE of camera reprojection against MediaPipe landmarks.
   * Used as quality gate for side views.
   */
  _computeCameraRMSE(landmarks, mapping, meshGen, camera) {
    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const verts = meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const { R, sx, sy, tx, ty } = camera;

    let totalErr = 0;
    let count = 0;
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;

      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      const x = bc[0] * verts[pi0 * 3] + bc[1] * verts[pi1 * 3] + bc[2] * verts[pi2 * 3];
      const y = bc[0] * verts[pi0 * 3 + 1] + bc[1] * verts[pi1 * 3 + 1] + bc[2] * verts[pi2 * 3 + 1];
      const z = bc[0] * verts[pi0 * 3 + 2] + bc[1] * verts[pi1 * 3 + 2] + bc[2] * verts[pi2 * 3 + 2];

      const xr = R[0] * x + R[1] * y + R[2] * z;
      const yr = R[3] * x + R[4] * y + R[5] * z;

      const predX = sx * xr + tx;
      const predY = sy * yr + ty;
      const ex = predX - lm.x;
      const ey = predY - lm.y;
      totalErr += ex * ex + ey * ey;
      count++;
    }

    return count > 0 ? Math.sqrt(totalErr / count) : Infinity;
  }

  // -----------------------------------------------------------------------
  // FLAME shape fitting — landmark-only PCA optimisation
  // -----------------------------------------------------------------------

  /**
   * Fit FLAME shape PCA parameters (beta) to 105 MediaPipe landmarks.
   *
   * Uses alternating linear least-squares:
   *   1) Fix beta -> estimate camera (R, sx, sy, tx, ty)
   *   2) Fix camera -> solve for beta via regularised normal equations
   * Repeats for 4 iterations. All sub-steps are closed-form (no gradient descent).
   *
   * @param {Array} landmarks  - MediaPipe 478 landmarks
   * @param {object} mapping   - flameMapping { landmark_indices, lmk_face_idx, lmk_b_coords }
   * @param {object} meshGen   - FlameMeshGenerator instance
   * @returns {Float64Array} fitted shape parameters (20 components)
   */
  _fitShapeFromLandmarks(landmarks, mapping, meshGen) {
    const NUM_COMPONENTS = 25;   // more components -> finer face shape control
    const NUM_ITERATIONS = 6;
    const LAMBDA = 0.005;       // lighter regularisation -> shape absorbs more geometry

    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    const templateVerts = meshGen._flameTemplateVertices ?? meshGen.flameTemplateVertices;
    const shapeBasis = meshGen._flameShapeBasis;
    const { shapeComponents } = meshGen._flameMeta;

    if (!shapeBasis || !templateVerts) {
      console.warn('Shape fitting: FLAME data not available');
      return null;
    }

    const C = Math.min(NUM_COMPONENTS, shapeComponents);

    // --- Extract valid 2D-3D landmark correspondences ---
    const validIndices = []; // indices into mpIndices
    const pts2D = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      validIndices.push(i);
      pts2D.push([lm.x, lm.y]);
    }

    const K = validIndices.length;
    if (K < 10) {
      console.warn(`Shape fitting: Only ${K} valid landmarks, need >=10`);
      return null;
    }
    console.log(`Shape fitting: ${K} valid landmarks, ${C} PCA components`);

    // --- Pre-compute per-landmark template positions and shape basis Jacobian ---
    // templateLmk[k] = barycentric interpolation of template vertices at landmark k
    // J[k][coord][c] = barycentric interpolation of shapeBasis for component c at landmark k
    const templateLmk = new Array(K); // K x 3
    const J = new Array(K);           // K x 3 x C

    for (let ki = 0; ki < K; ki++) {
      const i = validIndices[ki];
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      // Template position (barycentric interpolation)
      const tx = bc[0] * templateVerts[pi0 * 3]     + bc[1] * templateVerts[pi1 * 3]     + bc[2] * templateVerts[pi2 * 3];
      const ty = bc[0] * templateVerts[pi0 * 3 + 1] + bc[1] * templateVerts[pi1 * 3 + 1] + bc[2] * templateVerts[pi2 * 3 + 1];
      const tz = bc[0] * templateVerts[pi0 * 3 + 2] + bc[1] * templateVerts[pi1 * 3 + 2] + bc[2] * templateVerts[pi2 * 3 + 2];
      templateLmk[ki] = [tx, ty, tz];

      // Shape basis Jacobian: J[k][coord][c] = interpolated basis for this landmark
      // basis layout: basis[(vertex*3 + coord) * shapeComponents + component]
      const Jk = [new Float64Array(C), new Float64Array(C), new Float64Array(C)]; // x, y, z
      for (let c = 0; c < C; c++) {
        for (let coord = 0; coord < 3; coord++) {
          Jk[coord][c] =
            bc[0] * shapeBasis[(pi0 * 3 + coord) * shapeComponents + c] +
            bc[1] * shapeBasis[(pi1 * 3 + coord) * shapeComponents + c] +
            bc[2] * shapeBasis[(pi2 * 3 + coord) * shapeComponents + c];
        }
      }
      J[ki] = Jk;
    }

    // --- Alternating optimisation ---
    let beta = new Float64Array(C); // start from mean shape

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      // 1. Compute current 3D landmark positions: lmk3D[k] = templateLmk[k] + J[k] . beta
      const lmk3D = new Array(K);
      for (let ki = 0; ki < K; ki++) {
        const p = [templateLmk[ki][0], templateLmk[ki][1], templateLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * beta[c];
          p[1] += J[ki][1][c] * beta[c];
          p[2] += J[ki][2][c] * beta[c];
        }
        lmk3D[ki] = p;
      }

      // 2. Estimate anisotropic weak-perspective camera (fix beta)
      const cam = this._estimateShapeFitCamera(lmk3D, pts2D);
      if (!cam) {
        console.warn(`Shape fitting: Camera estimation failed at iter ${iter}`);
        break;
      }
      const { R, sx, sy, tx, ty } = cam;

      // 3. Build design matrix M (2K x C) and target vector d (2K x 1)
      // Anisotropic weak-perspective: predX = sx * (R.p).x + tx
      //                               predY = sy * (R.p).y + ty
      const rows = 2 * K;
      const M = new Float64Array(rows * C);
      const d = new Float64Array(rows);

      for (let ki = 0; ki < K; ki++) {
        const [tmx, tmy, tmz] = templateLmk[ki];
        const txr = R[0] * tmx + R[1] * tmy + R[2] * tmz;
        const tyr = R[3] * tmx + R[4] * tmy + R[5] * tmz;

        d[2 * ki]     = pts2D[ki][0] - (sx * txr + tx);
        d[2 * ki + 1] = pts2D[ki][1] - (sy * tyr + ty);

        for (let c = 0; c < C; c++) {
          const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
          M[(2 * ki) * C + c]     = sx * (R[0] * jx + R[1] * jy + R[2] * jz);
          M[(2 * ki + 1) * C + c] = sy * (R[3] * jx + R[4] * jy + R[5] * jz);
        }
      }

      // 4. Normal equations: (M^T M + lambda * Lambda) beta = M^T d
      const MtM = new Float64Array(C * C);
      const Mtd = new Float64Array(C);
      for (let a = 0; a < C; a++) {
        for (let b = a; b < C; b++) {
          let sum = 0;
          for (let r = 0; r < rows; r++) sum += M[r * C + a] * M[r * C + b];
          MtM[a * C + b] = sum;
          MtM[b * C + a] = sum;
        }
        let rhs = 0;
        for (let r = 0; r < rows; r++) rhs += M[r * C + a] * d[r];
        Mtd[a] = rhs;
      }

      // Tikhonov regularisation: lambda * 2^(c/5)
      for (let c = 0; c < C; c++) {
        MtM[c * C + c] += LAMBDA * Math.pow(2, c / 5);
      }

      // 5. Solve for full beta (linear problem with weak-perspective)
      beta = this._solveCholesky(MtM, Mtd, C);

      // Log progress
      let reproj = 0;
      for (let ki = 0; ki < K; ki++) {
        const p = [templateLmk[ki][0], templateLmk[ki][1], templateLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * beta[c];
          p[1] += J[ki][1][c] * beta[c];
          p[2] += J[ki][2][c] * beta[c];
        }
        const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
        const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
        const ex = (sx * xrP + tx) - pts2D[ki][0];
        const ey = (sy * yrP + ty) - pts2D[ki][1];
        reproj += Math.sqrt(ex * ex + ey * ey);
      }
      console.log(`Shape fitting: iter ${iter}, sx=${sx.toFixed(4)}, sy=${sy.toFixed(4)}, reproj=${(reproj / K).toFixed(6)}`);
    }

    // --- Clamp beta to valid range ---
    for (let c = 0; c < C; c++) {
      const limit = c < 10 ? 3.0 : 2.0;
      beta[c] = Math.max(-limit, Math.min(limit, beta[c]));
    }

    // --- Apply to mesh ---
    const shapeParams = new Array(shapeComponents).fill(0);
    for (let c = 0; c < C; c++) shapeParams[c] = beta[c];
    meshGen.applyShapeParams(shapeParams);

    console.log(`Shape fitting: beta[0..4] = [${beta.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
    console.log('Shape fitting: Applied shape params to mesh');

    return beta;
  }

  // -----------------------------------------------------------------------
  // FLAME expression fitting
  // -----------------------------------------------------------------------

  /**
   * Fit FLAME expression PCA parameters (epsilon) to 105 MediaPipe landmarks.
   *
   * Same alternating least-squares as shape fitting, but:
   *   - Uses expression basis (not shape basis)
   *   - Starts from shaped vertices (not template)
   *   - Fits 10 components with higher regularisation (lambda=0.1)
   *   - Adds symmetry penalty to prevent lopsided expressions
   *
   * Must be called AFTER _fitShapeFromLandmarks().
   */
  _fitExpressionFromLandmarks(landmarks, mapping, meshGen) {
    const NUM_COMPONENTS = 20;
    const NUM_ITERATIONS = 6;
    const LAMBDA = 0.03;       // slightly lighter regularisation for better fit
    const LAMBDA_SYM = 0.05;
    const W_LIP = 3.0; // Weight boost for inner lip landmarks

    const mpIndices = mapping.landmark_indices;
    const faceIndices = mapping.lmk_face_idx;
    const baryCoords = mapping.lmk_b_coords;
    const posFaces = meshGen.flameFaces;
    // Use shaped vertices (after shape fitting) as the base
    const shapedVerts = meshGen._flameShapedVertices ?? meshGen._flameCurrentVertices ?? meshGen.flameTemplateVertices;
    const exprBasis = meshGen._flameExpressionBasis;
    const { expressionComponents } = meshGen._flameMeta;

    if (!exprBasis || !shapedVerts) {
      console.warn('Expression fitting: FLAME expression data not available');
      return null;
    }

    const C = Math.min(NUM_COMPONENTS, expressionComponents);

    // --- Extract valid 2D-3D landmark correspondences ---
    const validIndices = [];
    const pts2D = [];
    for (let i = 0; i < mpIndices.length; i++) {
      const mpIdx = mpIndices[i];
      if (mpIdx >= landmarks.length) continue;
      const lm = landmarks[mpIdx];
      if (isNaN(lm.x) || isNaN(lm.y)) continue;
      validIndices.push(i);
      pts2D.push([lm.x, lm.y]);
    }

    const K = validIndices.length;
    if (K < 10) {
      console.warn(`Expression fitting: Only ${K} valid landmarks, need >=10`);
      return null;
    }
    console.log(`Expression fitting: ${K} landmarks, ${C} PCA components`);

    // --- Pre-compute per-landmark shaped positions and expression Jacobian ---
    const baseLmk = new Array(K); // K x 3 (shaped vertex positions at landmarks)
    const J = new Array(K);       // K x 3 x C (expression basis Jacobian)

    for (let ki = 0; ki < K; ki++) {
      const i = validIndices[ki];
      const fi = faceIndices[i];
      const bc = baryCoords[i];
      const pi0 = posFaces[fi * 3], pi1 = posFaces[fi * 3 + 1], pi2 = posFaces[fi * 3 + 2];

      // Shaped position (barycentric interpolation on shaped vertices)
      const bx = bc[0] * shapedVerts[pi0 * 3]     + bc[1] * shapedVerts[pi1 * 3]     + bc[2] * shapedVerts[pi2 * 3];
      const by = bc[0] * shapedVerts[pi0 * 3 + 1] + bc[1] * shapedVerts[pi1 * 3 + 1] + bc[2] * shapedVerts[pi2 * 3 + 1];
      const bz = bc[0] * shapedVerts[pi0 * 3 + 2] + bc[1] * shapedVerts[pi1 * 3 + 2] + bc[2] * shapedVerts[pi2 * 3 + 2];
      baseLmk[ki] = [bx, by, bz];

      // Expression Jacobian (same layout as shape basis)
      const Jk = [new Float64Array(C), new Float64Array(C), new Float64Array(C)];
      for (let c = 0; c < C; c++) {
        for (let coord = 0; coord < 3; coord++) {
          Jk[coord][c] =
            bc[0] * exprBasis[(pi0 * 3 + coord) * expressionComponents + c] +
            bc[1] * exprBasis[(pi1 * 3 + coord) * expressionComponents + c] +
            bc[2] * exprBasis[(pi2 * 3 + coord) * expressionComponents + c];
        }
      }
      J[ki] = Jk;
    }

    // --- Build symmetry pairs for regularisation ---
    // Find left/right landmark pairs by checking X-coordinate symmetry
    // Landmarks with similar Y but opposite X are symmetric pairs
    const symPairs = [];
    const used = new Set();
    for (let a = 0; a < K; a++) {
      if (used.has(a)) continue;
      const xa = baseLmk[a][0], ya = baseLmk[a][1];
      if (Math.abs(xa) < 0.005) continue; // Skip midline landmarks
      let bestB = -1, bestDist = 0.01;
      for (let b = a + 1; b < K; b++) {
        if (used.has(b)) continue;
        const xb = baseLmk[b][0], yb = baseLmk[b][1];
        // Mirror: similar Y, opposite X
        const dist = Math.abs(xa + xb) + Math.abs(ya - yb);
        if (dist < bestDist) { bestDist = dist; bestB = b; }
      }
      if (bestB >= 0) {
        symPairs.push([a, bestB]);
        used.add(a);
        used.add(bestB);
      }
    }
    console.log(`Expression fitting: ${symPairs.length} symmetric landmark pairs`);

    // --- Identify inner lip landmarks for weighting ---
    // MediaPipe 13 (upper lip top), 14 (lower lip bottom), 78, 308 (lip corners)
    const LIP_MP_INDICES = new Set([13, 14, 78, 308]);
    const lipKiSet = new Set();
    for (let ki = 0; ki < K; ki++) {
      const mpIdx = mpIndices[validIndices[ki]];
      if (LIP_MP_INDICES.has(mpIdx)) lipKiSet.add(ki);
    }

    // --- Detect closed mouth using face-relative threshold ---
    // Use face height (brow to chin) as reference, not absolute pixels
    const lipTop = landmarks[13], lipBot = landmarks[14];
    const browLmk = landmarks[10], chinLmk = landmarks[152]; // forehead & chin
    const faceH = Math.sqrt(
      Math.pow((browLmk.x - chinLmk.x), 2) + Math.pow((browLmk.y - chinLmk.y), 2)
    );
    const lipGap = Math.sqrt(
      Math.pow((lipTop.x - lipBot.x), 2) + Math.pow((lipTop.y - lipBot.y), 2)
    );
    const lipRatio = faceH > 0.01 ? lipGap / faceH : 0;
    const mouthClosed = lipRatio < 0.02; // 2% of face height
    console.log(`Expression fitting: lip/face ratio = ${(lipRatio * 100).toFixed(2)}% (faceH=${faceH.toFixed(4)}), mouth ${mouthClosed ? 'CLOSED' : 'OPEN'}`);

    // --- Alternating optimisation ---
    let epsilon = new Float64Array(C);

    for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
      // 1. Compute current 3D landmark positions: lmk3D = baseLmk + J . epsilon
      const lmk3D = new Array(K);
      for (let ki = 0; ki < K; ki++) {
        const p = [baseLmk[ki][0], baseLmk[ki][1], baseLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * epsilon[c];
          p[1] += J[ki][1][c] * epsilon[c];
          p[2] += J[ki][2][c] * epsilon[c];
        }
        lmk3D[ki] = p;
      }

      // 2. Estimate anisotropic weak-perspective camera (fix epsilon)
      const cam = this._estimateShapeFitCamera(lmk3D, pts2D);
      if (!cam) {
        console.warn(`Expression fitting: Camera failed at iter ${iter}`);
        break;
      }
      const { R, sx, sy, tx, ty } = cam;

      // 3. Build design matrix M and target vector d (anisotropic weak-perspective)
      const rows = 2 * K;
      const M = new Float64Array(rows * C);
      const d = new Float64Array(rows);

      for (let ki = 0; ki < K; ki++) {
        const w = lipKiSet.has(ki) ? W_LIP : 1.0;
        const [bmx, bmy, bmz] = baseLmk[ki];
        const bxr = R[0] * bmx + R[1] * bmy + R[2] * bmz;
        const byr = R[3] * bmx + R[4] * bmy + R[5] * bmz;

        d[2 * ki]     = w * (pts2D[ki][0] - (sx * bxr + tx));
        d[2 * ki + 1] = w * (pts2D[ki][1] - (sy * byr + ty));

        for (let c = 0; c < C; c++) {
          const jx = J[ki][0][c], jy = J[ki][1][c], jz = J[ki][2][c];
          M[(2 * ki) * C + c]     = w * sx * (R[0] * jx + R[1] * jy + R[2] * jz);
          M[(2 * ki + 1) * C + c] = w * sy * (R[3] * jx + R[4] * jy + R[5] * jz);
        }
      }

      // 4. Normal equations: (M^T M + lambda*Lambda + lambda_sym S) epsilon = M^T d
      const MtM = new Float64Array(C * C);
      const Mtd = new Float64Array(C);
      for (let a = 0; a < C; a++) {
        for (let b = a; b < C; b++) {
          let sum = 0;
          for (let r = 0; r < rows; r++) sum += M[r * C + a] * M[r * C + b];
          MtM[a * C + b] = sum;
          MtM[b * C + a] = sum;
        }
        let rhs = 0;
        for (let r = 0; r < rows; r++) rhs += M[r * C + a] * d[r];
        Mtd[a] = rhs;
      }

      // Tikhonov: lambda * 2^(c/5)
      for (let c = 0; c < C; c++) {
        MtM[c * C + c] += LAMBDA * Math.pow(2, c / 5);
      }

      // Symmetry regularisation: penalise asymmetric expression effects
      // For each symmetric pair (a, b), penalty: || (J[a] - J[b]_mirrored) . epsilon ||^2
      // This adds the outer product S[ca][cb] = Sum_pairs (diff_ca . diff_cb) to MtM
      for (const [a, b] of symPairs) {
        for (let ca = 0; ca < C; ca++) {
          // diff_ca = J[a][:,ca] - mirror(J[b][:,ca])  (mirror = negate X)
          const dXa = J[a][0][ca] + J[b][0][ca]; // J[a].x - (-J[b].x) = J[a].x + J[b].x
          const dYa = J[a][1][ca] - J[b][1][ca];
          const dZa = J[a][2][ca] - J[b][2][ca];
          for (let cb = ca; cb < C; cb++) {
            const dXb = J[a][0][cb] + J[b][0][cb];
            const dYb = J[a][1][cb] - J[b][1][cb];
            const dZb = J[a][2][cb] - J[b][2][cb];
            const pen = LAMBDA_SYM * (dXa * dXb + dYa * dYb + dZa * dZb);
            MtM[ca * C + cb] += pen;
            if (ca !== cb) MtM[cb * C + ca] += pen; // symmetric matrix
          }
        }
      }

      // 5. Solve for full epsilon (linear problem with weak-perspective)
      epsilon = this._solveCholesky(MtM, Mtd, C);

      // Log progress
      let reproj = 0;
      for (let ki = 0; ki < K; ki++) {
        const p = [baseLmk[ki][0], baseLmk[ki][1], baseLmk[ki][2]];
        for (let c = 0; c < C; c++) {
          p[0] += J[ki][0][c] * epsilon[c];
          p[1] += J[ki][1][c] * epsilon[c];
          p[2] += J[ki][2][c] * epsilon[c];
        }
        const xrP = R[0] * p[0] + R[1] * p[1] + R[2] * p[2];
        const yrP = R[3] * p[0] + R[4] * p[1] + R[5] * p[2];
        const ex = (sx * xrP + tx) - pts2D[ki][0];
        const ey = (sy * yrP + ty) - pts2D[ki][1];
        reproj += Math.sqrt(ex * ex + ey * ey);
      }
      console.log(`Expression fitting: iter ${iter}, sx=${sx.toFixed(4)}, sy=${sy.toFixed(4)}, reproj=${(reproj / K).toFixed(6)}`);
    }

    // --- Clamp epsilon to valid range ---
    for (let c = 0; c < C; c++) {
      epsilon[c] = Math.max(-2.0, Math.min(2.0, epsilon[c]));
    }
    // Hard constraint: if mouth is closed in photo, zero out jaw-opening components
    if (mouthClosed) {
      epsilon[0] = 0; // jaw open -> force zero
      epsilon[1] = 0; // secondary jaw -> force zero
      console.log('Expression fitting: mouth closed -> hard set epsilon[0]=epsilon[1]=0');
    }

    // --- Apply expression to mesh ---
    const exprParams = new Array(expressionComponents).fill(0);
    for (let c = 0; c < C; c++) exprParams[c] = epsilon[c];
    meshGen.applyExpressionParams(exprParams);

    console.log(`Expression fitting: epsilon[0..4] = [${epsilon.slice(0, 5).map(v => v.toFixed(3)).join(', ')}]`);
    console.log('Expression fitting: Applied expression params to mesh');

    return epsilon;
  }
}
