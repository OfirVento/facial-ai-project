#!/usr/bin/env python3
"""
convert_flame_to_web.py

Converts FLAME model files (pickle format) to web-friendly JSON + binary
formats for use in a Three.js browser application.

Reads from:
    models/flame/flame2023/      -- FLAME 2023 model pickle
    models/flame/vertex_masks/   -- FLAME vertex masks pickle
    models/flame/mediapipe/      -- MediaPipe landmark embedding
    models/flame/albedo/         -- AlbedoMM FLAME albedo model (optional)
    models/flame/texture_space/  -- FLAME texture space model (optional)

Outputs to:
    models/flame/web/            -- JSON + binary files for the browser

Usage:
    cd facial-ai-project
    python scripts/convert_flame_to_web.py

    Or from any directory:
    python scripts/convert_flame_to_web.py --flame-dir /path/to/models/flame
"""

import argparse
import glob
import json
import os
import pickle
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed. Sparse matrices (J_regressor) will be skipped.")

# ---------------------------------------------------------------------------
# Chumpy compatibility
# ---------------------------------------------------------------------------
# FLAME pickles often store arrays as chumpy objects. We try to import chumpy,
# but if it is not installed we patch the unpickler to convert them to numpy.

try:
    import chumpy
    HAS_CHUMPY = True
except ImportError:
    HAS_CHUMPY = False

def _to_numpy(obj):
    """Convert a value to a plain numpy array, handling chumpy and scipy sparse."""
    if obj is None:
        return None
    if HAS_CHUMPY and isinstance(obj, chumpy.Ch):
        return np.array(obj.r)  # .r gives the raw numpy array
    if HAS_SCIPY and sparse.issparse(obj):
        return np.array(obj.todense())
    # Handle our ChumbyShim objects
    if hasattr(obj, '_data') and hasattr(obj, 'r'):
        try:
            arr = obj.r
            if isinstance(arr, np.ndarray):
                return arr
            return np.array(arr)
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.ndim == 0:
            # Scalar object array wrapping something — try to extract
            try:
                inner = obj.item()
                if isinstance(inner, np.ndarray):
                    return inner
                if hasattr(inner, '_data'):
                    return np.array(inner._data)
                if hasattr(inner, 'r'):
                    return np.array(inner.r)
            except Exception:
                pass
        return obj
    try:
        return np.array(obj)
    except Exception:
        return obj


class _ChumbyUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing chumpy module gracefully.

    When chumpy is not installed, FLAME pickles will fail because they reference
    chumpy classes. This unpickler intercepts those references and substitutes
    a shim that produces plain numpy arrays.
    """

    class _ChumbyShim:
        """Minimal shim that pretends to be a chumpy array."""
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            # chumpy.Ch stores its value under the key 'x' in __getstate__
            if isinstance(state, dict) and 'x' in state:
                self._data = np.array(state['x'])
            elif isinstance(state, dict):
                self._data = state
            else:
                self._data = state

        def __reduce_ex__(self, protocol):
            return (np.array, (self._data,))

        @property
        def r(self):
            return np.array(self._data) if isinstance(self._data, np.ndarray) else self._data

    class _ChumbyModule:
        """Fake module that returns shims for any attribute lookup."""
        def __getattr__(self, name):
            return _ChumbyUnpickler._ChumbyShim

    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return self._ChumbyShim
        return super().find_class(module, name)


def _load_pickle(filepath):
    """Load a pickle file, handling chumpy references gracefully."""
    with open(filepath, 'rb') as f:
        if HAS_CHUMPY:
            data = pickle.load(f, encoding='latin1')
        else:
            try:
                data = _ChumbyUnpickler(f, encoding='latin1').load()
            except Exception:
                f.seek(0)
                try:
                    data = pickle.load(f, encoding='latin1')
                except Exception as e:
                    print(f"[ERROR] Failed to load pickle {filepath}: {e}")
                    print("        Install chumpy to handle FLAME pickles: pip install chumpy")
                    return None
    return data


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_file(directory, patterns):
    """Find the first file matching any of the given glob patterns in directory.

    Also looks inside ZIP files if no extracted file is found, and extracts
    the matching file.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    # First try to find an already-extracted file
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]

    # If not found, try extracting from ZIP files in the directory
    for zf_path in sorted(directory.glob("*.zip")):
        try:
            with zipfile.ZipFile(zf_path, 'r') as zf:
                for name in zf.namelist():
                    basename = os.path.basename(name)
                    for pattern in patterns:
                        # Convert glob pattern to a simple check
                        import fnmatch
                        if fnmatch.fnmatch(basename.lower(), pattern.lower()):
                            print(f"  Extracting {name} from {zf_path.name}...")
                            zf.extract(name, directory)
                            extracted_path = directory / name
                            if extracted_path.exists():
                                return extracted_path
        except zipfile.BadZipFile:
            continue

    return None


def find_flame_model(flame_dir):
    """Locate the FLAME model pickle file."""
    model_dir = Path(flame_dir) / "flame2023"
    patterns = [
        "generic_model.pkl",
        "FLAME2023.pkl",
        "flame2023.pkl",
        "FLAME*.pkl",
        "flame*.pkl",
        "*.pkl",
    ]
    result = _find_file(model_dir, patterns)
    if result is None:
        # Check if there is a subdirectory inside flame2023 (some zips nest)
        for sub in model_dir.iterdir() if model_dir.is_dir() else []:
            if sub.is_dir():
                result = _find_file(sub, patterns)
                if result:
                    break
    return result


def find_vertex_masks(flame_dir):
    """Locate the FLAME vertex masks file."""
    masks_dir = Path(flame_dir) / "vertex_masks"
    patterns = [
        "FLAME_masks.pkl",
        "flame_masks.pkl",
        "FLAME_masks.npz",
        "flame_masks.npz",
        "*masks*.pkl",
        "*masks*.npz",
    ]
    result = _find_file(masks_dir, patterns)
    if result is None:
        for sub in masks_dir.iterdir() if masks_dir.is_dir() else []:
            if sub.is_dir():
                result = _find_file(sub, patterns)
                if result:
                    break
    return result


def find_mediapipe_embedding(flame_dir):
    """Locate the MediaPipe landmark embedding file."""
    mp_dir = Path(flame_dir) / "mediapipe"
    patterns = [
        "mediapipe_landmark_embedding.npz",
        "mediapipe_landmark_embedding.npy",
        "landmark_embedding.npz",
        "landmark_embedding.npy",
        "*mediapipe*.npz",
        "*mediapipe*.npy",
        "*landmark*.npz",
        "*landmark*.npy",
    ]
    result = _find_file(mp_dir, patterns)
    if result is None:
        for sub in mp_dir.iterdir() if mp_dir.is_dir() else []:
            if sub.is_dir():
                result = _find_file(sub, patterns)
                if result:
                    break
    return result


def find_albedo_model(flame_dir):
    """Locate the AlbedoMM FLAME albedo model file."""
    albedo_dir = Path(flame_dir) / "albedo"
    patterns = [
        "albedoModel2020_FLAME*.npz",
        "*albedo*FLAME*.npz",
        "*albedo*.npz",
    ]
    result = _find_file(albedo_dir, patterns)
    if result is None:
        for sub in albedo_dir.iterdir() if albedo_dir.is_dir() else []:
            if sub.is_dir():
                result = _find_file(sub, patterns)
                if result:
                    break
    return result


def find_texture_space(flame_dir):
    """Locate the FLAME texture space model file."""
    tex_dir = Path(flame_dir) / "texture_space"
    patterns = [
        "FLAME_texture.npz",
        "flame_texture.npz",
        "*texture*.npz",
    ]
    result = _find_file(tex_dir, patterns)
    if result is None:
        for sub in tex_dir.iterdir() if tex_dir.is_dir() else []:
            if sub.is_dir():
                result = _find_file(sub, patterns)
                if result:
                    break
    return result


# ---------------------------------------------------------------------------
# Clinical sub-region classification
# ---------------------------------------------------------------------------

# The 61 clinical zone names used in our application (includes full_face
# as the root, plus 60 sub-regions).  The regions dict maps each name to
# a list of vertex indices.

ALL_CLINICAL_ZONES = [
    "full_face",
    "forehead", "forehead_left", "forehead_right", "forehead_center",
    "brow_left", "brow_right", "brow_inner_left", "brow_inner_right",
    "eye_left_upper", "eye_left_lower", "eye_right_upper", "eye_right_lower",
    "eye_left_corner_inner", "eye_left_corner_outer",
    "eye_right_corner_inner", "eye_right_corner_outer",
    "under_eye_left", "under_eye_right",
    "tear_trough_left", "tear_trough_right",
    "nose_bridge", "nose_bridge_upper", "nose_bridge_lower",
    "nose_tip", "nose_tip_left", "nose_tip_right",
    "nostril_left", "nostril_right", "nose_dorsum",
    "cheek_left", "cheek_right", "cheekbone_left", "cheekbone_right",
    "cheek_hollow_left", "cheek_hollow_right",
    "nasolabial_left", "nasolabial_right",
    "lip_upper", "lip_upper_left", "lip_upper_right", "lip_upper_center",
    "lip_lower", "lip_lower_left", "lip_lower_right", "lip_lower_center",
    "lip_corner_left", "lip_corner_right",
    "chin", "chin_center", "chin_left", "chin_right",
    "jaw_left", "jaw_right", "jawline_left", "jawline_right",
    "temple_left", "temple_right",
    "ear_left", "ear_right",
    "neck",
]


def _compute_vertex_stats(v_template):
    """Compute per-vertex statistics used for subdivision.

    Returns x, y, z arrays and useful derived quantities.
    FLAME coordinate system: X = left/right, Y = up/down, Z = forward/back.
    """
    x = v_template[:, 0]
    y = v_template[:, 1]
    z = v_template[:, 2]

    # Compute useful reference points from the mesh
    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    # Normalize to [0, 1] range for easier threshold setting
    y_range = y_max - y_min
    x_range = x_max - x_min

    return {
        'x': x, 'y': y, 'z': z,
        'y_min': y_min, 'y_max': y_max,
        'x_min': x_min, 'x_max': x_max,
        'z_min': z_min, 'z_max': z_max,
        'y_range': y_range, 'x_range': x_range,
        # Normalized coordinates (0 = bottom/left, 1 = top/right)
        'yn': (y - y_min) / (y_range if y_range > 0 else 1),
        'xn': (x - x_min) / (x_range if x_range > 0 else 1),
    }


def subdivide_flame_masks_to_clinical_zones(v_template, flame_masks):
    """Map FLAME vertex masks (~12 broad regions) into our 52+ clinical zones.

    Uses vertex positions from v_template to subdivide FLAME's broad regions
    into finer clinical sub-regions.

    Parameters
    ----------
    v_template : ndarray, shape (N, 3)
        Mean vertex positions from the FLAME model.
    flame_masks : dict
        Keys are FLAME region names (e.g., 'nose', 'lips', 'forehead', 'face',
        'left_eye_region', 'right_eye_region', 'neck', 'left_ear', 'right_ear',
        'scalp', 'boundary', 'left_eyeball', 'right_eyeball').
        Values are arrays of vertex indices (or boolean masks).

    Returns
    -------
    dict
        Maps each of our clinical zone names to a sorted list of vertex indices.
    """
    n_verts = v_template.shape[0]
    stats = _compute_vertex_stats(v_template)
    x, y, z = stats['x'], stats['y'], stats['z']

    # Convert FLAME mask values to index arrays
    # FLAME masks can be boolean arrays or index arrays
    def _mask_to_indices(mask_val):
        if mask_val is None:
            return np.array([], dtype=np.int64)
        arr = _to_numpy(mask_val)
        if arr is None or arr.size == 0:
            return np.array([], dtype=np.int64)
        if arr.dtype == bool or (arr.ndim == 1 and arr.shape[0] == n_verts and arr.max() <= 1):
            # Boolean mask
            return np.where(arr.astype(bool))[0]
        # Already index array
        return arr.astype(np.int64).flatten()

    # Extract FLAME mask index arrays
    fm = {}
    for key, val in flame_masks.items():
        fm[key] = _mask_to_indices(val)

    # Convenience: gather broad regions
    nose_idx = fm.get('nose', np.array([], dtype=np.int64))
    lips_idx = fm.get('lips', np.array([], dtype=np.int64))
    forehead_idx = fm.get('forehead', np.array([], dtype=np.int64))
    left_eye_idx = fm.get('left_eye_region', np.array([], dtype=np.int64))
    right_eye_idx = fm.get('right_eye_region', np.array([], dtype=np.int64))
    face_idx = fm.get('face', np.array([], dtype=np.int64))
    neck_idx = fm.get('neck', np.array([], dtype=np.int64))
    left_ear_idx = fm.get('left_ear', np.array([], dtype=np.int64))
    right_ear_idx = fm.get('right_ear', np.array([], dtype=np.int64))
    scalp_idx = fm.get('scalp', np.array([], dtype=np.int64))
    left_eyeball_idx = fm.get('left_eyeball', np.array([], dtype=np.int64))
    right_eyeball_idx = fm.get('right_eyeball', np.array([], dtype=np.int64))

    # Build a set of all classified vertices to identify remaining "face" vertices
    specific_regions = set()
    for key in ['nose', 'lips', 'forehead', 'left_eye_region', 'right_eye_region',
                'neck', 'left_ear', 'right_ear', 'left_eyeball', 'right_eyeball']:
        if key in fm:
            specific_regions.update(fm[key].tolist())

    # "Face" remainder = face mask minus specific sub-regions
    if len(face_idx) > 0:
        face_remainder = np.array(sorted(set(face_idx.tolist()) - specific_regions), dtype=np.int64)
    else:
        face_remainder = np.array([], dtype=np.int64)

    # Initialize the output regions
    regions = {name: [] for name in ALL_CLINICAL_ZONES}

    # -----------------------------------------------------------------------
    # Helper: subdivide an index set using position-based thresholds.
    # -----------------------------------------------------------------------
    def _split_lr(indices, threshold=0.0):
        """Split indices into left (x > threshold) and right (x <= threshold)."""
        if len(indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        mask_left = x[indices] > threshold
        return indices[mask_left], indices[~mask_left]

    def _split_upper_lower(indices, threshold):
        """Split indices into upper (y > threshold) and lower (y <= threshold)."""
        if len(indices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        mask_upper = y[indices] > threshold
        return indices[mask_upper], indices[~mask_upper]

    def _filter_x_range(indices, x_lo, x_hi):
        """Keep only vertices with x in [x_lo, x_hi]."""
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        mask = (x[indices] >= x_lo) & (x[indices] <= x_hi)
        return indices[mask]

    def _filter_y_range(indices, y_lo, y_hi):
        """Keep only vertices with y in [y_lo, y_hi]."""
        if len(indices) == 0:
            return np.array([], dtype=np.int64)
        mask = (y[indices] >= y_lo) & (y[indices] <= y_hi)
        return indices[mask]

    # -----------------------------------------------------------------------
    # Compute reference landmarks from vertex positions
    # -----------------------------------------------------------------------
    # Median positions within each FLAME mask give us stable reference points
    # for placing subdivision thresholds.

    def _median_y(idx):
        if len(idx) == 0:
            return 0.0
        return float(np.median(y[idx]))

    def _median_x(idx):
        if len(idx) == 0:
            return 0.0
        return float(np.median(x[idx]))

    def _percentile_y(idx, pct):
        if len(idx) == 0:
            return 0.0
        return float(np.percentile(y[idx], pct))

    def _percentile_x(idx, pct):
        if len(idx) == 0:
            return 0.0
        return float(np.percentile(x[idx], pct))

    nose_med_y = _median_y(nose_idx)
    nose_med_x = _median_x(nose_idx)
    lips_med_y = _median_y(lips_idx)

    # X = 0 is roughly the midline of the face in FLAME
    midline_x = 0.0
    if len(nose_idx) > 0:
        midline_x = float(np.median(x[nose_idx]))

    # -----------------------------------------------------------------------
    # FOREHEAD subdivision
    # -----------------------------------------------------------------------
    if len(forehead_idx) > 0:
        fh_y_med = _median_y(forehead_idx)
        fh_y_lo = _percentile_y(forehead_idx, 15)
        fh_x_lo = _percentile_x(forehead_idx, 25)
        fh_x_hi = _percentile_x(forehead_idx, 75)

        # Center strip: middle third by X
        fh_third_width = (fh_x_hi - fh_x_lo) / 3.0
        center_lo = fh_x_lo + fh_third_width
        center_hi = fh_x_hi - fh_third_width

        regions['forehead'] = sorted(forehead_idx.tolist())

        for vi in forehead_idx:
            vx, vy = x[vi], y[vi]
            if vx > center_hi:
                # In FLAME, positive X can be left or right depending on convention.
                # We follow the convention: positive X = subject's left
                regions['forehead_left'].append(int(vi))
            elif vx < center_lo:
                regions['forehead_right'].append(int(vi))
            else:
                regions['forehead_center'].append(int(vi))

        # Brows: lower portion of forehead region
        brow_y_threshold = fh_y_lo + (fh_y_med - fh_y_lo) * 0.3
        brow_candidates = forehead_idx[y[forehead_idx] <= brow_y_threshold]

        brow_left_cands, brow_right_cands = _split_lr(brow_candidates, midline_x)
        regions['brow_left'] = sorted(brow_left_cands.tolist())
        regions['brow_right'] = sorted(brow_right_cands.tolist())

        # Inner brows: closest to midline
        inner_brow_x_range = (fh_x_hi - fh_x_lo) * 0.15
        if len(brow_left_cands) > 0:
            regions['brow_inner_left'] = sorted(
                brow_left_cands[x[brow_left_cands] < midline_x + inner_brow_x_range].tolist()
            )
        if len(brow_right_cands) > 0:
            regions['brow_inner_right'] = sorted(
                brow_right_cands[x[brow_right_cands] > midline_x - inner_brow_x_range].tolist()
            )

        # Temples: outermost vertices of the forehead (and scalp if available)
        temple_x_threshold_hi = _percentile_x(forehead_idx, 85)
        temple_x_threshold_lo = _percentile_x(forehead_idx, 15)
        temple_y_lo = _percentile_y(forehead_idx, 10)
        temple_y_hi = _percentile_y(forehead_idx, 70)

        temple_candidates = forehead_idx[
            (y[forehead_idx] >= temple_y_lo) & (y[forehead_idx] <= temple_y_hi)
        ]
        if len(temple_candidates) > 0:
            regions['temple_left'] = sorted(
                temple_candidates[x[temple_candidates] > temple_x_threshold_hi].tolist()
            )
            regions['temple_right'] = sorted(
                temple_candidates[x[temple_candidates] < temple_x_threshold_lo].tolist()
            )

    # -----------------------------------------------------------------------
    # EYE REGION subdivision
    # -----------------------------------------------------------------------
    for side, eye_idx, prefix_upper, prefix_lower, corner_inner, corner_outer, \
        under_eye_name, tear_trough_name in [
        ('left', left_eye_idx, 'eye_left_upper', 'eye_left_lower',
         'eye_left_corner_inner', 'eye_left_corner_outer',
         'under_eye_left', 'tear_trough_left'),
        ('right', right_eye_idx, 'eye_right_upper', 'eye_right_lower',
         'eye_right_corner_inner', 'eye_right_corner_outer',
         'under_eye_right', 'tear_trough_right'),
    ]:
        if len(eye_idx) == 0:
            continue

        eye_med_y = _median_y(eye_idx)
        eye_med_x = _median_x(eye_idx)
        eye_x_lo = _percentile_x(eye_idx, 10)
        eye_x_hi = _percentile_x(eye_idx, 90)
        eye_y_lo = _percentile_y(eye_idx, 10)
        eye_y_hi = _percentile_y(eye_idx, 90)

        # Remove eyeball vertices from the eye region if they exist
        eyeball_set = set()
        if side == 'left' and len(left_eyeball_idx) > 0:
            eyeball_set = set(left_eyeball_idx.tolist())
        elif side == 'right' and len(right_eyeball_idx) > 0:
            eyeball_set = set(right_eyeball_idx.tolist())

        eye_skin_idx = np.array([vi for vi in eye_idx if vi not in eyeball_set], dtype=np.int64)
        if len(eye_skin_idx) == 0:
            eye_skin_idx = eye_idx

        upper_eye, lower_eye = _split_upper_lower(eye_skin_idx, eye_med_y)
        regions[prefix_upper] = sorted(upper_eye.tolist())
        regions[prefix_lower] = sorted(lower_eye.tolist())

        # Eye corners: innermost and outermost by X
        # Inner corner = closest to midline, outer = farthest from midline
        corner_band = eye_skin_idx[
            np.abs(y[eye_skin_idx] - eye_med_y) < (eye_y_hi - eye_y_lo) * 0.3
        ]
        if len(corner_band) > 0:
            if side == 'left':
                # Left eye: inner corner = low X (toward midline), outer = high X
                inner_thresh = _percentile_x(corner_band, 15)
                outer_thresh = _percentile_x(corner_band, 85)
                regions[corner_inner] = sorted(
                    corner_band[x[corner_band] <= inner_thresh].tolist()
                )
                regions[corner_outer] = sorted(
                    corner_band[x[corner_band] >= outer_thresh].tolist()
                )
            else:
                # Right eye: inner corner = high X (toward midline), outer = low X
                inner_thresh = _percentile_x(corner_band, 85)
                outer_thresh = _percentile_x(corner_band, 15)
                regions[corner_inner] = sorted(
                    corner_band[x[corner_band] >= inner_thresh].tolist()
                )
                regions[corner_outer] = sorted(
                    corner_band[x[corner_band] <= outer_thresh].tolist()
                )

        # Under-eye: lower portion of eye region
        under_eye_candidates = lower_eye
        regions[under_eye_name] = sorted(under_eye_candidates.tolist())

        # Tear trough: inner-lower portion of eye region
        if len(under_eye_candidates) > 0:
            ue_med_x = _median_x(under_eye_candidates)
            if side == 'left':
                # Tear trough is medial (toward nose)
                regions[tear_trough_name] = sorted(
                    under_eye_candidates[x[under_eye_candidates] < ue_med_x].tolist()
                )
            else:
                regions[tear_trough_name] = sorted(
                    under_eye_candidates[x[under_eye_candidates] > ue_med_x].tolist()
                )

    # -----------------------------------------------------------------------
    # NOSE subdivision
    # -----------------------------------------------------------------------
    if len(nose_idx) > 0:
        nose_y_lo = _percentile_y(nose_idx, 5)
        nose_y_hi = _percentile_y(nose_idx, 95)
        nose_y_range = nose_y_hi - nose_y_lo
        nose_x_lo = _percentile_x(nose_idx, 10)
        nose_x_hi = _percentile_x(nose_idx, 90)

        # Nose bridge: upper 60% of nose, narrow central strip
        bridge_y_thresh = nose_y_lo + nose_y_range * 0.40
        bridge_x_half = (nose_x_hi - nose_x_lo) * 0.35
        bridge_cands = nose_idx[
            (y[nose_idx] > bridge_y_thresh) &
            (np.abs(x[nose_idx] - midline_x) < bridge_x_half)
        ]
        regions['nose_bridge'] = sorted(bridge_cands.tolist())
        regions['nose_dorsum'] = sorted(bridge_cands.tolist())  # dorsum spans full bridge

        bridge_mid_y = _median_y(bridge_cands) if len(bridge_cands) > 0 else nose_med_y
        if len(bridge_cands) > 0:
            regions['nose_bridge_upper'] = sorted(
                bridge_cands[y[bridge_cands] > bridge_mid_y].tolist()
            )
            regions['nose_bridge_lower'] = sorted(
                bridge_cands[y[bridge_cands] <= bridge_mid_y].tolist()
            )

        # Nose tip: lower-center of nose
        tip_y_thresh = nose_y_lo + nose_y_range * 0.30
        tip_cands = nose_idx[
            (y[nose_idx] <= tip_y_thresh) &
            (y[nose_idx] > nose_y_lo + nose_y_range * 0.10)
        ]
        # Also filter to central X band for the tip
        if len(tip_cands) > 0:
            tip_x_half = (nose_x_hi - nose_x_lo) * 0.35
            tip_cands = tip_cands[np.abs(x[tip_cands] - midline_x) < tip_x_half]

        regions['nose_tip'] = sorted(tip_cands.tolist())
        if len(tip_cands) > 0:
            regions['nose_tip_left'] = sorted(
                tip_cands[x[tip_cands] > midline_x].tolist()
            )
            regions['nose_tip_right'] = sorted(
                tip_cands[x[tip_cands] <= midline_x].tolist()
            )

        # Nostrils: lower-lateral parts of nose
        nostril_y_thresh = nose_y_lo + nose_y_range * 0.30
        nostril_cands = nose_idx[y[nose_idx] <= nostril_y_thresh]
        nostril_x_inner = (nose_x_hi - nose_x_lo) * 0.15
        if len(nostril_cands) > 0:
            regions['nostril_left'] = sorted(
                nostril_cands[x[nostril_cands] > midline_x + nostril_x_inner].tolist()
            )
            regions['nostril_right'] = sorted(
                nostril_cands[x[nostril_cands] < midline_x - nostril_x_inner].tolist()
            )

    # -----------------------------------------------------------------------
    # LIPS subdivision
    # -----------------------------------------------------------------------
    if len(lips_idx) > 0:
        lips_y_med = _median_y(lips_idx)
        lips_x_lo = _percentile_x(lips_idx, 5)
        lips_x_hi = _percentile_x(lips_idx, 95)
        lips_x_third = (lips_x_hi - lips_x_lo) / 3.0

        upper_lip, lower_lip = _split_upper_lower(lips_idx, lips_y_med)
        regions['lip_upper'] = sorted(upper_lip.tolist())
        regions['lip_lower'] = sorted(lower_lip.tolist())

        # Upper lip sub-regions
        center_lo_x = lips_x_lo + lips_x_third
        center_hi_x = lips_x_hi - lips_x_third

        if len(upper_lip) > 0:
            regions['lip_upper_left'] = sorted(
                upper_lip[x[upper_lip] > center_hi_x].tolist()
            )
            regions['lip_upper_right'] = sorted(
                upper_lip[x[upper_lip] < center_lo_x].tolist()
            )
            regions['lip_upper_center'] = sorted(
                upper_lip[(x[upper_lip] >= center_lo_x) & (x[upper_lip] <= center_hi_x)].tolist()
            )

        # Lower lip sub-regions
        if len(lower_lip) > 0:
            regions['lip_lower_left'] = sorted(
                lower_lip[x[lower_lip] > center_hi_x].tolist()
            )
            regions['lip_lower_right'] = sorted(
                lower_lip[x[lower_lip] < center_lo_x].tolist()
            )
            regions['lip_lower_center'] = sorted(
                lower_lip[(x[lower_lip] >= center_lo_x) & (x[lower_lip] <= center_hi_x)].tolist()
            )

        # Lip corners: outermost vertices near the lip midline Y
        corner_band_y = (lips_y_med - (y[lips_idx].max() - y[lips_idx].min()) * 0.20,
                         lips_y_med + (y[lips_idx].max() - y[lips_idx].min()) * 0.20)
        corner_cands = lips_idx[
            (y[lips_idx] >= corner_band_y[0]) & (y[lips_idx] <= corner_band_y[1])
        ]
        if len(corner_cands) > 0:
            corner_x_thresh_hi = _percentile_x(corner_cands, 85)
            corner_x_thresh_lo = _percentile_x(corner_cands, 15)
            regions['lip_corner_left'] = sorted(
                corner_cands[x[corner_cands] > corner_x_thresh_hi].tolist()
            )
            regions['lip_corner_right'] = sorted(
                corner_cands[x[corner_cands] < corner_x_thresh_lo].tolist()
            )

    # -----------------------------------------------------------------------
    # FACE remainder -> cheeks, nasolabial, chin, jaw
    # -----------------------------------------------------------------------
    if len(face_remainder) > 0:
        fr_y = y[face_remainder]
        fr_x = x[face_remainder]
        fr_y_lo = _percentile_y(face_remainder, 5)
        fr_y_hi = _percentile_y(face_remainder, 95)
        fr_y_range = fr_y_hi - fr_y_lo

        # Chin: bottom portion of face remainder, central
        chin_y_thresh = fr_y_lo + fr_y_range * 0.25
        chin_x_half = (x[face_remainder].max() - x[face_remainder].min()) * 0.25
        chin_cands = face_remainder[
            (fr_y <= chin_y_thresh) &
            (np.abs(fr_x - midline_x) < chin_x_half)
        ]
        regions['chin'] = sorted(chin_cands.tolist())
        if len(chin_cands) > 0:
            chin_third = chin_x_half * 2 / 3
            regions['chin_center'] = sorted(
                chin_cands[np.abs(x[chin_cands] - midline_x) < chin_third / 2].tolist()
            )
            regions['chin_left'] = sorted(
                chin_cands[x[chin_cands] > midline_x + chin_third / 2].tolist()
            )
            regions['chin_right'] = sorted(
                chin_cands[x[chin_cands] < midline_x - chin_third / 2].tolist()
            )

        # Jaw: lower-lateral portions of face
        jaw_y_thresh = fr_y_lo + fr_y_range * 0.40
        jaw_x_inner = chin_x_half
        jaw_cands = face_remainder[
            (fr_y <= jaw_y_thresh) &
            (np.abs(fr_x - midline_x) >= jaw_x_inner)
        ]
        jaw_left, jaw_right = _split_lr(jaw_cands, midline_x)
        regions['jaw_left'] = sorted(jaw_left.tolist())
        regions['jaw_right'] = sorted(jaw_right.tolist())

        # Jawline: the lower edge of the jaw
        if len(jaw_left) > 0:
            jl_y_thresh = _percentile_y(jaw_left, 35)
            regions['jawline_left'] = sorted(
                jaw_left[y[jaw_left] < jl_y_thresh].tolist()
            )
        if len(jaw_right) > 0:
            jr_y_thresh = _percentile_y(jaw_right, 35)
            regions['jawline_right'] = sorted(
                jaw_right[y[jaw_right] < jr_y_thresh].tolist()
            )

        # Cheeks: mid-lateral portions of face
        cheek_y_lo = fr_y_lo + fr_y_range * 0.25
        cheek_y_hi = fr_y_lo + fr_y_range * 0.70
        cheek_x_inner = (x[face_remainder].max() - x[face_remainder].min()) * 0.10
        cheek_cands = face_remainder[
            (fr_y >= cheek_y_lo) & (fr_y <= cheek_y_hi) &
            (np.abs(fr_x - midline_x) > cheek_x_inner)
        ]
        cheek_left, cheek_right = _split_lr(cheek_cands, midline_x)
        regions['cheek_left'] = sorted(cheek_left.tolist())
        regions['cheek_right'] = sorted(cheek_right.tolist())

        # Cheekbone: upper cheek
        cheek_mid_y = _median_y(cheek_cands) if len(cheek_cands) > 0 else (cheek_y_lo + cheek_y_hi) / 2
        if len(cheek_left) > 0:
            regions['cheekbone_left'] = sorted(
                cheek_left[y[cheek_left] > cheek_mid_y].tolist()
            )
            regions['cheek_hollow_left'] = sorted(
                cheek_left[y[cheek_left] <= cheek_mid_y].tolist()
            )
        if len(cheek_right) > 0:
            regions['cheekbone_right'] = sorted(
                cheek_right[y[cheek_right] > cheek_mid_y].tolist()
            )
            regions['cheek_hollow_right'] = sorted(
                cheek_right[y[cheek_right] <= cheek_mid_y].tolist()
            )

        # Nasolabial folds: narrow strip between nose and cheek
        if len(nose_idx) > 0 and len(lips_idx) > 0:
            nl_y_lo = _percentile_y(lips_idx, 50)
            nl_y_hi = _percentile_y(nose_idx, 30)
            nose_x_range = _percentile_x(nose_idx, 90) - _percentile_x(nose_idx, 10)
            nl_x_inner = nose_x_range * 0.4
            nl_x_outer = nose_x_range * 0.9

            nl_cands = face_remainder[
                (fr_y >= nl_y_lo) & (fr_y <= nl_y_hi) &
                (np.abs(fr_x - midline_x) >= nl_x_inner) &
                (np.abs(fr_x - midline_x) <= nl_x_outer)
            ]
            nl_left, nl_right = _split_lr(nl_cands, midline_x)
            regions['nasolabial_left'] = sorted(nl_left.tolist())
            regions['nasolabial_right'] = sorted(nl_right.tolist())

    # -----------------------------------------------------------------------
    # Ears and neck (pass-through from FLAME masks)
    # -----------------------------------------------------------------------
    regions['ear_left'] = sorted(left_ear_idx.tolist())
    regions['ear_right'] = sorted(right_ear_idx.tolist())
    regions['neck'] = sorted(neck_idx.tolist())

    # -----------------------------------------------------------------------
    # Temples: if not already populated from forehead, try using scalp + face
    # -----------------------------------------------------------------------
    if not regions['temple_left'] and len(scalp_idx) > 0:
        # Temples from scalp: low-lateral scalp vertices
        scalp_y_lo = _percentile_y(scalp_idx, 5)
        scalp_y_med = _median_y(scalp_idx)
        scalp_low = scalp_idx[y[scalp_idx] < scalp_y_med]
        if len(scalp_low) > 0:
            temple_x_thresh = _percentile_x(scalp_low, 70)
            regions['temple_left'] = sorted(
                scalp_low[x[scalp_low] > temple_x_thresh].tolist()
            )
            temple_x_thresh_r = _percentile_x(scalp_low, 30)
            regions['temple_right'] = sorted(
                scalp_low[x[scalp_low] < temple_x_thresh_r].tolist()
            )

    # -----------------------------------------------------------------------
    # full_face: union of all facial regions (excluding neck and ears)
    # -----------------------------------------------------------------------
    full_face_set = set()
    exclude_from_full = {'neck', 'ear_left', 'ear_right', 'full_face'}
    for name in ALL_CLINICAL_ZONES:
        if name not in exclude_from_full:
            full_face_set.update(regions[name])
    # Also include any remaining face vertices
    if len(face_idx) > 0:
        full_face_set.update(face_idx.tolist())
    if len(forehead_idx) > 0:
        full_face_set.update(forehead_idx.tolist())
    if len(nose_idx) > 0:
        full_face_set.update(nose_idx.tolist())
    if len(lips_idx) > 0:
        full_face_set.update(lips_idx.tolist())
    if len(left_eye_idx) > 0:
        full_face_set.update(left_eye_idx.tolist())
    if len(right_eye_idx) > 0:
        full_face_set.update(right_eye_idx.tolist())
    regions['full_face'] = sorted(full_face_set)

    return regions


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def convert_flame_model(flame_dir, output_dir, n_shape_components=50, n_expr_components=50):
    """Convert FLAME model files to web-friendly format.

    Parameters
    ----------
    flame_dir : str or Path
        Root directory containing flame2023/, vertex_masks/, mediapipe/
    output_dir : str or Path
        Output directory for web-friendly files.
    n_shape_components : int
        Number of shape basis components to export (default 50).
    n_expr_components : int
        Number of expression basis components to export (default 50).
    """
    flame_dir = Path(flame_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    file_sizes = {}
    albedo_uv_coords = None
    albedo_uv_faces = None

    # -----------------------------------------------------------------------
    # 1. Load FLAME model
    # -----------------------------------------------------------------------
    model_path = find_flame_model(flame_dir)
    model_data = None
    v_template = None
    faces = None
    shapedirs = None
    exprdirs = None
    uv_coords = None

    if model_path:
        print(f"[INFO] Loading FLAME model from: {model_path}")
        model_data = _load_pickle(str(model_path))

        if model_data is not None:
            # Extract arrays from the model dictionary
            v_template = _to_numpy(model_data.get('v_template'))
            faces = _to_numpy(model_data.get('f'))
            shapedirs_raw = _to_numpy(model_data.get('shapedirs'))
            exprdirs = _to_numpy(model_data.get('exprdirs'))

            # FLAME 2023 stores 300 shape + 100 expression = 400 total in shapedirs
            # Older versions may have separate exprdirs. Handle both cases.
            if shapedirs_raw is not None and shapedirs_raw.ndim == 3:
                total_components = shapedirs_raw.shape[2]
                if exprdirs is None and total_components > 300:
                    # Split: first 300 = shape, rest = expression
                    n_shape_total = 300
                    n_expr_total = total_components - 300
                    shapedirs = shapedirs_raw[:, :, :n_shape_total]
                    exprdirs = shapedirs_raw[:, :, n_shape_total:]
                    print(f"  [INFO] Split shapedirs ({total_components}) into "
                          f"shape ({n_shape_total}) + expression ({n_expr_total})")
                else:
                    shapedirs = shapedirs_raw
            else:
                shapedirs = shapedirs_raw

            # UV coordinates may be stored under various keys
            for uv_key in ['vt', 'uv', 'texcoords', 'texture_coordinates']:
                if uv_key in model_data:
                    uv_coords = _to_numpy(model_data[uv_key])
                    break

            # Also check for face UV indices
            ft = _to_numpy(model_data.get('ft'))

            if v_template is not None:
                print(f"  v_template: {v_template.shape} ({v_template.dtype})")
                summary['vertex_count'] = int(v_template.shape[0])
            if faces is not None:
                print(f"  faces (f): {faces.shape} ({faces.dtype})")
                summary['face_count'] = int(faces.shape[0])
            if shapedirs is not None:
                print(f"  shapedirs: {shapedirs.shape} ({shapedirs.dtype})")
                summary['total_shape_components'] = int(shapedirs.shape[2]) if shapedirs.ndim == 3 else 0
            if exprdirs is not None:
                print(f"  exprdirs: {exprdirs.shape} ({exprdirs.dtype})")
                summary['total_expression_components'] = int(exprdirs.shape[2]) if exprdirs.ndim == 3 else 0
            if uv_coords is not None:
                print(f"  UV coords: {uv_coords.shape}")

            # Print all available keys for reference
            if isinstance(model_data, dict):
                print(f"  Available keys: {list(model_data.keys())}")
        else:
            print("[WARN] Failed to parse FLAME model pickle.")
    else:
        print("[WARN] FLAME model pickle not found in flame2023/. Skipping model conversion.")

    # -----------------------------------------------------------------------
    # 2. Write binary files
    # -----------------------------------------------------------------------

    # -- flame_template_vertices.bin --
    if v_template is not None:
        verts_flat = v_template.astype(np.float32).flatten()
        out_path = output_dir / "flame_template_vertices.bin"
        verts_flat.tofile(str(out_path))
        file_sizes['flame_template_vertices.bin'] = out_path.stat().st_size
        print(f"  Wrote {out_path.name}: {file_sizes['flame_template_vertices.bin']:,} bytes "
              f"({v_template.shape[0]} vertices x 3 floats)")

    # -- flame_shape_basis.bin --
    if shapedirs is not None and shapedirs.ndim == 3:
        n_shape = min(n_shape_components, shapedirs.shape[2])
        shape_subset = shapedirs[:, :, :n_shape].astype(np.float32)
        shape_flat = shape_subset.flatten()
        out_path = output_dir / "flame_shape_basis.bin"
        shape_flat.tofile(str(out_path))
        file_sizes['flame_shape_basis.bin'] = out_path.stat().st_size
        summary['exported_shape_components'] = n_shape
        print(f"  Wrote {out_path.name}: {file_sizes['flame_shape_basis.bin']:,} bytes "
              f"({v_template.shape[0]} x 3 x {n_shape} floats)")

    # -- flame_expression_basis.bin --
    if exprdirs is not None and exprdirs.ndim == 3:
        n_expr = min(n_expr_components, exprdirs.shape[2])
        expr_subset = exprdirs[:, :, :n_expr].astype(np.float32)
        expr_flat = expr_subset.flatten()
        out_path = output_dir / "flame_expression_basis.bin"
        expr_flat.tofile(str(out_path))
        file_sizes['flame_expression_basis.bin'] = out_path.stat().st_size
        summary['exported_expression_components'] = n_expr
        print(f"  Wrote {out_path.name}: {file_sizes['flame_expression_basis.bin']:,} bytes "
              f"({v_template.shape[0]} x 3 x {n_expr} floats)")

    # -- flame_faces.bin --
    if faces is not None:
        faces_uint32 = faces.astype(np.uint32).flatten()
        out_path = output_dir / "flame_faces.bin"
        faces_uint32.tofile(str(out_path))
        file_sizes['flame_faces.bin'] = out_path.stat().st_size
        print(f"  Wrote {out_path.name}: {file_sizes['flame_faces.bin']:,} bytes "
              f"({faces.shape[0]} triangles x 3 uints)")

    # -- flame_uv.bin --
    # Note: UV coords typically come from the albedo/texture model, not from the
    # FLAME pickle itself. We write them here if found in the model, or later
    # when processing the albedo model.
    if uv_coords is not None:
        uv_flat = uv_coords.astype(np.float32).flatten()
        out_path = output_dir / "flame_uv.bin"
        uv_flat.tofile(str(out_path))
        file_sizes['flame_uv.bin'] = out_path.stat().st_size
        print(f"  Wrote {out_path.name}: {file_sizes['flame_uv.bin']:,} bytes "
              f"({uv_coords.shape[0]} UV coords)")
    else:
        print("  [SKIP] No UV coordinates in FLAME model. Will try albedo/texture model later.")

    # -- flame_template.json (built here, written after all data is loaded) --
    template_json = None
    if v_template is not None:
        template_json = {
            'vertex_count': int(v_template.shape[0]),
            'face_count': int(faces.shape[0]) if faces is not None else 0,
            'shape_param_count': summary.get('exported_shape_components', 0),
            'expression_param_count': summary.get('exported_expression_components', 0),
            'total_shape_components': summary.get('total_shape_components', 0),
            'total_expression_components': summary.get('total_expression_components', 0),
            'has_uv': uv_coords is not None,
            'coordinate_system': 'FLAME: X=right, Y=up, Z=toward-viewer',
            'binary_files': {
                'vertices': 'flame_template_vertices.bin',
                'shape_basis': 'flame_shape_basis.bin',
                'expression_basis': 'flame_expression_basis.bin',
                'faces': 'flame_faces.bin',
                'uv': 'flame_uv.bin' if uv_coords is not None else None,
            },
            'data_types': {
                'vertices': 'Float32Array',
                'shape_basis': 'Float32Array',
                'expression_basis': 'Float32Array',
                'faces': 'Uint32Array',
                'uv': 'Float32Array' if uv_coords is not None else None,
            },
            'vertex_positions_sample': {
                'first_3': v_template[:3].tolist(),
                'last_3': v_template[-3:].tolist(),
            },
        }

        if faces is not None:
            template_json['face_indices_sample'] = {
                'first_3': faces[:3].tolist(),
            }

    # -----------------------------------------------------------------------
    # 3. Load and convert vertex masks -> clinical zones
    # -----------------------------------------------------------------------
    masks_path = find_vertex_masks(flame_dir)
    flame_masks = None

    if masks_path:
        print(f"\n[INFO] Loading vertex masks from: {masks_path}")
        if str(masks_path).endswith('.npz'):
            flame_masks = dict(np.load(str(masks_path), allow_pickle=True))
        else:
            flame_masks = _load_pickle(str(masks_path))

        if flame_masks is not None:
            if isinstance(flame_masks, dict):
                print(f"  Mask regions found: {list(flame_masks.keys())}")
            else:
                print(f"  [WARN] Unexpected mask format: {type(flame_masks)}")
                flame_masks = None
    else:
        print("\n[WARN] Vertex masks file not found. Skipping region conversion.")

    # Convert to clinical zones
    if flame_masks is not None and v_template is not None:
        print("\n[INFO] Subdividing FLAME masks into 52+ clinical zones...")
        clinical_regions = subdivide_flame_masks_to_clinical_zones(v_template, flame_masks)

        # Print region stats
        total_assigned = 0
        empty_regions = []
        for name in ALL_CLINICAL_ZONES:
            count = len(clinical_regions.get(name, []))
            if count == 0:
                empty_regions.append(name)
            total_assigned += count

        print(f"  Total zone assignments: {total_assigned}")
        print(f"  Populated zones: {len(ALL_CLINICAL_ZONES) - len(empty_regions)}/{len(ALL_CLINICAL_ZONES)}")
        if empty_regions:
            print(f"  Empty zones: {empty_regions}")

        # Write flame_regions.json
        regions_json = {
            'zone_count': len(ALL_CLINICAL_ZONES),
            'vertex_count': int(v_template.shape[0]),
            'zones': {},
            'flame_mask_names': list(flame_masks.keys()) if isinstance(flame_masks, dict) else [],
        }
        for name in ALL_CLINICAL_ZONES:
            indices = clinical_regions.get(name, [])
            regions_json['zones'][name] = {
                'vertex_indices': indices,
                'vertex_count': len(indices),
            }

        out_path = output_dir / "flame_regions.json"
        with open(str(out_path), 'w') as f:
            json.dump(regions_json, f, indent=2)
        file_sizes['flame_regions.json'] = out_path.stat().st_size
        print(f"  Wrote {out_path.name}: {file_sizes['flame_regions.json']:,} bytes")
    elif v_template is not None and flame_masks is None:
        # Fallback: create regions using just vertex positions (no FLAME masks)
        print("\n[INFO] No FLAME masks available. Creating position-only region map...")
        print("       (This will be less accurate than mask-based subdivision)")

        # Use a simplified position-based classification
        clinical_regions = _position_only_regions(v_template)

        regions_json = {
            'zone_count': len(ALL_CLINICAL_ZONES),
            'vertex_count': int(v_template.shape[0]),
            'zones': {},
            'flame_mask_names': [],
            'note': 'Position-only classification (no FLAME masks available)',
        }
        for name in ALL_CLINICAL_ZONES:
            indices = clinical_regions.get(name, [])
            regions_json['zones'][name] = {
                'vertex_indices': indices,
                'vertex_count': len(indices),
            }

        out_path = output_dir / "flame_regions.json"
        with open(str(out_path), 'w') as f:
            json.dump(regions_json, f, indent=2)
        file_sizes['flame_regions.json'] = out_path.stat().st_size
        print(f"  Wrote {out_path.name}: {file_sizes['flame_regions.json']:,} bytes")

    # -----------------------------------------------------------------------
    # 4. Load and convert MediaPipe embedding
    # -----------------------------------------------------------------------
    mp_path = find_mediapipe_embedding(flame_dir)

    if mp_path:
        print(f"\n[INFO] Loading MediaPipe embedding from: {mp_path}")
        mp_data = None

        if str(mp_path).endswith('.npz'):
            mp_data = dict(np.load(str(mp_path), allow_pickle=True))
        elif str(mp_path).endswith('.npy'):
            mp_data = np.load(str(mp_path), allow_pickle=True)
            if isinstance(mp_data, np.ndarray) and mp_data.dtype == object:
                mp_data = mp_data.item()
        else:
            mp_data = _load_pickle(str(mp_path))

        if mp_data is not None:
            mp_json = {}

            if isinstance(mp_data, dict):
                print(f"  Keys: {list(mp_data.keys())}")
                # Common keys in FLAME MediaPipe embedding:
                # 'lmk_faces_idx' - face index for each landmark
                # 'lmk_bary_coords' - barycentric coordinates within the face
                # 'landmark_indices' - direct vertex indices
                for key, val in mp_data.items():
                    arr = _to_numpy(val)
                    if arr is not None:
                        mp_json[key] = arr.tolist()
                        print(f"  {key}: shape={arr.shape if hasattr(arr, 'shape') else 'scalar'}")
            elif isinstance(mp_data, np.ndarray):
                mp_json['mapping'] = mp_data.tolist()
                print(f"  Array shape: {mp_data.shape}")
            else:
                print(f"  [WARN] Unexpected MediaPipe data format: {type(mp_data)}")

            if mp_json:
                out_path = output_dir / "flame_mediapipe_mapping.json"
                with open(str(out_path), 'w') as f:
                    json.dump(mp_json, f, indent=2)
                file_sizes['flame_mediapipe_mapping.json'] = out_path.stat().st_size
                print(f"  Wrote {out_path.name}: {file_sizes['flame_mediapipe_mapping.json']:,} bytes")
        else:
            print("  [WARN] Failed to parse MediaPipe embedding file.")
    else:
        print("\n[WARN] MediaPipe embedding file not found. Skipping.")

    # -----------------------------------------------------------------------
    # 5. Load and convert Albedo model (AlbedoMM)
    # -----------------------------------------------------------------------
    albedo_path = find_albedo_model(flame_dir)
    texture_space_path = find_texture_space(flame_dir)
    albedo_uv_coords = None
    albedo_uv_faces = None

    if albedo_path:
        print(f"\n[INFO] Loading AlbedoMM albedo model from: {albedo_path}")
        albedo_data = dict(np.load(str(albedo_path), allow_pickle=True))
        print(f"  Keys: {list(albedo_data.keys())}")

        # Mean diffuse albedo texture (512x512x3, float64, range 0-1)
        mean_diffuse = albedo_data.get('MU')
        # Mean specular albedo texture (512x512x3, float64, range 0-1)
        mean_specular = albedo_data.get('specMU')
        # PCA bases
        diffuse_pc = albedo_data.get('PC')    # (512, 512, 3, 145)
        specular_pc = albedo_data.get('specPC')  # (512, 512, 3, 145)
        # UV mapping
        albedo_uv_coords = albedo_data.get('vt')   # (5118, 2)
        albedo_uv_faces = albedo_data.get('ft')     # (9976, 3)

        # -- Export mean diffuse albedo as raw uint8 RGB (768KB) --
        if mean_diffuse is not None:
            print(f"  Mean diffuse: {mean_diffuse.shape}, range [{mean_diffuse.min():.4f}, {mean_diffuse.max():.4f}]")
            # Flip vertically (texture coordinate convention) and convert to uint8
            tex_diffuse = np.flipud(mean_diffuse)
            tex_diffuse = np.clip(tex_diffuse * 255.0, 0, 255).astype(np.uint8)
            out_path = output_dir / "flame_albedo_diffuse.bin"
            tex_diffuse.tofile(str(out_path))
            file_sizes['flame_albedo_diffuse.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_diffuse.bin']:,} bytes "
                  f"({mean_diffuse.shape[0]}x{mean_diffuse.shape[1]} RGB uint8)")

        # -- Export mean specular albedo as raw uint8 RGB --
        if mean_specular is not None:
            print(f"  Mean specular: {mean_specular.shape}, range [{mean_specular.min():.4f}, {mean_specular.max():.4f}]")
            tex_specular = np.flipud(mean_specular)
            # Specular values are typically lower; scale to full range for better precision
            spec_max = max(mean_specular.max(), 0.01)
            tex_specular_scaled = np.clip(tex_specular / spec_max * 255.0, 0, 255).astype(np.uint8)
            out_path = output_dir / "flame_albedo_specular.bin"
            tex_specular_scaled.tofile(str(out_path))
            file_sizes['flame_albedo_specular.bin'] = out_path.stat().st_size
            summary['specular_scale_factor'] = float(spec_max)
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_specular.bin']:,} bytes "
                  f"(scale factor: {spec_max:.6f})")

        # -- Export UV coordinates from albedo model --
        if albedo_uv_coords is not None:
            print(f"  UV coords (vt): {albedo_uv_coords.shape}")
            uv_flat = albedo_uv_coords.astype(np.float32).flatten()
            out_path = output_dir / "flame_uv.bin"
            uv_flat.tofile(str(out_path))
            file_sizes['flame_uv.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_uv.bin']:,} bytes "
                  f"({albedo_uv_coords.shape[0]} UV coords)")

        if albedo_uv_faces is not None:
            print(f"  UV face indices (ft): {albedo_uv_faces.shape}")
            ft_flat = albedo_uv_faces.astype(np.uint32).flatten()
            out_path = output_dir / "flame_uv_faces.bin"
            ft_flat.tofile(str(out_path))
            file_sizes['flame_uv_faces.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_uv_faces.bin']:,} bytes "
                  f"({albedo_uv_faces.shape[0]} face UV indices)")

        # -- Export top N diffuse PCA components (downsampled to 256x256) --
        n_albedo_pca = 20  # Top 20 diffuse components
        if diffuse_pc is not None:
            n_available = diffuse_pc.shape[3]
            n_export = min(n_albedo_pca, n_available)
            print(f"  Diffuse PCA: {diffuse_pc.shape} ({n_available} components)")
            # Downsample from 512x512 to 256x256 using simple 2x2 averaging
            h, w = diffuse_pc.shape[0] // 2, diffuse_pc.shape[1] // 2
            pc_small = (diffuse_pc[0::2, 0::2, :, :n_export] +
                        diffuse_pc[1::2, 0::2, :, :n_export] +
                        diffuse_pc[0::2, 1::2, :, :n_export] +
                        diffuse_pc[1::2, 1::2, :, :n_export]) / 4.0
            pc_small = np.flipud(pc_small)
            pc_flat = pc_small.astype(np.float16).flatten()
            out_path = output_dir / "flame_albedo_diffuse_pca.bin"
            pc_flat.tofile(str(out_path))
            file_sizes['flame_albedo_diffuse_pca.bin'] = out_path.stat().st_size
            summary['albedo_diffuse_pca_components'] = n_export
            summary['albedo_pca_resolution'] = [h, w]
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_diffuse_pca.bin']:,} bytes "
                  f"({h}x{w}x3x{n_export} float16)")

        # -- Export top N specular PCA components --
        n_spec_pca = 10
        if specular_pc is not None:
            n_available = specular_pc.shape[3]
            n_export = min(n_spec_pca, n_available)
            print(f"  Specular PCA: {specular_pc.shape} ({n_available} components)")
            h, w = specular_pc.shape[0] // 2, specular_pc.shape[1] // 2
            spc_small = (specular_pc[0::2, 0::2, :, :n_export] +
                         specular_pc[1::2, 0::2, :, :n_export] +
                         specular_pc[0::2, 1::2, :, :n_export] +
                         specular_pc[1::2, 1::2, :, :n_export]) / 4.0
            spc_small = np.flipud(spc_small)
            spc_flat = spc_small.astype(np.float16).flatten()
            out_path = output_dir / "flame_albedo_specular_pca.bin"
            spc_flat.tofile(str(out_path))
            file_sizes['flame_albedo_specular_pca.bin'] = out_path.stat().st_size
            summary['albedo_specular_pca_components'] = n_export
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_specular_pca.bin']:,} bytes "
                  f"({h}x{w}x3x{n_export} float16)")

    elif texture_space_path:
        # Fallback: use FLAME texture space if albedo model not available
        print(f"\n[INFO] AlbedoMM not found. Loading FLAME texture space from: {texture_space_path}")
        tex_data = dict(np.load(str(texture_space_path), allow_pickle=True))
        print(f"  Keys: {list(tex_data.keys())}")

        mean_tex = tex_data.get('mean')
        tex_pca = tex_data.get('tex_dir')
        albedo_uv_coords = tex_data.get('vt')
        albedo_uv_faces = tex_data.get('ft')

        if mean_tex is not None:
            print(f"  Mean texture: {mean_tex.shape}")
            tex = np.flipud(mean_tex)
            tex = np.clip(tex * 255.0, 0, 255).astype(np.uint8)
            out_path = output_dir / "flame_albedo_diffuse.bin"
            tex.tofile(str(out_path))
            file_sizes['flame_albedo_diffuse.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_diffuse.bin']:,} bytes")

        # UV coords
        if albedo_uv_coords is not None:
            uv_flat = albedo_uv_coords.astype(np.float32).flatten()
            out_path = output_dir / "flame_uv.bin"
            uv_flat.tofile(str(out_path))
            file_sizes['flame_uv.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_uv.bin']:,} bytes")

        if albedo_uv_faces is not None:
            ft_flat = albedo_uv_faces.astype(np.uint32).flatten()
            out_path = output_dir / "flame_uv_faces.bin"
            ft_flat.tofile(str(out_path))
            file_sizes['flame_uv_faces.bin'] = out_path.stat().st_size
            print(f"  Wrote {out_path.name}: {file_sizes['flame_uv_faces.bin']:,} bytes")

        # Texture PCA
        if tex_pca is not None:
            n_export = min(20, tex_pca.shape[3])
            h, w = tex_pca.shape[0] // 2, tex_pca.shape[1] // 2
            pc_small = (tex_pca[0::2, 0::2, :, :n_export] +
                        tex_pca[1::2, 0::2, :, :n_export] +
                        tex_pca[0::2, 1::2, :, :n_export] +
                        tex_pca[1::2, 1::2, :, :n_export]) / 4.0
            pc_small = np.flipud(pc_small)
            pc_flat = pc_small.astype(np.float16).flatten()
            out_path = output_dir / "flame_albedo_diffuse_pca.bin"
            pc_flat.tofile(str(out_path))
            file_sizes['flame_albedo_diffuse_pca.bin'] = out_path.stat().st_size
            summary['albedo_diffuse_pca_components'] = n_export
            summary['albedo_pca_resolution'] = [h, w]
            print(f"  Wrote {out_path.name}: {file_sizes['flame_albedo_diffuse_pca.bin']:,} bytes")
    else:
        print("\n[INFO] No albedo or texture space model found. Skipping texture conversion.")

    # -----------------------------------------------------------------------
    # 6. Finalize and write flame_template.json
    # -----------------------------------------------------------------------
    if template_json is not None:
        # Add albedo/texture metadata now that all data has been loaded
        if albedo_uv_coords is not None:
            template_json['uv_vertex_count'] = int(albedo_uv_coords.shape[0])
            template_json['has_uv'] = True
            template_json['binary_files']['uv'] = 'flame_uv.bin'
            template_json['binary_files']['uv_faces'] = 'flame_uv_faces.bin'
            template_json['data_types']['uv'] = 'Float32Array'
            template_json['data_types']['uv_faces'] = 'Uint32Array'

        if 'flame_albedo_diffuse.bin' in file_sizes:
            template_json['has_albedo'] = True
            template_json['albedo_resolution'] = [512, 512]
            template_json['binary_files']['albedo_diffuse'] = 'flame_albedo_diffuse.bin'
            template_json['binary_files']['albedo_specular'] = (
                'flame_albedo_specular.bin' if 'flame_albedo_specular.bin' in file_sizes else None
            )
            template_json['data_types']['albedo_diffuse'] = 'Uint8Array'
            template_json['data_types']['albedo_specular'] = 'Uint8Array'
            if 'specular_scale_factor' in summary:
                template_json['specular_scale_factor'] = summary['specular_scale_factor']
            if 'albedo_diffuse_pca_components' in summary:
                template_json['albedo_pca'] = {
                    'diffuse_components': summary.get('albedo_diffuse_pca_components', 0),
                    'specular_components': summary.get('albedo_specular_pca_components', 0),
                    'resolution': summary.get('albedo_pca_resolution', [256, 256]),
                    'diffuse_file': 'flame_albedo_diffuse_pca.bin',
                    'specular_file': (
                        'flame_albedo_specular_pca.bin'
                        if 'flame_albedo_specular_pca.bin' in file_sizes else None
                    ),
                    'dtype': 'Float16Array',
                }

        out_path = output_dir / "flame_template.json"
        with open(str(out_path), 'w') as f:
            json.dump(template_json, f, indent=2)
        file_sizes['flame_template.json'] = out_path.stat().st_size
        print(f"\n  Wrote {out_path.name}: {file_sizes['flame_template.json']:,} bytes (final with all metadata)")

    # -----------------------------------------------------------------------
    # 7. Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"  Input directory:  {flame_dir}")
    print(f"  Output directory: {output_dir}")
    print()

    if summary:
        print("  Model stats:")
        for key, val in summary.items():
            print(f"    {key}: {val}")
        print()

    if file_sizes:
        print("  Output files:")
        total_size = 0
        for fname, size in sorted(file_sizes.items()):
            total_size += size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"    {fname:40s} {size_str:>10s}")
        print(f"    {'---':40s} {'---':>10s}")
        if total_size > 1024 * 1024:
            print(f"    {'TOTAL':40s} {total_size/(1024*1024):.1f} MB")
        else:
            print(f"    {'TOTAL':40s} {total_size/1024:.1f} KB")
    else:
        print("  No files were written. Check that model files exist in the expected locations.")

    print()
    return file_sizes


def _position_only_regions(v_template):
    """Fallback: classify vertices into clinical zones using only vertex positions.

    This is used when FLAME vertex masks are not available. It produces a rougher
    classification based on the mesh geometry alone.
    """
    n_verts = v_template.shape[0]
    x, y, z = v_template[:, 0], v_template[:, 1], v_template[:, 2]

    y_min, y_max = y.min(), y.max()
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    y_range = y_max - y_min
    x_range = x_max - x_min
    midline = (x_min + x_max) / 2.0

    # Only classify front-facing vertices
    z_mid = (z_min + z_max) / 2.0

    regions = {name: [] for name in ALL_CLINICAL_ZONES}

    for vi in range(n_verts):
        vx, vy, vz = x[vi], y[vi], z[vi]
        yn = (vy - y_min) / y_range  # 0 = bottom, 1 = top
        xn = (vx - midline) / (x_range / 2.0)  # -1 = right, +1 = left
        is_front = vz > z_mid

        # Neck: bottom 10%
        if yn < 0.10:
            regions['neck'].append(vi)
            continue

        if not is_front:
            continue

        # Full face: everything front-facing above neck
        if yn > 0.10:
            regions['full_face'].append(vi)

        # Forehead: top 20% of front face
        if yn > 0.75:
            regions['forehead'].append(vi)
            if xn > 0.15:
                regions['forehead_left'].append(vi)
            elif xn < -0.15:
                regions['forehead_right'].append(vi)
            else:
                regions['forehead_center'].append(vi)

        # Brows: narrow band
        if 0.65 < yn < 0.75:
            if xn > 0.10:
                regions['brow_left'].append(vi)
                if xn < 0.30:
                    regions['brow_inner_left'].append(vi)
            if xn < -0.10:
                regions['brow_right'].append(vi)
                if xn > -0.30:
                    regions['brow_inner_right'].append(vi)

        # Eyes: approximate eye region
        if 0.55 < yn < 0.68:
            if 0.15 < xn < 0.55:
                if yn > 0.62:
                    regions['eye_left_upper'].append(vi)
                else:
                    regions['eye_left_lower'].append(vi)
                if xn < 0.25:
                    regions['eye_left_corner_inner'].append(vi)
                if xn > 0.45:
                    regions['eye_left_corner_outer'].append(vi)
            if -0.55 < xn < -0.15:
                if yn > 0.62:
                    regions['eye_right_upper'].append(vi)
                else:
                    regions['eye_right_lower'].append(vi)
                if xn > -0.25:
                    regions['eye_right_corner_inner'].append(vi)
                if xn < -0.45:
                    regions['eye_right_corner_outer'].append(vi)

        # Under-eye
        if 0.48 < yn < 0.58:
            if 0.10 < xn < 0.50:
                regions['under_eye_left'].append(vi)
                if xn < 0.30:
                    regions['tear_trough_left'].append(vi)
            if -0.50 < xn < -0.10:
                regions['under_eye_right'].append(vi)
                if xn > -0.30:
                    regions['tear_trough_right'].append(vi)

        # Nose
        if 0.35 < yn < 0.60 and abs(xn) < 0.15:
            regions['nose_bridge'].append(vi)
            regions['nose_dorsum'].append(vi)
            if yn > 0.48:
                regions['nose_bridge_upper'].append(vi)
            else:
                regions['nose_bridge_lower'].append(vi)

        if 0.30 < yn < 0.40 and abs(xn) < 0.15:
            regions['nose_tip'].append(vi)
            if xn > 0:
                regions['nose_tip_left'].append(vi)
            else:
                regions['nose_tip_right'].append(vi)

        if 0.28 < yn < 0.38:
            if 0.08 < xn < 0.22:
                regions['nostril_left'].append(vi)
            if -0.22 < xn < -0.08:
                regions['nostril_right'].append(vi)

        # Cheeks
        if 0.30 < yn < 0.55 and abs(xn) > 0.25:
            if xn > 0:
                regions['cheek_left'].append(vi)
                if yn > 0.42:
                    regions['cheekbone_left'].append(vi)
                else:
                    regions['cheek_hollow_left'].append(vi)
            else:
                regions['cheek_right'].append(vi)
                if yn > 0.42:
                    regions['cheekbone_right'].append(vi)
                else:
                    regions['cheek_hollow_right'].append(vi)

        # Nasolabial
        if 0.28 < yn < 0.48 and 0.15 < abs(xn) < 0.28:
            if xn > 0:
                regions['nasolabial_left'].append(vi)
            else:
                regions['nasolabial_right'].append(vi)

        # Lips
        if 0.22 < yn < 0.32 and abs(xn) < 0.25:
            if yn > 0.27:
                regions['lip_upper'].append(vi)
                if xn > 0.06:
                    regions['lip_upper_left'].append(vi)
                elif xn < -0.06:
                    regions['lip_upper_right'].append(vi)
                else:
                    regions['lip_upper_center'].append(vi)
            else:
                regions['lip_lower'].append(vi)
                if xn > 0.06:
                    regions['lip_lower_left'].append(vi)
                elif xn < -0.06:
                    regions['lip_lower_right'].append(vi)
                else:
                    regions['lip_lower_center'].append(vi)
            if abs(yn - 0.27) < 0.03 and abs(xn) > 0.18:
                if xn > 0:
                    regions['lip_corner_left'].append(vi)
                else:
                    regions['lip_corner_right'].append(vi)

        # Chin
        if 0.12 < yn < 0.24 and abs(xn) < 0.30:
            regions['chin'].append(vi)
            if abs(xn) < 0.10:
                regions['chin_center'].append(vi)
            elif xn > 0:
                regions['chin_left'].append(vi)
            else:
                regions['chin_right'].append(vi)

        # Jaw
        if 0.10 < yn < 0.30 and abs(xn) > 0.25:
            if xn > 0:
                regions['jaw_left'].append(vi)
                if yn < 0.18:
                    regions['jawline_left'].append(vi)
            else:
                regions['jaw_right'].append(vi)
                if yn < 0.18:
                    regions['jawline_right'].append(vi)

        # Temples
        if 0.60 < yn < 0.78 and abs(xn) > 0.50:
            if xn > 0:
                regions['temple_left'].append(vi)
            else:
                regions['temple_right'].append(vi)

        # Ears
        if 0.40 < yn < 0.70 and abs(xn) > 0.80:
            if xn > 0:
                regions['ear_left'].append(vi)
            else:
                regions['ear_right'].append(vi)

    return regions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert FLAME model files to web-friendly JSON + binary formats."
    )
    parser.add_argument(
        '--flame-dir',
        type=str,
        default=None,
        help='Path to the FLAME models directory (default: auto-detect from script location)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: <flame-dir>/web/)'
    )
    parser.add_argument(
        '--shape-components',
        type=int,
        default=50,
        help='Number of shape basis components to export (default: 50)'
    )
    parser.add_argument(
        '--expr-components',
        type=int,
        default=50,
        help='Number of expression basis components to export (default: 50)'
    )

    args = parser.parse_args()

    # Auto-detect flame directory
    if args.flame_dir:
        flame_dir = Path(args.flame_dir)
    else:
        # Try to find the models/flame directory relative to this script
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent / 'models' / 'flame',
            script_dir / 'models' / 'flame',
            Path.cwd() / 'models' / 'flame',
        ]
        flame_dir = None
        for c in candidates:
            if c.is_dir():
                flame_dir = c
                break
        if flame_dir is None:
            print("[ERROR] Could not find models/flame/ directory.")
            print("        Run this script from the project root, or use --flame-dir.")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else flame_dir / 'web'

    print("=" * 60)
    print("FLAME to Web Converter")
    print("=" * 60)
    print(f"  FLAME dir: {flame_dir}")
    print(f"  Output:    {output_dir}")
    print(f"  Shape components: {args.shape_components}")
    print(f"  Expression components: {args.expr_components}")
    print(f"  chumpy available: {HAS_CHUMPY}")
    print(f"  scipy available: {HAS_SCIPY}")
    print()

    convert_flame_model(
        flame_dir=flame_dir,
        output_dir=output_dir,
        n_shape_components=args.shape_components,
        n_expr_components=args.expr_components,
    )


if __name__ == '__main__':
    main()
