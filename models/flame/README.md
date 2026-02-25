# FLAME Model Files

Download models from https://flame.is.tue.mpg.de/ and place them in the appropriate subdirectories.

## Required Downloads

### 1. `flame2023/` — FLAME 2023 Model (103 MB)
The core 3D morphable face model with 5,023 vertices.
- Download: "FLAME 2023 (revised eye region, improved expressions)"
- Contains: `generic_model.pkl`, shape/expression basis vectors, face topology

### 2. `vertex_masks/` — FLAME Vertex Masks (1.1 MB)
Defines vertex groups for face regions (forehead, nose, lips, etc.)
- Download: "FLAME Vertex Masks"
- Contains: Vertex group assignments for the FLAME topology

### 3. `mediapipe/` — MediaPipe Landmark Embedding (3.1 KB)
Maps MediaPipe 478 face landmarks to FLAME mesh vertices.
- Download: "FLAME Mediapipe Landmark Embedding"
- Contains: Landmark-to-vertex correspondence file

### 4. `texture_space/` — FLAME Texture Space (1.2 GB)
UV texture model for generating realistic skin from parameters.
- Download: "FLAME texture space (non-commercial)"
- Contains: Texture PCA basis, mean texture, UV mapping

### 5. `albedo/` — Morphable Albedo (CVPR 2020)
Learned albedo texture model for photorealistic skin rendering.
- Download: "Morphable Albedo texture space [CVPR 2020] (FLAME version)"
- Contains: Albedo PCA basis and mean

## Conversion

After downloading, run the conversion script to generate web-friendly formats:

```bash
cd models/flame
python ../../scripts/convert_flame_to_web.py
```

This generates `web/` folder with JSON + binary files that the browser app consumes.

## Note
These files are gitignored due to their size and licensing.
Non-commercial use only (except FLAME 2023 Open which is CC-BY-4.0).
