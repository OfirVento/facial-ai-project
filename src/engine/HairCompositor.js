/**
 * HairCompositor — Extracts hair from patient photo and overlays on 3D model.
 *
 * Strategy:
 *   1. Use MediaPipe face landmarks to identify the hairline boundary
 *   2. Create a mask: everything above the hairline + outside the face oval = hair region
 *   3. Extract hair pixels from the original photo as an alpha-masked layer
 *   4. Render as a billboard sprite in Three.js that tracks head rotation
 *
 * This approach avoids 3D hair modeling entirely. It's convincing for ±30° rotation.
 *
 * Usage:
 *   const compositor = new HairCompositor(scene, camera);
 *   await compositor.extractFromPhoto(photoCanvas, landmarks);
 *   // Hair sprite auto-tracks camera rotation via the update loop
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** MediaPipe landmark indices that trace the face boundary / hairline. */
const FACE_OVAL_INDICES = [
  10,   // forehead center (top)
  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,  // right side
  397, 365, 379, 378, 400, 377, 152,  // chin
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162,  // left side
  21, 54, 103, 67, 109,  // left forehead
  10  // back to top
];

/** How far above the forehead to extend the hair region (as fraction of face height). */
const HAIR_EXTEND_ABOVE = 0.35;

/** How far to the sides to extend (as fraction of face width). */
const HAIR_EXTEND_SIDES = 0.15;

/** Feather radius for mask edge softening (pixels at 1024px resolution). */
const FEATHER_RADIUS = 8;

// ---------------------------------------------------------------------------
// HairCompositor
// ---------------------------------------------------------------------------

export class HairCompositor {
  /**
   * @param {THREE.Scene} scene
   * @param {THREE.PerspectiveCamera} camera
   */
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;

    /** @type {THREE.Sprite|null} */
    this._hairSprite = null;

    /** @type {THREE.SpriteMaterial|null} */
    this._hairMaterial = null;

    /** @type {THREE.Texture|null} */
    this._hairTexture = null;

    /** Hair extraction result metadata */
    this._hairData = null;
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Extract hair from photo and create billboard sprite.
   *
   * @param {HTMLCanvasElement|HTMLImageElement} photoSource - Original face photo
   * @param {Array<{x:number, y:number, z:number}>} landmarks - 478 MediaPipe landmarks (normalized 0-1)
   * @param {object} [options]
   * @param {number} [options.spriteScale=0.2] - Scale of the sprite in 3D units
   * @returns {Promise<boolean>} true if hair was successfully extracted
   */
  async extractFromPhoto(photoSource, landmarks, options = {}) {
    if (!landmarks || landmarks.length < 468) {
      console.warn('HairCompositor: Insufficient landmarks for hair extraction');
      return false;
    }

    const spriteScale = options.spriteScale ?? 0.2;

    // Step 1: Create hair mask from landmarks
    const { hairCanvas, maskCanvas, bounds } = this._createHairMask(photoSource, landmarks);
    if (!hairCanvas) return false;

    // Step 2: Create Three.js texture from hair canvas
    if (this._hairTexture) this._hairTexture.dispose();
    this._hairTexture = new THREE.CanvasTexture(hairCanvas);
    this._hairTexture.colorSpace = THREE.SRGBColorSpace;
    this._hairTexture.needsUpdate = true;

    // Step 3: Create sprite material with transparency
    if (this._hairMaterial) this._hairMaterial.dispose();
    this._hairMaterial = new THREE.SpriteMaterial({
      map: this._hairTexture,
      transparent: true,
      depthWrite: false,
      depthTest: true,
    });

    // Step 4: Create or update sprite
    this._removeSprite();
    this._hairSprite = new THREE.Sprite(this._hairMaterial);

    // Scale based on photo aspect ratio
    const aspect = hairCanvas.width / hairCanvas.height;
    this._hairSprite.scale.set(spriteScale * aspect, spriteScale, 1);

    // Position: centered at face position, slightly in front of the mesh
    this._hairSprite.position.set(0, spriteScale * 0.25, 0.01);

    // Store hair data for position adjustments
    this._hairData = {
      bounds,
      aspect,
      spriteScale,
    };

    this.scene.add(this._hairSprite);
    console.log('HairCompositor: Hair sprite created from photo');

    return true;
  }

  /**
   * Update hair sprite position/orientation (call from render loop).
   * The sprite auto-faces the camera as a Three.js Sprite, but we
   * may need to adjust position based on camera angle.
   */
  update() {
    if (!this._hairSprite || !this._hairData) return;

    // Sprite auto-faces camera (billboard behavior).
    // No additional update needed for basic use.
    // For advanced: could adjust opacity based on camera angle
    // to fade out at extreme angles where the 2D overlay fails.
  }

  /**
   * Set hair sprite visibility.
   */
  setVisible(visible) {
    if (this._hairSprite) {
      this._hairSprite.visible = visible;
    }
  }

  /**
   * Remove hair overlay.
   */
  remove() {
    this._removeSprite();
    this._hairData = null;
  }

  /**
   * Clean up all resources.
   */
  dispose() {
    this._removeSprite();
    if (this._hairTexture) { this._hairTexture.dispose(); this._hairTexture = null; }
    if (this._hairMaterial) { this._hairMaterial.dispose(); this._hairMaterial = null; }
    this._hairData = null;
  }

  // =========================================================================
  // Private: Hair Mask Creation
  // =========================================================================

  /**
   * Create a hair mask from face landmarks.
   *
   * Algorithm:
   *   1. Draw the face oval polygon (from FACE_OVAL_INDICES landmarks)
   *   2. Create a "non-face" mask: everything outside the face oval
   *   3. Restrict to the upper region (above nose bridge) — this is where hair is
   *   4. Feather the mask edges
   *   5. Apply mask to source photo → hair-only RGBA canvas
   */
  _createHairMask(photoSource, landmarks) {
    // Get source dimensions
    let srcW, srcH;
    if (photoSource instanceof HTMLCanvasElement) {
      srcW = photoSource.width;
      srcH = photoSource.height;
    } else {
      srcW = photoSource.naturalWidth || photoSource.width;
      srcH = photoSource.naturalHeight || photoSource.height;
    }

    if (srcW === 0 || srcH === 0) return { hairCanvas: null };

    // Work at a fixed resolution for performance
    const maxDim = 1024;
    const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
    const w = Math.round(srcW * scale);
    const h = Math.round(srcH * scale);

    // Create mask canvas
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = w;
    maskCanvas.height = h;
    const maskCtx = maskCanvas.getContext('2d');

    // Step 1: Draw the face oval
    maskCtx.fillStyle = 'white';
    maskCtx.fillRect(0, 0, w, h); // start with everything as "hair"

    // Step 2: Cut out the face oval (mark face region as black = no hair)
    maskCtx.fillStyle = 'black';
    maskCtx.beginPath();

    for (let i = 0; i < FACE_OVAL_INDICES.length; i++) {
      const idx = FACE_OVAL_INDICES[i];
      if (idx >= landmarks.length) continue;
      const lm = landmarks[idx];
      const px = lm.x * w;
      const py = lm.y * h;
      if (i === 0) maskCtx.moveTo(px, py);
      else maskCtx.lineTo(px, py);
    }
    maskCtx.closePath();
    maskCtx.fill();

    // Step 3: Cut out the lower half (below nose bridge = no hair)
    // Nose bridge is approximately at landmark 6
    const noseBridge = landmarks[6] || landmarks[1];
    const cutoffY = noseBridge.y * h;
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, cutoffY, w, h - cutoffY);

    // Step 4: Get mask image data and feather edges
    const maskData = maskCtx.getImageData(0, 0, w, h);
    this._featherMask(maskData, FEATHER_RADIUS);
    maskCtx.putImageData(maskData, 0, 0);

    // Step 5: Apply mask to source photo
    const hairCanvas = document.createElement('canvas');
    hairCanvas.width = w;
    hairCanvas.height = h;
    const hairCtx = hairCanvas.getContext('2d');

    // Draw source photo (scaled)
    hairCtx.drawImage(photoSource, 0, 0, w, h);

    // Apply mask as alpha channel
    const hairData = hairCtx.getImageData(0, 0, w, h);
    const mask = maskData.data;
    const pixels = hairData.data;

    for (let i = 0; i < pixels.length; i += 4) {
      // Use red channel of mask as alpha
      pixels[i + 3] = mask[i]; // R channel = hair mask intensity
    }

    hairCtx.putImageData(hairData, 0, 0);

    // Calculate bounding box of hair region
    const forehead = landmarks[10];
    const bounds = {
      top: Math.max(0, (forehead.y - HAIR_EXTEND_ABOVE) * h),
      centerX: forehead.x * w,
      centerY: forehead.y * h,
    };

    return { hairCanvas, maskCanvas, bounds };
  }

  /**
   * Feather (soft-edge) a mask using a simple box blur.
   * @param {ImageData} maskData
   * @param {number} radius
   */
  _featherMask(maskData, radius) {
    if (radius < 1) return;

    const { width, height, data } = maskData;
    const copy = new Uint8ClampedArray(data);

    // Horizontal pass
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0, count = 0;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx >= 0 && nx < width) {
            sum += copy[(y * width + nx) * 4]; // R channel
            count++;
          }
        }
        const avg = sum / count;
        const idx = (y * width + x) * 4;
        data[idx] = avg;
        data[idx + 1] = avg;
        data[idx + 2] = avg;
      }
    }

    // Vertical pass
    const copy2 = new Uint8ClampedArray(data);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0, count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          const ny = y + dy;
          if (ny >= 0 && ny < height) {
            sum += copy2[(ny * width + x) * 4];
            count++;
          }
        }
        const avg = sum / count;
        const idx = (y * width + x) * 4;
        data[idx] = avg;
        data[idx + 1] = avg;
        data[idx + 2] = avg;
      }
    }
  }

  /**
   * Remove sprite from scene.
   */
  _removeSprite() {
    if (this._hairSprite) {
      this.scene.remove(this._hairSprite);
      this._hairSprite = null;
    }
  }
}
