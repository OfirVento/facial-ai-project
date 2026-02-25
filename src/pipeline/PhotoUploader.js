/**
 * Photo Upload & Reconstruction Pipeline
 * Handles photo capture/upload, sends to cloud reconstruction,
 * and processes the returned FLAME mesh + texture
 */

export class PhotoUploader {
  constructor() {
    this.photos = { front: null, left45: null, right45: null };
    this.reconstructionStatus = 'idle'; // idle | uploading | reconstructing | done | error
    this.listeners = new Set();
    this.reconstructionResult = null;
  }

  /**
   * Set a photo for a specific angle
   * @param {string} angle - 'front' | 'left45' | 'right45'
   * @param {File|Blob} file - The image file
   */
  async setPhoto(angle, file) {
    if (!['front', 'left45', 'right45'].includes(angle)) {
      throw new Error(`Invalid angle: ${angle}. Use 'front', 'left45', or 'right45'`);
    }

    // Create preview URL
    const url = URL.createObjectURL(file);

    // Get image dimensions
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });

    this.photos[angle] = {
      file,
      url,
      width: img.naturalWidth,
      height: img.naturalHeight,
      name: file.name,
      size: file.size
    };

    this._notify({ type: 'photo_set', angle, photo: this.photos[angle] });
    return this.photos[angle];
  }

  /**
   * Capture photo from webcam
   */
  async captureFromCamera(angle) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: 'user'
        }
      });

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      await new Promise(r => { video.onloadedmetadata = r; });
      await video.play();

      // Wait a moment for camera to adjust
      await new Promise(r => setTimeout(r, 500));

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      // Stop camera
      stream.getTracks().forEach(t => t.stop());

      // Convert to blob
      const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.95));
      const file = new File([blob], `capture_${angle}.jpg`, { type: 'image/jpeg' });

      return this.setPhoto(angle, file);
    } catch (err) {
      this._notify({ type: 'error', error: `Camera access failed: ${err.message}` });
      throw err;
    }
  }

  /**
   * Check if we have the minimum required photos
   */
  getStatus() {
    const hasFront = !!this.photos.front;
    const hasLeft = !!this.photos.left45;
    const hasRight = !!this.photos.right45;
    const photoCount = [hasFront, hasLeft, hasRight].filter(Boolean).length;

    return {
      hasFront,
      hasLeft,
      hasRight,
      photoCount,
      canReconstruct: hasFront, // Minimum: front photo
      isOptimal: hasFront && hasLeft && hasRight,
      reconstructionStatus: this.reconstructionStatus
    };
  }

  /**
   * Generate a texture from the front photo by projecting it onto UV space
   * This is the browser-side fallback when no cloud reconstruction is available
   */
  async generateTextureFromPhoto() {
    if (!this.photos.front) {
      throw new Error('Front photo required');
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = this.photos.front.url;
    });

    // Create texture canvas (1024x1024 UV space)
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Fill with base skin color
    ctx.fillStyle = '#e8b89d';
    ctx.fillRect(0, 0, size, size);

    // Project the face photo onto the UV layout
    // The face occupies roughly the center 60% of the UV space
    const faceRegion = {
      x: size * 0.2,
      y: size * 0.05,
      w: size * 0.6,
      h: size * 0.8
    };

    // Crop the face from the photo (assuming face is centered, takes ~60% width)
    const srcFaceX = img.width * 0.2;
    const srcFaceY = img.height * 0.05;
    const srcFaceW = img.width * 0.6;
    const srcFaceH = img.height * 0.85;

    ctx.drawImage(
      img,
      srcFaceX, srcFaceY, srcFaceW, srcFaceH,
      faceRegion.x, faceRegion.y, faceRegion.w, faceRegion.h
    );

    // Blend edges with skin color for smooth transition
    const gradient = ctx.createRadialGradient(
      size / 2, size * 0.45, size * 0.25,
      size / 2, size * 0.45, size * 0.5
    );
    gradient.addColorStop(0, 'rgba(232, 184, 157, 0)');
    gradient.addColorStop(1, 'rgba(232, 184, 157, 1)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/jpeg', 0.9),
      width: size,
      height: size
    };
  }

  /**
   * Generate a normal map from the photo for surface detail
   */
  async generateNormalMapFromPhoto() {
    if (!this.photos.front) return null;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = this.photos.front.url;
    });

    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Draw source image
    ctx.drawImage(img, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size);
    const pixels = imageData.data;

    // Convert to grayscale heightmap
    const heights = new Float32Array(size * size);
    for (let i = 0; i < size * size; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      heights[i] = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    }

    // Compute normals from height differences (Sobel-like)
    const normalData = ctx.createImageData(size, size);
    const strength = 2.0;

    for (let y = 1; y < size - 1; y++) {
      for (let x = 1; x < size - 1; x++) {
        const idx = y * size + x;
        const left = heights[idx - 1];
        const right = heights[idx + 1];
        const up = heights[idx - size];
        const down = heights[idx + size];

        let nx = (left - right) * strength;
        let ny = (up - down) * strength;
        let nz = 1.0;

        // Normalize
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        nx /= len; ny /= len; nz /= len;

        // Encode as RGB (normal map convention)
        const pi = idx * 4;
        normalData.data[pi] = Math.round((nx * 0.5 + 0.5) * 255);
        normalData.data[pi + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        normalData.data[pi + 2] = Math.round((nz * 0.5 + 0.5) * 255);
        normalData.data[pi + 3] = 255;
      }
    }

    ctx.putImageData(normalData, 0, 0);

    return {
      canvas,
      dataUrl: canvas.toDataURL('image/png'),
      width: size,
      height: size
    };
  }

  /**
   * Process reconstruction result from Colab/Modal
   * Expected format: { mesh: OBJ string or URL, texture: image URL, normalMap: image URL, params: {} }
   */
  async loadReconstructionResult(result) {
    this.reconstructionResult = result;
    this.reconstructionStatus = 'done';
    this._notify({ type: 'reconstruction_complete', result });
    return result;
  }

  /**
   * Load reconstruction from uploaded files (mesh + texture)
   */
  async loadFromFiles(meshFile, textureFile, normalMapFile) {
    const result = {};

    if (meshFile) {
      result.meshUrl = URL.createObjectURL(meshFile);
      result.meshType = meshFile.name.endsWith('.glb') ? 'glb' : 'obj';
    }

    if (textureFile) {
      result.textureUrl = URL.createObjectURL(textureFile);
    }

    if (normalMapFile) {
      result.normalMapUrl = URL.createObjectURL(normalMapFile);
    }

    return this.loadReconstructionResult(result);
  }

  /**
   * Clear all photos and results
   */
  clear() {
    for (const angle of Object.keys(this.photos)) {
      if (this.photos[angle]?.url) {
        URL.revokeObjectURL(this.photos[angle].url);
      }
      this.photos[angle] = null;
    }
    this.reconstructionStatus = 'idle';
    this.reconstructionResult = null;
    this._notify({ type: 'cleared' });
  }

  onChange(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  _notify(event) {
    for (const listener of this.listeners) {
      try { listener(event); } catch (e) { console.error(e); }
    }
  }
}

export default PhotoUploader;
