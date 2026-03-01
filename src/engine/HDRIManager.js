/**
 * HDRIManager.js
 *
 * Loads .hdr environment maps and generates prefiltered PBR environment textures
 * via PMREMGenerator. The resulting envMap can be shared across multiple renderers.
 *
 * Usage:
 *   const hdri = new HDRIManager(renderer);
 *   const envMap = await hdri.load('/hdri/studio_small_09_2k.hdr');
 *   renderer.setEnvironment(envMap);
 */

import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export class HDRIManager {
  /**
   * @param {THREE.WebGLRenderer} renderer - WebGL renderer for PMREMGenerator.
   */
  constructor(renderer) {
    if (!renderer) throw new Error('HDRIManager requires a WebGLRenderer.');
    this._renderer = renderer;
    this._pmremGenerator = new THREE.PMREMGenerator(renderer);
    this._pmremGenerator.compileEquirectangularShader();
    this._cache = new Map(); // url → envMap
    this._rgbeLoader = new RGBELoader();
  }

  /**
   * Load an HDR environment map and generate a prefiltered cubemap.
   * Results are cached — repeated calls with the same URL return the cached texture.
   *
   * @param {string} url - Path to .hdr file.
   * @returns {Promise<THREE.Texture>} Prefiltered environment map texture.
   */
  async load(url) {
    // Return cached if available
    if (this._cache.has(url)) {
      return this._cache.get(url);
    }

    // Load the equirectangular HDR texture
    const hdrTexture = await this._rgbeLoader.loadAsync(url);
    hdrTexture.mapping = THREE.EquirectangularReflectionMapping;

    // Generate prefiltered environment map for PBR
    const envMap = this._pmremGenerator.fromEquirectangular(hdrTexture).texture;

    // Dispose the raw HDR texture — we only need the prefiltered version
    hdrTexture.dispose();

    // Cache the result
    this._cache.set(url, envMap);

    console.log(`HDRIManager: Loaded and prefiltered "${url}"`);
    return envMap;
  }

  /**
   * Dispose all cached environment maps and the PMREMGenerator.
   */
  dispose() {
    for (const [url, envMap] of this._cache) {
      envMap.dispose();
    }
    this._cache.clear();
    this._pmremGenerator.dispose();
    console.log('HDRIManager: Disposed all resources');
  }
}

export default HDRIManager;
