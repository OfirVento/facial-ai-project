/**
 * LabApp.js — Benchmark Viewer (Phase 1)
 *
 * Split-screen viewer comparing Lumirithmic (GLB) output against
 * the custom FLAME engine. Both panels share the same HDRI environment
 * and camera angle via sync.
 *
 * Left panel  = Lumirithmic GLB (drag-drop or button)
 * Right panel = Custom FLAME pipeline (default mesh, or photo upload)
 */

import { FaceRenderer } from '../engine/FaceRenderer.js';
import { FlameMeshGenerator } from '../engine/FlameMeshGenerator.js';
import { HDRIManager } from '../engine/HDRIManager.js';
import { PhotoUploader } from '../pipeline/PhotoUploader.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HDRI_PATH = '/hdri/studio_small_09_2k.hdr';
const CAMERA_SYNC_RATE = 1000 / 60; // sync at 60fps

// ---------------------------------------------------------------------------
// LabApp
// ---------------------------------------------------------------------------

class LabApp {
  constructor() {
    // Renderers
    this.leftRenderer = null;   // Lumirithmic GLB panel
    this.rightRenderer = null;  // Custom FLAME panel

    // Shared resources
    this.hdriManager = null;
    this.envMap = null;

    // FLAME mesh generator (for right panel)
    this.meshGenerator = new FlameMeshGenerator();
    this.photoUploader = new PhotoUploader({ meshGenerator: this.meshGenerator });

    // State
    this._syncInterval = null;
    this._fpsFrames = 0;
    this._fpsTime = performance.now();
    this._leftHasModel = false;
    this._isDividerDragging = false;
  }

  async init() {
    console.log('LabApp: Initializing...');

    // --- Create two FaceRenderer instances ---
    const leftContainer = document.getElementById('viewport-left');
    const rightContainer = document.getElementById('viewport-right');

    if (!leftContainer || !rightContainer) {
      console.error('LabApp: Missing viewport containers');
      return;
    }

    this.leftRenderer = new FaceRenderer(leftContainer);
    this.leftRenderer.init();

    this.rightRenderer = new FaceRenderer(rightContainer);
    this.rightRenderer.init();

    // --- Load shared HDRI environment ---
    try {
      // Use the left renderer's WebGL renderer for PMREMGenerator (any renderer works)
      this.hdriManager = new HDRIManager(this.leftRenderer.renderer);
      this.envMap = await this.hdriManager.load(HDRI_PATH);

      this.leftRenderer.setEnvironment(this.envMap, 1.0);
      this.rightRenderer.setEnvironment(this.envMap, 1.0);

      this._updateStat('stat-hdri', 'HDRI: loaded');
      console.log('LabApp: Shared HDRI environment applied to both renderers');
    } catch (err) {
      this._updateStat('stat-hdri', 'HDRI: failed');
      console.warn('LabApp: HDRI loading failed:', err.message);
    }

    // --- Load FLAME mesh on right panel ---
    await this._loadFlameMesh();

    // --- Camera sync: left = leader, right = follower ---
    this._startCameraSync();

    // --- Wire up UI ---
    this._bindUI();

    // --- FPS counter ---
    this._startFpsCounter();

    console.log('LabApp: Initialization complete');
  }

  // -----------------------------------------------------------------------
  // FLAME Mesh (Right Panel)
  // -----------------------------------------------------------------------

  async _loadFlameMesh() {
    try {
      const result = await this.meshGenerator.loadFLAME();
      if (result && result.geometry) {
        this.rightRenderer.loadFromGeometry(result.geometry);
        const vertCount = result.geometry.attributes.position.count;
        this._updateStat('stat-right-verts', `R verts: ${vertCount}`);
        this._updateStat('right-status', 'FLAME loaded');
        console.log('LabApp: FLAME mesh loaded on right panel');

        // Apply FLAME albedo if available
        if (result.diffuseData) {
          this.rightRenderer.applyAlbedoTexture(
            result.diffuseData,
            result.specularData || null,
            result.meta || null
          );
        }
      }
    } catch (err) {
      console.warn('LabApp: FLAME loading failed, right panel stays empty:', err.message);
      this._updateStat('right-status', 'FLAME failed');
    }
  }

  // -----------------------------------------------------------------------
  // GLB Loading (Left Panel)
  // -----------------------------------------------------------------------

  async _loadGLB(file) {
    if (!file) return;

    const url = URL.createObjectURL(file);
    try {
      // Hide drop zone
      const dropZone = document.getElementById('drop-zone-left');
      if (dropZone) dropZone.classList.add('hidden');

      await this.leftRenderer.loadFromGLB(url);

      // Update stats
      if (this.leftRenderer.faceMesh) {
        const vertCount = this.leftRenderer.faceMesh.geometry.attributes.position.count;
        this._updateStat('stat-left-verts', `L verts: ${vertCount}`);
      }
      this._updateStat('left-status', file.name);
      this._leftHasModel = true;

      console.log(`LabApp: GLB loaded on left panel — ${file.name}`);
    } catch (err) {
      console.error('LabApp: GLB loading failed:', err);
      this._updateStat('left-status', 'Load failed');
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // -----------------------------------------------------------------------
  // Photo Upload (Right Panel — FLAME pipeline)
  // -----------------------------------------------------------------------

  async _uploadPhoto(file) {
    if (!file) return;

    this._updateStat('right-status', 'Processing...');

    try {
      const reader = new FileReader();
      const dataUrl = await new Promise((resolve, reject) => {
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      // Run the photo → texture pipeline
      const result = await this.photoUploader.generateTextureFromPhoto(
        dataUrl,
        this.meshGenerator
      );

      if (result && result.textureDataUrl) {
        await this.rightRenderer.loadTexture(result.textureDataUrl, 'albedo', { flipY: false });
        this._updateStat('right-status', 'Photo applied');
        console.log('LabApp: Photo texture applied to FLAME mesh');
      }

      // Update shape if shape params were fitted
      if (result && result.shapeParams && this.meshGenerator) {
        const shaped = this.meshGenerator.applyShapeParams(result.shapeParams);
        if (shaped && shaped.geometry) {
          this.rightRenderer.loadFromGeometry(shaped.geometry);
          const vertCount = shaped.geometry.attributes.position.count;
          this._updateStat('stat-right-verts', `R verts: ${vertCount}`);

          // Reload texture onto the reshaped mesh
          if (result.textureDataUrl) {
            await this.rightRenderer.loadTexture(result.textureDataUrl, 'albedo', { flipY: false });
          }
        }
      }
    } catch (err) {
      console.error('LabApp: Photo processing failed:', err);
      this._updateStat('right-status', 'Processing failed');
    }
  }

  // -----------------------------------------------------------------------
  // Camera Sync
  // -----------------------------------------------------------------------

  _startCameraSync() {
    // Left renderer is the leader: we intercept its pointer events
    // and copy its spherical state to the right renderer each frame.

    // Use requestAnimationFrame-linked sync for smooth results
    const syncLoop = () => {
      if (this.leftRenderer && this.rightRenderer) {
        const state = this.leftRenderer.getSphericalState();
        this.rightRenderer.setSphericalState(state);
      }
      this._syncRafId = requestAnimationFrame(syncLoop);
    };
    this._syncRafId = requestAnimationFrame(syncLoop);
  }

  _stopCameraSync() {
    if (this._syncRafId) {
      cancelAnimationFrame(this._syncRafId);
      this._syncRafId = null;
    }
  }

  // -----------------------------------------------------------------------
  // UI Bindings
  // -----------------------------------------------------------------------

  _bindUI() {
    // --- GLB file button ---
    const glbBtn = document.getElementById('load-glb-btn');
    const glbInput = document.getElementById('glb-file-input');
    if (glbBtn && glbInput) {
      glbBtn.addEventListener('click', () => glbInput.click());
      glbInput.addEventListener('change', (e) => {
        if (e.target.files[0]) this._loadGLB(e.target.files[0]);
      });
    }

    // --- Photo file button ---
    const photoBtn = document.getElementById('load-photo-btn');
    const photoInput = document.getElementById('photo-file-input');
    if (photoBtn && photoInput) {
      photoBtn.addEventListener('click', () => photoInput.click());
      photoInput.addEventListener('change', (e) => {
        if (e.target.files[0]) this._uploadPhoto(e.target.files[0]);
      });
    }

    // --- Drag & drop on left viewport ---
    const leftViewport = document.getElementById('viewport-left');
    if (leftViewport) {
      leftViewport.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const dropZone = document.getElementById('drop-zone-left');
        if (dropZone) dropZone.classList.add('active');
      });

      leftViewport.addEventListener('dragleave', (e) => {
        e.preventDefault();
        const dropZone = document.getElementById('drop-zone-left');
        if (dropZone) dropZone.classList.remove('active');
      });

      leftViewport.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const dropZone = document.getElementById('drop-zone-left');
        if (dropZone) dropZone.classList.remove('active');

        const file = e.dataTransfer?.files?.[0];
        if (file && (file.name.endsWith('.glb') || file.name.endsWith('.gltf'))) {
          this._loadGLB(file);
        }
      });
    }

    // --- Divider drag for resizing panels ---
    this._bindDivider();
  }

  _bindDivider() {
    const divider = document.getElementById('lab-divider');
    const container = document.querySelector('.lab-viewports');
    const leftPanel = document.querySelector('.lab-panel-left');
    const rightPanel = document.querySelector('.lab-panel-right');

    if (!divider || !container || !leftPanel || !rightPanel) return;

    let startX = 0;
    let startLeftWidth = 0;

    const onMouseDown = (e) => {
      e.preventDefault();
      this._isDividerDragging = true;
      divider.classList.add('dragging');
      startX = e.clientX;
      startLeftWidth = leftPanel.getBoundingClientRect().width;

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };

    const onMouseMove = (e) => {
      if (!this._isDividerDragging) return;
      const dx = e.clientX - startX;
      const containerWidth = container.getBoundingClientRect().width - 6; // minus divider
      const newLeftWidth = Math.max(200, Math.min(containerWidth - 200, startLeftWidth + dx));
      const leftPct = (newLeftWidth / containerWidth) * 100;

      leftPanel.style.flex = 'none';
      leftPanel.style.width = `${leftPct}%`;
      rightPanel.style.flex = '1';

      // Trigger resize on both renderers
      this.leftRenderer?._onResize();
      this.rightRenderer?._onResize();
    };

    const onMouseUp = () => {
      this._isDividerDragging = false;
      divider.classList.remove('dragging');
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };

    divider.addEventListener('mousedown', onMouseDown);
  }

  // -----------------------------------------------------------------------
  // FPS Counter
  // -----------------------------------------------------------------------

  _startFpsCounter() {
    const update = () => {
      this._fpsFrames++;
      const now = performance.now();
      if (now - this._fpsTime >= 1000) {
        const fps = Math.round(this._fpsFrames * 1000 / (now - this._fpsTime));
        this._updateStat('stat-fps', `FPS: ${fps}`);
        this._fpsFrames = 0;
        this._fpsTime = now;
      }
      requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  _updateStat(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  // -----------------------------------------------------------------------
  // Cleanup
  // -----------------------------------------------------------------------

  destroy() {
    this._stopCameraSync();
    this.leftRenderer?.destroy();
    this.rightRenderer?.destroy();
    this.hdriManager?.dispose();
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

const app = new LabApp();
app.init().catch((err) => console.error('LabApp init failed:', err));
