/**
 * Main Application — Wires FLAME mesh + renderer + agents + UI
 * Facial AI Project v2.0
 */

import { FlameMeshGenerator } from './engine/FlameMeshGenerator.js';
import { FaceRenderer } from './engine/FaceRenderer.js';
import { NLUAgent } from './agents/NLUAgent.js';
import { MedicalAdvisorAgent } from './agents/MedicalAdvisorAgent.js';
import { ExpressionAgent } from './agents/ExpressionAgent.js';
import { ComparisonAgent } from './agents/ComparisonAgent.js';
import { VoiceAgent } from './agents/VoiceAgent.js';
import { ConversationDirector } from './agents/ConversationDirector.js';
import { PhotoUploader } from './pipeline/PhotoUploader.js';
import { MediaPipeBridge } from './engine/MediaPipeBridge.js';
import { HDRIManager } from './engine/HDRIManager.js';
import { FaceDNA } from './data/FaceDNA.js';

class FacialAIProject {
  constructor() {
    // Engine
    this.meshGenerator = new FlameMeshGenerator();
    this.renderer = null;
    this.meshData = null; // { geometry, regions, regionMeta }

    // Agents
    this.nluAgent = new NLUAgent();
    this.medicalAdvisor = new MedicalAdvisorAgent();
    this.expressionAgent = new ExpressionAgent();
    this.comparisonAgent = new ComparisonAgent();
    this.voiceAgent = new VoiceAgent();

    // Pipeline
    this.photoUploader = new PhotoUploader({ meshGenerator: this.meshGenerator });
    this.mediaPipeBridge = null; // initialized lazily when needed

    // State
    this.flameLoaded = false; // true when real FLAME 2023 data is loaded
    this.morphState = {}; // regionName → { inflate, translateX, translateY, translateZ }
    this.morphHistory = [];
    this.morphHistoryIndex = -1;
    this.versions = [];
    this.activeRegion = null;
    this.activeRegionTab = 'all';
    this.isVoiceMode = false;
    this.comparisonMode = false;

    // FaceDNA — digital face identity
    this.currentFaceDNA = null;

    // Director (initialized after all agents ready)
    this.director = null;
  }

  async init() {
    // Init 3D viewport
    const viewport = document.getElementById('face-viewport');
    if (viewport) {
      this.renderer = new FaceRenderer(viewport);
      this.renderer.init();
    }

    // Load HDRI environment for PBR lighting (Phase 0)
    if (this.renderer) {
      try {
        this.hdriManager = new HDRIManager(this.renderer.renderer);
        const envMap = await this.hdriManager.load('/hdri/studio_small_09_2k.hdr');
        this.renderer.setEnvironment(envMap, 1.0);
        console.log('App: HDRI environment loaded successfully');
      } catch (err) {
        console.warn('App: HDRI loading failed, continuing without IBL:', err.message);
      }
    }

    // Generate demo face mesh (tries real FLAME first, falls back to analytic)
    await this._generateDemoFace();

    // Init director — pass morphEngine (this) and agent instances
    this.director = new ConversationDirector(this, {
      nlu: this.nluAgent,
      medical: this.medicalAdvisor,
      expression: this.expressionAgent,
      comparison: this.comparisonAgent,
      voice: this.voiceAgent
    });
    this.director.setModelLoaded(true); // demo face is loaded

    // Voice onChange listener (VoiceAgent initializes in constructor)
    this.voiceAgent.onChange((event) => {
      if (event.type === 'transcript_final') {
        this._handleInput(event.text);
      }
      if (event.type === 'transcript_interim') {
        const el = document.getElementById('interim-transcript');
        if (el) { el.textContent = event.text; el.style.display = 'block'; }
      }
    });

    // Setup all UI
    this._setupUploadUI();
    this._setupControlsUI();
    this._setupTreatmentUI();
    this._setupChatUI();
    this._setupTopBarUI();
    this._setupViewportUI();

    // Welcome message — mention FLAME status
    const flameStatus = this.flameLoaded
      ? `Using **FLAME 2023** model (${this.meshGenerator.vertexCount.toLocaleString()} vertices, ${Object.keys(this.meshData.regions).length} clinical regions)`
      : 'Using demo mesh (upload FLAME data or run Colab notebook for photorealistic reconstruction)';

    this._addMessage('assistant', `Welcome to your facial design consultation!

${flameStatus}

Upload your photos on the left panel, or start exploring with the current face model.

Try typing commands like:
• "Make my nose thinner"
• "Add subtle lip filler to the right side"
• "Lift my left cheekbone"
• "What procedures would I need?"

You can also click the quick-action buttons below, use the sliders on the left, or switch to voice mode.`);

    // Save initial state
    this._saveMorphHistory();
  }

  // ============================================================
  // MESH GENERATION
  // ============================================================

  async _generateDemoFace() {
    // Try loading real FLAME 2023 data first, fall back to analytic generation
    try {
      if (typeof this.meshGenerator.loadFLAME === 'function') {
        this.meshData = await this.meshGenerator.loadFLAME('/models/flame/web');
        // Check if real FLAME data was actually loaded (vs internal fallback to analytic)
        this.flameLoaded = !!this.meshGenerator.isFlameLoaded;
        if (this.flameLoaded) {
          console.log('✅ Real FLAME 2023 mesh loaded: %d vertices, %d regions',
            this.meshGenerator.vertexCount,
            Object.keys(this.meshData.regions).length);
        } else {
          console.log('ℹ️ FLAME data not available, using analytic demo mesh (%d vertices)',
            this.meshGenerator.vertexCount);
        }
      }
    } catch (err) {
      console.warn('FLAME load error, using analytic fallback:', err.message);
    }

    // Fallback to analytic mesh generation
    if (!this.meshData) {
      this.meshData = this.meshGenerator.generate();
      this.flameLoaded = false;
    }

    if (this.renderer) {
      this.renderer.loadFromGeometry(this.meshData.geometry);

      // Apply real FLAME albedo texture if available, otherwise fall back to demo
      if (this.flameLoaded && this.meshGenerator.hasAlbedo) {
        const meta = this.meshGenerator.flameMeta;
        this.renderer.applyAlbedoTexture(
          this.meshGenerator.albedoDiffuseData,
          this.meshGenerator.albedoSpecularData,
          meta
        );
        console.log('✅ Applied FLAME albedo texture (diffuse + specular)');
      } else {
        this.renderer.applyDemoTexture();
      }
    }
    this._renderRegionControls();
    this._updatePhaseIndicator('exploration');
  }

  async _loadReconstructedMesh(meshUrl, textureUrl, normalMapUrl, meshType = 'obj') {
    if (!this.renderer) return;

    try {
      if (meshType === 'obj') {
        await this.renderer.loadFromOBJ(meshUrl);
      } else {
        await this.renderer.loadFromGLB(meshUrl);
      }

      if (textureUrl) {
        await this.renderer.loadTexture(textureUrl, 'albedo');
      }
      if (normalMapUrl) {
        await this.renderer.loadTexture(normalMapUrl, 'normal');
      }

      this._addMessage('system', '3D reconstruction loaded successfully.');
      this._updatePhaseIndicator('exploration');
    } catch (err) {
      this._addMessage('system', `Error loading reconstruction: ${err.message}`);
      console.error(err);
    }
  }

  /**
   * Load DECA reconstruction results — applies shape/expression params from face_params.json.
   * Called when user uploads the Colab output ZIP.
   */
  async loadDECAReconstruction(paramsJson) {
    if (!this.flameLoaded || !this.meshGenerator.applyShapeParams) {
      this._addMessage('system', 'FLAME model not loaded. Please ensure web-ready FLAME files are in /models/flame/web/');
      return;
    }

    try {
      const params = typeof paramsJson === 'string' ? JSON.parse(paramsJson) : paramsJson;

      // Apply shape parameters to morph the template into the patient's face
      if (params.shape_params) {
        this.meshGenerator.applyShapeParams(params.shape_params);
        this._addMessage('system', `Applied ${params.shape_params.length} shape parameters from DECA reconstruction`);
      }

      // Apply expression parameters (usually neutral for clinic use)
      if (params.expression_params) {
        this.meshGenerator.applyExpressionParams(params.expression_params);
      }

      // Update renderer with the new geometry
      if (this.renderer && this.meshGenerator.geometry) {
        this.renderer.loadFromGeometry(this.meshGenerator.geometry);
      }

      this._renderRegionControls();
      this._updatePhaseIndicator('exploration');
      this._addMessage('assistant', `DECA reconstruction loaded successfully.
The 3D model now matches your photo with ${params.vertex_count?.toLocaleString() || '5,023'} vertices.
You can now modify specific regions — try saying "make my nose thinner" or use the sliders.`);
    } catch (err) {
      console.error('Failed to load DECA reconstruction:', err);
      this._addMessage('system', `Error loading reconstruction: ${err.message}`);
    }
  }

  // ============================================================
  // MORPH ENGINE (interface for agents)
  // ============================================================

  applyChanges(changes) {
    // changes: { regionName: { inflate, translateX, translateY, translateZ } }
    const applied = [];

    for (const [regionName, params] of Object.entries(changes)) {
      if (!this.morphState[regionName]) {
        this.morphState[regionName] = { inflate: 0, translateX: 0, translateY: 0, translateZ: 0 };
      }

      for (const [key, value] of Object.entries(params)) {
        this.morphState[regionName][key] = (this.morphState[regionName][key] || 0) + value;
      }

      applied.push({ region: regionName, params: this.morphState[regionName] });

      // Apply to renderer
      if (this.renderer && this.meshData) {
        const vertices = this.meshData.regions[regionName];
        if (vertices) {
          this.renderer.deformRegion(vertices, this.morphState[regionName]);
        }
      }
    }

    this._saveMorphHistory();
    this._updateControlsFromState();
    this._updateChangeCount();
    return applied;
  }

  setRegionValue(regionName, key, value) {
    if (!this.morphState[regionName]) {
      this.morphState[regionName] = { inflate: 0, translateX: 0, translateY: 0, translateZ: 0 };
    }

    this.morphState[regionName][key] = value;

    if (this.renderer && this.meshData) {
      // Use applyMorphState to reset geometry first, then apply all regions fresh.
      // This avoids cumulative deformation errors from bare deformRegion() calls.
      this.renderer.applyMorphState(this.morphState, this.meshData.regions);
    }

    this._saveMorphHistory();
    this._updateChangeCount();
  }

  getChanges() {
    const changes = {};
    for (const [region, params] of Object.entries(this.morphState)) {
      const hasChange = Object.values(params).some(v => Math.abs(v) > 0.001);
      if (hasChange) {
        changes[region] = { ...params };
      }
    }
    return changes;
  }

  getClinicalMapping() {
    // Map morphed regions to clinical procedures
    const changes = this.getChanges();
    const mapping = {};
    for (const region of Object.keys(changes)) {
      const meta = this.meshData?.regionMeta?.[region];
      if (meta?.clinical) {
        if (!mapping[meta.clinical]) mapping[meta.clinical] = [];
        mapping[meta.clinical].push(region);
      }
    }
    return mapping;
  }

  undo() {
    if (this.morphHistoryIndex > 0) {
      this.morphHistoryIndex--;
      this.morphState = JSON.parse(JSON.stringify(this.morphHistory[this.morphHistoryIndex]));
      this._applyFullState();
      this._updateControlsFromState();
      this._updateChangeCount();
      return true;
    }
    return false;
  }

  redo() {
    if (this.morphHistoryIndex < this.morphHistory.length - 1) {
      this.morphHistoryIndex++;
      this.morphState = JSON.parse(JSON.stringify(this.morphHistory[this.morphHistoryIndex]));
      this._applyFullState();
      this._updateControlsFromState();
      this._updateChangeCount();
      return true;
    }
    return false;
  }

  reset() {
    this.morphState = {};
    this._saveMorphHistory();
    if (this.renderer) {
      this.renderer.resetDeformations?.();
      // Regenerate to clear all deformations
      this._generateDemoFace();
    }
    this._updateControlsFromState();
    this._updateChangeCount();
  }

  saveVersion(name) {
    const version = {
      name: name || `Version ${this.versions.length + 1}`,
      state: JSON.parse(JSON.stringify(this.morphState)),
      timestamp: Date.now()
    };
    this.versions.push(version);
    this._updateVersionsList();
    return version;
  }

  loadVersion(index) {
    if (index >= 0 && index < this.versions.length) {
      this.morphState = JSON.parse(JSON.stringify(this.versions[index].state));
      this._saveMorphHistory();
      this._applyFullState();
      this._updateControlsFromState();
      this._updateChangeCount();
    }
  }

  _applyFullState() {
    if (!this.renderer || !this.meshData) return;
    // Reset first
    this.renderer.resetDeformations?.();
    // Apply all regions
    for (const [regionName, params] of Object.entries(this.morphState)) {
      const vertices = this.meshData.regions[regionName];
      if (vertices) {
        this.renderer.deformRegion(vertices, params);
      }
    }
  }

  _saveMorphHistory() {
    this.morphHistory = this.morphHistory.slice(0, this.morphHistoryIndex + 1);
    this.morphHistory.push(JSON.parse(JSON.stringify(this.morphState)));
    this.morphHistoryIndex = this.morphHistory.length - 1;
    if (this.morphHistory.length > 50) {
      this.morphHistory.shift();
      this.morphHistoryIndex--;
    }
  }

  // ============================================================
  // UI SETUP
  // ============================================================

  _setupUploadUI() {
    // Photo upload slots
    document.querySelectorAll('.upload-slot').forEach(slot => {
      const angle = slot.dataset.angle;
      const input = slot.querySelector('.upload-input');

      slot.addEventListener('click', () => input.click());

      input.addEventListener('change', async (e) => {
        if (e.target.files[0]) {
          const photo = await this.photoUploader.setPhoto(angle, e.target.files[0]);
          // Show preview
          const existing = slot.querySelector('.upload-preview');
          if (existing) existing.remove();
          const img = document.createElement('img');
          img.className = 'upload-preview';
          img.src = photo.url;
          slot.appendChild(img);
          slot.classList.add('has-photo');

          // Enable generate button if front photo exists
          const status = this.photoUploader.getStatus();
          const genBtn = document.getElementById('generate-btn');
          genBtn.disabled = !status.canReconstruct;
          if (status.canReconstruct) {
            genBtn.textContent = 'Generate 3D Model';  // Reset text for re-generation
          }
        }
      });
    });

    // ===== Guided Capture Button =====
    this._setupGuidedCaptureButton();

    // Generate button
    document.getElementById('generate-btn')?.addEventListener('click', async () => {
      const btn = document.getElementById('generate-btn');
      console.log('Generate button clicked');
      btn.textContent = 'Loading AI model...';
      btn.disabled = true;

      try {
        // Lazy-init MediaPipeBridge and pass to uploader
        if (!this.mediaPipeBridge) {
          console.log('App: Creating MediaPipeBridge...');
          this.mediaPipeBridge = new MediaPipeBridge();
        }
        this.photoUploader._mediaPipeBridge = this.mediaPipeBridge;

        // Pre-init MediaPipe so we can show progress
        if (!this.mediaPipeBridge.isReady) {
          console.log('App: Initializing MediaPipe (first time, may take 10-20s)...');
          btn.textContent = 'Loading face AI (first time)...';
          const ok = await this.mediaPipeBridge.init();
          if (!ok) {
            console.warn('App: MediaPipe init failed, will use fallback projection');
          } else {
            console.log('App: MediaPipe ready');
          }
        }

        // Show multi-view status if side photos available
        const status = this.photoUploader.getStatus();
        if (status.photoCount > 1) {
          btn.textContent = `Processing ${status.photoCount} views...`;
        } else {
          btn.textContent = 'Detecting face...';
        }
        console.log(`App: Calling generateTextureFromPhoto (${status.photoCount} view(s))...`);

        // Generate texture from photo — pass render mode for delighting adjustment
        const renderMode = this.renderer?._photoRenderMode || 'hybrid';
        const texture = await this.photoUploader.generateTextureFromPhoto({ renderMode });
        console.log('App: Texture generated, applying...');

        btn.textContent = 'Applying texture...';

        // Expert round 3: normal map is computed from raw image-space photo (wrong!).
        // It doesn't align with the UV-projected texture → waxy look + pseudo artifacts.
        // Disable in hybrid mode. Only re-enable once computed from UV-space texture.
        const normalMap = renderMode !== 'hybrid'
          ? await this.photoUploader.generateNormalMapFromPhoto()
          : null;

        // Apply photo texture to existing mesh
        // flipY=false because the rasterizer writes UV V directly to canvas Y
        // (row 0 = V=0 = chin), matching WebGL's native texture layout
        if (this.renderer) {
          await this.renderer.loadTexture(texture.dataUrl, 'albedo', { flipY: false });
          if (normalMap) {
            await this.renderer.loadTexture(normalMap.dataUrl, 'normal', { flipY: false });
          }
        }

        // ===== DIAGNOSTIC: Show debug panel with overlays =====
        this._showTextureDebugPanel(this.photoUploader);

        // Expert diagnostics: toggle buttons available but NOT auto-shown
        this._addExpertDiagButtons(this.photoUploader);

        // ===== FaceDNA: capture digital face identity =====
        try {
          this.currentFaceDNA = FaceDNA.fromCurrentState(
            this.meshGenerator,
            this.photoUploader,
            this.mediaPipeBridge
          );
          const summary = this.currentFaceDNA.getSummary();
          console.log('App: FaceDNA created —', summary);
          this._addMessage('system',
            `FaceDNA captured: ${summary.views} view(s), ` +
            `shape=${summary.hasShape ? '✓' : '✗'}, ` +
            `texture=${summary.hasTexture ? '✓' : '✗'}, ` +
            `${Object.keys(summary.measurements).length} measurements`
          );
          // Enable FaceDNA buttons
          document.getElementById('facedna-save-btn')?.removeAttribute('disabled');
          document.getElementById('facedna-export-btn')?.removeAttribute('disabled');
        } catch (fdnaErr) {
          console.warn('App: FaceDNA creation failed:', fdnaErr.message);
        }

        this._addMessage('system', 'Photo texture projected onto 3D model.');
        btn.textContent = '✓ Generated';
        console.log('App: Generation complete');
      } catch (err) {
        console.error('App: Generation failed:', err);
        this._addMessage('system', `Error: ${err.message}`);
        btn.textContent = 'Generate 3D Model';
        btn.disabled = false;
      }
    });

    // ===== DIAGNOSTIC: UV Checkerboard test button =====
    document.getElementById('generate-btn')?.insertAdjacentHTML('afterend',
      ' <button id="uv-checker-btn" style="margin-left:6px;padding:4px 8px;font-size:11px;background:#553;color:#ff0;border:1px solid #882;border-radius:4px;cursor:pointer" title="Test 3: Load UV checkerboard to verify orientation">UV Check</button>'
    );
    document.getElementById('uv-checker-btn')?.addEventListener('click', async () => {
      const { PhotoUploader } = await import('./pipeline/PhotoUploader.js');
      const checkerUrl = PhotoUploader.generateUVCheckerboard();
      if (this.renderer) {
        await this.renderer.loadTexture(checkerUrl, 'albedo', { flipY: false });
        this._addMessage('system', 'UV Checkerboard loaded. Check: TL=chin-right, TR=chin-left, center=nose. If wrong → flipY/axis bug.');
      }
    });

    // Load reconstruction files
    document.getElementById('load-recon-btn')?.addEventListener('click', () => {
      document.getElementById('recon-mesh-input').click();
    });

    document.getElementById('recon-mesh-input')?.addEventListener('change', async (e) => {
      if (e.target.files[0]) {
        const meshFile = e.target.files[0];
        const meshType = meshFile.name.endsWith('.glb') ? 'glb' : 'obj';
        const meshUrl = URL.createObjectURL(meshFile);

        // Prompt for texture
        const textureInput = document.getElementById('recon-texture-input');
        this._addMessage('system', 'Mesh loaded. Now select the texture file (face_texture.png)...');
        textureInput.click();

        textureInput.addEventListener('change', async (te) => {
          const textureUrl = te.target.files[0] ? URL.createObjectURL(te.target.files[0]) : null;
          await this._loadReconstructedMesh(meshUrl, textureUrl, null, meshType);
        }, { once: true });
      }
    });
  }

  _setupControlsUI() {
    // Region tabs
    document.querySelectorAll('.region-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        this.activeRegionTab = e.target.dataset.region;
        document.querySelectorAll('.region-tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
        this._renderRegionControls();
      });
    });

    this._renderRegionControls();
  }

  /**
   * Setup the "Guided Capture" button in the upload section.
   * Dynamically imports CaptureGuide on click for code splitting.
   */
  _setupGuidedCaptureButton() {
    const genBtn = document.getElementById('generate-btn');
    if (!genBtn) return;

    const btn = document.createElement('button');
    btn.id = 'guided-capture-btn';
    btn.className = 'btn-secondary';
    btn.textContent = '📸 Guided Capture';
    btn.style.cssText = 'margin: 0 12px 8px; width: calc(100% - 24px); padding: 8px 16px; font-size: 13px; background: var(--bg-card); color: var(--accent-light); border: 1px solid var(--accent-dim); border-radius: var(--radius-sm); cursor: pointer; font-family: var(--font); transition: all 0.2s ease;';

    // Insert before the generate button's parent or before generate-btn
    genBtn.parentNode.insertBefore(btn, genBtn);

    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'var(--accent-dim)';
      btn.style.color = 'white';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'var(--bg-card)';
      btn.style.color = 'var(--accent-light)';
    });

    btn.addEventListener('click', async () => {
      btn.textContent = 'Loading...';
      btn.disabled = true;

      try {
        // Dynamic import for code splitting
        const { CaptureGuide } = await import('./capture/CaptureGuide.js');

        // Lazy-init MediaPipeBridge
        if (!this.mediaPipeBridge) {
          this.mediaPipeBridge = new MediaPipeBridge();
        }
        if (!this.mediaPipeBridge.isReady) {
          btn.textContent = 'Loading face AI...';
          await this.mediaPipeBridge.init();
        }

        const guide = new CaptureGuide({
          photoUploader: this.photoUploader,
          mediaPipeBridge: this.mediaPipeBridge,
          onComplete: async (captures) => {
            // Set photos on PhotoUploader
            for (const [angle, capture] of Object.entries(captures)) {
              if (capture && capture.blob) {
                const file = new File([capture.blob], `guided_${angle}.jpg`, { type: 'image/jpeg' });
                await this.photoUploader.setPhoto(angle, file);

                // Store quality score on photo metadata
                if (this.photoUploader.photos[angle]) {
                  this.photoUploader.photos[angle]._qualityScore = capture.quality?.overallScore / 100 || 1.0;
                }

                // Update upload slot UI
                const slot = document.querySelector(`.upload-slot[data-angle="${angle}"]`);
                if (slot) {
                  const existing = slot.querySelector('.upload-preview');
                  if (existing) existing.remove();
                  const img = document.createElement('img');
                  img.className = 'upload-preview';
                  img.src = capture.dataUrl;
                  slot.appendChild(img);
                  slot.classList.add('has-photo');
                }
              }
            }

            // Enable generate button
            const status = this.photoUploader.getStatus();
            const generateBtn = document.getElementById('generate-btn');
            if (generateBtn) {
              generateBtn.disabled = !status.canReconstruct;
              generateBtn.textContent = 'Generate 3D Model';
            }
            this._addMessage('system', '📸 Guided capture complete. Click "Generate 3D Model" to create 3D face.');
          },
          onCancel: () => {
            this._addMessage('system', 'Guided capture cancelled.');
          },
        });

        guide.start();
      } catch (err) {
        console.error('Guided capture error:', err);
        this._addMessage('system', `Guided capture failed: ${err.message}`);
      } finally {
        btn.textContent = '📸 Guided Capture';
        btn.disabled = false;
      }
    });
  }

  /**
   * Setup the Treatment Predictor UI panel.
   * Dynamically imports GeometryPredictor and ConfidenceScorer on use.
   */
  _setupTreatmentUI() {
    const select = document.getElementById('treatment-select');
    const volumeRow = document.getElementById('treatment-volume-row');
    const volumeSlider = document.getElementById('treatment-volume');
    const volumeLabel = document.getElementById('treatment-volume-label');
    const confidenceEl = document.getElementById('treatment-confidence');
    const previewBtn = document.getElementById('treatment-preview-btn');
    const resetBtn = document.getElementById('treatment-reset-btn');

    if (!select) return;

    // Lazy-loaded modules
    let GeometryPredictor = null;
    let ConfidenceScorer = null;
    let currentTreatment = null;

    // Populate dropdown on first click (lazy load)
    select.addEventListener('focus', async () => {
      if (GeometryPredictor) return; // already loaded

      try {
        const gpMod = await import('./prediction/GeometryPredictor.js');
        const csMod = await import('./prediction/ConfidenceScorer.js');
        GeometryPredictor = gpMod.GeometryPredictor;
        ConfidenceScorer = csMod.ConfidenceScorer;

        // Populate options grouped by category
        const byCategory = GeometryPredictor.getByCategory();
        select.innerHTML = '<option value="">— Select Treatment —</option>';
        for (const [category, treatments] of Object.entries(byCategory)) {
          const group = document.createElement('optgroup');
          group.label = category;
          for (const t of treatments) {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.label;
            group.appendChild(opt);
          }
          select.appendChild(group);
        }
      } catch (err) {
        console.error('Failed to load treatment modules:', err);
      }
    }, { once: true });

    // On treatment selection change
    select.addEventListener('change', () => {
      const id = select.value;
      if (!id || !GeometryPredictor) {
        volumeRow.style.display = 'none';
        confidenceEl.style.display = 'none';
        previewBtn.disabled = true;
        currentTreatment = null;
        return;
      }

      const treatment = GeometryPredictor.treatments[id];
      if (!treatment) return;

      currentTreatment = { id, ...treatment };

      // Show volume slider
      volumeRow.style.display = 'block';
      previewBtn.disabled = false;

      // Configure slider range
      const maxVol = treatment.maxVolume || 5;
      const defaultVol = treatment.defaultVolume || 1;
      const unit = treatment.unit || 'ml';

      volumeSlider.min = 0;
      volumeSlider.max = 100;
      volumeSlider.value = Math.round((defaultVol / maxVol) * 100);
      volumeLabel.textContent = `${defaultVol.toFixed(1)} ${unit}`;

      // Show confidence
      this._updateTreatmentConfidence(id, defaultVol, maxVol);
    });

    // Volume slider change
    volumeSlider?.addEventListener('input', () => {
      if (!currentTreatment || !GeometryPredictor) return;
      const maxVol = currentTreatment.maxVolume || 5;
      const unit = currentTreatment.unit || 'ml';
      const volume = (volumeSlider.value / 100) * maxVol;
      volumeLabel.textContent = `${volume.toFixed(1)} ${unit}`;
      this._updateTreatmentConfidence(currentTreatment.id, volume, maxVol);
    });

    // Preview button
    previewBtn?.addEventListener('click', async () => {
      if (!currentTreatment || !GeometryPredictor || !this.renderer) return;

      const maxVol = currentTreatment.maxVolume || 5;
      const volume = (volumeSlider.value / 100) * maxVol;

      const result = GeometryPredictor.predict(currentTreatment.id, { volume });
      if (!result) return;

      // Apply morph state via renderer with animation
      if (this.meshData?.regions) {
        // Merge with current morph state
        const treatmentMorphState = { ...this.morphState, ...result.morphState };

        // Use animated morph transition (500ms smooth lerp)
        if (this.renderer.animateMorphTo) {
          this.renderer.animateMorphTo(treatmentMorphState, this.meshData.regions, {
            duration: 500,
          });
        } else {
          this.renderer.applyMorphState(treatmentMorphState, this.meshData.regions);
        }

        // Store in FaceDNA if available
        if (this.currentFaceDNA) {
          const unit = currentTreatment.unit || 'ml';
          this.currentFaceDNA.addTreatment(
            result.morphState,
            [`${currentTreatment.label} (${volume.toFixed(1)} ${unit})`],
            `Predicted via GeometryPredictor`
          );
        }

        resetBtn.disabled = false;
        this._addMessage('system',
          `💉 Applied ${currentTreatment.label} at ${volume.toFixed(1)} ${currentTreatment.unit || 'ml'}`
        );
      }
    });

    // Reset button
    resetBtn?.addEventListener('click', () => {
      if (this.renderer) {
        this.renderer.resetDeformation();
        // Re-apply existing morph state without treatment
        if (this.meshData?.regions && Object.keys(this.morphState).length > 0) {
          this.renderer.applyMorphState(this.morphState, this.meshData.regions);
        }
        resetBtn.disabled = true;
        this._addMessage('system', 'Treatment preview reset.');
      }
    });
  }

  /**
   * Update the treatment confidence display.
   */
  _updateTreatmentConfidence(treatmentId, volume, maxVolume) {
    const confidenceEl = document.getElementById('treatment-confidence');
    if (!confidenceEl) return;

    // Dynamic import already done by _setupTreatmentUI
    import('./prediction/ConfidenceScorer.js').then(({ ConfidenceScorer }) => {
      const captureQuality = this.currentFaceDNA?.capture?.views?.[0]?.qualityScore ?? 1.0;
      const hasMultiView = (this.currentFaceDNA?.capture?.views?.length || 0) > 1;

      const conf = ConfidenceScorer.score(treatmentId, {
        volume,
        maxVolume,
        captureQuality,
        hasMultiView,
      });

      confidenceEl.style.display = 'block';
      confidenceEl.innerHTML = `
        <span style="color:${ConfidenceScorer.colorForLevel(conf.level)}">
          Confidence: ${Math.round(conf.score * 100)}% (${conf.level})
        </span>
      `;
    });
  }

  _renderRegionControls() {
    const container = document.getElementById('region-controls');
    if (!container || !this.meshData) return;

    container.innerHTML = '';

    const regionMeta = this.meshData.regionMeta || {};
    const regionTabMap = {
      all: null,
      forehead: ['forehead', 'forehead_left', 'forehead_right', 'forehead_center', 'temple_left', 'temple_right'],
      eyes: ['brow_left', 'brow_right', 'brow_inner_left', 'brow_inner_right', 'eye_left_upper', 'eye_left_lower', 'eye_right_upper', 'eye_right_lower', 'eye_left_corner_inner', 'eye_left_corner_outer', 'eye_right_corner_inner', 'eye_right_corner_outer', 'under_eye_left', 'under_eye_right', 'tear_trough_left', 'tear_trough_right'],
      nose: ['nose_bridge', 'nose_bridge_upper', 'nose_bridge_lower', 'nose_tip', 'nose_tip_left', 'nose_tip_right', 'nostril_left', 'nostril_right', 'nose_dorsum'],
      cheeks: ['cheek_left', 'cheek_right', 'cheekbone_left', 'cheekbone_right', 'cheek_hollow_left', 'cheek_hollow_right', 'nasolabial_left', 'nasolabial_right'],
      lips: ['lip_upper', 'lip_upper_left', 'lip_upper_right', 'lip_upper_center', 'lip_lower', 'lip_lower_left', 'lip_lower_right', 'lip_lower_center', 'lip_corner_left', 'lip_corner_right'],
      jaw: ['chin', 'chin_center', 'chin_left', 'chin_right', 'jaw_left', 'jaw_right', 'jawline_left', 'jawline_right'],
      skin: ['full_face']
    };

    const activeRegions = this.activeRegionTab === 'all'
      ? Object.keys(regionMeta)
      : (regionTabMap[this.activeRegionTab] || []);

    for (const regionName of activeRegions) {
      const meta = regionMeta[regionName];
      if (!meta) continue;

      const state = this.morphState[regionName] || { inflate: 0, translateX: 0, translateY: 0, translateZ: 0 };

      // Create control for "inflate" (volume)
      const row = document.createElement('div');
      const isModified = Math.abs(state.inflate) > 0.001 || Math.abs(state.translateZ) > 0.001;
      row.className = `control-row ${isModified ? 'modified' : ''}`;
      row.innerHTML = `
        <label class="control-label" title="${regionName}">${meta.label || regionName}</label>
        <input type="range" class="control-slider" data-region="${regionName}" data-key="inflate"
               min="-50" max="50" value="${Math.round(state.inflate * 100)}" step="1">
        <span class="control-value">${Math.round(state.inflate * 100)}</span>
      `;

      const slider = row.querySelector('.control-slider');
      const valueSpan = row.querySelector('.control-value');

      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value) / 100;
        this.setRegionValue(regionName, 'inflate', val);
        valueSpan.textContent = Math.round(val * 100);
        row.classList.toggle('modified', Math.abs(val) > 0.001);
      });

      // Highlight region on hover
      row.querySelector('.control-label').addEventListener('mouseenter', () => {
        if (this.renderer && this.meshData.regions[regionName]) {
          this.renderer.highlightRegion(this.meshData.regions[regionName], 0x7c5cff);
          const indicator = document.getElementById('region-indicator');
          const indicatorText = document.getElementById('region-indicator-text');
          if (indicator && indicatorText) {
            indicatorText.textContent = meta.label || regionName;
            indicator.style.display = 'block';
          }
        }
      });

      row.querySelector('.control-label').addEventListener('mouseleave', () => {
        this.renderer?.clearHighlight?.();
        const indicator = document.getElementById('region-indicator');
        if (indicator) indicator.style.display = 'none';
      });

      container.appendChild(row);
    }
  }

  _updateControlsFromState() {
    document.querySelectorAll('.control-slider').forEach(slider => {
      const region = slider.dataset.region;
      const key = slider.dataset.key;
      const state = this.morphState[region] || {};
      const val = (state[key] || 0) * 100;
      slider.value = Math.round(val);
      const valueSpan = slider.parentElement.querySelector('.control-value');
      if (valueSpan) valueSpan.textContent = Math.round(val);
      slider.parentElement.classList.toggle('modified', Math.abs(val) > 0.1);
    });
  }

  _updateChangeCount() {
    const changes = this.getChanges();
    const count = Object.keys(changes).length;
    const badge = document.getElementById('change-count');
    if (badge) badge.textContent = count;
  }

  _updateVersionsList() {
    const list = document.getElementById('versions-list');
    if (!list) return;

    if (this.versions.length === 0) {
      list.innerHTML = '<span class="empty-state">No saved versions</span>';
      return;
    }

    list.innerHTML = this.versions.map((v, i) => `
      <div class="version-item">
        <span class="version-name">${v.name}</span>
        <button class="version-load" data-index="${i}">Load</button>
      </div>
    `).join('');

    list.querySelectorAll('.version-load').forEach(btn => {
      btn.addEventListener('click', () => this.loadVersion(parseInt(btn.dataset.index)));
    });
  }

  _setupChatUI() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const voiceBtn = document.getElementById('voice-btn');

    chatInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this._sendChat();
      }
    });

    sendBtn?.addEventListener('click', () => this._sendChat());

    voiceBtn?.addEventListener('click', () => {
      this.isVoiceMode = !this.isVoiceMode;
      if (this.isVoiceMode) {
        this.voiceAgent.startListening('patient');
        voiceBtn.classList.add('active');
        voiceBtn.textContent = '⏹';
      } else {
        this.voiceAgent.stopListening();
        voiceBtn.classList.remove('active');
        voiceBtn.textContent = '🎤';
      }
    });

    // Quick action buttons
    document.querySelectorAll('.quick-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const prompt = btn.dataset.prompt;
        document.getElementById('chat-input').value = prompt;
        this._sendChat();
      });
    });
  }

  _setupTopBarUI() {
    document.getElementById('undo-btn')?.addEventListener('click', () => {
      if (this.undo()) this._addMessage('system', 'Change undone.');
    });
    document.getElementById('redo-btn')?.addEventListener('click', () => {
      if (this.redo()) this._addMessage('system', 'Change redone.');
    });
    document.getElementById('reset-btn')?.addEventListener('click', () => {
      this.reset();
      this._addMessage('system', 'Face reset to original.');
    });
    document.getElementById('save-btn')?.addEventListener('click', () => {
      const v = this.saveVersion();
      this._addMessage('system', `Saved as "${v.name}"`);
    });
    document.getElementById('compare-btn')?.addEventListener('click', () => {
      this.comparisonMode = !this.comparisonMode;
      this.renderer?.setComparisonMode?.(this.comparisonMode);
      document.getElementById('compare-btn').classList.toggle('active', this.comparisonMode);
    });
    document.getElementById('report-btn')?.addEventListener('click', () => {
      this._generateReport();
    });

    // ===== FaceDNA buttons (injected dynamically) =====
    this._setupFaceDNAButtons();

    // Mode tabs
    document.querySelectorAll('.mode-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        e.target.classList.add('active');
      });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z') { e.preventDefault(); this.undo(); }
        if (e.key === 'y') { e.preventDefault(); this.redo(); }
      }
    });
  }

  // ============================================================
  // FACEDNA UI
  // ============================================================

  _setupFaceDNAButtons() {
    const topBarRight = document.querySelector('.top-bar-right');
    if (!topBarRight) return;

    // Insert FaceDNA button group before the report button
    const reportBtn = document.getElementById('report-btn');
    const facednaGroup = document.createElement('span');
    facednaGroup.className = 'facedna-btn-group';
    facednaGroup.style.cssText = 'display:inline-flex;gap:4px;margin-left:8px;border-left:1px solid #444;padding-left:8px;';

    facednaGroup.innerHTML = `
      <button class="top-btn" id="facedna-save-btn" disabled title="Save FaceDNA to browser">🧬 Save DNA</button>
      <button class="top-btn" id="facedna-export-btn" disabled title="Export .fdna file">📥 Export</button>
      <button class="top-btn" id="facedna-import-btn" title="Import .fdna file">📤 Import</button>
      <button class="top-btn" id="facedna-load-btn" title="Load saved FaceDNA">📂 Load DNA</button>
      <input type="file" id="facedna-file-input" accept=".fdna" style="display:none">
    `;

    if (reportBtn) {
      topBarRight.insertBefore(facednaGroup, reportBtn.nextSibling);
    } else {
      topBarRight.appendChild(facednaGroup);
    }

    // -- Save to IndexedDB --
    document.getElementById('facedna-save-btn')?.addEventListener('click', async () => {
      if (!this.currentFaceDNA) {
        this._addMessage('system', 'No FaceDNA to save. Generate a 3D model from a photo first.');
        return;
      }
      try {
        // Record current morph state as treatment if there are changes
        if (Object.keys(this.morphState).length > 0) {
          const hasChanges = Object.values(this.morphState).some(s =>
            s.inflate || s.translateX || s.translateY || s.translateZ
          );
          if (hasChanges) {
            this.currentFaceDNA.addTreatment(this.morphState, ['morphing'], 'Auto-saved morph state');
          }
        }

        await this.currentFaceDNA.saveToDB();
        this._addMessage('system', `FaceDNA saved (id: ${this.currentFaceDNA.id.slice(0, 12)}...)`);
      } catch (err) {
        this._addMessage('system', `FaceDNA save failed: ${err.message}`);
      }
    });

    // -- Export as .fdna file --
    document.getElementById('facedna-export-btn')?.addEventListener('click', () => {
      if (!this.currentFaceDNA) {
        this._addMessage('system', 'No FaceDNA to export. Generate a 3D model from a photo first.');
        return;
      }
      this.currentFaceDNA.downloadFile();
      this._addMessage('system', 'FaceDNA exported as .fdna file.');
    });

    // -- Import .fdna file --
    document.getElementById('facedna-import-btn')?.addEventListener('click', () => {
      document.getElementById('facedna-file-input')?.click();
    });

    document.getElementById('facedna-file-input')?.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      try {
        const dna = await FaceDNA.fromFile(file);
        this.currentFaceDNA = dna;

        // Apply geometry if we have FLAME loaded
        if (this.flameLoaded && dna.geometry.shapeParams) {
          dna.applyGeometry(this.meshGenerator);
        }

        // Apply texture
        if (dna.texture.albedoDataUrl && this.renderer) {
          await this.renderer.loadTexture(dna.texture.albedoDataUrl, 'albedo', { flipY: false });
        }

        const summary = dna.getSummary();
        this._addMessage('system',
          `FaceDNA imported: ${summary.views} view(s), ` +
          `${summary.treatmentCount} treatment(s), ` +
          `${Object.keys(summary.measurements).length} measurements`
        );

        // Enable buttons
        document.getElementById('facedna-save-btn')?.removeAttribute('disabled');
        document.getElementById('facedna-export-btn')?.removeAttribute('disabled');
      } catch (err) {
        this._addMessage('system', `FaceDNA import failed: ${err.message}`);
      }

      // Reset file input so same file can be re-selected
      e.target.value = '';
    });

    // -- Load from IndexedDB --
    document.getElementById('facedna-load-btn')?.addEventListener('click', async () => {
      try {
        const entries = await FaceDNA.listFromDB();
        if (entries.length === 0) {
          this._addMessage('system', 'No saved FaceDNA records found.');
          return;
        }

        // Show a simple selection in chat
        const entryList = entries
          .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
          .slice(0, 10)
          .map((e, i) => `${i + 1}. ${e.id.slice(0, 12)}... (${new Date(e.createdAt).toLocaleDateString()}, ${e.views} views, texture: ${e.hasTexture ? '✓' : '✗'})`)
          .join('\n');

        this._addMessage('system',
          `Saved FaceDNA records:\n\n${entryList}\n\nClick a record to load:`
        );

        // Create clickable buttons for each entry
        const chatMessages = document.querySelector('.chat-messages');
        const lastMsg = chatMessages?.lastElementChild;
        if (lastMsg) {
          const btnContainer = document.createElement('div');
          btnContainer.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;';

          entries.slice(0, 10).forEach((entry) => {
            const loadBtn = document.createElement('button');
            loadBtn.style.cssText = 'padding:4px 8px;font-size:11px;background:#234;color:#8cf;border:1px solid #456;border-radius:4px;cursor:pointer;';
            loadBtn.textContent = `${entry.id.slice(0, 8)}… (${new Date(entry.createdAt).toLocaleDateString()})`;
            loadBtn.addEventListener('click', async () => {
              await this._loadFaceDNA(entry.id);
              btnContainer.remove();
            });
            btnContainer.appendChild(loadBtn);
          });

          lastMsg.appendChild(btnContainer);
        }
      } catch (err) {
        this._addMessage('system', `Failed to list FaceDNA records: ${err.message}`);
      }
    });
  }

  /**
   * Load a FaceDNA from IndexedDB and apply to the current scene.
   */
  async _loadFaceDNA(id) {
    try {
      const dna = await FaceDNA.loadFromDB(id);
      if (!dna) {
        this._addMessage('system', `FaceDNA record ${id} not found.`);
        return;
      }

      this.currentFaceDNA = dna;

      // Apply geometry
      if (this.flameLoaded && dna.geometry.shapeParams) {
        dna.applyGeometry(this.meshGenerator);
      }

      // Apply texture
      if (dna.texture.albedoDataUrl && this.renderer) {
        await this.renderer.loadTexture(dna.texture.albedoDataUrl, 'albedo', { flipY: false });
      }

      const summary = dna.getSummary();
      this._addMessage('system',
        `FaceDNA loaded: ${summary.views} view(s), ` +
        `${summary.treatmentCount} treatment(s). ` +
        `Symmetry: ${summary.symmetry?.overall !== undefined ? (summary.symmetry.overall * 100).toFixed(1) + '%' : 'N/A'}`
      );

      // Enable buttons
      document.getElementById('facedna-save-btn')?.removeAttribute('disabled');
      document.getElementById('facedna-export-btn')?.removeAttribute('disabled');
    } catch (err) {
      this._addMessage('system', `FaceDNA load failed: ${err.message}`);
    }
  }

  _setupViewportUI() {
    // View buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const view = e.target.dataset.view;
        this.renderer?.setCameraView?.(view);
        document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
      });
    });

    // Expression buttons
    document.querySelectorAll('.expr-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const exprKey = e.target.dataset.expression;
        const expr = this.expressionAgent.animate(exprKey);
        if (expr) {
          this.renderer?.playExpression?.(expr);
          document.querySelectorAll('.expr-btn').forEach(b => b.classList.remove('active'));
          e.target.classList.add('active');
          const duration = expr.animationParams?.duration || 2000;
          setTimeout(() => e.target.classList.remove('active'), duration);
        }
      });
    });
  }

  // ============================================================
  // CHAT
  // ============================================================

  async _sendChat() {
    const input = document.getElementById('chat-input');
    if (!input || !input.value.trim()) return;

    const text = input.value.trim();
    input.value = '';
    this._handleInput(text);
  }

  async _handleInput(text) {
    this._addMessage('user', text);
    this._setThinking(true);

    try {
      const result = await this.director.processMessage(text);
      this._setThinking(false);

      if (result.response) {
        this._addMessage('assistant', result.response);
      }

      // Handle actions from director
      for (const action of (result.actions || [])) {
        if (action.type === 'expression' && action.data) {
          this.renderer?.playExpression?.(action.data);
        }
        if (action.type === 'highlight_region') {
          // Scroll to region tab if applicable
        }
        if (action.type === 'compare') {
          this.comparisonMode = !this.comparisonMode;
          this.renderer?.setComparisonMode?.(this.comparisonMode);
          document.getElementById('compare-btn')?.classList.toggle('active', this.comparisonMode);
        }
        if (action.type === 'report') {
          this._generateReport();
        }
      }

      // Update UI state after modification
      if (result.morphResult) {
        this._updateControlsFromState();
        this._updateChangeCount();
      }

      // Voice feedback
      if (this.isVoiceMode && result.response) {
        const cleanText = result.response.replace(/[*•💉✨🔧⚠️📋🎛️📸📁📦📂]/g, '').replace(/\*\*/g, '');
        this.voiceAgent.speak(cleanText);
      }
    } catch (err) {
      this._setThinking(false);
      this._addMessage('system', 'Something went wrong. Please try again.');
      console.error(err);
    }
  }

  // -----------------------------------------------------------------------
  // DIAGNOSTIC: Debug panel for texture projection
  // -----------------------------------------------------------------------

  _showTextureDebugPanel(uploader) {
    // Remove old panel if present
    document.getElementById('texture-debug-panel')?.remove();

    const panel = document.createElement('div');
    panel.id = 'texture-debug-panel';
    panel.style.cssText = `
      position:fixed; bottom:0; left:0; right:0; z-index:10000;
      background:#111; border-top:2px solid #f80; padding:10px;
      display:flex; gap:12px; overflow-x:auto; max-height:45vh;
    `;

    const makeSection = (title, dataUrl) => {
      if (!dataUrl) return '';
      return `<div style="flex-shrink:0;text-align:center">
        <div style="color:#ff0;font-size:11px;margin-bottom:4px;font-family:monospace">${title}</div>
        <img src="${dataUrl}" style="max-height:35vh;border:1px solid #555;image-rendering:pixelated" />
      </div>`;
    };

    let html = '';
    html += makeSection('TEST 1a: Photo + Landmarks', uploader._debugPhotoLandmarks);
    html += makeSection('TEST 1b: UV Atlas + Points', uploader._debugUVLandmarks);
    html += makeSection('TEST 2a: Texture (pre-fill)', uploader._debugPreFillTexture);
    html += makeSection('TEST 2b: Texture (final)', uploader._debugFinalTexture);

    // Close button
    html += `<div style="flex-shrink:0;display:flex;align-items:center">
      <button onclick="this.closest('#texture-debug-panel').remove()"
        style="padding:8px 16px;background:#800;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:14px">
        ✕ Close
      </button>
    </div>`;

    panel.innerHTML = html;
    document.body.appendChild(panel);
    console.log('PhotoUploader DIAG: Debug panel displayed');
  }

  // -----------------------------------------------------------------------
  // EXPERT DIAGNOSTICS: Texture export + Quad/Sphere toggles
  // -----------------------------------------------------------------------

  _addExpertDiagButtons(uploader) {
    // Remove existing
    document.getElementById('expert-diag-panel')?.remove();

    const panel = document.createElement('div');
    panel.id = 'expert-diag-panel';
    panel.style.cssText = `
      position:fixed; top:52px; right:10px; z-index:10001;
      background:#1a1a3a; border:1px solid #444; border-radius:8px;
      padding:10px; display:flex; flex-direction:column; gap:6px;
      font-family:'JetBrains Mono',monospace; font-size:11px;
    `;

    const makeBtn = (label, onClick) => {
      const btn = document.createElement('button');
      btn.textContent = label;
      btn.style.cssText = 'padding:4px 10px;background:#333;color:#0f0;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:11px;font-family:monospace;text-align:left';
      btn.addEventListener('click', onClick);
      return btn;
    };

    // Download helper
    const downloadDataUrl = (dataUrl, filename) => {
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = filename;
      a.click();
    };

    panel.appendChild(makeBtn('📥 Export: photoUV_raw', () => {
      if (uploader._debugPhotoUV_raw) downloadDataUrl(uploader._debugPhotoUV_raw, 'photoUV_raw.png');
      else alert('No raw UV texture available — generate first');
    }));
    panel.appendChild(makeBtn('📥 Export: alpha_coverage', () => {
      if (uploader._debugAlphaCoverage) downloadDataUrl(uploader._debugAlphaCoverage, 'alpha_coverage.png');
      else alert('No alpha coverage available — generate first');
    }));
    panel.appendChild(makeBtn('📥 Export: albedo_tinted', () => {
      if (uploader._debugAlbedoTinted) downloadDataUrl(uploader._debugAlbedoTinted, 'albedo_tinted.png');
      else alert('No albedo tinted available — generate first');
    }));
    panel.appendChild(makeBtn('📥 Export: photoUV_final', () => {
      if (uploader._debugFinalTexture) downloadDataUrl(uploader._debugFinalTexture, 'photoUV_final.png');
      else alert('No final texture available — generate first');
    }));

    // Separator
    const sep = document.createElement('div');
    sep.style.cssText = 'height:1px;background:#444;margin:4px 0';
    panel.appendChild(sep);

    // Toggle buttons
    panel.appendChild(makeBtn('🔲 Toggle Quad Test', () => {
      if (this.renderer?._quadTestMesh) this.renderer.hideQuadTest();
      else this.renderer?.showQuadTest();
    }));
    panel.appendChild(makeBtn('⚪ Toggle Gray Sphere', () => {
      if (this.renderer?._graySphere) this.renderer.hideGraySphere();
      else this.renderer?.showGraySphere();
    }));

    // Separator: P1 Diagnostic Overlays
    const sep3 = document.createElement('div');
    sep3.style.cssText = 'height:1px;background:#444;margin:4px 0';
    panel.appendChild(sep3);

    const diagLabel = document.createElement('div');
    diagLabel.style.cssText = 'color:#888;font-size:10px;margin-bottom:2px';
    diagLabel.textContent = 'Diagnostic Overlays:';
    panel.appendChild(diagLabel);

    panel.appendChild(makeBtn('🟢 Coverage Heatmap', () => {
      const url = uploader.generateCoverageHeatmap();
      if (url) this.renderer?.applyDiagOverlay(url);
      else alert('No coverage data — generate texture first');
    }));
    panel.appendChild(makeBtn('🔴 N·V Visibility', () => {
      const url = uploader.generateNVHeatmap();
      if (url) this.renderer?.applyDiagOverlay(url);
      else alert('No visibility data — generate texture first');
    }));
    panel.appendChild(makeBtn('🔵 UV Landmarks', () => {
      const url = uploader.generateLandmarkOverlay();
      if (url) this.renderer?.applyDiagOverlay(url);
      else alert('No landmark data — generate texture first');
    }));
    panel.appendChild(makeBtn('🔄 Restore Photo', () => {
      this.renderer?.restoreDiagOverlay();
    }));

    // Separator
    const sep2 = document.createElement('div');
    sep2.style.cssText = 'height:1px;background:#444;margin:4px 0';
    panel.appendChild(sep2);

    // Render mode label
    const modeLabel = document.createElement('div');
    modeLabel.style.cssText = 'color:#888;font-size:10px;margin-bottom:2px';
    modeLabel.textContent = `Mode: ${this.renderer?._photoRenderMode || 'hybrid'}`;
    modeLabel.id = 'render-mode-label';
    panel.appendChild(modeLabel);

    // Mode toggle buttons (A/B/C comparison)
    for (const mode of ['hybrid', 'pbr', 'emissive']) {
      panel.appendChild(makeBtn(`🎨 ${mode}`, () => {
        this.renderer?.setPhotoRenderMode(mode);
        const label = document.getElementById('render-mode-label');
        if (label) label.textContent = `Mode: ${mode}`;
      }));
    }

    // Close
    panel.appendChild(makeBtn('✕ Close Diag Panel', () => {
      panel.remove();
      this.renderer?.hideQuadTest();
      this.renderer?.hideGraySphere();
    }));

    document.body.appendChild(panel);
  }

  _addMessage(role, text) {
    const messages = document.getElementById('chat-messages');
    if (!messages) return;

    const msg = document.createElement('div');
    msg.className = `message message-${role}`;

    const labels = { user: 'You', assistant: 'AI Assistant', system: 'System' };
    const label = labels[role] || role;

    if (role === 'system') {
      msg.innerHTML = `<div class="message-body">${this._formatText(text)}</div>`;
    } else {
      msg.innerHTML = `
        <div class="message-header">${label}</div>
        <div class="message-body">${this._formatText(text)}</div>
      `;
    }

    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  }

  _formatText(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>')
      .replace(/• /g, '&bull; ');
  }

  _setThinking(isThinking) {
    const el = document.getElementById('thinking-indicator');
    if (el) el.style.display = isThinking ? 'flex' : 'none';
    const interim = document.getElementById('interim-transcript');
    if (interim && !isThinking) interim.style.display = 'none';
  }

  _updatePhaseIndicator(phase) {
    const el = document.getElementById('phase-indicator');
    if (!el) return;
    const labels = {
      upload: 'Upload Photos',
      capture: 'Capturing...',
      exploration: 'Explore & Modify',
      comparison: 'Comparing',
      report: 'Report'
    };
    el.querySelector('.phase-label').textContent = labels[phase] || phase;
  }

  _generateReport() {
    const changes = this.getChanges();
    const changeCount = Object.keys(changes).length;

    const regionLabels = Object.entries(changes).map(([region]) => {
      const meta = this.meshData?.regionMeta?.[region];
      return meta?.label || region;
    });

    const modal = document.getElementById('report-modal');
    const content = document.getElementById('report-content');
    if (!modal || !content) return;

    content.innerHTML = `
      <button class="modal-close" onclick="document.getElementById('report-modal').style.display='none'">&times;</button>
      <h1>Facial Design Consultation Report</h1>
      <p class="report-date">${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>

      <div class="report-section">
        <h2>Consultation Summary</h2>
        <div class="report-section-body">
          ${changeCount > 0
            ? `This consultation explored ${changeCount} regional modification(s) across: ${regionLabels.join(', ')}.`
            : 'No modifications were made during this session.'}
        </div>
      </div>

      <div class="report-section">
        <h2>Modifications Applied</h2>
        <div class="report-section-body">
          ${changeCount > 0
            ? Object.entries(changes).map(([region, params]) => {
                const meta = this.meshData?.regionMeta?.[region];
                const label = meta?.label || region;
                const details = Object.entries(params)
                  .filter(([, v]) => Math.abs(v) > 0.001)
                  .map(([k, v]) => `${k}: ${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`)
                  .join(', ');
                return `&bull; <strong>${label}</strong>: ${details}`;
              }).join('<br>')
            : '<em>No modifications</em>'}
        </div>
      </div>

      <div class="report-section">
        <h2>Versions Saved</h2>
        <div class="report-section-body">
          ${this.versions.length > 0
            ? this.versions.map(v => `&bull; ${v.name}`).join('<br>')
            : '<em>No versions saved</em>'}
        </div>
      </div>

      <div class="report-disclaimer">
        ⚠️ This report is generated by an AI visualization tool and is not medical advice.
        All treatment decisions must be made in consultation with a qualified medical professional.
      </div>
    `;

    modal.style.display = 'flex';

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.style.display = 'none';
    }, { once: true });
  }
}

// ============================================================
// BOOT
// ============================================================
window.addEventListener('DOMContentLoaded', () => {
  window.app = new FacialAIProject();
  window.app.init();
});
