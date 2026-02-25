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
          document.getElementById('generate-btn').disabled = !status.canReconstruct;
        }
      });
    });

    // Generate button
    document.getElementById('generate-btn')?.addEventListener('click', async () => {
      const btn = document.getElementById('generate-btn');
      btn.textContent = 'Detecting face...';
      btn.disabled = true;

      try {
        // Lazy-init MediaPipeBridge and pass to uploader
        if (!this.mediaPipeBridge) {
          this.mediaPipeBridge = new MediaPipeBridge();
        }
        this.photoUploader._mediaPipeBridge = this.mediaPipeBridge;

        btn.textContent = 'Generating texture...';

        // Generate texture from photo (uses piecewise affine warp if possible)
        const texture = await this.photoUploader.generateTextureFromPhoto();
        const normalMap = await this.photoUploader.generateNormalMapFromPhoto();

        // Apply photo texture to existing mesh
        if (this.renderer) {
          await this.renderer.loadTexture(texture.dataUrl, 'albedo');
          if (normalMap) {
            await this.renderer.loadTexture(normalMap.dataUrl, 'normal');
          }
        }

        this._addMessage('system', 'Photo texture projected onto 3D model. You can now modify the face using chat or sliders.');
        btn.textContent = '✓ Generated';
      } catch (err) {
        console.error(err);
        this._addMessage('system', `Error: ${err.message}`);
        btn.textContent = 'Generate 3D Model';
        btn.disabled = false;
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
