/**
 * KioskApp — Touch-optimized clinic kiosk interface.
 *
 * Simplified, fullscreen flow designed for a tablet in a clinic:
 *   1. Check-in (patient lookup or new patient)
 *   2. Guided Face Capture (reuses CaptureGuide)
 *   3. 3D Preview + Treatment Selection
 *   4. Treatment Prediction Preview
 *   5. Report Generation
 *   6. Session End (auto-logout on inactivity)
 *
 * All state is ephemeral — data is synced to the backend API
 * and cleared from the device after each session.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INACTIVITY_TIMEOUT = 5 * 60 * 1000; // 5 minutes

const KIOSK_STEPS = [
  { id: 'checkin',    label: 'Check In',     icon: '👤' },
  { id: 'capture',    label: 'Face Scan',    icon: '📸' },
  { id: 'preview',    label: '3D Preview',   icon: '🎨' },
  { id: 'treatment',  label: 'Treatment',    icon: '💉' },
  { id: 'report',     label: 'Report',       icon: '📄' },
];

const API_BASE = '/api/v1';

// ---------------------------------------------------------------------------
// KioskApp
// ---------------------------------------------------------------------------

class KioskApp {
  constructor() {
    this._root = document.getElementById('kiosk-app');
    this._currentStep = 0;
    this._patientId = null;
    this._sessionToken = null;
    this._inactivityTimer = null;

    // Inject base styles
    this._injectStyles();
  }

  async init() {
    console.log('KioskApp: Initializing...');
    this._resetInactivityTimer();
    this._setupInteractionListeners();
    this._showStep('checkin');
  }

  // =========================================================================
  // Step rendering
  // =========================================================================

  _showStep(stepId) {
    this._currentStep = KIOSK_STEPS.findIndex(s => s.id === stepId);
    this._root.innerHTML = '';

    // Progress bar
    const progress = this._createProgressBar();
    this._root.appendChild(progress);

    // Step content
    const content = document.createElement('div');
    content.className = 'kiosk-content';
    this._root.appendChild(content);

    switch (stepId) {
      case 'checkin':   this._renderCheckin(content); break;
      case 'capture':   this._renderCapture(content); break;
      case 'preview':   this._renderPreview(content); break;
      case 'treatment': this._renderTreatment(content); break;
      case 'report':    this._renderReport(content); break;
    }
  }

  _createProgressBar() {
    const bar = document.createElement('div');
    bar.className = 'kiosk-progress';

    KIOSK_STEPS.forEach((step, i) => {
      const dot = document.createElement('div');
      dot.className = 'kiosk-progress-step';
      if (i < this._currentStep) dot.classList.add('completed');
      if (i === this._currentStep) dot.classList.add('active');

      dot.innerHTML = `
        <span class="kiosk-step-icon">${step.icon}</span>
        <span class="kiosk-step-label">${step.label}</span>
      `;
      bar.appendChild(dot);

      if (i < KIOSK_STEPS.length - 1) {
        const line = document.createElement('div');
        line.className = 'kiosk-progress-line';
        if (i < this._currentStep) line.classList.add('completed');
        bar.appendChild(line);
      }
    });

    return bar;
  }

  // =========================================================================
  // Step 1: Check-in
  // =========================================================================

  _renderCheckin(container) {
    container.innerHTML = `
      <div class="kiosk-card kiosk-card-center">
        <h1 class="kiosk-title">Welcome</h1>
        <p class="kiosk-subtitle">Please check in to begin your consultation</p>

        <div class="kiosk-form">
          <input type="text" id="kiosk-name" class="kiosk-input" placeholder="Your full name" autocomplete="off">
          <input type="email" id="kiosk-email" class="kiosk-input" placeholder="Email (optional)" autocomplete="off">
          <input type="tel" id="kiosk-phone" class="kiosk-input" placeholder="Phone (optional)" autocomplete="off">
        </div>

        <button class="kiosk-btn kiosk-btn-primary" id="kiosk-checkin-btn">
          Start Consultation →
        </button>
      </div>
    `;

    container.querySelector('#kiosk-checkin-btn').addEventListener('click', async () => {
      const name = container.querySelector('#kiosk-name').value.trim();
      if (!name) {
        container.querySelector('#kiosk-name').style.borderColor = 'var(--kiosk-danger)';
        return;
      }

      // In production: create patient via API
      // For now: store locally and proceed
      this._patientId = `local-${Date.now()}`;
      this._patientName = name;

      this._showStep('capture');
    });
  }

  // =========================================================================
  // Step 2: Capture
  // =========================================================================

  _renderCapture(container) {
    container.innerHTML = `
      <div class="kiosk-card kiosk-card-full">
        <h2 class="kiosk-title">Face Scan</h2>
        <p class="kiosk-subtitle">We'll capture your face from 3 angles for the best 3D model</p>
        <div id="kiosk-capture-area" style="flex:1;display:flex;align-items:center;justify-content:center;">
          <button class="kiosk-btn kiosk-btn-primary kiosk-btn-large" id="kiosk-start-capture">
            📸 Start Face Capture
          </button>
        </div>
        <button class="kiosk-btn kiosk-btn-ghost" id="kiosk-skip-capture">
          Skip (use manual upload later)
        </button>
      </div>
    `;

    container.querySelector('#kiosk-start-capture').addEventListener('click', async () => {
      // Dynamic import of CaptureGuide
      try {
        const { CaptureGuide } = await import('../capture/CaptureGuide.js');
        const { MediaPipeBridge } = await import('../engine/MediaPipeBridge.js');

        const btn = container.querySelector('#kiosk-start-capture');
        btn.textContent = 'Loading AI model...';
        btn.disabled = true;

        const bridge = new MediaPipeBridge();
        await bridge.init();

        const guide = new CaptureGuide({
          photoUploader: null, // Kiosk manages photos independently
          mediaPipeBridge: bridge,
          onComplete: (captures) => {
            this._captures = captures;
            this._showStep('preview');
          },
          onCancel: () => {
            btn.textContent = '📸 Start Face Capture';
            btn.disabled = false;
          },
        });

        guide.start();
      } catch (err) {
        console.error('Kiosk capture error:', err);
        container.querySelector('#kiosk-capture-area').innerHTML =
          `<p style="color:var(--kiosk-danger)">Camera error: ${err.message}</p>`;
      }
    });

    container.querySelector('#kiosk-skip-capture').addEventListener('click', () => {
      this._showStep('preview');
    });
  }

  // =========================================================================
  // Step 3: Preview
  // =========================================================================

  _renderPreview(container) {
    container.innerHTML = `
      <div class="kiosk-card kiosk-card-full">
        <h2 class="kiosk-title">3D Face Model</h2>
        <p class="kiosk-subtitle">Your 3D facial model is ready for consultation</p>
        <div id="kiosk-3d-viewport" style="flex:1;min-height:300px;background:#0a0a1a;border-radius:12px;margin:16px 0;"></div>
        <button class="kiosk-btn kiosk-btn-primary" id="kiosk-to-treatment">
          Continue to Treatment Options →
        </button>
      </div>
    `;

    // Initialize 3D renderer in the viewport
    this._init3DPreview(container.querySelector('#kiosk-3d-viewport'));

    container.querySelector('#kiosk-to-treatment').addEventListener('click', () => {
      this._showStep('treatment');
    });
  }

  async _init3DPreview(viewport) {
    try {
      const { FaceRenderer } = await import('../engine/FaceRenderer.js');
      const { FlameMeshGenerator } = await import('../engine/FlameMeshGenerator.js');

      const renderer = new FaceRenderer(viewport);
      renderer.init();

      const meshGen = new FlameMeshGenerator();
      const result = await meshGen.loadFLAME();
      if (result?.geometry) {
        renderer.loadFromGeometry(result.geometry);
      }

      this._kioskRenderer = renderer;
      this._kioskMeshGen = meshGen;
    } catch (err) {
      console.error('Kiosk 3D preview error:', err);
      viewport.innerHTML = `<p style="color:var(--kiosk-danger);padding:20px;">3D preview unavailable: ${err.message}</p>`;
    }
  }

  // =========================================================================
  // Step 4: Treatment
  // =========================================================================

  _renderTreatment(container) {
    container.innerHTML = `
      <div class="kiosk-card kiosk-card-full">
        <h2 class="kiosk-title">Treatment Options</h2>
        <p class="kiosk-subtitle">Select a treatment to preview the predicted result</p>
        <div id="kiosk-treatment-list" class="kiosk-treatment-grid">
          <!-- Populated dynamically -->
        </div>
        <button class="kiosk-btn kiosk-btn-primary" id="kiosk-to-report" disabled>
          Generate Report →
        </button>
      </div>
    `;

    this._populateTreatments(container);

    container.querySelector('#kiosk-to-report').addEventListener('click', () => {
      this._showStep('report');
    });
  }

  async _populateTreatments(container) {
    const { GeometryPredictor } = await import('../prediction/GeometryPredictor.js');

    const grid = container.querySelector('#kiosk-treatment-list');
    const categories = GeometryPredictor.getByCategory();

    for (const [category, treatments] of Object.entries(categories)) {
      const catHeader = document.createElement('div');
      catHeader.className = 'kiosk-cat-header';
      catHeader.textContent = category;
      grid.appendChild(catHeader);

      for (const t of treatments) {
        if (t.isCombo) continue; // Skip combos for now

        const card = document.createElement('button');
        card.className = 'kiosk-treatment-card';
        card.innerHTML = `
          <span class="kiosk-treatment-name">${t.label}</span>
          <span class="kiosk-treatment-desc">${t.description || ''}</span>
          <span class="kiosk-treatment-range">${t.defaultVolume} ${t.unit || 'ml'}</span>
        `;

        card.addEventListener('click', () => {
          // Deselect others
          grid.querySelectorAll('.kiosk-treatment-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          this._selectedTreatment = t.id;
          container.querySelector('#kiosk-to-report').disabled = false;
        });

        grid.appendChild(card);
      }
    }
  }

  // =========================================================================
  // Step 5: Report
  // =========================================================================

  _renderReport(container) {
    container.innerHTML = `
      <div class="kiosk-card kiosk-card-center">
        <div class="kiosk-report-icon">📄</div>
        <h2 class="kiosk-title">Consultation Complete</h2>
        <p class="kiosk-subtitle">
          Thank you, ${this._patientName || 'Patient'}!<br>
          Your consultation report has been generated.
        </p>
        <div class="kiosk-report-summary">
          <p><strong>Treatment discussed:</strong> ${this._selectedTreatment || 'None selected'}</p>
          <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
        </div>
        <div class="kiosk-report-actions">
          <button class="kiosk-btn kiosk-btn-primary" id="kiosk-email-report">
            📧 Email Report
          </button>
          <button class="kiosk-btn kiosk-btn-ghost" id="kiosk-end-session">
            End Session
          </button>
        </div>
      </div>
    `;

    container.querySelector('#kiosk-email-report').addEventListener('click', () => {
      alert('Report email feature coming soon!');
    });

    container.querySelector('#kiosk-end-session').addEventListener('click', () => {
      this._endSession();
    });
  }

  // =========================================================================
  // Session management
  // =========================================================================

  _endSession() {
    // Clear all patient data
    this._patientId = null;
    this._patientName = null;
    this._captures = null;
    this._selectedTreatment = null;

    // Cleanup renderer
    if (this._kioskRenderer) {
      this._kioskRenderer.destroy();
      this._kioskRenderer = null;
    }

    // Reset to check-in
    this._showStep('checkin');
  }

  _resetInactivityTimer() {
    clearTimeout(this._inactivityTimer);
    this._inactivityTimer = setTimeout(() => {
      console.log('KioskApp: Inactivity timeout — ending session');
      this._endSession();
    }, INACTIVITY_TIMEOUT);
  }

  _setupInteractionListeners() {
    ['click', 'touchstart', 'keydown'].forEach(event => {
      document.addEventListener(event, () => this._resetInactivityTimer(), { passive: true });
    });
  }

  // =========================================================================
  // Styles
  // =========================================================================

  _injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      :root {
        --kiosk-bg: #08081a;
        --kiosk-card: #141438;
        --kiosk-accent: #7c5cff;
        --kiosk-accent-light: #a08cff;
        --kiosk-success: #4ade80;
        --kiosk-danger: #f87171;
        --kiosk-text: #eaeaf4;
        --kiosk-text-dim: #9898c0;
        --kiosk-border: rgba(255,255,255,0.08);
      }

      * { margin: 0; padding: 0; box-sizing: border-box; }

      body {
        font-family: 'Inter', sans-serif;
        background: var(--kiosk-bg);
        color: var(--kiosk-text);
        height: 100vh;
        overflow: hidden;
        -webkit-font-smoothing: antialiased;
        user-select: none;
        -webkit-user-select: none;
      }

      #kiosk-app {
        display: flex;
        flex-direction: column;
        height: 100vh;
        padding: 16px;
        gap: 16px;
      }

      /* Progress bar */
      .kiosk-progress {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 12px 0;
        flex-shrink: 0;
      }

      .kiosk-progress-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        opacity: 0.4;
        transition: opacity 0.3s;
      }

      .kiosk-progress-step.active { opacity: 1; }
      .kiosk-progress-step.completed { opacity: 0.7; }

      .kiosk-step-icon { font-size: 24px; }
      .kiosk-step-label { font-size: 11px; color: var(--kiosk-text-dim); }
      .kiosk-progress-step.active .kiosk-step-label { color: var(--kiosk-accent-light); }

      .kiosk-progress-line {
        width: 32px; height: 2px;
        background: var(--kiosk-border);
        margin-bottom: 18px;
      }
      .kiosk-progress-line.completed { background: var(--kiosk-success); }

      /* Content */
      .kiosk-content { flex: 1; display: flex; min-height: 0; }

      .kiosk-card {
        flex: 1;
        background: var(--kiosk-card);
        border-radius: 16px;
        padding: 32px;
        display: flex;
        flex-direction: column;
      }

      .kiosk-card-center { align-items: center; justify-content: center; text-align: center; }
      .kiosk-card-full { align-items: stretch; }

      .kiosk-title { font-size: 28px; font-weight: 600; margin-bottom: 8px; }
      .kiosk-subtitle { font-size: 16px; color: var(--kiosk-text-dim); margin-bottom: 24px; line-height: 1.5; }

      /* Form */
      .kiosk-form { display: flex; flex-direction: column; gap: 12px; width: 100%; max-width: 400px; margin-bottom: 24px; }

      .kiosk-input {
        padding: 14px 16px;
        font-size: 16px;
        font-family: inherit;
        background: rgba(255,255,255,0.05);
        color: var(--kiosk-text);
        border: 1px solid var(--kiosk-border);
        border-radius: 12px;
        outline: none;
        transition: border-color 0.2s;
      }
      .kiosk-input:focus { border-color: var(--kiosk-accent); }

      /* Buttons */
      .kiosk-btn {
        padding: 14px 32px;
        font-size: 16px;
        font-family: inherit;
        font-weight: 500;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.2s;
      }
      .kiosk-btn:disabled { opacity: 0.4; cursor: not-allowed; }

      .kiosk-btn-primary { background: var(--kiosk-accent); color: white; }
      .kiosk-btn-primary:hover:not(:disabled) { background: var(--kiosk-accent-light); }

      .kiosk-btn-ghost {
        background: transparent;
        color: var(--kiosk-text-dim);
        border: 1px solid var(--kiosk-border);
        margin-top: 8px;
      }

      .kiosk-btn-large { font-size: 20px; padding: 20px 48px; }

      /* Treatment grid */
      .kiosk-treatment-grid {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 8px 0;
        align-content: flex-start;
      }

      .kiosk-cat-header {
        width: 100%;
        font-size: 13px;
        font-weight: 600;
        color: var(--kiosk-accent-light);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
      }

      .kiosk-treatment-card {
        display: flex;
        flex-direction: column;
        gap: 4px;
        padding: 14px 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--kiosk-border);
        border-radius: 12px;
        cursor: pointer;
        text-align: left;
        color: var(--kiosk-text);
        font-family: inherit;
        transition: all 0.2s;
        width: calc(50% - 6px);
      }

      .kiosk-treatment-card:hover { background: rgba(255,255,255,0.06); }
      .kiosk-treatment-card.selected {
        border-color: var(--kiosk-accent);
        background: rgba(124,92,255,0.1);
      }

      .kiosk-treatment-name { font-size: 14px; font-weight: 500; }
      .kiosk-treatment-desc { font-size: 12px; color: var(--kiosk-text-dim); }
      .kiosk-treatment-range { font-size: 11px; color: var(--kiosk-accent-light); }

      /* Report */
      .kiosk-report-icon { font-size: 64px; margin-bottom: 16px; }
      .kiosk-report-summary {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        text-align: left;
        width: 100%;
        max-width: 400px;
      }
      .kiosk-report-summary p { margin: 8px 0; font-size: 14px; color: var(--kiosk-text-dim); }
      .kiosk-report-actions { display: flex; flex-direction: column; gap: 8px; width: 100%; max-width: 300px; }

      /* Responsive */
      @media (max-width: 600px) {
        .kiosk-treatment-card { width: 100%; }
        .kiosk-title { font-size: 22px; }
      }
    `;
    document.head.appendChild(style);
  }
}

// Bootstrap
const kiosk = new KioskApp();
kiosk.init().catch(err => console.error('KioskApp init failed:', err));
