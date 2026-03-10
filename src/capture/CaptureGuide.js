/**
 * CaptureGuide — Fullscreen guided capture UI for multi-view face photos.
 *
 * Orchestrates a step-by-step capture flow:
 *   1. Welcome screen (camera permission, instructions)
 *   2. Front capture (face-outline overlay + real-time alignment feedback)
 *   3. Left 45° capture
 *   4. Right 45° capture
 *   5. Review screen (thumbnails + quality scores + retake)
 *
 * Creates all DOM elements programmatically. Uses existing MediaPipeBridge
 * for real-time landmark detection and AlignmentValidator / QualityValidator /
 * DepthEstimator for validation.
 *
 * Usage:
 *   const guide = new CaptureGuide({
 *     photoUploader,
 *     mediaPipeBridge,
 *     onComplete: (captures) => { ... },
 *     onCancel: () => { ... },
 *   });
 *   guide.start();
 */

import { AlignmentValidator } from './AlignmentValidator.js';
import { QualityValidator } from './QualityValidator.js';
import { DepthEstimator } from './DepthEstimator.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STATES = {
  WELCOME: 'welcome',
  CAPTURE: 'capture',
  REVIEW:  'review',
  DONE:    'done',
};

const CAPTURE_SEQUENCE = ['front', 'left45', 'right45'];

const VIEW_LABELS = {
  front:   'Front',
  left45:  'Left 45°',
  right45: 'Right 45°',
};

const VIEW_INSTRUCTIONS = {
  front:   'Look straight at the camera',
  left45:  'Turn your head 45° to the left',
  right45: 'Turn your head 45° to the right',
};

// ---------------------------------------------------------------------------
// CaptureGuide
// ---------------------------------------------------------------------------

export class CaptureGuide {
  /**
   * @param {object} options
   * @param {object} options.photoUploader - PhotoUploader instance
   * @param {object} options.mediaPipeBridge - MediaPipeBridge instance (must be initialized)
   * @param {Function} [options.onComplete] - Called with captures map when user confirms
   * @param {Function} [options.onCancel] - Called when user cancels
   */
  constructor(options) {
    this._photoUploader = options.photoUploader;
    this._bridge = options.mediaPipeBridge;
    this._onComplete = options.onComplete || null;
    this._onCancel = options.onCancel || null;

    // Sub-modules
    this._alignmentValidator = new AlignmentValidator();
    this._qualityValidator = new QualityValidator();
    this._depthEstimator = new DepthEstimator();

    // State
    this._state = STATES.WELCOME;
    this._currentViewIndex = 0;
    this._captures = {}; // { front: { blob, dataUrl, quality }, ... }

    // DOM
    this._overlay = null;
    this._videoEl = null;
    this._canvasEl = null;
    this._feedbackEl = null;
    this._confidenceBar = null;
    this._captureBtn = null;
    this._stepsContainer = null;
    this._instructionEl = null;

    // Detection loop
    this._rafId = null;
    this._stream = null;

    // Style element
    this._styleEl = null;
  }

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Launch the guided capture flow.
   * Creates fullscreen overlay, requests camera, starts detection.
   */
  async start() {
    this._injectStyles();
    this._buildOverlay();
    document.body.appendChild(this._overlay);
    this._showWelcome();
  }

  /**
   * Tear down everything and release camera.
   */
  destroy() {
    this._stopDetection();
    this._stopCamera();
    if (this._overlay && this._overlay.parentNode) {
      this._overlay.parentNode.removeChild(this._overlay);
    }
    if (this._styleEl && this._styleEl.parentNode) {
      this._styleEl.parentNode.removeChild(this._styleEl);
    }
    this._overlay = null;
    this._state = STATES.DONE;
  }

  // =========================================================================
  // Welcome screen
  // =========================================================================

  _showWelcome() {
    this._state = STATES.WELCOME;
    const content = this._overlay.querySelector('.cg-content');
    content.innerHTML = '';

    const welcome = document.createElement('div');
    welcome.className = 'cg-welcome';
    welcome.innerHTML = `
      <div class="cg-welcome-icon">📸</div>
      <h2 class="cg-welcome-title">Guided Face Capture</h2>
      <p class="cg-welcome-desc">
        We'll capture your face from 3 angles for the best 3D reconstruction.
        Position yourself in good, even lighting.
      </p>
      <div class="cg-welcome-steps-preview">
        <div class="cg-preview-step">
          <div class="cg-preview-icon">👤</div>
          <span>Front</span>
        </div>
        <div class="cg-preview-arrow">→</div>
        <div class="cg-preview-step">
          <div class="cg-preview-icon">👤</div>
          <span>Left 45°</span>
        </div>
        <div class="cg-preview-arrow">→</div>
        <div class="cg-preview-step">
          <div class="cg-preview-icon">👤</div>
          <span>Right 45°</span>
        </div>
      </div>
      <button class="cg-btn cg-btn-primary cg-start-btn">Start Capture</button>
      <button class="cg-btn cg-btn-ghost cg-cancel-welcome-btn">Cancel</button>
    `;
    content.appendChild(welcome);

    welcome.querySelector('.cg-start-btn').addEventListener('click', () => {
      this._startCaptureFlow();
    });
    welcome.querySelector('.cg-cancel-welcome-btn').addEventListener('click', () => {
      this.destroy();
      this._onCancel?.();
    });
  }

  // =========================================================================
  // Capture flow
  // =========================================================================

  async _startCaptureFlow() {
    // Request camera
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (err) {
      console.error('CaptureGuide: Camera access denied:', err);
      this._showError('Camera access is required for guided capture. Please allow camera permissions and try again.');
      return;
    }

    this._currentViewIndex = 0;
    this._showCaptureView(CAPTURE_SEQUENCE[0]);
  }

  _showCaptureView(viewType) {
    this._state = STATES.CAPTURE;
    this._alignmentValidator.reset();
    const content = this._overlay.querySelector('.cg-content');
    content.innerHTML = '';

    // -- Steps indicator --
    const steps = document.createElement('div');
    steps.className = 'cg-steps';
    for (let i = 0; i < CAPTURE_SEQUENCE.length; i++) {
      const step = document.createElement('div');
      step.className = 'cg-step';
      if (i < this._currentViewIndex) step.classList.add('completed');
      if (i === this._currentViewIndex) step.classList.add('active');

      const dot = document.createElement('div');
      dot.className = 'cg-step-dot';
      dot.textContent = i < this._currentViewIndex ? '✓' : (i + 1);
      step.appendChild(dot);

      const label = document.createElement('span');
      label.className = 'cg-step-label';
      label.textContent = VIEW_LABELS[CAPTURE_SEQUENCE[i]];
      step.appendChild(label);

      if (i < CAPTURE_SEQUENCE.length - 1) {
        const line = document.createElement('div');
        line.className = 'cg-step-line';
        if (i < this._currentViewIndex) line.classList.add('completed');
        steps.appendChild(step);
        steps.appendChild(line);
      } else {
        steps.appendChild(step);
      }
    }
    this._stepsContainer = steps;

    // -- Close button --
    const closeBtn = document.createElement('button');
    closeBtn.className = 'cg-close-btn';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', () => {
      this.destroy();
      this._onCancel?.();
    });

    // -- Header --
    const header = document.createElement('div');
    header.className = 'cg-header';
    header.appendChild(steps);
    header.appendChild(closeBtn);

    // -- Viewport (video + canvas overlay) --
    const viewport = document.createElement('div');
    viewport.className = 'cg-viewport';

    this._videoEl = document.createElement('video');
    this._videoEl.className = 'cg-video';
    this._videoEl.autoplay = true;
    this._videoEl.playsInline = true;
    this._videoEl.muted = true;
    this._videoEl.srcObject = this._stream;

    this._canvasEl = document.createElement('canvas');
    this._canvasEl.className = 'cg-canvas-overlay';

    // Instruction
    this._instructionEl = document.createElement('div');
    this._instructionEl.className = 'cg-instruction';
    this._instructionEl.textContent = VIEW_INSTRUCTIONS[viewType];

    // Feedback
    this._feedbackEl = document.createElement('div');
    this._feedbackEl.className = 'cg-feedback';
    this._feedbackEl.textContent = 'Detecting face...';

    // Confidence bar
    const confidenceContainer = document.createElement('div');
    confidenceContainer.className = 'cg-confidence-container';
    this._confidenceBar = document.createElement('div');
    this._confidenceBar.className = 'cg-confidence-fill';
    this._confidenceBar.style.width = '0%';
    confidenceContainer.appendChild(this._confidenceBar);

    viewport.appendChild(this._videoEl);
    viewport.appendChild(this._canvasEl);
    viewport.appendChild(this._instructionEl);
    viewport.appendChild(this._feedbackEl);
    viewport.appendChild(confidenceContainer);

    // -- Controls --
    const controls = document.createElement('div');
    controls.className = 'cg-controls';

    this._captureBtn = document.createElement('button');
    this._captureBtn.className = 'cg-btn cg-btn-capture';
    this._captureBtn.textContent = 'Capture';
    this._captureBtn.disabled = true;
    this._captureBtn.addEventListener('click', () => this._captureFrame(viewType));

    const skipBtn = document.createElement('button');
    skipBtn.className = 'cg-btn cg-btn-ghost cg-skip-btn';
    skipBtn.textContent = viewType === 'front' ? 'Cancel' : 'Skip This View';
    skipBtn.addEventListener('click', () => {
      if (viewType === 'front') {
        this.destroy();
        this._onCancel?.();
      } else {
        this._advanceToNextView();
      }
    });

    controls.appendChild(this._captureBtn);
    controls.appendChild(skipBtn);

    // Assemble
    content.appendChild(header);
    content.appendChild(viewport);
    content.appendChild(controls);

    // Wait for video to be ready, then start detection
    this._videoEl.addEventListener('loadedmetadata', () => {
      this._canvasEl.width = this._videoEl.videoWidth;
      this._canvasEl.height = this._videoEl.videoHeight;
      this._startDetection(viewType);
    }, { once: true });
  }

  // =========================================================================
  // Detection loop
  // =========================================================================

  _startDetection(viewType) {
    // Initialize MediaPipeBridge for video mode if needed
    if (this._bridge && this._bridge.isReady) {
      this._bridge.startCamera(this._videoEl).then(() => {
        this._runDetectionLoop(viewType);
      }).catch(err => {
        console.warn('CaptureGuide: MediaPipe camera start failed, using image mode:', err);
        this._runDetectionLoop(viewType);
      });
    } else {
      this._runDetectionLoop(viewType);
    }
  }

  _runDetectionLoop(viewType) {
    const loop = async () => {
      if (this._state !== STATES.CAPTURE) return;

      let landmarks = null;
      let headPose = null;

      // Try to get landmarks from the bridge
      if (this._bridge) {
        landmarks = this._bridge.lastLandmarks;
        if (landmarks) {
          headPose = this._bridge.landmarksToHeadPose(landmarks);
        } else {
          // Fallback: detect from current video frame
          try {
            landmarks = await this._bridge.detectFromImage(this._videoEl);
            if (landmarks) {
              headPose = this._bridge.landmarksToHeadPose(landmarks);
            }
          } catch { /* ignore */ }
        }
      }

      // Validate alignment
      const frameSize = {
        width: this._videoEl.videoWidth || 640,
        height: this._videoEl.videoHeight || 480,
      };
      const validation = this._alignmentValidator.validate(viewType, landmarks, headPose, frameSize);

      // Estimate depth
      const depth = landmarks ? this._depthEstimator.estimate(landmarks) : null;

      // Update UI
      this._updateCaptureUI(validation, depth, viewType);

      // Draw guide overlay
      this._drawGuideOverlay(landmarks, validation, viewType);

      // Auto-capture when ready
      if (validation.readyToCapture) {
        this._captureFrame(viewType);
        return;
      }

      this._rafId = requestAnimationFrame(loop);
    };

    this._rafId = requestAnimationFrame(loop);
  }

  _stopDetection() {
    if (this._rafId) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  _stopCamera() {
    if (this._bridge) {
      try { this._bridge.stopCamera(); } catch { /* ignore */ }
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
  }

  // =========================================================================
  // Capture frame
  // =========================================================================

  async _captureFrame(viewType) {
    this._stopDetection();

    // Create un-mirrored capture from video
    const vw = this._videoEl.videoWidth;
    const vh = this._videoEl.videoHeight;
    const canvas = document.createElement('canvas');
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');

    // Flip horizontally to un-mirror (webcam is mirrored in CSS)
    ctx.translate(vw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(this._videoEl, 0, 0, vw, vh);

    // Analyze quality
    const quality = await this._qualityValidator.analyze(canvas);

    // Convert to blob
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);

    // Store capture
    this._captures[viewType] = {
      blob,
      dataUrl,
      quality,
      width: vw,
      height: vh,
    };

    // If quality is low, show warning but still allow proceeding
    if (!quality.acceptable) {
      const proceed = await this._showQualityWarning(viewType, quality);
      if (!proceed) {
        // Retake
        this._alignmentValidator.reset();
        this._startDetection(viewType);
        return;
      }
    }

    this._advanceToNextView();
  }

  _advanceToNextView() {
    this._currentViewIndex++;
    if (this._currentViewIndex < CAPTURE_SEQUENCE.length) {
      this._showCaptureView(CAPTURE_SEQUENCE[this._currentViewIndex]);
    } else {
      this._showReview();
    }
  }

  // =========================================================================
  // Quality warning
  // =========================================================================

  _showQualityWarning(viewType, quality) {
    return new Promise(resolve => {
      const modal = document.createElement('div');
      modal.className = 'cg-quality-modal';
      modal.innerHTML = `
        <div class="cg-quality-card">
          <div class="cg-quality-score" style="color: ${quality.overallScore < 30 ? 'var(--danger)' : 'var(--warning)'}">
            Quality: ${quality.overallScore}/100
          </div>
          <div class="cg-quality-feedback">
            ${quality.feedback.map(f => `<p>⚠ ${f}</p>`).join('')}
          </div>
          <div class="cg-quality-actions">
            <button class="cg-btn cg-btn-primary cg-retake-btn">Retake</button>
            <button class="cg-btn cg-btn-ghost cg-use-btn">Use Anyway</button>
          </div>
        </div>
      `;

      modal.querySelector('.cg-retake-btn').addEventListener('click', () => {
        modal.remove();
        resolve(false);
      });
      modal.querySelector('.cg-use-btn').addEventListener('click', () => {
        modal.remove();
        resolve(true);
      });

      this._overlay.querySelector('.cg-content').appendChild(modal);
    });
  }

  // =========================================================================
  // Review screen
  // =========================================================================

  _showReview() {
    this._state = STATES.REVIEW;
    this._stopDetection();
    this._stopCamera();

    const content = this._overlay.querySelector('.cg-content');
    content.innerHTML = '';

    const review = document.createElement('div');
    review.className = 'cg-review';

    // Title
    const title = document.createElement('h2');
    title.className = 'cg-review-title';
    title.textContent = 'Review Captures';
    review.appendChild(title);

    // Thumbnails grid
    const grid = document.createElement('div');
    grid.className = 'cg-review-grid';

    for (const viewType of CAPTURE_SEQUENCE) {
      const capture = this._captures[viewType];
      const card = document.createElement('div');
      card.className = 'cg-review-card';

      if (capture) {
        const img = document.createElement('img');
        img.className = 'cg-review-img';
        img.src = capture.dataUrl;

        const score = document.createElement('div');
        score.className = 'cg-review-score';
        const scoreVal = capture.quality.overallScore;
        score.style.color = scoreVal >= 70 ? 'var(--success)' : scoreVal >= 40 ? 'var(--warning)' : 'var(--danger)';
        score.textContent = `${scoreVal}/100`;

        const label = document.createElement('div');
        label.className = 'cg-review-label';
        label.textContent = VIEW_LABELS[viewType];

        const retakeBtn = document.createElement('button');
        retakeBtn.className = 'cg-btn cg-btn-sm';
        retakeBtn.textContent = 'Retake';
        retakeBtn.addEventListener('click', async () => {
          // Restart camera and redo this view
          try {
            this._stream = await navigator.mediaDevices.getUserMedia({
              video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
              audio: false,
            });
          } catch { return; }
          this._currentViewIndex = CAPTURE_SEQUENCE.indexOf(viewType);
          this._showCaptureView(viewType);
        });

        card.appendChild(img);
        card.appendChild(label);
        card.appendChild(score);
        card.appendChild(retakeBtn);
      } else {
        card.innerHTML = `
          <div class="cg-review-empty">Skipped</div>
          <div class="cg-review-label">${VIEW_LABELS[viewType]}</div>
        `;
      }

      grid.appendChild(card);
    }
    review.appendChild(grid);

    // Action buttons
    const actions = document.createElement('div');
    actions.className = 'cg-review-actions';

    const useBtn = document.createElement('button');
    useBtn.className = 'cg-btn cg-btn-primary';
    useBtn.textContent = 'Use These Photos';
    useBtn.addEventListener('click', () => {
      this.destroy();
      this._onComplete?.(this._captures);
    });

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'cg-btn cg-btn-ghost';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.addEventListener('click', () => {
      this.destroy();
      this._onCancel?.();
    });

    actions.appendChild(useBtn);
    actions.appendChild(cancelBtn);
    review.appendChild(actions);

    content.appendChild(review);
  }

  // =========================================================================
  // UI updates
  // =========================================================================

  _updateCaptureUI(validation, depth, viewType) {
    if (!this._feedbackEl || !this._confidenceBar || !this._captureBtn) return;

    // Feedback text
    const messages = [...validation.feedback];
    if (depth?.feedback && !validation.isAligned) {
      messages.push(depth.feedback);
    }
    this._feedbackEl.textContent = messages[0] || 'Align your face...';

    // Feedback color
    if (validation.readyToCapture) {
      this._feedbackEl.style.color = 'var(--success)';
    } else if (validation.isAligned) {
      this._feedbackEl.style.color = 'var(--warning)';
    } else {
      this._feedbackEl.style.color = 'var(--text-secondary)';
    }

    // Confidence bar
    const pct = Math.round(validation.confidence * 100);
    this._confidenceBar.style.width = `${pct}%`;
    if (pct >= 80) {
      this._confidenceBar.style.background = 'var(--success)';
    } else if (pct >= 50) {
      this._confidenceBar.style.background = 'var(--warning)';
    } else {
      this._confidenceBar.style.background = 'var(--danger)';
    }

    // Capture button
    this._captureBtn.disabled = !validation.checks.pose;
  }

  // =========================================================================
  // Canvas overlay drawing
  // =========================================================================

  _drawGuideOverlay(landmarks, validation, viewType) {
    if (!this._canvasEl) return;
    const ctx = this._canvasEl.getContext('2d');
    const w = this._canvasEl.width;
    const h = this._canvasEl.height;
    ctx.clearRect(0, 0, w, h);

    // Draw face outline oval
    const cx = w / 2;
    const cy = h * 0.45;
    const rx = w * 0.18;
    const ry = h * 0.32;

    ctx.save();

    // Rotate oval for angled views
    if (viewType === 'left45') {
      ctx.translate(cx, cy);
      ctx.scale(0.85, 1);
      ctx.translate(-cx, cy * 0.02);
    } else if (viewType === 'right45') {
      ctx.translate(cx, cy);
      ctx.scale(0.85, 1);
      ctx.translate(-cx, cy * 0.02);
    }

    // Oval color based on alignment
    if (validation.readyToCapture) {
      ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)'; // success
      ctx.lineWidth = 4;
    } else if (validation.isAligned) {
      ctx.strokeStyle = 'rgba(251, 191, 36, 0.7)'; // warning
      ctx.lineWidth = 3;
    } else {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 2;
    }

    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Draw a subtle semi-transparent mask outside the oval
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, w, h);

    ctx.restore();

    // Draw landmark dots if we have them (subtle)
    if (landmarks && validation.confidence > 0.3) {
      ctx.fillStyle = 'rgba(124, 92, 255, 0.4)';
      // Draw just a few key landmarks
      const keyPoints = [1, 33, 263, 152, 10]; // nose, eyes, chin, forehead
      for (const idx of keyPoints) {
        if (idx < landmarks.length) {
          // Mirror x for display (video is CSS-mirrored)
          const lx = (1 - landmarks[idx].x) * w;
          const ly = landmarks[idx].y * h;
          ctx.beginPath();
          ctx.arc(lx, ly, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  // =========================================================================
  // Error display
  // =========================================================================

  _showError(message) {
    const content = this._overlay.querySelector('.cg-content');
    content.innerHTML = `
      <div class="cg-welcome">
        <div class="cg-welcome-icon">⚠️</div>
        <h2 class="cg-welcome-title">Camera Required</h2>
        <p class="cg-welcome-desc">${message}</p>
        <button class="cg-btn cg-btn-primary cg-retry-btn">Try Again</button>
        <button class="cg-btn cg-btn-ghost cg-cancel-err-btn">Cancel</button>
      </div>
    `;
    content.querySelector('.cg-retry-btn').addEventListener('click', () => this._startCaptureFlow());
    content.querySelector('.cg-cancel-err-btn').addEventListener('click', () => {
      this.destroy();
      this._onCancel?.();
    });
  }

  // =========================================================================
  // DOM construction
  // =========================================================================

  _buildOverlay() {
    this._overlay = document.createElement('div');
    this._overlay.className = 'cg-overlay';

    const content = document.createElement('div');
    content.className = 'cg-content';
    this._overlay.appendChild(content);
  }

  // =========================================================================
  // Injected styles
  // =========================================================================

  _injectStyles() {
    if (document.getElementById('capture-guide-styles')) return;

    this._styleEl = document.createElement('style');
    this._styleEl.id = 'capture-guide-styles';
    this._styleEl.textContent = `
      /* ============ CaptureGuide Overlay ============ */
      .cg-overlay {
        position: fixed;
        inset: 0;
        z-index: 10000;
        background: var(--bg-primary, #08081a);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: var(--font, 'Inter', sans-serif);
        color: var(--text-primary, #eaeaf4);
      }

      .cg-content {
        width: 100%;
        max-width: 700px;
        height: 100%;
        display: flex;
        flex-direction: column;
        padding: 16px;
        position: relative;
      }

      /* ============ Welcome ============ */
      .cg-welcome {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 16px;
        padding: 32px;
      }

      .cg-welcome-icon { font-size: 64px; }
      .cg-welcome-title { font-size: 24px; font-weight: 600; }
      .cg-welcome-desc {
        font-size: 15px;
        color: var(--text-secondary, #9898c0);
        max-width: 400px;
        line-height: 1.5;
      }

      .cg-welcome-steps-preview {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 24px 0;
      }

      .cg-preview-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        font-size: 13px;
        color: var(--text-secondary);
      }

      .cg-preview-icon { font-size: 28px; opacity: 0.6; }
      .cg-preview-arrow { color: var(--text-muted); font-size: 18px; }

      /* ============ Buttons ============ */
      .cg-btn {
        padding: 10px 24px;
        border-radius: var(--radius-sm, 8px);
        border: none;
        font-family: inherit;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
      }

      .cg-btn:disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }

      .cg-btn-primary {
        background: var(--accent, #7c5cff);
        color: white;
      }

      .cg-btn-primary:hover:not(:disabled) {
        background: var(--accent-light, #a08cff);
      }

      .cg-btn-ghost {
        background: transparent;
        color: var(--text-secondary);
        border: 1px solid var(--border-light, rgba(255,255,255,0.1));
      }

      .cg-btn-ghost:hover { background: var(--bg-hover); }

      .cg-btn-capture {
        background: var(--success, #4ade80);
        color: #000;
        font-size: 16px;
        padding: 14px 40px;
        border-radius: 50px;
        font-weight: 600;
      }

      .cg-btn-capture:disabled {
        background: var(--bg-card);
        color: var(--text-muted);
      }

      .cg-btn-sm {
        padding: 6px 14px;
        font-size: 12px;
        background: var(--bg-hover);
        color: var(--text-secondary);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-xs);
      }

      .cg-btn-sm:hover { background: var(--accent-dim); color: white; }

      /* ============ Header / Steps ============ */
      .cg-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 12px;
        flex-shrink: 0;
      }

      .cg-steps {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .cg-step {
        display: flex;
        align-items: center;
        gap: 6px;
      }

      .cg-step-dot {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        background: var(--bg-card);
        color: var(--text-muted);
        border: 2px solid var(--border-light);
        transition: all 0.3s ease;
      }

      .cg-step.active .cg-step-dot {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
        box-shadow: 0 0 12px var(--accent-glow);
      }

      .cg-step.completed .cg-step-dot {
        background: var(--success);
        color: #000;
        border-color: var(--success);
      }

      .cg-step-label {
        font-size: 12px;
        color: var(--text-muted);
      }

      .cg-step.active .cg-step-label { color: var(--text-primary); }
      .cg-step.completed .cg-step-label { color: var(--success); }

      .cg-step-line {
        width: 24px;
        height: 2px;
        background: var(--border-light);
      }

      .cg-step-line.completed { background: var(--success); }

      .cg-close-btn {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: none;
        background: var(--bg-card);
        color: var(--text-secondary);
        font-size: 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .cg-close-btn:hover { background: var(--danger); color: white; }

      /* ============ Viewport ============ */
      .cg-viewport {
        flex: 1;
        position: relative;
        border-radius: var(--radius);
        overflow: hidden;
        background: #000;
        min-height: 0;
      }

      .cg-video {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scaleX(-1); /* mirror */
      }

      .cg-canvas-overlay {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
      }

      .cg-instruction {
        position: absolute;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(8px);
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        white-space: nowrap;
      }

      .cg-feedback {
        position: absolute;
        bottom: 60px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 15px;
        font-weight: 500;
        text-align: center;
        text-shadow: 0 2px 8px rgba(0,0,0,0.6);
        transition: color 0.3s ease;
      }

      .cg-confidence-container {
        position: absolute;
        bottom: 40px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 4px;
        background: rgba(255,255,255,0.1);
        border-radius: 2px;
        overflow: hidden;
      }

      .cg-confidence-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.15s ease, background 0.3s ease;
      }

      /* ============ Controls ============ */
      .cg-controls {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        padding: 16px 0 0;
        flex-shrink: 0;
      }

      /* ============ Quality Warning Modal ============ */
      .cg-quality-modal {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0,0,0,0.7);
        z-index: 10;
      }

      .cg-quality-card {
        background: var(--bg-panel);
        border-radius: var(--radius);
        padding: 24px;
        max-width: 360px;
        text-align: center;
      }

      .cg-quality-score { font-size: 20px; font-weight: 600; margin-bottom: 12px; }
      .cg-quality-feedback { font-size: 13px; color: var(--text-secondary); margin-bottom: 20px; }
      .cg-quality-feedback p { margin: 4px 0; }
      .cg-quality-actions { display: flex; gap: 12px; justify-content: center; }

      /* ============ Review ============ */
      .cg-review {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 24px;
        padding: 32px 0;
      }

      .cg-review-title { font-size: 22px; font-weight: 600; }

      .cg-review-grid {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        justify-content: center;
      }

      .cg-review-card {
        width: 180px;
        background: var(--bg-card);
        border-radius: var(--radius);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-bottom: 12px;
        gap: 8px;
      }

      .cg-review-img {
        width: 100%;
        height: 140px;
        object-fit: cover;
      }

      .cg-review-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--text-secondary);
      }

      .cg-review-score { font-size: 16px; font-weight: 600; }

      .cg-review-empty {
        width: 100%;
        height: 140px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        font-size: 14px;
        background: var(--bg-secondary);
      }

      .cg-review-actions {
        display: flex;
        gap: 12px;
        margin-top: 8px;
      }

      /* ============ Responsive ============ */
      @media (max-width: 600px) {
        .cg-content { padding: 8px; }
        .cg-review-grid { flex-direction: column; align-items: center; }
        .cg-review-card { width: 100%; max-width: 280px; }
        .cg-steps { gap: 4px; }
        .cg-step-label { display: none; }
        .cg-step-line { width: 16px; }
      }
    `;
    document.head.appendChild(this._styleEl);
  }
}
