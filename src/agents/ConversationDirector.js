/**
 * ConversationDirector.js
 * Central orchestrator that routes user input to the correct specialist agents,
 * manages consultation phases, handles undo/redo/reset/save, and coordinates
 * the NLU -> morph engine -> renderer pipeline.
 *
 * Consultation phases:
 *  1. INTAKE    — greeting, understand patient goals
 *  2. CAPTURE   — photo upload and 3D reconstruction
 *  3. EXPLORE   — interactive modifications via chat/voice
 *  4. COMPARE   — before/after review
 *  5. REPORT    — generate clinical summary
 */

import { NLUAgent, CLINICAL_ZONES } from './NLUAgent.js';
import { MedicalAdvisorAgent } from './MedicalAdvisorAgent.js';
import { ExpressionAgent } from './ExpressionAgent.js';
import { ComparisonAgent } from './ComparisonAgent.js';
import { VoiceAgent, SPEAKER } from './VoiceAgent.js';

// ---------------------------------------------------------------------------
// Phase enum
// ---------------------------------------------------------------------------
export const PHASES = {
  INTAKE:  'intake',
  CAPTURE: 'capture',
  EXPLORE: 'explore',
  COMPARE: 'compare',
  REPORT:  'report',
};

// ---------------------------------------------------------------------------
// Intent categories for routing (ordered by priority)
// ---------------------------------------------------------------------------
const INTENT_PATTERNS = [
  { intent: 'greeting',      pattern: /^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings)\b/i },
  { intent: 'farewell',      pattern: /^(bye|goodbye|thanks|thank you|that'?s all|done|finish)\b/i },
  { intent: 'help',          pattern: /(?:^help$|what can you do|how does this work|instructions|guide me|capabilities)/i },
  { intent: 'undo',          pattern: /(?:^undo$|undo\s+that|go\s+back|reverse|revert|take\s+that\s+back)/i },
  { intent: 'redo',          pattern: /(?:^redo$|redo\s+that|put\s+it\s+back)/i },
  { intent: 'reset',         pattern: /(?:^reset$|^reset\s+all$|start\s+over|clear\s+all|remove\s+everything)/i },
  { intent: 'save',          pattern: /(?:save|snapshot|bookmark|remember\s+this|keep\s+this|save\s+version)/i },
  { intent: 'report',        pattern: /(?:report|summary|generate\s+report|consultation\s+report|treatment\s+plan|procedure\s+list|what\s+do\s+i\s+need|what\s+procedures)/i },
  { intent: 'compare',       pattern: /(?:compare|before\s*\/?\s*after|difference|show\s+changes|what\s+changed|original)/i },
  { intent: 'expression',    pattern: /(?:smile|frown|surprise|laugh|wink|talk|speak|think|disgust|resting|concerned|expression|animate|show\s+me\s+(?:a\s+)?(?:smile|frown|surprise|laugh|wink|talking|thinking|resting|concerned|disgusted))/i },
  { intent: 'region_select', pattern: /(?:focus\s+on|select|zoom\s+to|show\s+me\s+the|highlight|look\s+at)\s+(?:my\s+)?(?:the\s+)?(forehead|eyes?|nose|cheeks?|lips?|jaw|chin|brows?|eyelids?|nasolabial|marionette|temples?|neck)/i },
  { intent: 'medical',       pattern: /(?:procedure|treatment|surgery|injection|recovery|cost|how\s+long|risk|side\s+effect|recovery\s+time|permanent|temporary|non.?surgical|minimally\s+invasive|what\s+would\s+it\s+take)/i },
  { intent: 'modify_face',   pattern: /(?:make|change|adjust|modify|increase|decrease|reduce|add|remove|thinner|wider|bigger|smaller|lift|lower|smooth|filler|botox|rhinoplasty|augment|contour|slim|sculpt|define|enhance|soften|sharpen|narrow|broaden|lengthen|shorten|plump|deflate|inflate|puff|tighten|loosen|reshape|refine|fuller|project|flatten|raise|drop)/i },
];

// ---------------------------------------------------------------------------
// Response templates
// ---------------------------------------------------------------------------
const RESPONSES = {
  greeting: [
    "Welcome to your facial design consultation! I'm here to help you explore aesthetic enhancements. You can upload your photos on the left panel, or describe what changes you'd like to see.",
    "Hello! I'm your AI consultation assistant. Let's work together to visualize your ideal facial enhancements. Start by uploading a front-facing photo, or tell me what you'd like to change.",
    "Hi there! Ready to explore your facial design options? Upload your photos to get started, or simply describe what you'd like to adjust.",
  ],
  farewell: [
    "Thank you for your consultation! You can generate a full report using the Report button. Feel free to come back anytime to continue exploring.",
    "It was great working with you! Remember, you can save your current design and generate a consultation report. See you next time!",
  ],
  help: `Here's what I can help you with:

  Face Modifications -- Describe changes naturally, like "make my nose thinner" or "add subtle lip filler to the right side"
  Expressions -- See how changes look with different expressions: "show me smiling" or "how does it look surprised"
  Medical Info -- Ask about procedures: "what treatment would achieve this?" or "what's the recovery time?"
  Undo/Redo -- Say "undo" to revert changes or "redo" to reapply
  Compare -- "Show me before and after" to see your original vs. modified face
  Save -- "Save this version" to bookmark the current state
  Report -- "Generate a report" for a full consultation summary

You can also use the region controls on the left panel for precise adjustments.`,

  capture_prompt: 'Great! Now upload a front-facing photo using the panel on the left. For best results, also add 45-degree angle photos.',
  no_model: 'Please upload a photo first so I can create your 3D model. You can use the upload panel on the left.',
  modification_applied: [
    'Done! I\'ve applied those changes. How does that look?',
    'Changes applied. Take a look and let me know if you\'d like any adjustments.',
    'There you go! Feel free to ask for more changes or say "undo" to revert.',
    'Applied! Rotate the model to see from different angles. Want to adjust anything else?',
  ],
  undo_done: 'Reverted the last change. Say "redo" to reapply it.',
  redo_done: 'Change reapplied. Continue exploring or ask for more modifications.',
  nothing_to_undo: 'Nothing to undo -- no changes have been made yet.',
  nothing_to_redo: 'Nothing to redo.',
  reset_done: 'All changes have been cleared. The model is back to its original state.',
  save_done: 'Version saved! You can find it in the Versions panel.',
  low_confidence: 'I\'m not quite sure what you\'d like me to change. Could you be more specific? For example: "make my nose bridge narrower" or "add volume to my upper lip".',
};

// ---------------------------------------------------------------------------
// ConversationDirector class
// ---------------------------------------------------------------------------
export class ConversationDirector {
  /**
   * @param {object} [morphEngine]  Reference to the 3D morph engine (or null).
   * @param {object} [agents]       Optional pre-constructed agent instances.
   *   If omitted, agents are created internally.
   *   Supported keys: nlu, medical, expression, comparison, voice
   */
  constructor(morphEngine = null, agents = {}) {
    // Morph engine reference
    this.morphEngine = morphEngine;

    // Sub-agents (use provided instances or create new ones)
    this.nlu         = agents.nlu         || new NLUAgent();
    this.medical     = agents.medical     || new MedicalAdvisorAgent();
    this.expression  = agents.expression  || new ExpressionAgent();
    this.comparison  = agents.comparison  || new ComparisonAgent();
    this.voice       = agents.voice       || new VoiceAgent();

    // Consultation state
    this.phase = PHASES.INTAKE;
    this.hasModel = false;
    this.changeCount = 0;

    // Conversation history: { role: 'user'|'assistant', text: string, timestamp: number, intent?: string }
    this.history = [];

    // Undo / redo stacks
    // Each entry: { regions: Record<string, displacement>, timestamp: number }
    this._undoStack = [];
    this._redoStack = [];

    // Saved versions
    // Each entry: { name: string, morphState: object, timestamp: number }
    this._savedVersions = [];

    // Current aggregate morph state: region -> { displacement, inflate }
    this._currentMorphState = {};

    // Patient info (optional, populated during consultation)
    this.patientInfo = {};

    // Callbacks set by the UI layer
    this._callbacks = {
      onApplyChanges:    null,  // (regions) => void
      onUndo:            null,  // () => void
      onRedo:            null,  // () => void
      onReset:           null,  // () => void
      onSave:            null,  // (version) => void
      onCompare:         null,  // (comparisonData) => void
      onExpression:      null,  // (expressionData) => void
      onHighlightRegion: null,  // (regionName) => void
      onGenerateReport:  null,  // (report) => void
      onPhaseChange:     null,  // (phase) => void
    };
  }

  // =========================================================================
  // CONFIGURATION
  // =========================================================================

  /**
   * Register UI callbacks.
   * @param {object} callbacks  Keyed by callback name.
   */
  setCallbacks(callbacks) {
    for (const [key, fn] of Object.entries(callbacks)) {
      if (key in this._callbacks) {
        this._callbacks[key] = fn;
      }
    }
  }

  /**
   * Update the morph engine reference.
   * @param {object} engine
   */
  setMorphEngine(engine) {
    this.morphEngine = engine;
  }

  /**
   * Notify that a 3D model has been loaded.
   * @param {boolean} [loaded=true]
   */
  setModelLoaded(loaded = true) {
    this.hasModel = loaded;
    if (loaded && this.phase === PHASES.CAPTURE) {
      this._setPhase(PHASES.EXPLORE);
    }
  }

  /**
   * Set patient info for report generation.
   * @param {{ name?: string, age?: number, gender?: string, concerns?: string, notes?: string }} info
   */
  setPatientInfo(info) {
    this.patientInfo = { ...this.patientInfo, ...info };
  }

  // =========================================================================
  // MAIN MESSAGE PROCESSING
  // =========================================================================

  /**
   * Process a user text input and return a structured response.
   * This is the primary entry point for the conversation pipeline.
   *
   * @param {string} text  User input.
   * @returns {Promise<{
   *   response: string,
   *   actions: Array<{type: string, data: any}>,
   *   intent: string,
   *   morphResult?: object
   * }>}
   */
  async processMessage(text) {
    const trimmed = text.trim();
    if (!trimmed) return { response: '', actions: [], intent: 'empty' };

    // Record user message
    this._addHistory('user', trimmed);

    // Detect intent
    const intent = this._detectIntent(trimmed);

    let response = '';
    const actions = [];
    let morphResult = undefined;

    switch (intent) {
      case 'greeting':
        response = this._pick(RESPONSES.greeting);
        if (this.phase === PHASES.INTAKE) {
          // Stay in intake
        }
        break;

      case 'farewell':
        response = this._pick(RESPONSES.farewell);
        break;

      case 'help':
        response = RESPONSES.help;
        break;

      case 'undo':
        response = this._handleUndo(actions);
        break;

      case 'redo':
        response = this._handleRedo(actions);
        break;

      case 'reset':
        response = this._handleReset(actions);
        break;

      case 'save':
        response = this._handleSave(actions);
        break;

      case 'compare':
        response = this._handleCompare(actions);
        break;

      case 'report':
        response = this._handleReport(actions);
        break;

      case 'expression':
        response = this._handleExpression(trimmed, actions);
        break;

      case 'region_select':
        response = this._handleRegionSelect(trimmed, actions);
        break;

      case 'medical':
        response = this._handleMedical(trimmed, actions);
        break;

      case 'modify_face':
        ({ response, morphResult } = this._handleModification(trimmed, actions));
        break;

      default:
        // Fallback: try NLU parse (maybe it's a modification phrased unusually)
        ({ response, morphResult } = this._handleModification(trimmed, actions));
        if (actions.length === 0) {
          response = RESPONSES.low_confidence;
        }
        break;
    }

    // Record assistant response
    this._addHistory('assistant', response, intent);

    return { response, actions, intent, morphResult };
  }

  // =========================================================================
  // INTENT HANDLERS
  // =========================================================================

  /** @private */
  _handleModification(text, actions) {
    if (!this.hasModel) {
      return { response: RESPONSES.no_model, morphResult: undefined };
    }

    const parsed = this.nlu.parse(text);

    if (!parsed || parsed.confidence < 0.2 || Object.keys(parsed.regions).length === 0) {
      return { response: '', morphResult: undefined };
    }

    // Handle undo/reset action returned by NLU
    if (parsed.action === 'undo') {
      return { response: this._handleUndo(actions), morphResult: parsed };
    }
    if (parsed.action === 'reset') {
      return { response: this._handleReset(actions), morphResult: parsed };
    }

    // Push current state to undo stack before applying changes
    this._undoStack.push(this._cloneMorphState());
    this._redoStack = []; // Clear redo on new change

    // Merge parsed regions into current morph state
    this._applyRegionsToState(parsed.regions);

    // Notify UI / morph engine
    if (this._callbacks.onApplyChanges) {
      this._callbacks.onApplyChanges(parsed.regions);
    }

    this.changeCount++;
    this._setPhase(PHASES.EXPLORE);

    actions.push({
      type: 'modify',
      data: {
        regions: parsed.regions,
        explanation: parsed.explanation,
        confidence: parsed.confidence,
      },
    });

    // Build response
    let response = parsed.explanation;
    if (!response) {
      response = this._pick(RESPONSES.modification_applied);
    }

    // Append medical context for the affected regions
    const regionNames = Object.keys(parsed.regions);
    const medicalHint = this._getQuickMedicalHint(regionNames);
    if (medicalHint) {
      response += '\n\n' + medicalHint;
    }

    return { response, morphResult: parsed };
  }

  /** @private */
  _handleExpression(text, actions) {
    const result = this.expression.animate(text);
    if (!result) {
      return "I couldn't detect which expression you'd like. Try: smile, frown, surprise, laugh, thinking, or talking.";
    }

    if (this._callbacks.onExpression) {
      this._callbacks.onExpression(result);
    }

    actions.push({ type: 'expression', data: result });

    const tips = {
      smile:         'Notice how the cheeks lift -- this shows how your enhancements look with natural movement.',
      big_smile:     'A big smile activates the entire mid-face. Great for evaluating cheek and lip work dynamically.',
      subtle_smile:  'A subtle smile is the most common everyday expression -- ideal for checking natural-looking results.',
      frown:         'Frowning reveals forehead lines and brow positions -- useful for botox or brow lift planning.',
      surprise:      'Surprise shows forehead dynamics and brow elevation -- key areas for wrinkle treatment.',
      thinking:      'The thinking expression creates subtle asymmetry -- useful for checking natural-looking results.',
      talking:       'Animating speech shows how lip modifications look during conversation.',
      resting:       'The resting face is what others see most often -- check that modifications look balanced here.',
      laugh:         'Laughing is the ultimate expression test -- if modifications look good here, they look great at rest.',
      concerned:     'The concerned expression engages brow and forehead muscles -- useful for wrinkle assessment.',
      disgust:       'This expression tests nose and upper lip dynamics.',
      wink_left:     'Winking tests eye and cheek muscle interaction on the left side.',
      wink_right:    'Winking tests eye and cheek muscle interaction on the right side.',
    };

    const tip = tips[result.expression] || '';
    return `Showing ${result.name} expression. ${tip}`;
  }

  /** @private */
  _handleMedical(text, actions) {
    // Determine which regions are currently modified
    const modifiedRegions = Object.keys(this._currentMorphState).filter(r => {
      const s = this._currentMorphState[r];
      if (!s) return false;
      const d = s.displacement || { x: 0, y: 0, z: 0 };
      return Math.abs(d.x) + Math.abs(d.y) + Math.abs(d.z) + Math.abs(s.inflate || 0) > 0.001;
    });

    if (modifiedRegions.length > 0) {
      // Analyze current changes
      const analysis = this.medical.analyze(this._currentMorphState, modifiedRegions);
      actions.push({ type: 'medical', data: analysis });

      if (analysis.recommendations.length > 0) {
        let response = 'Based on your current modifications, here are the most relevant procedures:\n\n';
        const topRecs = analysis.recommendations.slice(0, 4);
        for (const rec of topRecs) {
          response += `  ${rec.procedure.name}`;
          response += ` -- Feasibility: ${(rec.feasibility * 100).toFixed(0)}%`;
          response += `, Cost: $${rec.procedure.costRange.min}-$${rec.procedure.costRange.max}`;
          response += `, Recovery: ${rec.procedure.recoveryTime}\n`;
        }

        if (analysis.warnings.length > 0) {
          response += '\nNotes:\n';
          for (const warn of analysis.warnings) {
            response += `  - ${warn}\n`;
          }
        }

        response += '\nThis is informational only -- always consult with a qualified medical professional.';
        return response;
      }
    }

    // No current modifications -- try to answer based on text keywords
    // Search procedures by keyword
    const searchResults = this.medical.searchProcedures(text);
    if (searchResults.length > 0) {
      actions.push({ type: 'medical', data: { searchResults } });

      let response = 'Here are procedures related to your question:\n\n';
      for (const proc of searchResults.slice(0, 3)) {
        response += `  ${proc.name} (${proc.type})\n`;
        response += `    ${proc.description}\n`;
        response += `    Recovery: ${proc.recoveryTime}\n`;
        response += `    Permanence: ${proc.permanence}\n`;
        response += `    Cost: $${proc.costRange.min}-$${proc.costRange.max}\n\n`;
      }
      response += 'This is informational only -- always consult with a qualified medical professional.';
      return response;
    }

    return 'I can provide information about facial aesthetic procedures. Try asking about specific procedures like "What is cheek filler?" or make some modifications first and ask "What procedures would achieve this?"';
  }

  /** @private */
  _handleRegionSelect(text, actions) {
    const match = text.match(/(?:forehead|eyes?|nose|cheeks?|lips?|jaw|chin|brows?|eyelids?|nasolabial|marionette|temples?|neck)/i);
    if (match) {
      const region = match[0].toLowerCase();
      if (this._callbacks.onHighlightRegion) {
        this._callbacks.onHighlightRegion(region);
      }
      actions.push({ type: 'highlight_region', data: region });

      // Also show applicable procedures for that region
      const regionNames = this._mapFriendlyToZones(region);
      if (regionNames.length > 0) {
        const procInfo = this.medical.getProcedureByRegion(regionNames[0]);
        const procNames = procInfo.procedures.slice(0, 3).map(p => p.name).join(', ');
        return `Focusing on the ${region} area. ${procNames ? 'Common procedures for this area: ' + procNames + '.' : ''} Describe what changes you'd like or use the sliders.`;
      }

      return `Focusing on the ${region} area. Describe what changes you'd like or use the sliders for precise adjustments.`;
    }
    return 'Which region would you like to focus on? Options: forehead, eyes, nose, cheeks, lips, jaw, chin, temples, neck.';
  }

  /** @private */
  _handleCompare(actions) {
    if (!this.hasModel) return RESPONSES.no_model;

    const modifiedRegions = Object.keys(this._currentMorphState);
    if (modifiedRegions.length === 0) {
      return 'No changes have been made yet. Make some modifications first, then ask to compare.';
    }

    const comparisonData = this.comparison.generateComparison(this._currentMorphState, modifiedRegions);

    if (this._callbacks.onCompare) {
      this._callbacks.onCompare(comparisonData);
    }

    this._setPhase(PHASES.COMPARE);
    actions.push({ type: 'compare', data: comparisonData });

    const { summary } = comparisonData;
    return `Before/After comparison: ${summary.totalRegionsModified} region(s) modified. ` +
           `Most significant change: ${summary.maxMagnitudeRegion ? summary.maxMagnitudeRegion.replace(/_/g, ' ') : 'none'} ` +
           `(${(summary.maxMagnitude * 100).toFixed(0)}%). ` +
           `Average intensity: ${(summary.averageMagnitude * 100).toFixed(0)}%.`;
  }

  /** @private */
  _handleReport(actions) {
    if (!this.hasModel) return RESPONSES.no_model;

    const modifiedRegions = Object.keys(this._currentMorphState);
    const report = this.comparison.generateReport(
      this._currentMorphState,
      modifiedRegions,
      this.patientInfo
    );

    if (this._callbacks.onGenerateReport) {
      this._callbacks.onGenerateReport(report);
    }

    this._setPhase(PHASES.REPORT);
    actions.push({ type: 'report', data: report });

    return `Consultation report generated (ID: ${report.metadata.reportId}). ` +
           `${report.metadata.regionsModified} regions analyzed, ` +
           `${report.metadata.proceduresRecommended} procedures recommended. ` +
           `You can view the full report in the report panel.`;
  }

  /** @private */
  _handleUndo(actions) {
    if (this._undoStack.length === 0) {
      return RESPONSES.nothing_to_undo;
    }

    // Save current state to redo stack
    this._redoStack.push(this._cloneMorphState());

    // Pop previous state
    const previousState = this._undoStack.pop();
    this._currentMorphState = previousState;
    this.changeCount = Math.max(0, this.changeCount - 1);

    // Notify UI
    if (this._callbacks.onUndo) {
      this._callbacks.onUndo();
    }
    if (this._callbacks.onApplyChanges) {
      // Apply the full previous state (UI should interpret this as a complete state replacement)
      this._callbacks.onApplyChanges(previousState);
    }

    actions.push({ type: 'undo' });
    return RESPONSES.undo_done;
  }

  /** @private */
  _handleRedo(actions) {
    if (this._redoStack.length === 0) {
      return RESPONSES.nothing_to_redo;
    }

    // Save current state to undo stack
    this._undoStack.push(this._cloneMorphState());

    // Pop redo state
    const redoState = this._redoStack.pop();
    this._currentMorphState = redoState;
    this.changeCount++;

    // Notify UI
    if (this._callbacks.onRedo) {
      this._callbacks.onRedo();
    }
    if (this._callbacks.onApplyChanges) {
      this._callbacks.onApplyChanges(redoState);
    }

    actions.push({ type: 'redo' });
    return RESPONSES.redo_done;
  }

  /** @private */
  _handleReset(actions) {
    // Push current state for possible undo
    if (Object.keys(this._currentMorphState).length > 0) {
      this._undoStack.push(this._cloneMorphState());
    }

    this._currentMorphState = {};
    this._redoStack = [];
    this.changeCount = 0;

    // Reset NLU history too
    this.nlu.parse('reset');

    // Notify UI
    if (this._callbacks.onReset) {
      this._callbacks.onReset();
    }

    actions.push({ type: 'reset' });
    return RESPONSES.reset_done;
  }

  /** @private */
  _handleSave(actions) {
    const version = {
      name: `Version ${this._savedVersions.length + 1}`,
      morphState: this._cloneMorphState(),
      timestamp: Date.now(),
      changeCount: this.changeCount,
    };

    this._savedVersions.push(version);

    if (this._callbacks.onSave) {
      this._callbacks.onSave(version);
    }

    actions.push({ type: 'save', data: version });
    return `${RESPONSES.save_done} (${version.name})`;
  }

  // =========================================================================
  // PHASE MANAGEMENT
  // =========================================================================

  /** @private */
  _setPhase(phase) {
    if (this.phase !== phase) {
      this.phase = phase;
      if (this._callbacks.onPhaseChange) {
        this._callbacks.onPhaseChange(phase);
      }
    }
  }

  /** Get current phase. */
  getPhase() {
    return this.phase;
  }

  /** Get all phase constants. */
  static get PHASES() {
    return PHASES;
  }

  // =========================================================================
  // SAVED VERSIONS
  // =========================================================================

  /** Get all saved versions. */
  getSavedVersions() {
    return this._savedVersions.map(v => ({
      name: v.name,
      timestamp: v.timestamp,
      changeCount: v.changeCount,
    }));
  }

  /**
   * Load a saved version by index.
   * @param {number} index
   * @returns {boolean}
   */
  loadVersion(index) {
    if (index < 0 || index >= this._savedVersions.length) return false;

    // Save current state for undo
    this._undoStack.push(this._cloneMorphState());
    this._redoStack = [];

    const version = this._savedVersions[index];
    this._currentMorphState = JSON.parse(JSON.stringify(version.morphState));
    this.changeCount = version.changeCount;

    if (this._callbacks.onApplyChanges) {
      this._callbacks.onApplyChanges(this._currentMorphState);
    }

    return true;
  }

  // =========================================================================
  // HISTORY
  // =========================================================================

  /** @private */
  _addHistory(role, text, intent) {
    this.history.push({
      role,
      text,
      timestamp: Date.now(),
      intent: intent || undefined,
    });

    // Cap at 200 messages
    if (this.history.length > 200) {
      this.history = this.history.slice(-200);
    }
  }

  /**
   * Get conversation history.
   * @param {number} [limit=50]
   * @returns {Array<{role: string, text: string, timestamp: number}>}
   */
  getHistory(limit = 50) {
    return this.history.slice(-limit);
  }

  /**
   * Get formatted transcript (for reports).
   * @returns {string}
   */
  getFormattedHistory() {
    return this.history
      .map(h => `[${h.role === 'user' ? 'Patient' : 'AI'}] ${h.text}`)
      .join('\n\n');
  }

  // =========================================================================
  // REPORT GENERATION (convenience)
  // =========================================================================

  /**
   * Generate a consultation report from the current state.
   * @returns {object}  Report from ComparisonAgent.
   */
  generateReport() {
    const modifiedRegions = Object.keys(this._currentMorphState);
    return this.comparison.generateReport(
      this._currentMorphState,
      modifiedRegions,
      this.patientInfo
    );
  }

  /**
   * Export the report as plain text.
   * @returns {string}
   */
  exportReportText() {
    const report = this.generateReport();
    return this.comparison.exportToText(report);
  }

  // =========================================================================
  // CURRENT STATE ACCESSORS
  // =========================================================================

  /**
   * Get the current morph state (all accumulated changes).
   * @returns {Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>}
   */
  getMorphState() {
    return this._cloneMorphState();
  }

  /**
   * Get list of currently modified regions.
   * @returns {string[]}
   */
  getModifiedRegions() {
    return Object.keys(this._currentMorphState).filter(r => {
      const s = this._currentMorphState[r];
      const d = s.displacement || { x: 0, y: 0, z: 0 };
      return Math.abs(d.x) + Math.abs(d.y) + Math.abs(d.z) + Math.abs(s.inflate || 0) > 0.001;
    });
  }

  // =========================================================================
  // VOICE INTEGRATION (convenience passthrough)
  // =========================================================================

  /**
   * Start voice-driven consultation.
   * Listens continuously and routes each utterance through processMessage.
   * @param {object} [uiCallbacks]  Optional UI hooks.
   */
  startVoiceMode(uiCallbacks = {}) {
    this.voice.startLiveMode(
      async (text) => {
        const result = await this.processMessage(text);
        return result.response;
      },
      uiCallbacks
    );
  }

  /** Stop voice mode. */
  stopVoiceMode() {
    this.voice.stopLiveMode();
  }

  // =========================================================================
  // RESET
  // =========================================================================

  /**
   * Full reset of the director to initial state.
   */
  reset() {
    this.phase = PHASES.INTAKE;
    this.history = [];
    this.hasModel = false;
    this.changeCount = 0;
    this._undoStack = [];
    this._redoStack = [];
    this._savedVersions = [];
    this._currentMorphState = {};
    this.patientInfo = {};
  }

  // =========================================================================
  // INTERNAL HELPERS
  // =========================================================================

  /** Detect intent from text by testing patterns in priority order. @private */
  _detectIntent(text) {
    for (const { intent, pattern } of INTENT_PATTERNS) {
      if (pattern.test(text)) {
        return intent;
      }
    }
    return 'unknown';
  }

  /** Apply parsed region changes to the accumulated morph state. @private */
  _applyRegionsToState(regions) {
    for (const [regionName, change] of Object.entries(regions)) {
      if (!this._currentMorphState[regionName]) {
        this._currentMorphState[regionName] = {
          displacement: { x: 0, y: 0, z: 0 },
          inflate: 0,
        };
      }

      const existing = this._currentMorphState[regionName];
      existing.displacement.x += change.displacement.x;
      existing.displacement.y += change.displacement.y;
      existing.displacement.z += change.displacement.z;
      existing.inflate += change.inflate;
    }
  }

  /** Deep clone the current morph state. @private */
  _cloneMorphState() {
    return JSON.parse(JSON.stringify(this._currentMorphState));
  }

  /** Map friendly region names (e.g., "lips") to clinical zone names. @private */
  _mapFriendlyToZones(friendly) {
    const map = {
      forehead:    ['forehead_center', 'forehead_left', 'forehead_right'],
      eye:         ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
      eyes:        ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
      nose:        ['nasal_tip', 'nasal_dorsum', 'nasal_bridge', 'nasal_ala_left', 'nasal_ala_right'],
      cheek:       ['cheek_left', 'cheek_right'],
      cheeks:      ['cheek_left', 'cheek_right'],
      lip:         ['upper_lip_center', 'lower_lip_center'],
      lips:        ['upper_lip_center', 'upper_lip_left', 'upper_lip_right', 'lower_lip_center', 'lower_lip_left', 'lower_lip_right'],
      jaw:         ['jawline_left', 'jawline_right', 'jaw_angle_left', 'jaw_angle_right'],
      chin:        ['chin_center', 'chin_left', 'chin_right'],
      brow:        ['brow_left', 'brow_right'],
      brows:       ['brow_left', 'brow_right'],
      eyelid:      ['upper_eyelid_left', 'upper_eyelid_right'],
      eyelids:     ['upper_eyelid_left', 'upper_eyelid_right', 'lower_eyelid_left', 'lower_eyelid_right'],
      nasolabial:  ['nasolabial_left', 'nasolabial_right'],
      marionette:  ['marionette_left', 'marionette_right'],
      temple:      ['temple_left', 'temple_right'],
      temples:     ['temple_left', 'temple_right'],
      neck:        ['neck_left', 'neck_right'],
    };
    return map[friendly] || [];
  }

  /** Get a quick medical hint for the modified regions. @private */
  _getQuickMedicalHint(regionNames) {
    if (regionNames.length === 0) return null;

    // Get procedures for the first region
    const info = this.medical.getProcedureByRegion(regionNames[0]);
    if (!info.procedures || info.procedures.length === 0) return null;

    // Pick the least invasive option
    const sorted = [...info.procedures].sort((a, b) => a.invasiveness - b.invasiveness);
    const top = sorted[0];
    return `Tip: ${top.name} ($${top.costRange.min}-$${top.costRange.max}, ${top.recoveryTime} recovery) could help achieve this change.`;
  }

  /** Pick a random entry from an array (or return the string as-is). @private */
  _pick(arr) {
    if (Array.isArray(arr)) {
      return arr[Math.floor(Math.random() * arr.length)];
    }
    return arr;
  }
}

export default ConversationDirector;
