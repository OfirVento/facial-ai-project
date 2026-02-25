/**
 * VoiceAgent.js
 * Voice input/output agent using the Web Speech API.
 *
 * Provides:
 *  - Speech-to-text via SpeechRecognition (continuous listening)
 *  - Text-to-speech via SpeechSynthesis (professional voice selection)
 *  - Speaker diarization placeholder (doctor vs patient labeling)
 *  - onChange listener pattern for reactive integrations
 *  - Live consultation mode with automatic listen/respond cycling
 */

// ---------------------------------------------------------------------------
// Speaker roles for diarization
// ---------------------------------------------------------------------------
export const SPEAKER = {
  DOCTOR:  'doctor',
  PATIENT: 'patient',
  UNKNOWN: 'unknown',
};

// ---------------------------------------------------------------------------
// Voice configuration defaults
// ---------------------------------------------------------------------------
const VOICE_CONFIG = {
  recognition: {
    lang: 'en-US',
    continuous: true,
    interimResults: true,
    maxAlternatives: 3,
  },
  synthesis: {
    rate: 0.95,
    pitch: 1.0,
    volume: 0.9,
    preferredVoices: [
      'Samantha', 'Karen', 'Daniel', 'Moira', 'Tessa',       // macOS premium
      'Google UK English Female', 'Google US English',          // Chrome
      'Microsoft Aria', 'Microsoft Jenny', 'Microsoft Guy',    // Edge
    ],
  },
};

// ---------------------------------------------------------------------------
// VoiceAgent class
// ---------------------------------------------------------------------------
export class VoiceAgent {
  /**
   * @param {object} [options]
   * @param {string} [options.lang]               Recognition language (default 'en-US').
   * @param {boolean} [options.continuous]         Keep listening after each result (default true).
   * @param {boolean} [options.interimResults]     Emit partial transcripts (default true).
   * @param {string} [options.preferredVoiceName]  Preferred TTS voice name substring.
   * @param {number} [options.speechRate]          TTS speech rate (default 0.95).
   * @param {number} [options.speechPitch]         TTS pitch (default 1.0).
   * @param {number} [options.speechVolume]        TTS volume 0-1 (default 0.9).
   */
  constructor(options = {}) {
    // Merge options with defaults
    this._config = {
      lang: options.lang || VOICE_CONFIG.recognition.lang,
      continuous: options.continuous !== undefined ? options.continuous : VOICE_CONFIG.recognition.continuous,
      interimResults: options.interimResults !== undefined ? options.interimResults : VOICE_CONFIG.recognition.interimResults,
      speechRate: options.speechRate || VOICE_CONFIG.synthesis.rate,
      speechPitch: options.speechPitch || VOICE_CONFIG.synthesis.pitch,
      speechVolume: options.speechVolume !== undefined ? options.speechVolume : VOICE_CONFIG.synthesis.volume,
      preferredVoiceName: options.preferredVoiceName || null,
    };

    // State
    this.isListening = false;
    this.isSpeaking = false;

    // Internal references
    this._recognition = null;
    this._synthesis = (typeof window !== 'undefined') ? window.speechSynthesis : null;
    this._selectedVoice = null;
    this._restartTimeout = null;

    // Current speaker for diarization
    this._currentSpeaker = SPEAKER.UNKNOWN;

    // Transcript buffer & diarization history
    this._transcript = '';
    this._interimTranscript = '';
    this._speakerHistory = [];

    // onChange listener registry
    /** @type {Array<(event: VoiceEvent) => void>} */
    this._listeners = [];

    // Legacy callback support (for backward compat with startListening(callbacks))
    this._onResult = null;
    this._onInterim = null;
    this._onEnd = null;
    this._onError = null;

    // Initialize
    this._initRecognition();
    this._initVoices();
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================

  /** @private */
  _initRecognition() {
    const SpeechRecognition = this._getSpeechRecognitionCtor();
    if (!SpeechRecognition) {
      console.warn('[VoiceAgent] Speech recognition not supported in this browser.');
      return;
    }

    this._recognition = new SpeechRecognition();
    this._recognition.lang = this._config.lang;
    this._recognition.continuous = this._config.continuous;
    this._recognition.interimResults = this._config.interimResults;
    this._recognition.maxAlternatives = VOICE_CONFIG.recognition.maxAlternatives;

    // --- onresult ---
    this._recognition.onresult = (event) => {
      let interimText = '';
      let finalText = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0].transcript;
        const confidence = result[0].confidence;

        if (result.isFinal) {
          finalText += transcript;

          // Diarization history entry
          this._speakerHistory.push({
            speaker: this._currentSpeaker,
            text: transcript.trim(),
            confidence,
            timestamp: Date.now(),
          });

          // Emit to onChange listeners
          this._emit({
            type: 'transcript_final',
            text: transcript.trim(),
            confidence,
            speaker: this._currentSpeaker,
          });

          // Legacy callback
          if (this._onResult) {
            this._onResult(transcript.trim(), confidence);
          }
        } else {
          interimText += transcript;
        }
      }

      if (finalText) {
        this._transcript += finalText;
      }
      this._interimTranscript = interimText;

      if (interimText) {
        this._emit({
          type: 'transcript_interim',
          text: interimText.trim(),
          speaker: this._currentSpeaker,
        });

        // Legacy callback
        if (this._onInterim) {
          this._onInterim(interimText.trim());
        }
      }
    };

    // --- onend ---
    this._recognition.onend = () => {
      if (this.isListening && this._config.continuous) {
        // Auto-restart after brief pause
        this._restartTimeout = setTimeout(() => {
          try {
            this._recognition.start();
          } catch (e) {
            // Already started or other issue
          }
        }, 100);
        return;
      }

      this.isListening = false;
      this._emit({ type: 'listening_end', speaker: this._currentSpeaker });

      if (this._onEnd) this._onEnd();
    };

    // --- onerror ---
    this._recognition.onerror = (event) => {
      console.warn('[VoiceAgent] Recognition error:', event.error);

      const errorMsg = this._mapErrorMessage(event.error);

      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        this.isListening = false;
        this._emit({ type: 'error', message: errorMsg, errorCode: event.error });
        if (this._onError) this._onError(errorMsg);
      } else if (event.error === 'no-speech' || event.error === 'aborted') {
        // Recoverable — will auto-restart via onend
      } else if (event.error === 'network') {
        this._emit({ type: 'error', message: errorMsg, errorCode: event.error });
        if (this._onError) this._onError(errorMsg);
      } else {
        this._emit({ type: 'error', message: errorMsg, errorCode: event.error });
        if (this._onError) this._onError(errorMsg);
      }
    };
  }

  /** @private */
  _initVoices() {
    if (!this._synthesis) return;

    const loadVoices = () => {
      const voices = this._synthesis.getVoices();
      if (voices.length === 0) return;

      this._selectedVoice = this._findProfessionalVoice(voices);
      this._emit({ type: 'voices_loaded', count: voices.length });
    };

    loadVoices();

    // Chrome fires voiceschanged asynchronously
    if (this._synthesis.onvoiceschanged !== undefined) {
      this._synthesis.onvoiceschanged = loadVoices;
    }
  }

  // =========================================================================
  // PUBLIC API: Listening (Speech-to-Text)
  // =========================================================================

  /**
   * Start listening for speech input.
   * Supports two calling styles:
   *   1) startListening(speaker)            — uses onChange listeners
   *   2) startListening({ onResult, ... })  — legacy callback style
   *
   * @param {string|object} [speakerOrCallbacks]
   *   If string: speaker label ('doctor' | 'patient' | 'unknown').
   *   If object: { onResult, onInterim, onEnd, onError } callbacks.
   * @returns {boolean}  True if started successfully.
   */
  startListening(speakerOrCallbacks) {
    if (!this._recognition) {
      const msg = 'Speech recognition not supported in this browser.';
      this._emit({ type: 'error', message: msg });
      return false;
    }

    if (this.isListening) return true;

    // Determine calling style
    if (typeof speakerOrCallbacks === 'string') {
      this._currentSpeaker = speakerOrCallbacks;
    } else if (typeof speakerOrCallbacks === 'object' && speakerOrCallbacks !== null) {
      // Legacy callback style
      this._onResult = speakerOrCallbacks.onResult || null;
      this._onInterim = speakerOrCallbacks.onInterim || null;
      this._onEnd = speakerOrCallbacks.onEnd || null;
      this._onError = speakerOrCallbacks.onError || null;
    }

    try {
      this._recognition.start();
      this.isListening = true;
      this._emit({ type: 'listening_start', speaker: this._currentSpeaker });
      return true;
    } catch (e) {
      console.error('[VoiceAgent] Failed to start recognition:', e);
      this._emit({ type: 'error', message: 'Failed to start speech recognition.' });
      return false;
    }
  }

  /**
   * Stop listening.
   */
  stopListening() {
    this.isListening = false;

    if (this._restartTimeout) {
      clearTimeout(this._restartTimeout);
      this._restartTimeout = null;
    }

    if (this._recognition) {
      try {
        this._recognition.stop();
      } catch (e) {
        // Already stopped
      }
    }
  }

  /**
   * Toggle listening on/off.
   * @param {string|object} [speakerOrCallbacks]
   * @returns {boolean}  New listening state.
   */
  toggleListening(speakerOrCallbacks) {
    if (this.isListening) {
      this.stopListening();
      return false;
    } else {
      this.startListening(speakerOrCallbacks);
      return true;
    }
  }

  // =========================================================================
  // PUBLIC API: Speaking (Text-to-Speech)
  // =========================================================================

  /**
   * Speak text aloud using SpeechSynthesis.
   * @param {string} text            Text to speak.
   * @param {object} [options]       Override synthesis config.
   * @param {number} [options.rate]  Speech rate override.
   * @param {number} [options.pitch] Pitch override.
   * @param {number} [options.volume] Volume override (0-1).
   * @returns {Promise<void>}        Resolves when speech finishes.
   */
  speak(text, options = {}) {
    return new Promise((resolve, reject) => {
      if (!this._synthesis) {
        resolve(); // Silently skip if not supported
        return;
      }

      // Cancel any current speech
      this._synthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.voice = this._selectedVoice;
      utterance.lang = this._config.lang;
      utterance.rate = options.rate ?? this._config.speechRate;
      utterance.pitch = options.pitch ?? this._config.speechPitch;
      utterance.volume = options.volume ?? this._config.speechVolume;

      utterance.onstart = () => {
        this.isSpeaking = true;
        this._emit({ type: 'speaking_start', text });
      };

      utterance.onend = () => {
        this.isSpeaking = false;
        this._emit({ type: 'speaking_end', text });
        resolve();
      };

      utterance.onerror = (event) => {
        this.isSpeaking = false;
        if (event.error === 'interrupted' || event.error === 'canceled') {
          resolve(); // Normal cancellation
        } else {
          console.warn('[VoiceAgent] Speech synthesis error:', event.error);
          this._emit({ type: 'error', message: `Speech synthesis error: ${event.error}` });
          resolve(); // Don't reject — keep pipeline flowing
        }
      };

      this._synthesis.speak(utterance);
    });
  }

  /**
   * Speak a longer response with sentence-level chunking for natural pauses.
   * @param {string} text       Full response text.
   * @param {object} [options]  Synthesis options.
   * @returns {Promise<void>}
   */
  async speakResponse(text, options = {}) {
    if (!this._synthesis) return;

    // Split into sentences
    const sentences = text
      .replace(/([.!?])\s+/g, '$1|')
      .split('|')
      .map(s => s.trim())
      .filter(s => s.length > 0);

    for (let i = 0; i < sentences.length; i++) {
      // Check if cancelled between sentences
      if (i > 0 && !this.isSpeaking) break;
      await this.speak(sentences[i], options);
    }
  }

  /**
   * Stop any current speech.
   */
  stopSpeaking() {
    if (this._synthesis) {
      this._synthesis.cancel();
      this.isSpeaking = false;
      this._emit({ type: 'speaking_end', text: '' });
    }
  }

  // =========================================================================
  // PUBLIC API: Speaker Diarization (Placeholder)
  // =========================================================================

  /**
   * Set the active speaker label.
   * In a future version this could be driven by ML-based speaker identification
   * (e.g., voice embeddings, spectral clustering).
   * @param {string} speaker  'doctor' | 'patient' | 'unknown'
   */
  setSpeaker(speaker) {
    if (Object.values(SPEAKER).includes(speaker)) {
      this._currentSpeaker = speaker;
      this._emit({ type: 'speaker_change', speaker });
    }
  }

  /**
   * Get the current speaker label.
   * @returns {string}
   */
  getCurrentSpeaker() {
    return this._currentSpeaker;
  }

  /**
   * Get the full diarized transcript history.
   * Each entry: { speaker, text, confidence, timestamp }
   * @returns {Array<{speaker: string, text: string, confidence: number, timestamp: number}>}
   */
  getTranscriptHistory() {
    return [...this._speakerHistory];
  }

  /**
   * Get transcript entries for a specific speaker.
   * @param {string} speaker
   * @returns {Array<{text: string, confidence: number, timestamp: number}>}
   */
  getTranscriptBySpeaker(speaker) {
    return this._speakerHistory
      .filter(entry => entry.speaker === speaker)
      .map(({ text, confidence, timestamp }) => ({ text, confidence, timestamp }));
  }

  /**
   * Get the full transcript as a formatted string with speaker labels.
   * @returns {string}
   */
  getFormattedTranscript() {
    return this._speakerHistory
      .map(entry => {
        const label = entry.speaker === SPEAKER.DOCTOR ? 'Doctor' :
                      entry.speaker === SPEAKER.PATIENT ? 'Patient' : 'Speaker';
        return `[${label}]: ${entry.text}`;
      })
      .join('\n');
  }

  /**
   * Clear transcript history.
   */
  clearHistory() {
    this._speakerHistory = [];
    this._transcript = '';
    this._interimTranscript = '';
  }

  // =========================================================================
  // PUBLIC API: onChange Listener Pattern
  // =========================================================================

  /**
   * Register a listener for voice events.
   * @param {(event: VoiceEvent) => void} callback
   * @returns {() => void}  Unsubscribe function.
   *
   * Event types emitted:
   *   listening_start   { type, speaker }
   *   listening_end     { type, speaker }
   *   transcript_final  { type, text, confidence, speaker }
   *   transcript_interim { type, text, speaker }
   *   speaking_start    { type, text }
   *   speaking_end      { type, text }
   *   speaker_change    { type, speaker }
   *   voices_loaded     { type, count }
   *   error             { type, message, errorCode? }
   *   warning           { type, message }
   */
  onChange(callback) {
    if (typeof callback !== 'function') {
      throw new Error('onChange requires a function callback');
    }
    this._listeners.push(callback);

    // Return unsubscribe function
    return () => {
      this._listeners = this._listeners.filter(fn => fn !== callback);
    };
  }

  /**
   * Remove all onChange listeners.
   */
  removeAllListeners() {
    this._listeners = [];
  }

  // =========================================================================
  // PUBLIC API: Live Consultation Mode
  // =========================================================================

  /**
   * Start live consultation mode.
   * Listens continuously and processes commands in real-time.
   * @param {(text: string) => Promise<string>} processCallback
   *   Called with final transcription text, should return AI response string.
   * @param {object} [uiCallbacks]
   * @param {(text: string) => void} [uiCallbacks.onInterim]     Show interim transcript.
   * @param {(text: string) => void} [uiCallbacks.onUserMessage]  Show final user message.
   * @param {(text: string) => void} [uiCallbacks.onAIMessage]    Show AI response.
   * @param {(error: string) => void} [uiCallbacks.onError]       Show error.
   */
  startLiveMode(processCallback, uiCallbacks = {}) {
    let processingLock = false;

    this.startListening({
      onInterim: (text) => {
        if (uiCallbacks.onInterim) uiCallbacks.onInterim(text);
      },
      onResult: async (text, confidence) => {
        if (processingLock) return;
        processingLock = true;

        try {
          // Record user utterance
          if (uiCallbacks.onUserMessage) uiCallbacks.onUserMessage(text);

          // Pause listening while processing
          this.stopListening();

          // Get AI response
          const response = await processCallback(text);

          // Display and speak response
          if (uiCallbacks.onAIMessage) uiCallbacks.onAIMessage(response);
          await this.speakResponse(response);

          // Resume listening
          this.startListening({
            onInterim: uiCallbacks.onInterim ? (t) => uiCallbacks.onInterim(t) : null,
            onResult: this._onResult,
            onEnd: this._onEnd,
            onError: uiCallbacks.onError || null,
          });
        } catch (e) {
          console.error('[VoiceAgent] Live mode error:', e);
          if (uiCallbacks.onError) uiCallbacks.onError(e.message);
        } finally {
          processingLock = false;
        }
      },
      onEnd: () => {},
      onError: (error) => {
        if (uiCallbacks.onError) uiCallbacks.onError(error);
      },
    });
  }

  /**
   * Stop live consultation mode.
   */
  stopLiveMode() {
    this.stopListening();
    this.stopSpeaking();
  }

  // =========================================================================
  // PUBLIC API: Voice Management
  // =========================================================================

  /**
   * Get available TTS voices (filtered to English).
   * @returns {Array<{name: string, lang: string, isSelected: boolean, localService: boolean}>}
   */
  getAvailableVoices() {
    if (!this._synthesis) return [];

    return this._synthesis.getVoices()
      .filter(v => v.lang.startsWith('en'))
      .map(v => ({
        name: v.name,
        lang: v.lang,
        isSelected: this._selectedVoice?.name === v.name,
        localService: v.localService,
      }));
  }

  /**
   * Set TTS voice by name.
   * @param {string} voiceName  Exact voice name.
   * @returns {boolean}  True if voice was found and set.
   */
  setVoice(voiceName) {
    if (!this._synthesis) return false;

    const voice = this._synthesis.getVoices().find(v => v.name === voiceName);
    if (voice) {
      this._selectedVoice = voice;
      return true;
    }
    return false;
  }

  // =========================================================================
  // PUBLIC API: Utility
  // =========================================================================

  /**
   * Check browser support for speech APIs.
   * @returns {{ recognition: boolean, synthesis: boolean }}
   */
  static checkSupport() {
    const hasWindow = typeof window !== 'undefined';
    return {
      recognition: hasWindow && !!(window.SpeechRecognition || window.webkitSpeechRecognition),
      synthesis: hasWindow && !!window.speechSynthesis,
    };
  }

  /**
   * Check if speech recognition is supported.
   * @returns {boolean}
   */
  isRecognitionSupported() {
    return !!this._recognition;
  }

  /**
   * Check if speech synthesis is supported.
   * @returns {boolean}
   */
  isSynthesisSupported() {
    return !!this._synthesis;
  }

  /**
   * Destroy the agent and release all resources.
   */
  destroy() {
    this.stopListening();
    this.stopSpeaking();
    this.removeAllListeners();
    this._speakerHistory = [];
    this._recognition = null;
    this._onResult = null;
    this._onInterim = null;
    this._onEnd = null;
    this._onError = null;
  }

  // =========================================================================
  // INTERNAL HELPERS
  // =========================================================================

  /** @private */
  _getSpeechRecognitionCtor() {
    if (typeof window === 'undefined') return null;
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  /**
   * Find a natural/professional English voice from the available list.
   * @private
   * @param {SpeechSynthesisVoice[]} voices
   * @returns {SpeechSynthesisVoice|null}
   */
  _findProfessionalVoice(voices) {
    // User preference
    if (this._config.preferredVoiceName) {
      const pref = voices.find(v =>
        v.name.toLowerCase().includes(this._config.preferredVoiceName.toLowerCase())
      );
      if (pref) return pref;
    }

    // Try preferred voices list
    for (const preferred of VOICE_CONFIG.synthesis.preferredVoices) {
      const found = voices.find(v => v.name.includes(preferred));
      if (found) return found;
    }

    // Fallback: first local English voice
    const localEnglish = voices.find(v => v.lang.startsWith('en') && v.localService);
    if (localEnglish) return localEnglish;

    // Fallback: any English voice
    const anyEnglish = voices.find(v => v.lang.startsWith('en'));
    if (anyEnglish) return anyEnglish;

    // Last resort
    return voices.find(v => v.default) || voices[0] || null;
  }

  /**
   * Map recognition error codes to user-friendly messages.
   * @private
   */
  _mapErrorMessage(errorCode) {
    const map = {
      'not-allowed': 'Microphone access denied. Please allow microphone access in your browser settings.',
      'service-not-allowed': 'Speech recognition service not allowed. Check browser permissions.',
      'network': 'Network error. Speech recognition requires an internet connection.',
      'no-speech': 'No speech detected.',
      'aborted': 'Speech recognition was aborted.',
      'audio-capture': 'No microphone found. Please check your audio input device.',
      'language-not-supported': 'The selected language is not supported.',
    };
    return map[errorCode] || `Speech recognition error: ${errorCode}`;
  }

  /**
   * Emit a voice event to all registered onChange listeners.
   * @private
   * @param {object} event
   */
  _emit(event) {
    for (const listener of this._listeners) {
      try {
        listener(event);
      } catch (err) {
        console.error('[VoiceAgent] Listener error:', err);
      }
    }
  }
}

export default VoiceAgent;
