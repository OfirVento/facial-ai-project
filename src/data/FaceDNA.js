/**
 * FaceDNA — The Digital Face Identity
 *
 * A structured data format encoding everything about a patient's face:
 * geometry (FLAME parameters), texture (composited photo), clinical
 * measurements, capture metadata, and treatment history.
 *
 * This is the foundation for the entire platform — treatment predictions,
 * store deployments, and patient records all operate on FaceDNA.
 */

import { ClinicalMeasurements } from './ClinicalMeasurements.js';
import { SymmetryAnalyzer } from './SymmetryAnalyzer.js';

const FACEDNA_VERSION = '1.0';
const FACEDNA_MAGIC = 'FDNA';

export class FaceDNA {
  /**
   * Create a new FaceDNA instance with default (empty) values.
   * Use FaceDNA.fromCurrentState() to populate from live pipeline data.
   */
  constructor() {
    this.version = FACEDNA_VERSION;
    this.id = FaceDNA._generateId();
    this.createdAt = new Date().toISOString();
    this.updatedAt = this.createdAt;

    // -- Geometry: FLAME parametric encoding --
    this.geometry = {
      flameVersion: '2023',
      shapeParams: null,           // Float32Array[50] — face identity (β)
      expressionParams: null,      // Float32Array[50] — neutral expression (ψ)
      perVertexDisplacements: null, // Float32Array[5023×3] — fine detail residuals
      globalPose: null,            // Float32Array[6] — [rx, ry, rz, tx, ty, tz]
      vertexCount: 5023,
      faceCount: 9976,
    };

    // -- Texture: composited photo atlas --
    this.texture = {
      resolution: 2048,
      albedoDataUrl: null,         // PNG data URL of final composited texture
      rawDataUrl: null,            // PNG data URL of pre-blend raw projection
    };

    // -- Clinical measurements --
    this.clinical = {
      measurements: null,          // { faceWidth, jawWidth, noseLength, ... } in mm
      symmetry: null,              // { overall: 0-1, perRegion: { ... } }
    };

    // -- Capture metadata --
    this.capture = {
      device: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
      views: [],                   // [{ angle, resolution, qualityScore, landmarks, cameraParams }]
      timestamp: this.createdAt,
    };

    // -- Treatment history (grows over time) --
    this.treatments = [];
  }

  // -----------------------------------------------------------------------
  // Factory: populate from live pipeline state
  // -----------------------------------------------------------------------

  /**
   * Harvest data from the current pipeline state to build a complete FaceDNA.
   *
   * @param {Object} meshGen     - FlameMeshGenerator instance
   * @param {Object} photoUploader - PhotoUploader instance
   * @param {Object} [mediaPipeBridge] - MediaPipeBridge instance (for mapping data)
   * @returns {FaceDNA}
   */
  static fromCurrentState(meshGen, photoUploader, mediaPipeBridge = null) {
    const dna = new FaceDNA();

    // -- Harvest geometry --
    if (meshGen) {
      // Shape parameters (the core face identity — 50 floats)
      if (meshGen._currentShapeParams) {
        dna.geometry.shapeParams = new Float32Array(meshGen._currentShapeParams);
      }

      // Expression parameters
      if (meshGen._currentExprParams) {
        dna.geometry.expressionParams = new Float32Array(meshGen._currentExprParams);
      }

      // Current deformed vertices (the full mesh state)
      if (meshGen._flameCurrentVertices) {
        // Compute per-vertex displacements from template
        const template = meshGen.flameTemplateVertices;
        const current = meshGen._flameCurrentVertices;
        if (template && current && template.length === current.length) {
          const displacements = new Float32Array(current.length);
          for (let i = 0; i < current.length; i++) {
            displacements[i] = current[i] - template[i];
          }
          dna.geometry.perVertexDisplacements = displacements;
        }
      }
    }

    // -- Harvest texture --
    if (photoUploader) {
      // Final composited texture
      if (photoUploader._debugFinalTexture) {
        dna.texture.albedoDataUrl = photoUploader._debugFinalTexture;
      }

      // Raw UV projection (before blending)
      if (photoUploader._debugPhotoUV_raw) {
        dna.texture.rawDataUrl = photoUploader._debugPhotoUV_raw;
      }
    }

    // -- Harvest capture metadata --
    if (photoUploader) {
      dna.capture.views = [];

      // Front view
      if (photoUploader.photos?.front) {
        const frontView = {
          angle: 'front',
          resolution: [
            photoUploader.photos.front.width || 0,
            photoUploader.photos.front.height || 0,
          ],
          qualityScore: photoUploader.photos.front._qualityScore ?? 1.0,
        };

        // Camera params from pipeline
        if (photoUploader._diagCameraParams) {
          const cam = photoUploader._diagCameraParams;
          frontView.cameraParams = {
            sx: cam.sx, sy: cam.sy,
            tx: cam.tx, ty: cam.ty,
            R: cam.R ? Array.from(cam.R) : null,
          };
        }

        dna.capture.views.push(frontView);
      }

      // Side views
      for (const angle of ['left45', 'right45']) {
        if (photoUploader.photos?.[angle]) {
          dna.capture.views.push({
            angle,
            resolution: [
              photoUploader.photos[angle].width || 0,
              photoUploader.photos[angle].height || 0,
            ],
            qualityScore: photoUploader.photos[angle]._qualityScore ?? 1.0,
          });
        }
      }
    }

    // -- Compute clinical measurements --
    if (meshGen && meshGen._flameCurrentVertices) {
      try {
        const mapping = mediaPipeBridge?._flameMapping || null;
        dna.clinical.measurements = ClinicalMeasurements.compute(meshGen, mapping);
        dna.clinical.symmetry = SymmetryAnalyzer.analyze(meshGen);
      } catch (err) {
        console.warn('FaceDNA: Clinical measurement computation failed:', err.message);
      }
    }

    dna.updatedAt = new Date().toISOString();
    console.log(`FaceDNA: Created from current state (${dna.capture.views.length} views, ` +
      `shape=${dna.geometry.shapeParams ? 'yes' : 'no'}, texture=${dna.texture.albedoDataUrl ? 'yes' : 'no'})`);

    return dna;
  }

  // -----------------------------------------------------------------------
  // Serialization: JSON
  // -----------------------------------------------------------------------

  /**
   * Serialize to a plain JSON-compatible object.
   * Float32Arrays are converted to regular arrays for JSON.stringify().
   */
  toJSON() {
    return {
      version: this.version,
      id: this.id,
      createdAt: this.createdAt,
      updatedAt: new Date().toISOString(),

      geometry: {
        flameVersion: this.geometry.flameVersion,
        shapeParams: this.geometry.shapeParams ? Array.from(this.geometry.shapeParams) : null,
        expressionParams: this.geometry.expressionParams ? Array.from(this.geometry.expressionParams) : null,
        perVertexDisplacements: this.geometry.perVertexDisplacements
          ? Array.from(this.geometry.perVertexDisplacements)
          : null,
        globalPose: this.geometry.globalPose ? Array.from(this.geometry.globalPose) : null,
        vertexCount: this.geometry.vertexCount,
        faceCount: this.geometry.faceCount,
      },

      texture: {
        resolution: this.texture.resolution,
        albedoDataUrl: this.texture.albedoDataUrl,
        rawDataUrl: this.texture.rawDataUrl,
      },

      clinical: this.clinical,

      capture: {
        device: this.capture.device,
        views: this.capture.views,
        timestamp: this.capture.timestamp,
      },

      treatments: this.treatments,
    };
  }

  /**
   * Restore a FaceDNA instance from a plain JSON object.
   * @param {Object} json - Output of toJSON()
   * @returns {FaceDNA}
   */
  static fromJSON(json) {
    const dna = new FaceDNA();

    dna.version = json.version || FACEDNA_VERSION;
    dna.id = json.id || FaceDNA._generateId();
    dna.createdAt = json.createdAt || dna.createdAt;
    dna.updatedAt = json.updatedAt || dna.updatedAt;

    // Geometry
    if (json.geometry) {
      dna.geometry.flameVersion = json.geometry.flameVersion || '2023';
      dna.geometry.shapeParams = json.geometry.shapeParams
        ? new Float32Array(json.geometry.shapeParams) : null;
      dna.geometry.expressionParams = json.geometry.expressionParams
        ? new Float32Array(json.geometry.expressionParams) : null;
      dna.geometry.perVertexDisplacements = json.geometry.perVertexDisplacements
        ? new Float32Array(json.geometry.perVertexDisplacements) : null;
      dna.geometry.globalPose = json.geometry.globalPose
        ? new Float32Array(json.geometry.globalPose) : null;
      dna.geometry.vertexCount = json.geometry.vertexCount || 5023;
      dna.geometry.faceCount = json.geometry.faceCount || 9976;
    }

    // Texture
    if (json.texture) {
      dna.texture.resolution = json.texture.resolution || 2048;
      dna.texture.albedoDataUrl = json.texture.albedoDataUrl || null;
      dna.texture.rawDataUrl = json.texture.rawDataUrl || null;
    }

    // Clinical
    if (json.clinical) {
      dna.clinical = json.clinical;
    }

    // Capture
    if (json.capture) {
      dna.capture = json.capture;
    }

    // Treatments
    dna.treatments = json.treatments || [];

    return dna;
  }

  // -----------------------------------------------------------------------
  // File export/import (.fdna format)
  // -----------------------------------------------------------------------

  /**
   * Export FaceDNA as a downloadable .fdna file (JSON format).
   * The .fdna file is a JSON file with embedded base64 textures.
   * @returns {Blob}
   */
  toFileBlob() {
    const json = JSON.stringify(this.toJSON(), null, 2);
    return new Blob([json], { type: 'application/json' });
  }

  /**
   * Trigger browser download of the .fdna file.
   * @param {string} [filename] - Optional filename override
   */
  downloadFile(filename) {
    const name = filename || `facedna_${this.id.slice(0, 8)}_${new Date().toISOString().slice(0, 10)}.fdna`;
    const blob = this.toFileBlob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`FaceDNA: Downloaded as ${name} (${(blob.size / 1024).toFixed(1)} KB)`);
  }

  /**
   * Import FaceDNA from a .fdna file.
   * @param {File} file - The .fdna file to import
   * @returns {Promise<FaceDNA>}
   */
  static async fromFile(file) {
    const text = await file.text();
    const json = JSON.parse(text);
    const dna = FaceDNA.fromJSON(json);
    console.log(`FaceDNA: Imported from file (id=${dna.id}, views=${dna.capture.views.length})`);
    return dna;
  }

  // -----------------------------------------------------------------------
  // Treatment history
  // -----------------------------------------------------------------------

  /**
   * Record a treatment simulation to the history.
   * @param {Object} morphState - Region morph state from the app
   * @param {string[]} procedures - List of procedure names
   * @param {string} [notes] - Optional notes
   */
  addTreatment(morphState, procedures = [], notes = '') {
    this.treatments.push({
      id: FaceDNA._generateId(),
      timestamp: new Date().toISOString(),
      morphState: JSON.parse(JSON.stringify(morphState)), // deep clone
      procedures,
      notes,
    });
    this.updatedAt = new Date().toISOString();
  }

  // -----------------------------------------------------------------------
  // Restoration: apply FaceDNA back to pipeline
  // -----------------------------------------------------------------------

  /**
   * Apply this FaceDNA's geometry (shape + expression params) back to a mesh generator.
   * This restores the face shape from the saved parameters.
   * @param {Object} meshGen - FlameMeshGenerator instance
   * @returns {boolean} true if successful
   */
  applyGeometry(meshGen) {
    if (!meshGen) return false;

    if (this.geometry.shapeParams && meshGen.applyShapeParams) {
      meshGen.applyShapeParams(this.geometry.shapeParams);
      console.log('FaceDNA: Applied shape params to mesh');
    }

    if (this.geometry.expressionParams && meshGen.applyExpressionParams) {
      meshGen.applyExpressionParams(this.geometry.expressionParams);
      console.log('FaceDNA: Applied expression params to mesh');
    }

    return true;
  }

  /**
   * Apply this FaceDNA's texture back to the renderer.
   * @param {Object} renderer - FaceRenderer instance
   * @returns {Promise<boolean>} true if successful
   */
  async applyTexture(renderer) {
    if (!renderer || !this.texture.albedoDataUrl) return false;

    await renderer.loadTexture(this.texture.albedoDataUrl, 'albedo', { flipY: false });
    console.log('FaceDNA: Applied texture to renderer');
    return true;
  }

  // -----------------------------------------------------------------------
  // Summary / display
  // -----------------------------------------------------------------------

  /**
   * Get a human-readable summary of this FaceDNA.
   * @returns {Object}
   */
  getSummary() {
    return {
      id: this.id,
      created: this.createdAt,
      views: this.capture.views.length,
      hasShape: !!this.geometry.shapeParams,
      hasTexture: !!this.texture.albedoDataUrl,
      hasMeasurements: !!this.clinical.measurements,
      hasSymmetry: !!this.clinical.symmetry,
      treatmentCount: this.treatments.length,
      measurements: this.clinical.measurements || {},
      symmetry: this.clinical.symmetry || {},
    };
  }

  // -----------------------------------------------------------------------
  // IndexedDB persistence
  // -----------------------------------------------------------------------

  /**
   * Save this FaceDNA to IndexedDB for browser persistence.
   * @returns {Promise<void>}
   */
  async saveToDB() {
    const db = await FaceDNA._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('facedna', 'readwrite');
      const store = tx.objectStore('facedna');
      store.put(this.toJSON());
      tx.oncomplete = () => {
        console.log(`FaceDNA: Saved to IndexedDB (id=${this.id})`);
        resolve();
      };
      tx.onerror = () => reject(tx.error);
    });
  }

  /**
   * Load a FaceDNA from IndexedDB by ID.
   * @param {string} id
   * @returns {Promise<FaceDNA|null>}
   */
  static async loadFromDB(id) {
    const db = await FaceDNA._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('facedna', 'readonly');
      const store = tx.objectStore('facedna');
      const req = store.get(id);
      req.onsuccess = () => {
        if (req.result) {
          resolve(FaceDNA.fromJSON(req.result));
        } else {
          resolve(null);
        }
      };
      req.onerror = () => reject(req.error);
    });
  }

  /**
   * List all saved FaceDNA entries from IndexedDB.
   * @returns {Promise<Array<{id, createdAt, views}>>}
   */
  static async listFromDB() {
    const db = await FaceDNA._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('facedna', 'readonly');
      const store = tx.objectStore('facedna');
      const req = store.getAll();
      req.onsuccess = () => {
        const entries = (req.result || []).map(r => ({
          id: r.id,
          createdAt: r.createdAt,
          views: r.capture?.views?.length || 0,
          hasTexture: !!r.texture?.albedoDataUrl,
        }));
        resolve(entries);
      };
      req.onerror = () => reject(req.error);
    });
  }

  /**
   * Delete a FaceDNA from IndexedDB by ID.
   * @param {string} id
   * @returns {Promise<void>}
   */
  static async deleteFromDB(id) {
    const db = await FaceDNA._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction('facedna', 'readwrite');
      tx.objectStore('facedna').delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  // -----------------------------------------------------------------------
  // Private helpers
  // -----------------------------------------------------------------------

  static _generateId() {
    return 'fdna_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 8);
  }

  static _dbInstance = null;

  static _openDB() {
    if (FaceDNA._dbInstance) return Promise.resolve(FaceDNA._dbInstance);

    return new Promise((resolve, reject) => {
      const request = indexedDB.open('FaceDNA_Store', 1);
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('facedna')) {
          db.createObjectStore('facedna', { keyPath: 'id' });
        }
      };
      request.onsuccess = () => {
        FaceDNA._dbInstance = request.result;
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }
}
