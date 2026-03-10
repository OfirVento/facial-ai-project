/**
 * Database layer — PostgreSQL connection + schema management.
 *
 * Tables:
 *   stores         — Clinic locations
 *   staff          — Staff accounts (login credentials)
 *   patients       — Patient profiles (PII encrypted)
 *   facedna        — FaceDNA records (versioned per patient)
 *   consultations  — Consultation sessions
 *   predictions    — Treatment prediction records
 *   audit_log      — Every data access logged
 */

import pg from 'pg';

const { Pool } = pg;

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

const pool = new Pool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '5432'),
  database: process.env.DB_NAME || 'facialai',
  user: process.env.DB_USER || 'facialai',
  password: process.env.DB_PASSWORD || 'facialai_dev',
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA_SQL = `
  -- Stores (clinic locations)
  CREATE TABLE IF NOT EXISTS stores (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL,
    address       TEXT,
    phone         TEXT,
    settings      JSONB DEFAULT '{}',
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
  );

  -- Staff (login accounts)
  CREATE TABLE IF NOT EXISTS staff (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id      UUID REFERENCES stores(id) ON DELETE CASCADE,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    name          TEXT NOT NULL,
    role          TEXT DEFAULT 'practitioner' CHECK (role IN ('admin', 'manager', 'practitioner', 'receptionist')),
    active        BOOLEAN DEFAULT true,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
  );

  -- Patients (PII encrypted at application level)
  CREATE TABLE IF NOT EXISTS patients (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    store_id      UUID REFERENCES stores(id) ON DELETE CASCADE,
    encrypted_data BYTEA,         -- AES-256-GCM encrypted JSON: { name, email, phone, dob, notes }
    encryption_iv  BYTEA,          -- Per-patient initialization vector
    created_by    UUID REFERENCES staff(id),
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
  );

  -- FaceDNA records (versioned)
  CREATE TABLE IF NOT EXISTS facedna (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id    UUID REFERENCES patients(id) ON DELETE CASCADE,
    version       INTEGER DEFAULT 1,
    data          JSONB NOT NULL,  -- FaceDNA JSON (geometry, clinical, capture metadata)
    texture_key   TEXT,            -- S3/storage key for albedo texture PNG
    file_size     INTEGER,         -- Total size in bytes
    quality_score REAL,            -- Overall capture quality (0-1)
    created_by    UUID REFERENCES staff(id),
    created_at    TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX IF NOT EXISTS idx_facedna_patient ON facedna(patient_id);

  -- Consultations
  CREATE TABLE IF NOT EXISTS consultations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id    UUID REFERENCES patients(id) ON DELETE CASCADE,
    store_id      UUID REFERENCES stores(id),
    staff_id      UUID REFERENCES staff(id),
    facedna_id    UUID REFERENCES facedna(id),
    status        TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
    notes         TEXT,
    started_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
  );

  -- Treatment predictions
  CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consultation_id UUID REFERENCES consultations(id) ON DELETE CASCADE,
    treatment_type  TEXT NOT NULL,
    parameters      JSONB NOT NULL,  -- { volume, intensity, ... }
    morph_state     JSONB,           -- Predicted morph state
    confidence      REAL,            -- Confidence score (0-1)
    created_at      TIMESTAMPTZ DEFAULT NOW()
  );

  -- Audit log (every data access)
  CREATE TABLE IF NOT EXISTS audit_log (
    id            BIGSERIAL PRIMARY KEY,
    staff_id      UUID,
    action        TEXT NOT NULL,    -- 'view_patient', 'create_facedna', 'delete_patient', etc.
    resource_type TEXT,             -- 'patient', 'facedna', 'consultation', etc.
    resource_id   UUID,
    details       JSONB,
    ip_address    TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX IF NOT EXISTS idx_audit_staff ON audit_log(staff_id);
  CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id);
`;

// ---------------------------------------------------------------------------
// DB Interface
// ---------------------------------------------------------------------------

export const db = {
  /**
   * Initialize database: run schema migration.
   */
  async init() {
    try {
      await pool.query(SCHEMA_SQL);
      console.log('Database: Schema initialized');
    } catch (err) {
      console.warn('Database: Schema init skipped (might need PostgreSQL running):', err.message);
      // Don't crash — the server can still serve static files
    }
  },

  /**
   * Execute a parameterized query.
   * @param {string} text - SQL with $1, $2 placeholders
   * @param {Array} params - Parameter values
   * @returns {Promise<pg.QueryResult>}
   */
  async query(text, params) {
    const start = Date.now();
    const result = await pool.query(text, params);
    const duration = Date.now() - start;
    if (duration > 1000) {
      console.warn(`Database: Slow query (${duration}ms):`, text.slice(0, 100));
    }
    return result;
  },

  /**
   * Get a client from the pool (for transactions).
   */
  async getClient() {
    return pool.connect();
  },

  /**
   * Log an audit event.
   */
  async audit(staffId, action, resourceType, resourceId, details, ipAddress) {
    try {
      await pool.query(
        'INSERT INTO audit_log (staff_id, action, resource_type, resource_id, details, ip_address) VALUES ($1, $2, $3, $4, $5, $6)',
        [staffId, action, resourceType, resourceId, details ? JSON.stringify(details) : null, ipAddress]
      );
    } catch (err) {
      console.warn('Audit log failed:', err.message);
    }
  },

  /**
   * Gracefully close pool.
   */
  async close() {
    await pool.end();
  },
};
