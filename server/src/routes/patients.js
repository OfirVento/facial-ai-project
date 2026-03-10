/**
 * Patient Routes — CRUD for patient profiles.
 *
 * Patient PII is encrypted at the application level (AES-256-GCM).
 * The database stores only encrypted blobs + IVs.
 */

import { db } from '../models/db.js';
import { encrypt, decrypt } from '../utils/crypto.js';

export async function patientRoutes(app) {
  // All patient routes require authentication
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/patients
   * List patients for the staff member's store.
   * Query: ?search=name&limit=50&offset=0
   */
  app.get('/', async (request) => {
    const storeId = request.user.store_id;
    const limit = Math.min(parseInt(request.query.limit) || 50, 100);
    const offset = parseInt(request.query.offset) || 0;

    const result = await db.query(
      `SELECT id, created_at, updated_at FROM patients
       WHERE store_id = $1
       ORDER BY updated_at DESC
       LIMIT $2 OFFSET $3`,
      [storeId, limit, offset]
    );

    // Decrypt patient data for each row
    const patients = result.rows.map(row => {
      // Note: in production, decrypt patient data here
      // For now, return metadata only
      return {
        id: row.id,
        created_at: row.created_at,
        updated_at: row.updated_at,
      };
    });

    const countResult = await db.query(
      'SELECT COUNT(*) FROM patients WHERE store_id = $1',
      [storeId]
    );

    return {
      patients,
      total: parseInt(countResult.rows[0].count),
      limit,
      offset,
    };
  });

  /**
   * GET /api/v1/patients/:id
   * Get a single patient with decrypted profile.
   */
  app.get('/:id', async (request, reply) => {
    const { id } = request.params;
    const storeId = request.user.store_id;

    const result = await db.query(
      'SELECT * FROM patients WHERE id = $1 AND store_id = $2',
      [id, storeId]
    );

    if (result.rows.length === 0) {
      return reply.status(404).send({ error: 'Patient not found' });
    }

    const row = result.rows[0];
    let profile = {};

    if (row.encrypted_data && row.encryption_iv) {
      try {
        const key = process.env.ENCRYPTION_KEY || 'dev-encryption-key-32b!';
        profile = JSON.parse(decrypt(row.encrypted_data, row.encryption_iv, key));
      } catch {
        profile = { error: 'Could not decrypt patient data' };
      }
    }

    await db.audit(request.user.id, 'view_patient', 'patient', id, null, request.ip);

    return {
      id: row.id,
      profile,
      created_at: row.created_at,
      updated_at: row.updated_at,
    };
  });

  /**
   * POST /api/v1/patients
   * Create a new patient.
   * Body: { name, email?, phone?, dob?, notes? }
   */
  app.post('/', async (request, reply) => {
    const storeId = request.user.store_id;
    const staffId = request.user.id;
    const patientData = request.body || {};

    if (!patientData.name) {
      return reply.status(400).send({ error: 'Patient name required' });
    }

    const key = process.env.ENCRYPTION_KEY || 'dev-encryption-key-32b!';
    const { encrypted, iv } = encrypt(JSON.stringify(patientData), key);

    const result = await db.query(
      `INSERT INTO patients (store_id, encrypted_data, encryption_iv, created_by)
       VALUES ($1, $2, $3, $4)
       RETURNING id, created_at`,
      [storeId, encrypted, iv, staffId]
    );

    await db.audit(staffId, 'create_patient', 'patient', result.rows[0].id, { name: patientData.name }, request.ip);

    return reply.status(201).send({
      id: result.rows[0].id,
      created_at: result.rows[0].created_at,
    });
  });

  /**
   * PUT /api/v1/patients/:id
   * Update patient profile.
   */
  app.put('/:id', async (request, reply) => {
    const { id } = request.params;
    const storeId = request.user.store_id;
    const patientData = request.body || {};

    // Verify patient belongs to store
    const check = await db.query(
      'SELECT id FROM patients WHERE id = $1 AND store_id = $2',
      [id, storeId]
    );

    if (check.rows.length === 0) {
      return reply.status(404).send({ error: 'Patient not found' });
    }

    const key = process.env.ENCRYPTION_KEY || 'dev-encryption-key-32b!';
    const { encrypted, iv } = encrypt(JSON.stringify(patientData), key);

    await db.query(
      'UPDATE patients SET encrypted_data = $1, encryption_iv = $2, updated_at = NOW() WHERE id = $3',
      [encrypted, iv, id]
    );

    await db.audit(request.user.id, 'update_patient', 'patient', id, null, request.ip);
    return { id, updated: true };
  });

  /**
   * DELETE /api/v1/patients/:id
   * Full deletion (GDPR compliance).
   * Cascades to facedna, consultations, predictions.
   */
  app.delete('/:id', async (request, reply) => {
    const { id } = request.params;
    const storeId = request.user.store_id;

    const check = await db.query(
      'SELECT id FROM patients WHERE id = $1 AND store_id = $2',
      [id, storeId]
    );

    if (check.rows.length === 0) {
      return reply.status(404).send({ error: 'Patient not found' });
    }

    await db.query('DELETE FROM patients WHERE id = $1', [id]);
    await db.audit(request.user.id, 'delete_patient', 'patient', id, null, request.ip);

    return { id, deleted: true };
  });
}
