/**
 * Consultation Routes — Manage consultation sessions and treatment predictions.
 */

import { db } from '../models/db.js';

export async function consultationRoutes(app) {
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/consultations
   * List consultations for the current store.
   * Query: ?patient_id=uuid&status=active&limit=50&offset=0
   */
  app.get('/', async (request) => {
    const storeId = request.user.store_id;
    const limit = Math.min(parseInt(request.query.limit) || 50, 100);
    const offset = parseInt(request.query.offset) || 0;
    const status = request.query.status;
    const patientId = request.query.patient_id;

    let sql = `SELECT c.*, s.name as staff_name
               FROM consultations c
               LEFT JOIN staff s ON c.staff_id = s.id
               WHERE c.store_id = $1`;
    const params = [storeId];
    let paramIdx = 2;

    if (status) {
      sql += ` AND c.status = $${paramIdx}`;
      params.push(status);
      paramIdx++;
    }

    if (patientId) {
      sql += ` AND c.patient_id = $${paramIdx}`;
      params.push(patientId);
      paramIdx++;
    }

    sql += ` ORDER BY c.started_at DESC LIMIT $${paramIdx} OFFSET $${paramIdx + 1}`;
    params.push(limit, offset);

    const result = await db.query(sql, params);
    return { consultations: result.rows };
  });

  /**
   * POST /api/v1/consultations
   * Start a new consultation.
   * Body: { patient_id, facedna_id?, notes? }
   */
  app.post('/', async (request, reply) => {
    const storeId = request.user.store_id;
    const staffId = request.user.id;
    const { patient_id, facedna_id, notes } = request.body || {};

    if (!patient_id) {
      return reply.status(400).send({ error: 'patient_id required' });
    }

    const result = await db.query(
      `INSERT INTO consultations (patient_id, store_id, staff_id, facedna_id, notes)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING *`,
      [patient_id, storeId, staffId, facedna_id || null, notes || null]
    );

    await db.audit(staffId, 'start_consultation', 'consultation', result.rows[0].id,
      { patient_id }, request.ip);

    return reply.status(201).send(result.rows[0]);
  });

  /**
   * PUT /api/v1/consultations/:id/complete
   * Mark a consultation as completed.
   */
  app.put('/:id/complete', async (request) => {
    const { id } = request.params;
    const { notes } = request.body || {};

    await db.query(
      `UPDATE consultations SET status = 'completed', completed_at = NOW(), notes = COALESCE($2, notes)
       WHERE id = $1`,
      [id, notes]
    );

    await db.audit(request.user.id, 'complete_consultation', 'consultation', id, null, request.ip);
    return { id, status: 'completed' };
  });

  /**
   * POST /api/v1/consultations/:id/predictions
   * Record a treatment prediction.
   * Body: { treatment_type, parameters, morph_state, confidence }
   */
  app.post('/:id/predictions', async (request, reply) => {
    const { id } = request.params;
    const { treatment_type, parameters, morph_state, confidence } = request.body || {};

    if (!treatment_type || !parameters) {
      return reply.status(400).send({ error: 'treatment_type and parameters required' });
    }

    const result = await db.query(
      `INSERT INTO predictions (consultation_id, treatment_type, parameters, morph_state, confidence)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING *`,
      [id, treatment_type, JSON.stringify(parameters), morph_state ? JSON.stringify(morph_state) : null, confidence]
    );

    await db.audit(request.user.id, 'create_prediction', 'prediction', result.rows[0].id,
      { consultation_id: id, treatment_type }, request.ip);

    return reply.status(201).send(result.rows[0]);
  });

  /**
   * GET /api/v1/consultations/:id/predictions
   * Get predictions for a consultation.
   */
  app.get('/:id/predictions', async (request) => {
    const { id } = request.params;
    const result = await db.query(
      'SELECT * FROM predictions WHERE consultation_id = $1 ORDER BY created_at',
      [id]
    );
    return { predictions: result.rows };
  });
}
