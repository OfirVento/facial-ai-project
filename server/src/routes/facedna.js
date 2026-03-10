/**
 * FaceDNA Routes — Upload, download, and manage FaceDNA records.
 *
 * FaceDNA is the core data format: geometry (FLAME params), clinical
 * measurements, capture metadata, and treatment history.
 *
 * Raw photos are NEVER stored — only FaceDNA + albedo texture PNG.
 */

import { db } from '../models/db.js';

export async function facednaRoutes(app) {
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/facedna/:patientId
   * List FaceDNA versions for a patient.
   */
  app.get('/:patientId', async (request) => {
    const { patientId } = request.params;
    const storeId = request.user.store_id;

    // Verify patient belongs to store
    const patientCheck = await db.query(
      'SELECT id FROM patients WHERE id = $1 AND store_id = $2',
      [patientId, storeId]
    );
    if (patientCheck.rows.length === 0) {
      return { versions: [] };
    }

    const result = await db.query(
      `SELECT id, version, quality_score, file_size, created_at
       FROM facedna WHERE patient_id = $1
       ORDER BY version DESC`,
      [patientId]
    );

    return { versions: result.rows };
  });

  /**
   * GET /api/v1/facedna/:patientId/:facednaId
   * Get a specific FaceDNA record (full JSON data).
   */
  app.get('/:patientId/:facednaId', async (request, reply) => {
    const { patientId, facednaId } = request.params;
    const storeId = request.user.store_id;

    const result = await db.query(
      `SELECT f.* FROM facedna f
       JOIN patients p ON f.patient_id = p.id
       WHERE f.id = $1 AND f.patient_id = $2 AND p.store_id = $3`,
      [facednaId, patientId, storeId]
    );

    if (result.rows.length === 0) {
      return reply.status(404).send({ error: 'FaceDNA record not found' });
    }

    await db.audit(request.user.id, 'view_facedna', 'facedna', facednaId, null, request.ip);

    return result.rows[0];
  });

  /**
   * POST /api/v1/facedna/:patientId
   * Upload a new FaceDNA version for a patient.
   * Body: FaceDNA JSON
   */
  app.post('/:patientId', async (request, reply) => {
    const { patientId } = request.params;
    const storeId = request.user.store_id;
    const staffId = request.user.id;

    // Verify patient belongs to store
    const patientCheck = await db.query(
      'SELECT id FROM patients WHERE id = $1 AND store_id = $2',
      [patientId, storeId]
    );
    if (patientCheck.rows.length === 0) {
      return reply.status(404).send({ error: 'Patient not found' });
    }

    const facednaData = request.body;
    if (!facednaData || !facednaData.geometry) {
      return reply.status(400).send({ error: 'Valid FaceDNA JSON required (must include geometry)' });
    }

    // Get next version number
    const versionResult = await db.query(
      'SELECT COALESCE(MAX(version), 0) + 1 as next_version FROM facedna WHERE patient_id = $1',
      [patientId]
    );
    const nextVersion = versionResult.rows[0].next_version;

    // Extract quality score from capture data
    const qualityScore = facednaData.capture?.views?.[0]?.qualityScore ?? null;

    // Calculate approximate data size
    const fileSize = JSON.stringify(facednaData).length;

    const result = await db.query(
      `INSERT INTO facedna (patient_id, version, data, quality_score, file_size, created_by)
       VALUES ($1, $2, $3, $4, $5, $6)
       RETURNING id, version, created_at`,
      [patientId, nextVersion, JSON.stringify(facednaData), qualityScore, fileSize, staffId]
    );

    await db.audit(staffId, 'create_facedna', 'facedna', result.rows[0].id,
      { patient_id: patientId, version: nextVersion }, request.ip);

    return reply.status(201).send(result.rows[0]);
  });

  /**
   * DELETE /api/v1/facedna/:patientId/:facednaId
   * Delete a specific FaceDNA version.
   */
  app.delete('/:patientId/:facednaId', async (request, reply) => {
    const { patientId, facednaId } = request.params;
    const storeId = request.user.store_id;

    const check = await db.query(
      `SELECT f.id FROM facedna f
       JOIN patients p ON f.patient_id = p.id
       WHERE f.id = $1 AND f.patient_id = $2 AND p.store_id = $3`,
      [facednaId, patientId, storeId]
    );

    if (check.rows.length === 0) {
      return reply.status(404).send({ error: 'FaceDNA record not found' });
    }

    await db.query('DELETE FROM facedna WHERE id = $1', [facednaId]);
    await db.audit(request.user.id, 'delete_facedna', 'facedna', facednaId, null, request.ip);

    return { id: facednaId, deleted: true };
  });
}
