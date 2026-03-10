/**
 * Store Routes — Manage clinic stores and view dashboard stats.
 */

import { db } from '../models/db.js';

export async function storeRoutes(app) {
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/stores/:id/stats
   * Dashboard statistics for a store.
   */
  app.get('/:id/stats', async (request, reply) => {
    const { id } = request.params;

    // Verify staff belongs to this store (or is admin)
    if (request.user.store_id !== id && request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Access denied to this store' });
    }

    const [patients, consultations, predictions, recentConsultations] = await Promise.all([
      db.query('SELECT COUNT(*) FROM patients WHERE store_id = $1', [id]),
      db.query('SELECT COUNT(*), status FROM consultations WHERE store_id = $1 GROUP BY status', [id]),
      db.query(
        `SELECT p.treatment_type, COUNT(*) as count
         FROM predictions p
         JOIN consultations c ON p.consultation_id = c.id
         WHERE c.store_id = $1
         GROUP BY p.treatment_type
         ORDER BY count DESC
         LIMIT 10`,
        [id]
      ),
      db.query(
        `SELECT c.*, s.name as staff_name
         FROM consultations c
         LEFT JOIN staff s ON c.staff_id = s.id
         WHERE c.store_id = $1
         ORDER BY c.started_at DESC LIMIT 10`,
        [id]
      ),
    ]);

    const consultationsByStatus = {};
    for (const row of consultations.rows) {
      consultationsByStatus[row.status] = parseInt(row.count);
    }

    return {
      store_id: id,
      totalPatients: parseInt(patients.rows[0].count),
      consultations: consultationsByStatus,
      popularTreatments: predictions.rows,
      recentConsultations: recentConsultations.rows,
    };
  });

  /**
   * GET /api/v1/stores
   * List all stores (admin only).
   */
  app.get('/', async (request, reply) => {
    if (request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Admin access required' });
    }

    const result = await db.query(
      'SELECT id, name, address, phone, created_at FROM stores ORDER BY name'
    );
    return { stores: result.rows };
  });

  /**
   * POST /api/v1/stores
   * Create a new store (admin only).
   */
  app.post('/', async (request, reply) => {
    if (request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Admin access required' });
    }

    const { name, address, phone } = request.body || {};
    if (!name) {
      return reply.status(400).send({ error: 'Store name required' });
    }

    const result = await db.query(
      'INSERT INTO stores (name, address, phone) VALUES ($1, $2, $3) RETURNING *',
      [name, address || null, phone || null]
    );

    await db.audit(request.user.id, 'create_store', 'store', result.rows[0].id, { name }, request.ip);
    return reply.status(201).send(result.rows[0]);
  });
}
