/**
 * Staff Routes — List and manage staff members.
 */

import { db } from '../models/db.js';

export async function staffRoutes(app) {
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/staff
   * List all staff. Admins see all, managers see own store only.
   */
  app.get('/', async (request, reply) => {
    const { role, store_id } = request.user;

    let result;
    if (role === 'admin') {
      result = await db.query(
        `SELECT s.id, s.store_id, s.email, s.name, s.role, s.active, s.created_at,
                st.name as store_name
         FROM staff s
         LEFT JOIN stores st ON s.store_id = st.id
         ORDER BY s.name`
      );
    } else if (role === 'manager') {
      result = await db.query(
        `SELECT s.id, s.store_id, s.email, s.name, s.role, s.active, s.created_at,
                st.name as store_name
         FROM staff s
         LEFT JOIN stores st ON s.store_id = st.id
         WHERE s.store_id = $1
         ORDER BY s.name`,
        [store_id]
      );
    } else {
      return reply.status(403).send({ error: 'Manager or admin role required' });
    }

    return { staff: result.rows };
  });

  /**
   * PUT /api/v1/staff/:id/deactivate
   * Deactivate a staff member (admin only).
   */
  app.put('/:id/deactivate', async (request, reply) => {
    if (request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Admin role required' });
    }

    const { id } = request.params;

    // Prevent self-deactivation
    if (id === request.user.id) {
      return reply.status(400).send({ error: 'Cannot deactivate your own account' });
    }

    const result = await db.query(
      'UPDATE staff SET active = false, updated_at = NOW() WHERE id = $1 RETURNING id, name, email, active',
      [id]
    );

    if (result.rows.length === 0) {
      return reply.status(404).send({ error: 'Staff member not found' });
    }

    await db.audit(request.user.id, 'deactivate_staff', 'staff', id, null, request.ip);
    return result.rows[0];
  });

  /**
   * PUT /api/v1/staff/:id/activate
   * Re-activate a staff member (admin only).
   */
  app.put('/:id/activate', async (request, reply) => {
    if (request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Admin role required' });
    }

    const { id } = request.params;
    const result = await db.query(
      'UPDATE staff SET active = true, updated_at = NOW() WHERE id = $1 RETURNING id, name, email, active',
      [id]
    );

    if (result.rows.length === 0) {
      return reply.status(404).send({ error: 'Staff member not found' });
    }

    await db.audit(request.user.id, 'activate_staff', 'staff', id, null, request.ip);
    return result.rows[0];
  });
}
