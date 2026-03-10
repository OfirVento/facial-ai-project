/**
 * Audit Routes — View security and access audit log.
 */

import { db } from '../models/db.js';

export async function auditRoutes(app) {
  app.addHook('preHandler', app.authenticate);

  /**
   * GET /api/v1/audit
   * List recent audit log entries. Admin/manager only.
   * Query params: limit (default 100), action, resource_type, staff_id
   */
  app.get('/', async (request, reply) => {
    const { role } = request.user;
    if (role !== 'admin' && role !== 'manager') {
      return reply.status(403).send({ error: 'Admin or manager role required' });
    }

    const limit = Math.min(parseInt(request.query.limit) || 100, 500);
    const conditions = [];
    const params = [];
    let paramIndex = 1;

    if (request.query.action) {
      conditions.push(`a.action = $${paramIndex++}`);
      params.push(request.query.action);
    }
    if (request.query.resource_type) {
      conditions.push(`a.resource_type = $${paramIndex++}`);
      params.push(request.query.resource_type);
    }
    if (request.query.staff_id) {
      conditions.push(`a.staff_id = $${paramIndex++}`);
      params.push(request.query.staff_id);
    }

    const whereClause = conditions.length > 0 ? 'WHERE ' + conditions.join(' AND ') : '';

    const result = await db.query(
      `SELECT a.*, s.name as staff_name, s.email as staff_email
       FROM audit_log a
       LEFT JOIN staff s ON a.staff_id = s.id
       ${whereClause}
       ORDER BY a.created_at DESC
       LIMIT $${paramIndex}`,
      [...params, limit]
    );

    return { entries: result.rows, count: result.rows.length };
  });

  /**
   * GET /api/v1/audit/summary
   * Aggregated audit stats for the dashboard.
   */
  app.get('/summary', async (request, reply) => {
    if (request.user.role !== 'admin' && request.user.role !== 'manager') {
      return reply.status(403).send({ error: 'Admin or manager role required' });
    }

    const [actionCounts, recentCount, uniqueStaff] = await Promise.all([
      db.query(
        `SELECT action, COUNT(*) as count
         FROM audit_log
         WHERE created_at > NOW() - INTERVAL '30 days'
         GROUP BY action
         ORDER BY count DESC`
      ),
      db.query(
        `SELECT COUNT(*) FROM audit_log WHERE created_at > NOW() - INTERVAL '24 hours'`
      ),
      db.query(
        `SELECT COUNT(DISTINCT staff_id) FROM audit_log WHERE created_at > NOW() - INTERVAL '24 hours'`
      ),
    ]);

    return {
      last30Days: actionCounts.rows,
      last24Hours: {
        totalActions: parseInt(recentCount.rows[0].count),
        activeStaff: parseInt(uniqueStaff.rows[0].count),
      },
    };
  });
}
