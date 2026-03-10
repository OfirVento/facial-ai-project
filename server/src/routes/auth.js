/**
 * Auth Routes — Staff authentication (login/register).
 */

import { db } from '../models/db.js';
import bcrypt from 'bcryptjs';

export async function authRoutes(app) {
  /**
   * POST /api/v1/auth/login
   * Body: { email, password }
   * Returns: { token, staff: { id, name, email, role, store_id } }
   */
  app.post('/login', async (request, reply) => {
    const { email, password } = request.body || {};
    if (!email || !password) {
      return reply.status(400).send({ error: 'Email and password required' });
    }

    const result = await db.query(
      'SELECT id, store_id, email, password_hash, name, role FROM staff WHERE email = $1 AND active = true',
      [email.toLowerCase()]
    );

    if (result.rows.length === 0) {
      return reply.status(401).send({ error: 'Invalid credentials' });
    }

    const staff = result.rows[0];
    const validPassword = await bcrypt.compare(password, staff.password_hash);
    if (!validPassword) {
      return reply.status(401).send({ error: 'Invalid credentials' });
    }

    const token = app.jwt.sign({
      id: staff.id,
      email: staff.email,
      role: staff.role,
      store_id: staff.store_id,
    });

    await db.audit(staff.id, 'login', 'staff', staff.id, null, request.ip);

    return {
      token,
      staff: {
        id: staff.id,
        name: staff.name,
        email: staff.email,
        role: staff.role,
        store_id: staff.store_id,
      },
    };
  });

  /**
   * POST /api/v1/auth/register
   * Body: { email, password, name, store_id, role? }
   * Requires: admin role (via JWT)
   */
  app.post('/register', { preHandler: [app.authenticate] }, async (request, reply) => {
    if (request.user.role !== 'admin') {
      return reply.status(403).send({ error: 'Admin role required to register staff' });
    }

    const { email, password, name, store_id, role } = request.body || {};
    if (!email || !password || !name || !store_id) {
      return reply.status(400).send({ error: 'email, password, name, and store_id required' });
    }

    const passwordHash = await bcrypt.hash(password, 12);

    try {
      const result = await db.query(
        'INSERT INTO staff (email, password_hash, name, store_id, role) VALUES ($1, $2, $3, $4, $5) RETURNING id, email, name, role',
        [email.toLowerCase(), passwordHash, name, store_id, role || 'practitioner']
      );

      await db.audit(request.user.id, 'register_staff', 'staff', result.rows[0].id, { email }, request.ip);
      return reply.status(201).send(result.rows[0]);
    } catch (err) {
      if (err.code === '23505') {
        return reply.status(409).send({ error: 'Email already registered' });
      }
      throw err;
    }
  });

  /**
   * GET /api/v1/auth/me
   * Returns current user info from JWT
   */
  app.get('/me', { preHandler: [app.authenticate] }, async (request) => {
    const result = await db.query(
      'SELECT id, store_id, email, name, role FROM staff WHERE id = $1',
      [request.user.id]
    );
    return result.rows[0] || null;
  });
}
