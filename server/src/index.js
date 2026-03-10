/**
 * FacialAI Store Platform — API Server
 *
 * REST API for managing patients, FaceDNA records, treatment predictions,
 * consultations, and store operations.
 *
 * Stack: Fastify + PostgreSQL
 * Privacy: FaceDNA-only storage (no raw photos), per-patient encryption, audit trail
 */

import Fastify from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import jwt from '@fastify/jwt';
import { patientRoutes } from './routes/patients.js';
import { facednaRoutes } from './routes/facedna.js';
import { consultationRoutes } from './routes/consultations.js';
import { authRoutes } from './routes/auth.js';
import { storeRoutes } from './routes/stores.js';
import { staffRoutes } from './routes/staff.js';
import { auditRoutes } from './routes/audit.js';
import { db } from './models/db.js';

// ---------------------------------------------------------------------------
// Server Setup
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3335;
const HOST = process.env.HOST || '127.0.0.1';
const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-change-in-production';

const app = Fastify({
  logger: {
    level: process.env.LOG_LEVEL || 'info',
    transport: process.env.NODE_ENV !== 'production'
      ? { target: 'pino-pretty', options: { colorize: true } }
      : undefined,
  },
});

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------

await app.register(cors, {
  origin: process.env.CORS_ORIGIN || 'http://localhost:3334',
  credentials: true,
});

await app.register(multipart, {
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max for FaceDNA uploads
    files: 1,
  },
});

await app.register(jwt, {
  secret: JWT_SECRET,
  sign: { expiresIn: '8h' },
});

// ---------------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------------

app.decorate('authenticate', async (request, reply) => {
  try {
    await request.jwtVerify();
  } catch (err) {
    reply.status(401).send({ error: 'Unauthorized', message: 'Valid token required' });
  }
});

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get('/api/v1/health', async () => ({
  status: 'ok',
  version: '1.0.0',
  timestamp: new Date().toISOString(),
}));

await app.register(authRoutes, { prefix: '/api/v1/auth' });
await app.register(patientRoutes, { prefix: '/api/v1/patients' });
await app.register(facednaRoutes, { prefix: '/api/v1/facedna' });
await app.register(consultationRoutes, { prefix: '/api/v1/consultations' });
await app.register(storeRoutes, { prefix: '/api/v1/stores' });
await app.register(staffRoutes, { prefix: '/api/v1/staff' });
await app.register(auditRoutes, { prefix: '/api/v1/audit' });

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

try {
  await db.init();
  await app.listen({ port: PORT, host: HOST });
  console.log(`\n  FacialAI API Server running at http://${HOST}:${PORT}`);
  console.log(`  Health check: http://${HOST}:${PORT}/api/v1/health\n`);
} catch (err) {
  app.log.error(err);
  process.exit(1);
}

export default app;
