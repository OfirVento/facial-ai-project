/**
 * Encryption utilities for patient PII.
 * AES-256-GCM with per-record initialization vectors.
 */

import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16;
const AUTH_TAG_LENGTH = 16;

/**
 * Encrypt a string using AES-256-GCM.
 *
 * @param {string} plaintext - Data to encrypt
 * @param {string} key - 32-byte key (or will be derived from shorter string)
 * @returns {{ encrypted: Buffer, iv: Buffer }}
 */
export function encrypt(plaintext, key) {
  // Derive a 32-byte key from the provided key
  const derivedKey = crypto.createHash('sha256').update(key).digest();
  const iv = crypto.randomBytes(IV_LENGTH);

  const cipher = crypto.createCipheriv(ALGORITHM, derivedKey, iv);
  let encrypted = cipher.update(plaintext, 'utf8');
  encrypted = Buffer.concat([encrypted, cipher.final()]);

  // Append auth tag to encrypted data
  const authTag = cipher.getAuthTag();
  const result = Buffer.concat([encrypted, authTag]);

  return { encrypted: result, iv };
}

/**
 * Decrypt a buffer using AES-256-GCM.
 *
 * @param {Buffer} encryptedData - Encrypted data (includes auth tag at end)
 * @param {Buffer} iv - Initialization vector
 * @param {string} key - Same key used for encryption
 * @returns {string} Decrypted plaintext
 */
export function decrypt(encryptedData, iv, key) {
  const derivedKey = crypto.createHash('sha256').update(key).digest();

  // Extract auth tag from end of encrypted data
  const authTag = encryptedData.subarray(encryptedData.length - AUTH_TAG_LENGTH);
  const encrypted = encryptedData.subarray(0, encryptedData.length - AUTH_TAG_LENGTH);

  const decipher = crypto.createDecipheriv(ALGORITHM, derivedKey, iv);
  decipher.setAuthTag(authTag);

  let decrypted = decipher.update(encrypted);
  decrypted = Buffer.concat([decrypted, decipher.final()]);

  return decrypted.toString('utf8');
}
