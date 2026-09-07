// Local encryption for .askdocs backups (spec §6.4). AES-GCM via Web Crypto,
// key derived from a user passphrase with PBKDF2. Never transmitted anywhere.

const enc = new TextEncoder();
const dec = new TextDecoder();

// TS DOM lib types Uint8Array.buffer as ArrayBufferLike (may be SharedArrayBuffer);
// Web Crypto wants a concrete ArrayBuffer. This cast satisfies the stricter lib.
const asBuf = (u: Uint8Array): BufferSource => u as unknown as BufferSource;

export async function deriveKey(
  passphrase: string,
  salt: Uint8Array,
): Promise<CryptoKey> {
  const baseKey = await crypto.subtle.importKey(
    "raw",
    asBuf(enc.encode(passphrase)),
    "PBKDF2",
    false,
    ["deriveKey"],
  );
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: asBuf(salt), iterations: 150_000, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
}

export async function encryptPayload(
  data: Uint8Array,
  passphrase: string,
): Promise<{ ciphertext: ArrayBuffer; salt: Uint8Array; iv: Uint8Array }> {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    asBuf(data),
  );
  return { ciphertext, salt, iv };
}

export async function decryptPayload(
  ciphertext: ArrayBuffer,
  passphrase: string,
  salt: Uint8Array,
  iv: Uint8Array,
): Promise<Uint8Array> {
  const key = await deriveKey(passphrase, salt);
  const plain = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: asBuf(iv) },
    key,
    ciphertext,
  );
  return new Uint8Array(plain);
}

export async function sha256Hex(data: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", asBuf(data));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}
