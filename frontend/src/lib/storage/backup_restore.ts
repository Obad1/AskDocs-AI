// Encrypted, compressed workspace backup/restore (spec §6.4).
// .askdocs files: fflate DEFLATE + WebCrypto AES-GCM (local_encryption.ts).

import { compress, decompress } from "fflate";
import {
  encryptPayload,
  decryptPayload,
  sha256Hex,
} from "./local_encryption";
import { db } from "./indexeddb";
import type { AskDocsWorkspace } from "../../types/schema";

interface BackupEnvelope {
  magic: "ASKDOCS1";
  salt: number[];
  iv: number[];
  sha256: string;
  ciphertext: number[];
}

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}
function fromBase64(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function createBackup(
  workspace: AskDocsWorkspace,
  passphrase: string,
): Promise<Blob> {
  const json = new TextEncoder().encode(JSON.stringify(workspace));
  const compressed = await new Promise<Uint8Array>((resolve, reject) =>
    compress(json, { level: 6 }, (err, data) =>
      err ? reject(err) : resolve(data),
    ),
  );
  const { ciphertext, salt, iv } = await encryptPayload(compressed, passphrase);
  const env: BackupEnvelope = {
    magic: "ASKDOCS1",
    salt: Array.from(salt),
    iv: Array.from(iv),
    sha256: await sha256Hex(compressed),
    ciphertext: Array.from(new Uint8Array(ciphertext)),
  };
  return new Blob([JSON.stringify(env)], { type: "application/octet-stream" });
}

export async function restoreBackup(
  file: Blob,
  passphrase: string,
): Promise<AskDocsWorkspace> {
  const text = await file.text();
  const env = JSON.parse(text) as BackupEnvelope;
  if (env.magic !== "ASKDOCS1") throw new Error("Invalid .askdocs backup");
  const compressed = await decryptPayload(
    new Uint8Array(env.ciphertext).buffer,
    passphrase,
    new Uint8Array(env.salt),
    new Uint8Array(env.iv),
  );
  const json = await new Promise<Uint8Array>((resolve, reject) =>
    decompress(compressed, (err, data) => (err ? reject(err) : resolve(data))),
  );
  const workspace = JSON.parse(new TextDecoder().decode(json)) as AskDocsWorkspace;
  await db.workspaces.put({ id: workspace.active_workspace, state: workspace });
  return workspace;
}
