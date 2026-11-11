"""Idempotent fictional AA01-AA04 cases, computed by the *actual* review engines.

Used only when AVAMALE_SEED_SYNTHETIC_CASES=1 and PostgreSQL is selected.
No customer uploads, fake review signoffs, outbound messages or external APIs.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .engine import run_product
from .postgres_store import PostgresStore, utc_now

ROOT = Path(__file__).resolve().parents[1] / "samples" / "creative_proof"
NAMESPACE = uuid.UUID("dff9c9e5-ecbf-4ef6-8103-7ce6baaa0104")

SAMPLES = {
    "AA01": ("Fictional security evidence review", {"data": "AA01_evidence_story.csv"}),
    "AA02": ("Fictional bank-to-ledger exception — NGN 21,500",
             {"bank": "AA02_bank_story.csv", "ledger": "AA02_ledger_story.csv"}),
    "AA03": ("Fictional freight discrepancy — 500 vs 480 cartons",
             {"data": "AA03_freight_story.csv"}),
    "AA04": ("Fictional payables review — 100 / 100 / 70 units",
             {"data": "AA04_payables_story.csv"}),
}

def demo_id(product: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "avamale-fictional-v1:" + product))

def seed_four_cases(db: PostgresStore) -> dict[str, str]:
    """Insert each generated case once. Repeated starts never duplicate records.

    A unique deterministic run id and transaction ensure that retries after
    interruptions safely resume without partial decisions or events.
    """
    if not isinstance(db, PostgresStore):
        raise RuntimeError("Persistent fictional cases require PostgreSQL")

    states = {}
    for product, (title, slots) in SAMPLES.items():
        identifier = demo_id(product)
        with db.conn() as conn:
            exists = conn.execute(
                "SELECT id FROM runs WHERE id=%s", (identifier,)
            ).fetchone()
        if exists:
            states[product] = "present"
            continue

        uploads = {slot: (name, (ROOT / name).read_bytes())
                   for slot, name in slots.items()}
        result = run_product(product, uploads, declared_synthetic=True,
                             as_of="2026-11-08", date_tolerance=0)
        findings = result["findings"]
        if not findings or not all(f.get("review_required") is True for f in findings):
            raise RuntimeError(f"{product} fictional case has invalid findings")
        stamp = utc_now()
        with db.conn() as conn:
            inserted = conn.execute(
                """INSERT INTO runs
                     (id,product,title,created_at,updated_at,input_hashes,
                      results,source_names,synthetic,state)
                     VALUES (%s,%s,%s,%s,%s,%s,%s,%s,1,'in_review')
                     ON CONFLICT (id) DO NOTHING RETURNING id""",
                (identifier, product, "FICTIONAL DEMO | " + title, stamp, stamp,
                 json.dumps(result["hashes"]),
                 json.dumps(findings, default=str),
                 json.dumps(result["names"])),
            ).fetchone()
            if inserted:
                for index in range(len(findings)):
                    conn.execute(
                        """INSERT INTO decisions
                           (run_id,finding_index,state,note,reviewed_by,updated_at)
                           VALUES (%s,%s,'pending','','',%s)""",
                        (identifier, index, stamp),
                    )
                conn.execute(
                    """INSERT INTO events
                       (run_id,finding_index,action,actor,detail,at)
                       VALUES (%s,NULL,'created','synthetic-seeder',%s,%s)""",
                    (identifier,
                     "Fictional source CSV processed by the real review engine. "
                     "All findings require a genuine human reviewer; no decisions approved.",
                     stamp),
                )
        states[product] = "inserted" if inserted else "present"
    return states
