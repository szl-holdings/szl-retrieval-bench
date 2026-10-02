"""Hash-chained run receipts. UNSIGNED_HONEST: proves integrity + order, not identity."""
import hashlib
import json
import time

# Both paths preserve this organ's UTF-8 canonical bytes for valid JSON.
# An absent optional package uses the local implementation; a broken installed
# package must fail visibly rather than silently changing the implementation.
try:
    from szl_evidence_core.canonical import CANON_UTF8 as _CANON_PROFILE
    from szl_evidence_core.canonical import canonical_json as _shared_canonical_json
except ModuleNotFoundError as exc:
    if exc.name != "szl_evidence_core":
        raise
    _CANON_PROFILE = "szl.lambda/v1"
    _shared_canonical_json = None


GENESIS = "0" * 64


def _canonical(obj):
    if _shared_canonical_json is not None:
        return _shared_canonical_json(obj, profile=_CANON_PROFILE, check=False)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


class ReceiptChain:
    def __init__(self):
        self.chain = []

    def emit(self, run_record):
        prev = self.chain[-1]["self_hash"] if self.chain else GENESIS
        body = {"prev_hash": prev, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "signature": "UNSIGNED_HONEST", "run": run_record}
        body["self_hash"] = hashlib.sha256(_canonical(body).encode()).hexdigest()
        self.chain.append(body)
        return body

    def verify(self):
        prev = GENESIS
        for r in self.chain:
            if r["prev_hash"] != prev:
                return False
            check = dict(r)
            h = check.pop("self_hash")
            if hashlib.sha256(_canonical(check).encode()).hexdigest() != h:
                return False
            prev = h
        return True
