"""Offline synthetic fixture scorer. External performance is UNMEASURED.

fixture_exact_match_v1 compares NFC/casefold/whitespace-normalized strings for
equality, retaining punctuation and signs. Its receipts are unsigned manifests.
The guard can inject the exact compiled source bytes before module execution;
without that injection source identity is explicitly an unverified file snapshot.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
from types import SimpleNamespace
import unicodedata
import uuid

SCHEMA = "szl.fixture-benchmark/v2"
METRIC_ID = "fixture_exact_match_v1"
NORMALIZATION = "NFC; casefold; collapse Unicode whitespace; retain punctuation/signs"
MAX_INPUT_BYTES = 1 << 20
MAX_ROWS = 1000
MAX_ARTIFACT_BYTES = 8 << 20
_SOURCE_BYTES = globals().get("__executed_source_bytes__")
_SOURCE_IDENTITY = "guard_compiled_bytes"
if _SOURCE_BYTES is None:
    _SOURCE_BYTES = Path(__file__).read_bytes()
    _SOURCE_IDENTITY = "module_file_snapshot_at_import_unverified"
_SOURCE_SHA256 = hashlib.sha256(_SOURCE_BYTES).hexdigest()
_COMPILED_SHA256 = globals().get("__executed_source_sha256__")
if _COMPILED_SHA256 is not None and _COMPILED_SHA256 != _SOURCE_SHA256:
    raise ValueError("guard source bytes/digest mismatch")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _json_loads(text):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number: " + value)
        return number
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject_constant,
                          parse_float=finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser limit") from exc


def parse_items(data, fmt):
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds byte limit")
    text = data.decode("utf-8")
    if fmt == "json":
        rows = _json_loads(text)
        if not isinstance(rows, list):
            raise ValueError("JSON input must be an array")
    elif fmt == "jsonl":
        rows = []
        for line_number, line in enumerate(text.split("\n"), 1):
            if line.strip():
                try:
                    rows.append(_json_loads(line))
                except ValueError as exc:
                    raise ValueError("invalid JSONL line %d: %s" % (line_number, exc)) from exc
    else:
        raise ValueError("unsupported input format")
    if len(rows) > MAX_ROWS:
        raise ValueError("input exceeds row limit")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every row must be an object")
    return rows


def _snapshot(path):
    path = Path(path)
    fmt = "jsonl" if path.suffix.lower() == ".jsonl" else "json"
    data = _read_bounded(path, MAX_INPUT_BYTES)
    digest = sha256_bytes(data)
    return data, fmt, parse_items(data, fmt), digest


def load_items(path):
    return _snapshot(path)[2]


def _n(value):
    if not isinstance(value, str):
        raise ValueError("answers and hypotheses must be strings")
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def judge_exact(question, gold, hypothesis):
    return bool(_n(gold)) and _n(gold) == _n(hypothesis)


def _nonblank(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")


def _id(row):
    value = row.get("question_id")
    _nonblank(value, "question_id")
    if value != value.strip() or len(value) > 256:
        raise ValueError("question_id must be unpadded and at most 256 characters")
    return value


def validate_rows(questions, predictions):
    if not questions or not predictions:
        raise ValueError("nonempty dataset and predictions are required")
    questions_by_id = {}
    for row in questions:
        qid = _id(row)
        if qid in questions_by_id:
            raise ValueError("duplicate dataset question_id: " + qid)
        _nonblank(row.get("question"), "question")
        _nonblank(row.get("answer"), "answer")
        _nonblank(row.get("question_type", "all"), "question_type")
        questions_by_id[qid] = row
    predictions_by_id = {}
    for row in predictions:
        qid = _id(row)
        if qid in predictions_by_id:
            raise ValueError("duplicate prediction question_id: " + qid)
        if qid not in questions_by_id:
            raise ValueError("unknown prediction question_id: " + qid)
        if "hypothesis" not in row or not isinstance(row["hypothesis"], str):
            raise ValueError("hypothesis must be an explicitly supplied string")
        predictions_by_id[qid] = row["hypothesis"]
    if not any(_n(hypothesis) for hypothesis in predictions_by_id.values()):
        raise ValueError("at least one matched nonempty hypothesis is required")
    return predictions_by_id


def _settings(a):
    if a.dataset_variant != "fixture" or a.judge not in ("exact", METRIC_ID):
        raise ValueError("only fixture/fixture_exact_match_v1 is supported")
    _nonblank(a.system, "system")
    _nonblank(a.system_version, "system_version")
    if a.top_k is not None and (type(a.top_k) is not int or a.top_k <= 0):
        raise ValueError("top_k must be a positive integer or null")
    if a.claimed is not None:
        if type(a.claimed) not in (int, float) or not math.isfinite(a.claimed) or not 0 <= a.claimed <= 100:
            raise ValueError("claimed must be a finite percentage from 0 to 100")
        _nonblank(a.claimed_source, "claimed_source")
    elif a.claimed_source is not None:
        raise ValueError("claimed_source requires claimed")


def _score(questions, predictions):
    by_type = collections.defaultdict(lambda: [0, 0])
    verdicts = []
    for row in questions:
        qid = row["question_id"]
        missing = qid not in predictions
        hypothesis = predictions.get(qid, "")
        empty = not missing and not _n(hypothesis)
        correct = not missing and not empty and judge_exact(row["question"], row["answer"], hypothesis)
        question_type = row.get("question_type", "all")
        by_type[question_type][0] += correct
        by_type[question_type][1] += 1
        verdicts.append({"question_id": qid, "question_type": question_type,
                         "correct": correct, "missing": missing, "empty": empty})
    total = len(verdicts)
    correct = sum(verdict["correct"] for verdict in verdicts)
    missing = sum(verdict["missing"] for verdict in verdicts)
    empty = sum(verdict["empty"] for verdict in verdicts)
    results = {"correct": correct, "total": total, "accuracy_pct": round(100 * correct / total, 2),
               "missing_predictions": missing, "empty_predictions": empty,
               "answered_predictions": total - missing - empty, "supplied_predictions": len(predictions),
               "coverage_fraction": (total - missing) / total,
               "by_question_type": {key: {"correct": counts[0], "total": counts[1]}
                                    for key, counts in sorted(by_type.items())}}
    return verdicts, results


def _serialize(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _write_new(path, data):
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_path(path):
    """Reject existing link/reparse components; parent must be caller-owned.

    This portable check is not protection against a concurrent hostile directory
    swap. Use a private local directory without untrusted writers.
    """
    absolute = Path(os.path.abspath(path))
    for part in [absolute, *absolute.parents]:
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if part.is_symlink() or (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise ValueError("path contains a symlink or reparse point")
    return absolute


def _read_bounded(path, limit):
    path = _safe_path(path)
    if not path.is_file():
        raise ValueError("input must be a regular file")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds byte limit")
    return data


def _safe_output_root(path):
    absolute = _safe_path(path)
    absolute.mkdir(parents=True, exist_ok=True)
    if not absolute.is_dir():
        raise ValueError("output root must be a directory")
    return absolute


def run(a):
    _settings(a)
    dataset_bytes, dataset_format, questions, dataset_digest = _snapshot(a.dataset)
    predictions_bytes, predictions_format, prediction_rows, predictions_digest = _snapshot(a.predictions)
    predictions = validate_rows(questions, prediction_rows)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    verdicts, results = _score(questions, predictions)
    verdict_bytes = b"".join(_serialize(verdict).replace(b"\n", b"") + b"\n" for verdict in verdicts)
    root = _safe_output_root(a.out)
    run_id = uuid.uuid4().hex
    directory = root / run_id
    directory.mkdir(mode=0o700, exist_ok=False)
    artifacts = {"dataset": ("dataset.snapshot", dataset_bytes), "predictions": ("predictions.snapshot", predictions_bytes),
                 "evaluator": ("evaluator.snapshot.py", _SOURCE_BYTES), "verdicts": ("verdicts.jsonl", verdict_bytes)}
    for name, data in artifacts.values():
        _write_new(directory / name, data)
    receipt = {"schema": SCHEMA, "status": "complete", "run_id": run_id,
               "external_performance": "UNMEASURED",
               "authentication": "none", "independent_verification": False,
               "started_at": started, "issued_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "metric": {"id": METRIC_ID, "normalization": NORMALIZATION, "unicode_version": unicodedata.unidata_version},
               "dataset": {"file": os.path.basename(a.dataset), "format": dataset_format, "variant": "fixture",
                           "n": len(questions), "sha256": dataset_digest,
                           "selected_ids_sha256": sha256_bytes(_serialize([row["question_id"] for row in questions]))},
               "system": {"name": a.system, "version": a.system_version,
                          "predictions_sha256": predictions_digest, "declared_metadata": True},
               "predictions": {"format": predictions_format},
               "judge": {"mode": METRIC_ID, "requested_mode": a.judge, "external_calls": 0},
               "retrieval": {"top_k": a.top_k, "status": "declared" if a.top_k is not None else "unknown_or_not_applicable"},
               "results": results, "verdicts_sha256": sha256_bytes(verdict_bytes),
               "evaluator": {"source_snapshot_sha256": _SOURCE_SHA256, "compiled_source_sha256": _COMPILED_SHA256,
                             "identity": _SOURCE_IDENTITY, "git_revision": "unknown", "dirty_state": "unknown"},
               "runtime": {"python": platform.python_version(), "dependencies": "Python standard library only"},
               "artifacts": {key: {"file": name, "sha256": sha256_bytes(data)} for key, (name, data) in artifacts.items()}}
    if a.claimed is not None:
        receipt["claimed"] = {"value_pct": a.claimed, "source": a.claimed_source,
                              "verified": False, "comparability": "not_established"}
    pending = directory / "receipt.pending.json"
    _write_new(pending, _serialize(receipt))
    os.replace(pending, directory / "receipt.json")
    print(json.dumps({"receipt": str(directory / "receipt.json"), "metric_id": METRIC_ID,
                      "accuracy_pct": results["accuracy_pct"], "missing": results["missing_predictions"]}))
    return receipt


def verify_receipt(path, expected_evaluator_sha256=None):
    """Check fixture consistency, optionally against an independently trusted evaluator digest.

    An unsigned coordinated rewrite can pass consistency checks. A supplied
    trusted digest checks source identity, not receipt authorship or execution.
    """
    path = Path(path)
    receipt = _json_loads(_read_bounded(path, MAX_ARTIFACT_BYTES).decode("utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("manifest must be an object")
    required = {"schema", "status", "run_id", "authentication", "independent_verification", "started_at", "issued_at",
                "metric", "dataset", "system", "predictions", "judge", "retrieval", "results", "verdicts_sha256",
                "evaluator", "runtime", "artifacts", "external_performance"}
    if not required <= set(receipt) or set(receipt) - required - {"claimed"}:
        raise ValueError("manifest fields mismatch")
    for key in ("metric", "dataset", "system", "predictions", "judge", "retrieval", "results", "evaluator", "runtime", "artifacts"):
        if not isinstance(receipt[key], dict):
            raise ValueError("manifest field must be an object: " + key)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "complete":
        raise ValueError("unsupported or incomplete manifest")
    if receipt["external_performance"] != "UNMEASURED":
        raise ValueError("unsupported external performance claim")
    if receipt.get("authentication") != "none" or receipt.get("independent_verification") is not False:
        raise ValueError("unsupported authentication claim")
    if receipt.get("metric") != {"id": METRIC_ID, "normalization": NORMALIZATION, "unicode_version": unicodedata.unidata_version}:
        raise ValueError("metric mismatch")
    run_id = receipt["run_id"]
    if not isinstance(run_id, str) or len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("invalid run_id")
    if run_id != path.parent.name or path.name != "receipt.json":
        raise ValueError("run path identity mismatch")
    timestamps = {}
    for key in ("started_at", "issued_at"):
        try:
            timestamp = datetime.datetime.fromisoformat(receipt[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid UTC timestamp") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() != datetime.timedelta(0):
            raise ValueError("timestamp must be UTC")
        timestamps[key] = timestamp
    if timestamps["issued_at"] < timestamps["started_at"]:
        raise ValueError("timestamp ordering mismatch")
    judge = receipt["judge"]
    if set(judge) != {"mode", "requested_mode", "external_calls"} or judge["mode"] != METRIC_ID or type(judge["external_calls"]) is not int or judge["external_calls"] != 0:
        raise ValueError("judge settings mismatch")
    system, retrieval = receipt["system"], receipt["retrieval"]
    if set(system) != {"name", "version", "predictions_sha256", "declared_metadata"} or system["declared_metadata"] is not True:
        raise ValueError("system settings mismatch")
    if set(retrieval) != {"top_k", "status"} or retrieval["status"] != ("declared" if retrieval["top_k"] is not None else "unknown_or_not_applicable"):
        raise ValueError("retrieval settings mismatch")
    claimed = receipt.get("claimed")
    if "claimed" in receipt and (not isinstance(claimed, dict) or set(claimed) != {"value_pct", "source", "verified", "comparability"} or claimed["verified"] is not False or claimed["comparability"] != "not_established"):
        raise ValueError("claimed score settings mismatch")
    _settings(SimpleNamespace(dataset_variant=receipt["dataset"].get("variant"), judge=judge["requested_mode"],
                              system=system["name"], system_version=system["version"], top_k=retrieval["top_k"],
                              claimed=None if claimed is None else claimed["value_pct"], claimed_source=None if claimed is None else claimed["source"]))
    if set(receipt["dataset"]) != {"file", "format", "variant", "n", "sha256", "selected_ids_sha256"} or type(receipt["dataset"]["n"]) is not int:
        raise ValueError("dataset metadata mismatch")
    _nonblank(receipt["dataset"]["file"], "dataset file")
    if set(receipt["predictions"]) != {"format"}:
        raise ValueError("prediction metadata mismatch")
    evaluator = receipt["evaluator"]
    if set(evaluator) != {"source_snapshot_sha256", "compiled_source_sha256", "identity", "git_revision", "dirty_state"}:
        raise ValueError("evaluator metadata mismatch")
    if evaluator["identity"] not in ("guard_compiled_bytes", "module_file_snapshot_at_import_unverified") or evaluator["git_revision"] != "unknown" or evaluator["dirty_state"] != "unknown":
        raise ValueError("evaluator identity mismatch")
    if (evaluator["compiled_source_sha256"] is not None) != (evaluator["identity"] == "guard_compiled_bytes"):
        raise ValueError("compiled identity claim mismatch")
    if set(receipt["runtime"]) != {"python", "dependencies"} or receipt["runtime"]["dependencies"] != "Python standard library only":
        raise ValueError("runtime metadata mismatch")
    _nonblank(receipt["runtime"]["python"], "runtime python")
    expected_names = {"dataset": "dataset.snapshot", "predictions": "predictions.snapshot",
                      "evaluator": "evaluator.snapshot.py", "verdicts": "verdicts.jsonl"}
    if set(receipt["artifacts"]) != set(expected_names):
        raise ValueError("unexpected artifact set")
    snapshots = {}
    for key, name in expected_names.items():
        artifact = receipt["artifacts"][key]
        if not isinstance(artifact, dict) or set(artifact) != {"file", "sha256"} or artifact["file"] != name:
            raise ValueError("artifact path mismatch")
        local_path = path.parent / name
        limit = MAX_INPUT_BYTES if key in ("dataset", "predictions") else MAX_ARTIFACT_BYTES
        snapshots[key] = _read_bounded(local_path, limit)
        if sha256_bytes(snapshots[key]) != artifact["sha256"]:
            raise ValueError("artifact digest mismatch: " + key)
    questions = parse_items(snapshots["dataset"], receipt["dataset"]["format"])
    predictions = validate_rows(questions, parse_items(snapshots["predictions"], receipt["predictions"]["format"]))
    verdicts, results = _score(questions, predictions)
    expected_verdict_bytes = b"".join(_serialize(verdict).replace(b"\n", b"") + b"\n" for verdict in verdicts)
    if snapshots["verdicts"] != expected_verdict_bytes or _serialize(receipt["results"]) != _serialize(results):
        raise ValueError("verdict or aggregate mismatch")
    if receipt["dataset"]["variant"] != "fixture" or receipt["dataset"]["n"] != len(questions):
        raise ValueError("dataset identity mismatch")
    if receipt["dataset"]["selected_ids_sha256"] != sha256_bytes(_serialize([row["question_id"] for row in questions])):
        raise ValueError("selected ID digest mismatch")
    identities = [(receipt["dataset"]["sha256"], "dataset"), (receipt["system"]["predictions_sha256"], "predictions"),
                  (receipt["verdicts_sha256"], "verdicts"), (receipt["evaluator"]["source_snapshot_sha256"], "evaluator")]
    if any(digest != sha256_bytes(snapshots[key]) for digest, key in identities):
        raise ValueError("identity digest mismatch")
    compiled = receipt["evaluator"]["compiled_source_sha256"]
    if compiled is not None and compiled != sha256_bytes(snapshots["evaluator"]):
        raise ValueError("compiled source digest mismatch")
    if expected_evaluator_sha256 is not None and expected_evaluator_sha256 != sha256_bytes(snapshots["evaluator"]):
        raise ValueError("trusted evaluator digest mismatch")
    return {"content_consistent": True, "authentication": "none", "source_identity_verified": expected_evaluator_sha256 is not None,
            "metric_id": METRIC_ID, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-variant", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--system-version", default="unknown")
    parser.add_argument("--judge", required=True, help="exact (strict fixture alias) | fixture_exact_match_v1")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--claimed", type=float)
    parser.add_argument("--claimed-source")
    parser.add_argument("--out", default="receipts")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
