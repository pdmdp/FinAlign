"""Checksummed JSON checkpoints. A mismatching checkpoint is never reused."""

import hashlib
import errno
import json
import os
from pathlib import Path


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError as exc:
            # Some Drive/FUSE mounts do not implement fsync. Other I/O failures
            # must stop the run; checksums still guard every saved checkpoint.
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                raise
    os.replace(temporary, path)


def write_json(path, value):
    atomic_bytes(path, canonical(value))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Store:
    def __init__(self, root, context):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.context_hash = digest(context)
        path = self.root / "context.json"
        if path.exists():
            if read_json(path) != context:
                raise ValueError("Run context changed (source, index, packages or device). "
                                 "Restore the original environment or use a new run directory.")
        else:
            if any(self.root.glob("cache/**/*.json")) or (self.root / "frozen_config.json").exists():
                raise ValueError("Run has checkpoints but no context.json; do not mix runs.")
            write_json(path, context)

    def identity(self, kind, row, extra=None):
        return {"context_sha256": self.context_hash, "kind": kind, "qid": row["qid"],
                "query": row["query"], "extra": extra}

    def path(self, kind, row):
        # Hash the ID rather than trusting it as a filesystem path.
        name = hashlib.sha256(str(row["qid"]).encode()).hexdigest()
        return self.root / "cache" / kind / name[:2] / (name + ".json")

    def get(self, kind, row, extra=None):
        path = self.path(kind, row)
        if not path.exists():
            return None
        item = read_json(path)
        expected = self.identity(kind, row, extra)
        if (item.get("identity") != expected or item.get("key") != digest(expected)
                or item.get("payload_sha256") != digest(item.get("payload"))):
            raise ValueError(f"Checkpoint mismatch or corruption: {path}")
        return item["payload"]

    def put(self, kind, row, payload, extra=None):
        identity = self.identity(kind, row, extra)
        value = {"identity": identity, "key": digest(identity), "payload": payload,
                 "payload_sha256": digest(payload)}
        write_json(self.path(kind, row), value)

    def freeze(self, value):
        path = self.root / "frozen_config.json"
        sidecar = self.root / "frozen_config.sha256"
        if path.exists():
            raise ValueError("The frozen configuration already exists; it cannot be overwritten.")
        write_json(path, value)
        atomic_bytes(sidecar, (digest(value) + "\n").encode())

    def frozen(self):
        path = self.root / "frozen_config.json"
        sidecar = self.root / "frozen_config.sha256"
        if not path.exists():
            raise ValueError("Run development selection and freeze it before evaluating test queries.")
        value = read_json(path)
        if not sidecar.exists():
            # Safe recovery if interrupted between the two atomic freeze writes.
            if (self.root / "test_started.json").exists() or any(self.root.glob("cache/results/**/*.json")):
                raise ValueError("Missing frozen hash after test started.")
            atomic_bytes(sidecar, (digest(value) + "\n").encode())
        if sidecar.read_text().strip() != digest(value) or value.get("context_sha256") != self.context_hash:
            raise ValueError("Frozen configuration checksum or context mismatch.")
        selection = read_json(self.root / "dev_selection.json")
        if digest(selection) != value.get("dev_selection_sha256"):
            raise ValueError("The retained dev selection no longer matches the freeze.")
        if value["weights"] != selection["weights"] or value["alpha"] != selection["alpha"]:
            raise ValueError("Frozen weights do not match dev selection.")
        return value, digest(value)
