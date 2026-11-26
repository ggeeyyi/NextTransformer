"""Wire identities and stable errors shared by client and control plane."""

import hashlib
import json
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def new_id():
    return uuid.uuid4().hex


class GymError(Exception):
    def __init__(self, code, hint, phase="control", retryable=False, details=None):
        self.code, self.hint, self.phase, self.retryable = code, hint, phase, retryable
        self.details = dict(details or {})
        super().__init__(hint)

    def wire(self):
        return {
            "schema_version": "gym.api.v1",
            "error": {
                "code": self.code,
                "hint": self.hint,
                "phase": self.phase,
                "retryable": self.retryable,
                "details": self.details,
            },
        }


def require(condition, code, hint):
    if not condition:
        raise GymError(code, hint)


def integer(value, minimum=0, maximum=None, name=None):
    """Return an integer within bounds; a named field reports its name and bounds."""
    if (
        type(value) is int
        and value >= minimum
        and (maximum is None or value <= maximum)
    ):
        return value
    if name is None:
        raise GymError("invalid_request", "Integer outside the supported range.")
    if maximum is None:
        bounds = f"of at least {minimum}"
    else:
        bounds = f"from {minimum} to {maximum}"
    raise GymError(
        "invalid_request",
        f"{name} must be an integer {bounds}.",
        details={"field": name, "minimum": minimum, "maximum": maximum},
    )


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
