"""Structured errors returned to the agent (design doc, appendix C)."""

RETRYABLE = {"BACKEND_UNAVAILABLE", "QUOTE_EXPIRED", "RATE_LIMITED", "INTERNAL"}
CODES = {"NOT_FOUND", "SPEC_VIOLATION", "CAUSALITY_VIOLATION", "NUMERIC_CHECK_FAILED", "AUDIT_UNKNOWN_OP",
         "FLOPS_OUT_OF_TOLERANCE", "CALIBRATION_FAILED", "DUPLICATE_ATTEMPT", "PAYLOAD_TOO_LARGE",
         "NOT_VERIFIED", "INVALID_WORLD_SIZE", "BACKEND_UNAVAILABLE", "QUOTE_EXPIRED", "INSUFFICIENT_BUDGET",
         "POLICY_DENIED", "RUN_CONFLICT", "IDEMPOTENCY_CONFLICT", "ALREADY_TERMINAL", "RATE_LIMITED",
         "INVALID_ARGUMENT", "INTERNAL"}


class GymError(Exception):
    def __init__(self, code, message, *, details=None, hint=None):
        if code not in CODES: raise ValueError(f"unknown error code {code}")
        super().__init__(message)
        self.code, self.message, self.details, self.hint = code, message, details or {}, hint

    def to_dict(self):
        out = dict(code=self.code, message=self.message, retryable=self.code in RETRYABLE)
        if self.details: out["details"] = self.details
        if self.hint: out["hint"] = self.hint
        return out


def error(code, message, **kwargs):
    return GymError(code, message, **kwargs).to_dict()
