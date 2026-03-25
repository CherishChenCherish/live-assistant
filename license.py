"""Activation code system for Live Assistant Pro.

Free tier:  Real-time transcription only
Pro tier:   AI responses, code generation, follow-up predictions,
            keyword hints, SOS, screen monitor, context upload

Activation codes are HMAC-SHA256 based, offline-verifiable.
No server needed — codes are self-validating.
"""

import hashlib
import os
import hmac
import json
import time
from pathlib import Path

# Secret key loaded from env or file; auto-generated on first run
def _load_secret() -> bytes:
    env_key = os.environ.get("LA_LICENSE_SECRET")
    if env_key:
        return env_key.encode()
    key_file = Path(__file__).parent / ".license_key"
    if key_file.exists():
        return key_file.read_text().strip().encode()
    import secrets
    new_key = secrets.token_hex(32)
    key_file.write_text(new_key)
    key_file.chmod(0o600)
    return new_key.encode()


_SECRET = _load_secret()

LICENSE_FILE = Path(__file__).parent / ".license"

# Feature tiers
FREE_FEATURES = {
    "transcription",
    "manual_question",
}

PRO_FEATURES = {
    "transcription",
    "manual_question",
    "ai_responses",
    "code_generation",
    "followup_prediction",
    "keyword_hints",
    "sos_button",
    "screen_monitor",
    "screenshot_ocr",
    "context_upload",
    "regenerate",
}


def generate_code(email: str, days: int = 365) -> str:
    """Generate an activation code for an email.

    Code format: BASE-EXPIRY-SIGNATURE
    - BASE: first 8 chars of email hash
    - EXPIRY: unix timestamp (hex)
    - SIGNATURE: HMAC verification (12 chars)

    Run this to generate codes for customers:
        python -c "from license import generate_code; print(generate_code('customer@email.com', 365))"
    """
    expiry = int(time.time()) + (days * 86400)
    base = hashlib.sha256(email.lower().encode()).hexdigest()[:8]
    expiry_hex = format(expiry, "x")
    payload = f"{base}-{expiry_hex}"
    sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:12]
    return f"{base}-{expiry_hex}-{sig}".upper()


def verify_code(code: str) -> dict:
    """Verify an activation code.

    Returns:
        {"valid": True, "expires": timestamp, "days_left": int}
        or {"valid": False, "reason": "..."}
    """
    try:
        parts = code.strip().upper().split("-")
        if len(parts) != 3:
            return {"valid": False, "reason": "Invalid code format"}

        base, expiry_hex, sig = parts
        payload = f"{base}-{expiry_hex}".lower()
        expected_sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:12]

        if sig.lower() != expected_sig:
            return {"valid": False, "reason": "Invalid activation code"}

        expiry = int(expiry_hex, 16)
        now = int(time.time())
        if now > expiry:
            return {"valid": False, "reason": "Code has expired"}

        days_left = (expiry - now) // 86400
        return {"valid": True, "expires": expiry, "days_left": days_left}

    except Exception:
        return {"valid": False, "reason": "Invalid code format"}


def activate(code: str) -> dict:
    """Activate a code and save to license file."""
    result = verify_code(code)
    if result["valid"]:
        LICENSE_FILE.write_text(json.dumps({
            "code": code.strip().upper(),
            "activated_at": int(time.time()),
            "expires": result["expires"],
        }))
    return result


def deactivate():
    """Remove license."""
    if LICENSE_FILE.exists():
        LICENSE_FILE.unlink()


def get_license_status() -> dict:
    """Check current license status.

    Returns:
        {"tier": "pro", "days_left": 340, ...}
        or {"tier": "free"}
    """
    if not LICENSE_FILE.exists():
        return {"tier": "free"}

    try:
        data = json.loads(LICENSE_FILE.read_text())
        result = verify_code(data["code"])
        if result["valid"]:
            return {
                "tier": "pro",
                "days_left": result["days_left"],
                "expires": result["expires"],
            }
        else:
            return {"tier": "free", "expired": True, "reason": result["reason"]}
    except Exception:
        return {"tier": "free"}


def has_feature(feature: str) -> bool:
    """Check if a feature is available in current tier."""
    status = get_license_status()
    if status["tier"] == "pro":
        return feature in PRO_FEATURES
    return feature in FREE_FEATURES


def get_available_features() -> set:
    """Get all available features for current tier."""
    status = get_license_status()
    if status["tier"] == "pro":
        return PRO_FEATURES
    return FREE_FEATURES
