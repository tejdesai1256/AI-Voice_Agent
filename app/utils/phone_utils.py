import re
from typing import Tuple, Optional


def normalize_phone_number(phone_raw: str, default_country_code: str = "+91") -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates and standardizes a phone number to E.164 format.
    Default country code is India (+91).
    
    Returns:
        (is_valid, normalized_e164_number, error_message)
    """
    if not phone_raw or not str(phone_raw).strip():
        return False, None, "Phone number cannot be empty."

    # Remove spaces, hyphens, parentheses, dots
    cleaned = re.sub(r"[\s\-\(\)\.]", "", str(phone_raw).strip())

    # Check for invalid characters
    if not re.match(r"^\+?[0-9]{10,15}$", cleaned):
        return (
            False,
            None,
            "Invalid phone number format. Number must contain only digits and optional '+' prefix (10-15 digits).",
        )

    # If it starts with '+', keep as is
    if cleaned.startswith("+"):
        e164 = cleaned
    elif cleaned.startswith("0") and len(cleaned) == 11:
        # Domestic 0-prefixed 10-digit number (e.g. 09876543210)
        e164 = f"{default_country_code}{cleaned[1:]}"
    elif len(cleaned) == 10:
        # Standard 10-digit mobile number
        e164 = f"{default_country_code}{cleaned}"
    elif len(cleaned) == 12 and cleaned.startswith("91"):
        e164 = f"+{cleaned}"
    else:
        e164 = f"+{cleaned}"

    # Basic length check: standard international E.164 is max 15 digits excluding '+'
    digits_only = re.sub(r"\D", "", e164)
    if len(digits_only) < 10 or len(digits_only) > 15:
        return False, None, f"Phone number length ({len(digits_only)} digits) is out of valid range (10-15 digits)."

    return True, e164, None


def mask_phone_number(phone: str) -> str:
    """
    Masks the middle digits of a phone number for privacy in logs and UI.
    e.g. +919876543210 -> +91******3210
    """
    if not phone:
        return "N/A"
    clean = str(phone).strip()
    if len(clean) <= 6:
        return clean
    visible_prefix = clean[:4]
    visible_suffix = clean[-4:]
    masked_length = max(2, len(clean) - 8)
    return f"{visible_prefix}{'*' * masked_length}{visible_suffix}"
