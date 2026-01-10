import re

def extract_claims(backstory_text):
    """
    Converts a backstory paragraph into a list of atomic claims.
    This is a simple rule-based first version.
    """

    # clean text
    text = backstory_text.strip()

    # split by common separators
    raw_parts = re.split(r"[.;]", text)

    claims = []

    for part in raw_parts:
        part = part.strip()

        # ignore very small fragments
        if len(part) < 20:
            continue

        claims.append(part)

    return claims
