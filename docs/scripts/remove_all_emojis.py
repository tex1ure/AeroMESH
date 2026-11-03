import re
import unicodedata

files = [
    "API_REFERENCE.md",
    "BENCHMARKS.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LIMITATIONS.md",
    "README.md",
    "SECURITY.md",
    "docs/architecture/fastapi-proxy-gateway.md"
]

def is_emoji(ch):
    cp = ord(ch)
    # Check emoji code point ranges
    if (0x1F000 <= cp <= 0x1FAFF or
        0x2600 <= cp <= 0x27BF or
        0x2300 <= cp <= 0x23FF or
        0x2B00 <= cp <= 0x2BFF or
        0x20D0 <= cp <= 0x20FF or
        0xFE00 <= cp <= 0xFE0F):
        return True
    return False

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    # Remove all emojis
    cleaned = []
    removed_count = 0
    for ch in content:
        if is_emoji(ch):
            removed_count += 1
        else:
            cleaned.append(ch)
    content = "".join(cleaned)

    # Normalize double spaces after markdown headers like '##  ' -> '## '
    content = re.sub(r'^(#+)\s{2,}', r'\1 ', content, flags=re.MULTILINE)

    # Clean up double spaces in links like '[ 1. ...]'
    content = re.sub(r'\[\s+', '[', content)

    if content != original:
        with open(fpath, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(content)
        print(f"Updated {fpath}: removed {removed_count} emoji(s)")
    else:
        print(f"No changes needed for {fpath}")

print("All files processed!")

