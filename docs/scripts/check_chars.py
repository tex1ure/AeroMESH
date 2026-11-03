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

chars = {}
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            for ch in line:
                if ord(ch) > 127:
                    chars.setdefault(ch, set()).add(fpath)

for ch in sorted(chars.keys(), key=lambda c: ord(c)):
    name = unicodedata.name(ch, "UNKNOWN")
    print(f"{ch} U+{ord(ch):04X} {name} (in {len(chars[ch])} files: {list(chars[ch])})")

