from __future__ import annotations

import re
import sys
from pathlib import Path


def count_words(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return len(re.findall(r"\b\S+\b", text))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("README.md")
    count = count_words(path)
    print(f"README word count: {count}")
    if 600 <= count <= 1000:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
