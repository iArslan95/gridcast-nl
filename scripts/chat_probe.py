"""End-to-end probe of the assistant: one real grounded answer.

Needs a key in .streamlit/secrets.toml or the GROQ_API_KEY environment
variable. Exits cleanly with a message when there is none, so CI without a
key does not fail on it.

    python scripts/chat_probe.py
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import assistant  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    key = assistant.get_api_key()
    if not key:
        print("no GROQ_API_KEY found; probe skipped (the app runs fine "
              "without one).")
        return 0

    meta = json.loads((ROOT / "data" / "processed" / "meta.json")
                      .read_text(encoding="utf-8"))
    context = assistant.build_context(meta, [
        "(probe run: alleen meta.json als digest; de app zelf levert meer.)"])
    vraag = ("Hoe verhoudt het model zich tot seizoensnaief, en wat is er "
             "mis met het 80%-interval?")
    print(f"Q: {vraag}\nA: ", end="", flush=True)
    n = 0
    for chunk in assistant.stream_reply(key, context,
                                        [{"role": "user", "content": vraag}]):
        print(chunk, end="", flush=True)
        n += len(chunk)
    print(f"\n\nCHAT PROBE OK ({n} tekens gestreamd)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
