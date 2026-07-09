"""Probe GPT-OSS on Groq to find why message.content comes back empty.

Runs one trivial question under several reasoning/token configurations and
dumps, for each, the raw response shape: finish_reason, message.content,
any reasoning field (message.reasoning / reasoning_content), and usage. This
pinpoints (a) where the answer text actually lands and (b) which param combo
returns it cleanly in `content`, so panels.py can be set correctly once.

No project imports — just the openai SDK pointed at Groq, so you can run it in
isolation.

    export GROQ_API_KEY=...
    python scripts/probe_gptoss.py
    python scripts/probe_gptoss.py --model openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
QUESTION = "What is the capital of France? Answer in one word."


def dump(tag: str, completion, err: str | None = None) -> None:
    print("\n" + "-" * 68)
    print(f"CONFIG: {tag}")
    print("-" * 68)
    if err is not None:
        print(f"  ERROR: {err}")
        return
    try:
        raw = completion.model_dump()
    except Exception:
        raw = None
    choice = completion.choices[0]
    msg = choice.message
    content = getattr(msg, "content", None)
    reasoning = getattr(msg, "reasoning", None)
    reasoning_content = getattr(msg, "reasoning_content", None)
    usage = getattr(completion, "usage", None)

    print(f"  finish_reason        : {getattr(choice, 'finish_reason', None)!r}")
    print(f"  message.content      : {(content or '')[:200]!r}")
    print(f"  message.reasoning    : {(reasoning or '')[:120]!r}"
          + ("" if reasoning is None else "   <-- answer/thoughts landed here"))
    print(f"  message.reasoning_content: {(reasoning_content or '')[:120]!r}")
    if usage is not None:
        print(f"  usage                : prompt={getattr(usage,'prompt_tokens',None)} "
              f"completion={getattr(usage,'completion_tokens',None)} "
              f"total={getattr(usage,'total_tokens',None)}")
    # Show any extra message keys we didn't anticipate.
    if raw:
        mkeys = set(raw["choices"][0]["message"].keys())
        known = {"role", "content", "reasoning", "reasoning_content",
                 "tool_calls", "function_call", "refusal", "annotations", "audio"}
        extra = mkeys - known
        if extra:
            print(f"  UNEXPECTED message keys: {sorted(extra)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY first.")
    client = OpenAI(base_url=GROQ_BASE_URL, api_key=key)
    msgs = [{"role": "user", "content": QUESTION}]
    mt = args.max_tokens

    # (label, create-kwargs). We vary how reasoning is suppressed and how the
    # token budget is expressed, since both are suspects.
    configs = [
        ("bare: no reasoning params, max_tokens",
         dict(max_tokens=mt)),
        ("reasoning_format=hidden (current panels setting)",
         dict(max_tokens=mt, extra_body={"reasoning_format": "hidden"})),
        ("reasoning_format=hidden + effort=low",
         dict(max_tokens=mt, extra_body={"reasoning_format": "hidden",
                                         "reasoning_effort": "low"})),
        ("reasoning_format=parsed (answer in content, reasoning separate)",
         dict(max_tokens=mt, extra_body={"reasoning_format": "parsed"})),
        ("include_reasoning=False (Groq's documented 'exclude' switch)",
         dict(max_tokens=mt, extra_body={"include_reasoning": False})),
        ("max_completion_tokens instead of max_tokens + hidden",
         dict(extra_body={"reasoning_format": "hidden", "max_completion_tokens": mt})),
    ]

    print(f"Model: {args.model}   Question: {QUESTION!r}   budget: {mt}")
    for tag, kwargs in configs:
        try:
            completion = client.chat.completions.create(
                model=args.model, messages=msgs, temperature=0.0, **kwargs
            )
            dump(tag, completion)
        except Exception as exc:  # noqa: BLE001 - want the 400 text, it's diagnostic
            dump(tag, None, err=f"{type(exc).__name__}: {exc}")

    print("\n" + "=" * 68)
    print("READ-OUT")
    print("=" * 68)
    print("  Pick the config whose `message.content` holds 'Paris' with")
    print("  finish_reason='stop'. That's the combo to bake into panels.py.")
    print("  If content is empty but message.reasoning has the answer, the fix")
    print("  is to read the reasoning field (or use reasoning_format=parsed).")
    print()


if __name__ == "__main__":
    main()