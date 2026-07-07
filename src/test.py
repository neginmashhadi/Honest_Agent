import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

client = OpenAI()
print("env key tail:", (os.environ.get("OPENAI_API_KEY") or "NONE")[-6:])
print("client key tail:", client.api_key[-6:])

anthropic_client = Anthropic()

# --- List ALL available models, sorted ---
print("\n=== ALL AVAILABLE MODELS (OpenAI key) ===")
try:
    ids = sorted(m.id for m in client.models.list().data)
    for mid in ids:
        print(" ", mid)
    print(f"\n  total: {len(ids)} models")
except Exception as e:
    print("Could not list models:", e)

# --- Test a specific GPT model end to end ---
print("\n=== GPT CALL TEST ===")
for model in ["gpt-4.1-2025-04-14", "gpt-4.1"]:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: API works"}],
            max_tokens=10,
        )
        print(f"  {model}: SUCCESS -> {r.choices[0].message.content!r}")
    except Exception as e:
        # print just the model + short reason, not the whole traceback
        msg = str(e)
        print(f"  {model}: FAILED -> {msg[:120]}")

# --- List ALL available models, sorted (Anthropic key) ---
print("\n=== ALL AVAILABLE MODELS (Anthropic key) ===")
try:
    ids = sorted(m.id for m in anthropic_client.models.list())
    for mid in ids:
        print(" ", mid)
    print(f"\n  total: {len(ids)} models")
except Exception as e:
    print("Could not list models:", e)