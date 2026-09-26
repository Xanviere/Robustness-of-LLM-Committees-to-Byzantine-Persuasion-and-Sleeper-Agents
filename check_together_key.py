import os
import httpx

api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")
print("1. Checking API Key presence...")
if not api_key:
    print("[FAIL] No API key found in OPENAI_API_KEY or TOGETHER_API_KEY environment variables.")
    print("Please set it with: $env:OPENAI_API_KEY='your-key-here'")
    exit(1)

print(f"[OK] API Key detected: {api_key[:6]}...{api_key[-4:] if len(api_key)>8 else ''}")

print("\n2. Sending test ping to Together AI (Llama-3.3-70B)...")
try:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    r = httpx.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=15.0)
    print(f"HTTP Status: {r.status_code}")
    if r.status_code == 200:
        print("\n>>> [SUCCESS] Together AI API is active, authenticated, and ready to run Stage 2! <<<")
    else:
        print(f"\n[ERROR Response from Together AI]:\n{r.text}")
except Exception as e:
    print(f"[Network Error]: {e}")
