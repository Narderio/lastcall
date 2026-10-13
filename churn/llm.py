"""Client LLM unico per il progetto: stesso codice per Gemini/OpenRouter (sviluppo) e vLLM su AMD (gara).

Si sceglie con tre variabili d'ambiente, nessuna chiave nel codice:

  # OpenRouter, sviluppo
  LLM_BASE_URL=https://openrouter.ai/api/v1
  LLM_API_KEY=<chiave OpenRouter>
  LLM_MODEL=<modello OpenRouter>

  # vLLM su AMD MI300X, versione finale
  LLM_BASE_URL=http://<IP-droplet>:8000/v1
  LLM_API_KEY=<VLLM_API_KEY>
  LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
"""
import json
import os
import urllib.request

BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def chat(messages, json_mode=False, **kw):
    body = {"model": MODEL, "messages": messages, **kw}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    return json.loads(text) if json_mode else text


if __name__ == "__main__":
    print(f"Endpoint: {BASE_URL}  modello: {MODEL}")
    print(chat([{"role": "user", "content": 'Rispondi solo con JSON: {"stato": "ok"}'}], json_mode=True))
