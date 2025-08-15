# Open WebUI ↔ Kyutai Unmute TTS Bridge

This bridge exposes an OpenAI-compatible TTS API and a low-latency WebSocket streaming API to connect Open WebUI to the Kyutai Unmute TTS server.

Bridge server module: `unmute.scripts.tts_bridge_server`

## Endpoints

- POST `/v1/audio/speech`

  - Input JSON (OpenAI-compatible):
    - `input` or `text`: string (required)
    - `voice`: string (optional, passed through to Kyutai)
    - `format`: only `"wav"` supported
  - Response: `audio/wav`

- GET `/v1/models` → minimal list with `kyutai-unmute-tts`
- GET `/healthz` → `{"status":"ok"}`

- WS `/v1/audio/speech/stream`
  - Client sends JSON once on connect: `{ "input"|"text": string, "voice"?: string }`
  - Server replies immediately with: `{ "type":"ready", "sample_rate":24000, "encoding":"pcm_s16le" }`
  - Server streams binary frames: PCM S16LE mono @ 24kHz
  - Completes with `{ "type":"eos" }` or `{ "type":"error", "detail": string }`

## Environment

- `KYUTAI_TTS_URL` (optional): WebSocket URL to Kyutai TTS (e.g., `ws://127.0.0.1:8020`).
  - If unset, the bridge tries: 127.0.0.1:8020 → localhost:8020 → 127.0.0.1:8089 → localhost:8089.

## Run the bridge

From repo root, either:

1. Use the helper script (bash):

```
dockerless/start_tts_bridge.sh
```

Environment overrides:

```
HOST=0.0.0.0 PORT=8070 dockerless/start_tts_bridge.sh
```

2. Or run the module directly (Python):

```
python -m unmute.scripts.tts_bridge_server --host 127.0.0.1 --port 8070
```

On Windows with repo venv:

```
.venv/Scripts/python.exe -m unmute.scripts.tts_bridge_server --host 127.0.0.1 --port 8070
```

## Open WebUI integration (additive)

Use the ready-to-copy files under `OWUI/` in this repo:

- `OWUI/src/lib/apis/openai/streamingTTS.ts`
- `OWUI/src/lib/audio/pcmPlayer.ts`
- `OWUI/src/lib/audio/kyutaiStreamingController.ts`
- `OWUI/src/lib/components/chat/StreamingResponsePlayer.svelte`

Copy them into the same paths in your Open WebUI repo. See `OWUI/INTEGRATION.md` for usage and interruption wiring notes.

## Quick smoke test (browser)

Open `OWUI/stream-demo.html` in a browser, set your bridge WS URL, enter text, and click Start. You should hear the streamed audio with low latency.

Troubleshooting:

- 503/unreachable: ensure Kyutai TTS container is up (host 127.0.0.1, port 8020:8080), or set `KYUTAI_TTS_URL`.
- No audio frames: check the `voice` value and that the Kyutai server is producing audio.

# Open WebUI ↔ Kyutai Unmute TTS Bridge

This lightweight FastAPI server exposes an OpenAI-compatible `/v1/audio/speech` endpoint and forwards requests to the Kyutai Unmute TTS WebSocket API. Use it to plug Unmute TTS into Open WebUI.

## Prereqs

- Unmute TTS container running and mapped as `-p 8020:8080`
- Python env for this repo (uv recommended)

## Run the bridge

By default the bridge targets `ws://localhost:8020`.

```bash
# From repo root
uv run python -m unmute.scripts.tts_bridge_server --host 0.0.0.0 --port 8070
```

- Health: http://localhost:8070/healthz
- Models (minimal): http://localhost:8070/v1/models
- TTS: POST http://localhost:8070/v1/audio/speech
  - Streaming (low latency): WS ws://localhost:8070/v1/audio/speech/stream

Example request body:

```json
{
  "model": "kyutai-unmute-tts",
  "input": "Hello from Kyutai TTS",
  "voice": "default",
  "format": "wav"
}
```

Response is `audio/wav`.

Streaming WS protocol:

- Client connects and sends JSON: `{"input"|"text": string, "voice"?: string}`
- Server replies JSON header: `{"type":"ready","sample_rate":24000,"encoding":"pcm_s16le"}`
- Then streams binary frames with PCM S16LE mono at 24kHz as audio becomes available
- On completion, server sends `{ "type": "eos" }` and closes

## Configure Open WebUI

There are two common options in Open WebUI:

1. Global OpenAI-compatible base URL

- Settings → Admin → LLM Providers
- Add provider (or edit existing OpenAI provider):
  - Base URL: `http://localhost:8070`
  - API Key: any non-empty string (the bridge doesn’t validate)
- In Audio/TTS settings, select the OpenAI provider and choose a voice (the bridge accepts `voice`).

2. If Open WebUI has a dedicated TTS provider using the OpenAI Audio API

- Set TTS Provider to OpenAI-compatible
- Base URL: `http://localhost:8070`
- Model: `kyutai-unmute-tts`
- Voice: any available value (mapped to Unmute voice if present)

Notes:

- The bridge currently supports only `wav` output.
- To target a remote TTS host/port, set `KYUTAI_TTS_URL=ws://<host>:<port>` before starting the bridge.

## OWUI helpers

See `OWUI/` for a minimal streaming demo (`stream-demo.html`) and integration notes.

## Troubleshooting

- No audio returned: ensure the TTS container is reachable at `ws://localhost:8020` and healthy.
- 400 error: missing or empty `input`. Provide text via `input` (preferred) or `text`.
- Voices: pass a `voice` string. For custom voice embeddings configured in Unmute (e.g. `custom:<id>`), the bridge forwards the value unchanged.
