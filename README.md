# Real-Time AI Safety Monitoring System

Computer-vision application that monitors a camera feed (webcam / video file /
RTSP) in real time to:

- Detect weapons (gun, knife, pistol, rifle, other configured classes) via YOLO
- Optionally detect fighting/violent activity via a pluggable action model
- Confirm detections over a temporal window (no single-frame false triggers)
- Capture annotated evidence frames + metadata into SQLite
- Provide a live dashboard (REST + WebSocket + MJPEG feed)
- Emit non-interventionist alerts (console + dashboard only)

> Surveillance / documentation only. This system observes and logs. It must not
> autonomously identify, accuse, or physically intervene against a person based
> on an AI prediction.

## Status

Handoff doc: `handoff.md` (approved for implementation).

## Architecture

```
fight/
├── app/
│   ├── main.py                 # FastAPI app factory + background capture/detection loop
│   ├── config.py               # dataclass config; YAML defaults ← .env/ENV override
│   ├── camera/                 # webcam index / video file / RTSP / synthetic
│   ├── detection/              # base types, weapon detector, violence detector
│   ├── events/                 # evidence frames + temporal confirmation/event manager
│   ├── alerts/                 # alert manager (console backend by default)
│   ├── database/               # SQLite schema + CRUD
│   └── api/                    # REST routes + WebSocket hub
├── frontend/                   # single-page dashboard (HTML/CSS/JS)
├── models/                     # place your trained weapon_model.pt here
├── events/                     # evidence images: events/YYYY/MM/DD/evt_<id>_<class>.jpg
├── tests/                      # pytest (no model / camera / network required)
├── config.yaml                 # defaults
├── .env.example
├── requirements.txt
└── run.py                      # CLI entry
```

Data flow: `camera → detectors → TemporalTracker → EventManager → SQLite + events/`
with a WebSocket/MJPEG dashboard fed from the shared monitor state.

## Install

Requires Python 3.10+ (developed on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run

```bash
python run.py                    # webcam 0 (stub mode if no model)
python run.py --test             # synthetic feed, no camera needed (demo)
python run.py --source path/video.mp4
python run.py --source rtsp://user:pass@host/stream
python run.py --config config.yaml --host 0.0.0.0 --port 8000 --db data/safety.db
```

Then open the dashboard: `http://127.0.0.1:8000/`

If no trained model exists at `models/weapon_model.pt`, the system runs in
**STUB demo mode**: a deterministic detector that reacts to bright blobs and
exercises the full pipeline. No fake model files are created.

## Training a weapon model

The pretrained COCO YOLO has poor gun/knife coverage and is not acceptable for
production claims. Train your own:

1. Collect a dataset (e.g. images of guns, knives, pistols, rifles) with class
   labels; Roboflow can export in YOLO format.
2. Train (Ultralytics):

   ```bash
   yolo detect train data=/path/to/data.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```

3. Copy the best weights to `models/weapon_model.pt` and restart.

`detection.weapon_classes` lists the classes the monitor treats as weapons
(default: gun, knife, pistol, rifle, weapon). Empty list → defaults.

## Configuration

Precedence: `config.yaml` < environment / `.env` < CLI flags.

Key settings (see `config.yaml`):
- `detection.confidence_threshold` (0.80) — positive-detection gate
- `confirmation.window` (5) / `confirmation.min_positive` (3) — temporal gate
- `events.cooldown_seconds` (10) — per-type minimum spacing between events
- `events.evidence_dir` — where annotated JPEGs are written
- `violence.enabled` (false) — enable when an action model is plugged in;
  weapon ∧ violence → `high_risk`
- `api.require_auth_token` / `api.auth_token` — optional API auth

Environment overrides: prefix the upper-cased field name, e.g. `CAMERA_SOURCE`,
`DETECT_CONFIDENCE_THRESHOLD`, `CONFIRM_WINDOW`, `EVENTS_COOLDOWN_SECONDS`,
`API_PORT`. Some also accept an `AI_` prefix. `CAMERA_SOURCE` is honoured at
the top level for convenience.

## REST / WebSocket API

- `GET /api/status` — monitor state, risk level, config summary
- `GET /api/events` — recorded events (`?limit=`, `?status=` filters)
- `GET /api/events/{id}` — event detail
- `POST /api/events/{id}/review` — mark reviewed
- `GET /video_feed` — MJPEG annotated stream
- `WS /api/ws` — live JSON updates (state, detections, events)

## Testing

```bash
python -m pytest
```

75 tests, no model / camera / network required (synthetic feed behind a flag).

## False Positive / False Negative Risks

- **FP**: hands/phones/tools mistaken for knives; silhouettes for guns; motion blur.
- **FN**: tiny / occluded / rotated objects; low light; fast motion.
- **Mitigations**: confidence threshold, class whitelist, temporal confirmation,
  per-type cooldown; consider ROI upscaling for far-field cameras.

## Troubleshooting

- Webcam not found: fall back to a video file or `--source synthetic`.
- RTSP drops: the monitor loop logs and retries the connection.
- Slow inference: lower `camera.fps` / `detection.max_det`, or run YOLO on GPU.
- Evidence dir missing: it is created automatically on first event.

## Privacy & Limitations

This system is a passive observer. Predictions are probabilistic and can be
wrong; do not use them as the sole basis for any decision about a person.
Prefer local-only deployment, encrypt evidence at rest, respect local
surveillance/privacy law, and apply retention policies. Alerts are advisory
(console + dashboard) only, by design.