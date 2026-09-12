# Handoff: Real-Time AI Safety Monitoring System

## Status: Approved for Implementation
Date: 2026-09-11

## Project Goal
Build a computer-vision application monitoring a camera feed (webcam / video file / RTSP) in real time to:
- Detect weapons (gun, knife, other configured weapon classes) via YOLO
- Optionally detect fighting/violent activity via a pluggable temporal/action-recognition model
- Confirm detections over a temporal window (no single-frame false triggers)
- Capture annotated evidence frames + metadata into SQLite
- Provide a live dashboard (REST + WebSocket + MJPEG feed)
- Non-interventionist alerts (console + dashboard only)

## Decisions Confirmed
- **Missing model behavior**: Stub/demo mode (deterministic stub detector) when `models/weapon_model.pt` is absent. Real inference activates once the user supplies a trained `.pt`. NO fake model files are created.
- Surveillance/documentation only. System must not autonomously identify, accuse, or physically intervene against a person based on AI predictions.

## Architecture

```
fight/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory + background capture/detection loop
│   ├── config.py               # dataclass config; YAML defaults ← .env/ENV override
│   ├── camera/
│   │   ├── __init__.py
│   │   └── camera_source.py    # webcam index / video file / RTSP; synthetic test mode
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── base.py             # Detection, FrameResult, BaseDetector
│   │   ├── weapon_detector.py  # Ultralytics YOLO + class whitelist + annotation + StubDetector
│   │   └── violence_detector.py# abstract ViolenceDetector + stub + combined-risk logic helper
│   ├── events/
│   │   ├── __init__.py
│   │   ├── evidence.py         # evidence_dir(), unique_filename(), save_annotated_frame()
│   │   └── event_manager.py    # TemporalTracker + EventManager (confirmation + cooldown)
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_manager.py    # AlertManager ABC + ConsoleAlertManager (extensible)
│   ├── database/
│   │   ├── __init__.py
│   │   └── database.py         # SQLite schema, insert/list/get/mark_reviewed
│   └── api/
│       ├── __init__.py
│       ├── routes.py           # REST endpoints + MJPEG feed
│       └── ws.py               # WebSocket broadcast hub
├── frontend/
│   ├── index.html              # single-page dashboard
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── models/                     # user places weapon_model.pt here
├── events/                     # evidence: events/YYYY/MM/DD/evt_<id>_<class>.jpg
├── tests/                      # pytest; config, detection logic, events, db, evidence, camera
├── .env.example
├── .gitignore
├── config.yaml
├── requirements.txt
├── run.py                      # CLI entry (--source, --test, --config)
└── README.md
```

## Key Mechanics

### Config precedence
1. `config.yaml` (defaults)
2. `.env` / environment variables (prefix `AI_` optional)
3. CLI flags via `run.py`

Core: `confidence_threshold=0.80`, `confirmation_window=5`, `min_positive=3`, `cooldown_seconds=10`.

### Temporal confirmation
`TemporalTracker` slides over last `window` frames; a frame is positive if it contains a detection with `confidence >= threshold` for a whitelisted class. Confirmed when `positives >= min_positive` over the window. Framed example: 5 frames, positives at 1,2,4,5 (≥3 → confirmed).

### Event/cooldown
`EventManager.confirm()` fires an event only if the same event-type's cooldown has elapsed (per-type cooldown). Evidence saved to `events/YYYY/MM/DD/`.

### Database schema
`id, event_type, detected_class, confidence, timestamp, image_path, camera_id, status`

### Combination logic
- weapon ∧ violence → `high_risk`
- weapon alone → `weapon`
- violence alone → config-gated (default off)

## Dependencies
- opencv-python, ultralytics, fastapi, uvicorn[standard], PyYAML, python-dotenv, pydantic
- Tests: pytest (+ no network, no camera required via synthetic mode)

## Model Strategy
- Pretrained COCO YOLO has poor gun/knife coverage → NOT acceptable for production claims.
- User fine-tunes `models/weapon_model.pt` (README covers dataset prep + training).
- StubDetector used until then.

## False Positive / False Negative Risks (document in README)
- FP: hands/phones/tools as knives; silhouettes as guns; motion blur.
- FN: tiny/occluded/rotated objects; low light.
- Mitigations: threshold + temporal confirmation + class whitelist; ROI upscaling suggestion.

## Build Order
1. Scaffold + config.py
2. database.py
3. camera/camera_source.py
4. detection/ (base, weapon, violence)
5. events/ (evidence, event_manager)
6. alerts/
7. api/ (routes, ws)
8. main.py + run.py
9. frontend
10. tests
11. config.yaml, .env.example, .gitignore, requirements.txt
12. README.md

## Definition of Done
- `pytest` green (no model, no camera, no network required)
- API + dashboard launch with `python run.py` (stub mode default)
- README complete (purpose, architecture, install, training, modes, config, troubleshooting, limitations, privacy)