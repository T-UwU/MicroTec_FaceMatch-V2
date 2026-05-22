# Face Anti-Spoofing API

API de detección de vida (liveness detection). Recibe un video y determina si es una persona real o un intento de suplantación (spoof).

## Requisitos

- Python 3.10+
- No requiere GPU

## Entrenar

# 1. Validar videos
python3 scripts/validate_videos.py

# 2. Generar manifiesto 
python3 scripts/generate_manifest.py

# 3. Pre-cachear (solo la primera vez, o cuando agregas videos nuevos)
python3 scripts/precache_dataset.py

# 4. Entrenar (ahora mucho más rápido)
python3 scripts/train.py


## Instalación

```bash
bash setup.sh
```

O manualmente:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Encender el servidor

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Para correr en background (producción):
```bash
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 > server.log 2>&1 &
```

Para detener:
```bash
pkill -f "uvicorn app.main:app"
```

## Endpoints

### POST /v1/liveness

Analiza un video y devuelve si es real o spoof.

```bash
curl -X POST http://localhost:8000/v1/liveness \
  -F "video=@video.mp4"
```

Respuesta:
```json
{
  "request_id": "fe477b56-5f70-4c82-9118-0a2b4b0d0cbb",
  "decision": "REAL",
  "confidence": {
    "real": 0.9942,
    "spoof": 0.0058
  },
  "processing_time_ms": 460
}
```

- `decision`: REAL o SPOOF
- `confidence.real`: confianza de que es real (0 a 1, donde 1 = seguro)
- `confidence.spoof`: confianza de que es spoof (0 a 1, donde 1 = seguro)

### GET /health

```bash
curl http://localhost:8000/health
```

## Formatos aceptados

- .mp4, .webm, .avi, .mov
- Máximo 50MB

## Estructura

```
face-antispoofing-api/
├── app/
│   ├── main.py            # FastAPI endpoints
│   ├── model.py           # Inferencia ONNX
│   └── preprocessing.py   # Extracción de frames + landmarks
├── model/
│   ├── model.onnx         # Modelo anti-spoofing
│   ├── model.onnx.data    # Pesos del modelo
│   └── face_landmarker.task  # MediaPipe landmarks
├── requirements.txt
└── setup.sh
```
