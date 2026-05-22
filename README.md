# Face Anti-Spoofing API

API de detección de vida (liveness detection). Recibe un video y determina si es una persona real o un intento de suplantación (spoof). Opcionalmente, si se envía también la foto de una INE, hace comparación de rostros (face match) entre el video y la credencial.

### Requisitos

- Python 3.10+
- No requiere GPU

### Entrenar

#### 1. Validar videos
python3 scripts/validate_videos.py

#### 2. Generar manifiesto 
python3 scripts/generate_manifest.py

#### 3. Pre-cachear (solo la primera vez, o cuando agregas videos nuevos)
python3 scripts/precache_dataset.py

#### 4. Entrenar (ahora mucho más rápido)
python3 scripts/train.py


### Instalación

```bash
bash setup.sh
```

O manualmente:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Encender el servidor

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

### Endpoints

#### POST /v1/liveness

Analiza un video y devuelve si es real o spoof. Si además se envía `ine_image`, cuando el video resulta REAL se compara el rostro del video contra la foto de la INE.

Solo liveness:
```bash
curl -X POST http://localhost:8000/v1/liveness \
  -F "video=@video.mp4"
```

Liveness + face match:
```bash
curl -X POST http://localhost:8000/v1/liveness \
  -F "video=@video.mp4" \
  -F "ine_image=@ine.jpg" \
  -F "match_threshold=50"
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

#### GET /health

```bash
curl http://localhost:8000/health
```

### Formatos aceptados

- Video: .mp4, .webm, .avi, .mov — máximo 50MB
- INE (imagen): .jpg, .jpeg, .png, .bmp, .webp

### Comparación de rostros (`face_comparison.py`)

Extrae los frames del video y los compara contra la foto de la INE usando InsightFace
(ArcFace + ONNX Runtime, sin dependencia de TensorFlow/Keras).

La comparación funciona como una **cascada de dos etapas**:

1. **Detección compartida.** `_detect_and_align()` corre la detección una sola vez
   sobre la INE y todos los frames, y entrega recortes alineados de 112×112 que
   alimentan ambos modelos de reconocimiento. La selección por rostro más grande
   maneja el retrato holográfico tenue de la INE (la foto principal es la detección
   más grande).
2. **Etapa 1 — modelo rápido (r50).** El modelo de reconocimiento de `buffalo_l`
   genera embeddings de todos los frames y los ordena por similitud real contra la
   INE.
3. **Etapa 2 — modelo fuerte (r100).** El modelo glint360k ResNet100 (`antelopev2`)
   re-evalúa solo la INE y los `K` mejores frames (`_CASCADE_K = 2`). El score final
   es la **similitud máxima** sobre esos frames; tomar el máximo —en lugar de
   promediar— recupera el frame bueno que un usuario genuino puede mostrar solo un
   instante. Si el modelo fuerte no está disponible, hace fallback a r50.

> El modelo `antelopev2` se descarga automáticamente la primera vez que se usa.
> `warmup_deepface()` precarga ambos modelos para que la primera petición sea rápida.

### Cambios recientes en `face_comparison.py`

Esta versión reemplaza por completo la estrategia anterior de comparación de rostros.

**Antes**
- Seleccionaba los 5 frames más nítidos mediante una heurística de imagen (nitidez
  Laplaciana + iluminación), sin usar ML.
- Corría un único modelo InsightFace (`buffalo_l` completo) sobre esos candidatos.
- Devolvía la similitud más alta, con salida temprana si superaba `threshold + 10`.

**Ahora**
- Cascada de dos etapas r50 → r100 (ver sección anterior). Los frames ya **no** se
  eligen por nitidez, sino por similitud real contra la INE.
- Nuevo modelo fuerte: glint360k ResNet100 (`antelopev2`), cargado de forma perezosa
  por `_get_strong_rec()` con descarga automática y fallback a r50.
- Detección y alineación compartidas (`_detect_and_align()`): se corren una sola vez
  y los recortes alimentan ambos modelos.
- Carga solo los módulos `detection` + `recognition` (`allowed_modules`); se omiten
  los modelos de landmarks y gender-age, que no se usan.
- `warmup_deepface()` ahora precarga ambos modelos de reconocimiento.
- Se eliminaron las funciones `_pick_candidate_indices()` y `_get_embedding()`, junto
  con la lógica de salida temprana.
- Nuevas constantes: `_DET_SIZE` y `_CASCADE_K`.

**Validación e impacto**
- Probado en 64 casos reales: el comparador r100 sube el piso de score genuino
  (mínimo 35 → 42) y recupera falsos rechazos que r50 producía en pares con brecha
  de apariencia difícil.
- La cascada mantiene el costo de r100 acotado a `K + 1` embeddings.

### Estructura

```
face-antispoofing-api/
├── app/
│   ├── main.py             # FastAPI endpoints
│   ├── model.py            # Inferencia ONNX (anti-spoofing)
│   ├── preprocessing.py    # Extracción de frames + landmarks
│   ├── face_comparison.py  # Comparación de rostros video vs INE (cascada r50→r100)
│   └── reasoning.py        # Generación de explicación del resultado
├── model/
│   ├── model.onnx          # Modelo anti-spoofing
│   ├── model.onnx.data     # Pesos del modelo
│   └── face_landmarker.task  # MediaPipe landmarks
├── web/                    # Interfaz de prueba
├── test_facematch.py
├── requirements.txt
└── setup.sh
```
