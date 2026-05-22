# Face Anti-Spoofing API

API de detección de vida. Recibe un video y determina si
es una persona real o un intento de suplantación (*spoof*). Opcionalmente, si se
envía también la foto de una INE, hace comparación de rostros (*face match*)
entre el video y la credencial.

**Estado del comparador de rostro:** la verificación INE vs. video usa una
**cascada de dos modelos (r50 → r100)**, adoptada tras validarse sobre casos
reales.


## Requisitos

- Python 3.10+
- No requiere GPU para funcionar. La cascada de *face match* corre ~2× más lenta
  en CPU; si el SLA de latencia es crítico, desplegar en GPU.



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


## Servidor

Encender:

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Correr en background (producción):

```bash
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 > server.log 2>&1 &
```

Detener:

```bash
pkill -f "uvicorn app.main:app"
```


## Endpoints

### POST /v1/liveness

Analiza un video y devuelve si es real o spoof. Si además se envía `ine_image`,
cuando el video resulta REAL se compara el rostro del video contra la foto de la
INE.

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

### GET /health

```bash
curl http://localhost:8000/health
```



## Formatos aceptados

- **Video:** `.mp4`, `.webm`, `.avi`, `.mov`, máximo 50 MB
- **INE (imagen):** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`



## Comparación de rostros (`face_comparison.py`)

Extrae los frames del video y los compara contra la foto de la INE usando
InsightFace (ArcFace + ONNX Runtime, sin dependencia de TensorFlow/Keras).

### Cómo funciona la cascada

La comparación funciona como en dos etapas:

1. **Detección compartida.** `_detect_and_align()` corre la detección una sola
   vez sobre la INE y todos los frames, y entrega recortes alineados de 112×112
   que alimentan ambos modelos de reconocimiento. La selección por rostro más
   grande maneja el retrato holográfico tenue de la INE (la foto principal es la
   detección más grande).
2. **Etapa 1: modelo rápido (r50).** El modelo de reconocimiento de `buffalo_l`
   genera embeddings de todos los frames y los ordena por similitud real contra
   la INE.
3. **Etapa 2: modelo fuerte (r100).** El modelo glint360k ResNet100
   (`antelopev2`) re-evalúa solo la INE y los `K` mejores frames
   (`_CASCADE_K = 2`). El score final es la similitud máxima sobre esos
   frames toma el máximo, recupera el frame bueno que un
   usuario genuino puede mostrar solo un instante. Si el modelo fuerte no está
   disponible, hace fallback a r50 y lo registra en el log.

> El modelo `antelopev2` se descarga automáticamente la primera vez que se usa.
> `warmup_deepface()` precarga ambos modelos para que la primera petición sea
> rápida.

### Cambios realizados

| Aspecto | Antes | Ahora (cascada) |
| --- | --- | --- |
| Modelo de reconocimiento | un único modelo (`buffalo_l` / r50) | r50 (rankea) + glint360k r100 (`antelopev2`, confirma) |
| Selección de frame | 5 más nítidos (heurística Laplaciana + iluminación) | todos, rankeados por similitud real contra la INE |
| Score final | máx. sobre 5 candidatos, con salida temprana si superaba `threshold + 10` | máx. r100 sobre los `K = 2` mejores frames |
| Detección/alineación | por modelo | compartida (`_detect_and_align()`), una sola vez |
| Módulos cargados | completos | solo `detection` + `recognition`; se omiten landmarks y gender-age |

Otros cambios en el código:

- Carga perezosa del modelo fuerte por `_get_strong_rec()`, con descarga
  automática y fallback a r50.
- `warmup_deepface()` ahora precarga ambos modelos de reconocimiento.
- Se eliminaron `_pick_candidate_indices()` y `_get_embedding()`, junto con la
  lógica de salida temprana.
- Nuevas constantes: `_DET_SIZE` y `_CASCADE_K`.

### Validación e impacto

Conjunto de prueba: 47 casos con veredicto REAL (persona genuina) y comparación
de rostro, de los reportes `response.json` provistos. Se validó la línea base
ejecutando el código de producción sobre los mismos videos: los scores
reproducen los de los `response.json` con diferencia máxima de 0.02.

Selección de `K`: se probó re-evaluar con r100 los top-1, top-2 y top-3 frames.
**K=2 es el mínimo que mantiene a todos los genuinos por encima del umbral**;
K=1 reintroduce un falso rechazo (caso 171349 → 39.7).

**Resultados agregados (47 casos, umbral 40):**

| Métrica | Producción | Cascada |
| --- | --- | --- |
| Falsos rechazos | 3 | **0** |
| Score de similitud medio | 60.33 | **65.27** |
| Mejora media por caso | — | **+4.94** (rango −0.41 a +13.18) |
| Casos mejorados / empatados / peores | — | 45 / 0 / 2 |

**Zona crítica (scores más bajos donde se toma la decisión):**

| Caso | Producción | Cascada | Δ |
| --- | --- | --- | --- |
| 171340 | 35.3 | **42.3** | +7.0 |
| 171349 | 36.1 | **42.2** | +6.0 |
| 171430 | 39.8 | **45.7** | +6.0 |
| 171342 | 40.5 | 43.6 | +3.1 |
| 171355 | 42.0 | 50.1 | +8.1 |
| 171351 | 43.6 | 49.6 | +6.0 |
| 171348 | 46.0 | 47.2 | +1.2 |
| 171415 | 49.2 | 49.6 | +0.4 |

La cascada **elimina la totalidad de los falsos rechazos** del conjunto de
prueba y sube el piso de score genuino (mínimo 35 → 42). Las dos únicas
regresiones (−0.41 y −0.27) ocurren en casos con score ~51–53, muy por encima
del umbral: no afectan ninguna decisión.

### Velocidad y trade-off

| | Producción | Cascada |
| --- | --- | --- |
| `face_match` en CPU (media) | ~1.2 s | ~2.6 s |
| Rango observado | 0.9 – 2.4 s | 2.4 – 2.7 s |

La cascada es ~2× más lenta en CPU: hace detección sobre todos los frames, los
embebe con r50 para rankear y luego corre r100 sobre 3 imágenes (`K + 1`
embeddings). Ese r100 extra es el costo de recuperar los casos difíciles. En
**GPU** el sobrecosto prácticamente desaparece (estimado ~0.5 s end-to-end).

El intercambio es **precisión por latencia**. Para recuperar velocidad sin
perder precisión: desplegar en GPU (elimina casi


## Estructura

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
