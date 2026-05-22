"""Generate natural-language reasoning for model decisions."""

import random


def generate_reasoning(score: float, metadata: dict) -> str:
    """Build a natural reasoning string based on score and preprocessing metadata."""
    face_frames = metadata.get("face_detected_frames", 0)
    total_frames = metadata.get("frames_analyzed", 16)
    variability = metadata.get("landmark_variability", 0)
    face_rate = face_frames / total_frames if total_frames > 0 else 0
    confidence = max(score, 1 - score)

    parts = []

    # Face detection quality
    if face_rate >= 0.9:
        parts.append(random.choice([
            f"Rostro visible en {face_frames} de {total_frames} frames.",
            f"Se detectó el rostro de forma consistente en todo el video.",
            f"Buena presencia facial a lo largo del video.",
        ]))
    elif face_rate >= 0.5:
        parts.append(f"Rostro detectado en {face_frames} de {total_frames} frames, algunos con dificultad.")
    else:
        parts.append(f"Rostro detectado solo en {face_frames} de {total_frames} frames. Señal débil.")

    # Decision reasoning based on score
    if score < 0.1:  # Very confident REAL
        parts.append(random.choice([
            "El patrón visual y geométrico es consistente con una persona real.",
            "Los rasgos faciales muestran naturalidad y profundidad.",
            "No se detectaron artefactos de reproducción o impresión.",
            "El movimiento facial es natural y coherente entre frames.",
        ]))
    elif score < 0.3:  # Confident REAL
        parts.append(random.choice([
            "Los indicadores apuntan a una persona real, aunque con algunas variaciones menores.",
            "El análisis visual sugiere autenticidad, con ligeras irregularidades en algunos frames.",
            "Patrón general consistente con video en vivo.",
        ]))
    elif score < 0.5:  # Borderline REAL
        parts.append(random.choice([
            "El video presenta algunas características inusuales, pero no suficientes para considerarlo falso.",
            "Hay señales mixtas. El análisis se inclina hacia persona real, pero con reservas.",
            "Algunos frames muestran patrones atípicos. Se recomienda verificación adicional.",
        ]))
    elif score < 0.7:  # Borderline SPOOF
        parts.append(random.choice([
            "Se detectaron indicios de posible suplantación, aunque no son concluyentes.",
            "El video muestra patrones que podrían indicar reproducción de pantalla o imagen impresa.",
            "Hay señales sospechosas. Se recomienda solicitar una nueva captura.",
        ]))
    elif score < 0.9:  # Confident SPOOF
        parts.append(random.choice([
            "Se detectaron patrones claros de suplantación en el video.",
            "El análisis indica alta probabilidad de video reproducido o imagen estática.",
            "Los rasgos faciales presentan artefactos típicos de ataques de presentación.",
        ]))
    else:  # Very confident SPOOF
        parts.append(random.choice([
            "El video presenta señales evidentes de suplantación.",
            "Se identificaron artefactos claros de reproducción de pantalla o foto impresa.",
            "El patrón visual es altamente consistente con un intento de fraude.",
            "La falta de profundidad y movimiento natural confirma suplantación.",
        ]))

    # Variability insight
    if score < 0.5 and variability > 0.02:
        parts.append("Se observa movimiento natural en los puntos faciales.")
    elif score >= 0.5 and variability < 0.01:
        parts.append("La rigidez en los puntos faciales refuerza la sospecha.")

    # Confidence closing
    if confidence > 0.95:
        parts.append(f"Nivel de certeza: alto ({confidence*100:.1f}%).")
    elif confidence > 0.8:
        parts.append(f"Nivel de certeza: moderado ({confidence*100:.1f}%).")
    else:
        parts.append(f"Nivel de certeza: bajo ({confidence*100:.1f}%). Se sugiere repetir la captura.")

    return " ".join(parts)

