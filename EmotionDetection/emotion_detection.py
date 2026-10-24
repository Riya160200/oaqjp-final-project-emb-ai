import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def emotion_detector(text_to_analyze):
    """Analyze text and return emotion scores plus the dominant emotion."""
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None, "disgust": None, "fear": None,
            "joy": None, "sadness": None, "dominant_emotion": None
        }

    try:
        response = requests.post(
            URL,
            headers=HEADERS,
            json={"raw_document": {"text": text_to_analyze}},
            timeout=20,
        )
    except requests.RequestException:
        return {
            "anger": None, "disgust": None, "fear": None,
            "joy": None, "sadness": None, "dominant_emotion": None
        }

    if response.status_code == 400:
        return {
            "anger": None, "disgust": None, "fear": None,
            "joy": None, "sadness": None, "dominant_emotion": None
        }
    if response.status_code != 200:
        response.raise_for_status()

    data = response.json()
    # Watson service response normally stores emotion scores under this path.
    emotions = data.get("emotionPredictions", [{}])[0].get("emotion", {})
    if not emotions:
        emotions = data.get("emotion", data.get("emotion_scores", {}))

    scores = {
        name: emotions.get(name)
        for name in ("anger", "disgust", "fear", "joy", "sadness")
    }
    valid_scores = {name: value for name, value in scores.items()
                    if isinstance(value, (int, float))}
    dominant = max(valid_scores, key=valid_scores.get) if valid_scores else None
    scores["dominant_emotion"] = dominant
    return scores
