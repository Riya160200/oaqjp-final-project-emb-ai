from flask import Flask, jsonify, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return (
        "<h1>Emotion Detector</h1>"
        "<p>Use /emotionDetector?textToAnalyze=I%20am%20happy</p>"
    )


@app.route("/emotionDetector", methods=["GET"])
def detect_emotion():
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return jsonify({"error": "Invalid input. Please provide text to analyze."}), 400

    result = emotion_detector(text_to_analyze)
    if not result or result.get("dominant_emotion") is None:
        return jsonify({"error": "Emotion analysis could not be completed."}), 502
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
