import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_happy_text(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01, "disgust": 0.01, "fear": 0.02,
                    "joy": 0.91, "sadness": 0.05
                }
            }]
        }
        mock_post.return_value = mock_response
        result = emotion_detector("I am happy")
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.91)

    def test_blank_input(self):
        result = emotion_detector("   ")
        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
