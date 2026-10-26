
import unittest
from app import app


class TestStudentPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 90,
                "previous_score": 88
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            response.json["prediction"],
            ["PASS", "FAIL"]
        )

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"attendance": 90}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
