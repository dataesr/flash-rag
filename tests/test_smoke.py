import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from src.mistral import batch_mistral_documents
from src.utils import parse_key_value_pair


class SmokeTests(unittest.TestCase):
    def test_mistral_batches_respect_request_limit(self):
        documents = [f"document-{index}" for index in range(9)]

        batches = batch_mistral_documents(documents, max_documents_per_batch=4)

        self.assertEqual([len(batch) for batch in batches], [4, 4, 1])
        self.assertEqual([document for batch in batches for document in batch], documents)

    def test_mistral_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            batch_mistral_documents(["document"], max_documents_per_batch=0)

    def test_filter_parser_preserves_equals_in_value(self):
        self.assertEqual(parse_key_value_pair("reference=ssmesr"), ("reference", "ssmesr"))
        self.assertEqual(parse_key_value_pair("query=a=b"), ("query", "a=b"))

    def test_query_endpoint_returns_contract_without_external_services(self):
        expected = ([{"id": "source-1"}], "mistral_not_enabled", [])
        with patch("main.run_query", return_value=expected):
            response = TestClient(app).post("/query", json={"query": "test", "top_k": 1})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "query": "test",
            "sources": [{"id": "source-1"}],
            "answer": "mistral_not_enabled",
            "citations": [],
        })

    def test_static_frontend_is_served(self):
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Flash Notes", response.text)


if __name__ == "__main__":
    unittest.main()