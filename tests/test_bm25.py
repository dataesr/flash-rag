import os
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import bm25


class BM25IndexTests(unittest.TestCase):
    def test_build_bm25_index_reads_collection_in_batches(self):
        documents = [f"document {index}" for index in range(2501)]
        ids = [f"id-{index}" for index in range(len(documents))]
        metadatas = [{} for _ in documents]
        collection = Mock()
        collection.count.return_value = len(documents)

        def get_batch(*, limit, offset, include):
            self.assertEqual(include, ["documents", "metadatas"])
            return {
                "ids": ids[offset : offset + limit],
                "documents": documents[offset : offset + limit],
                "metadatas": metadatas[offset : offset + limit],
            }

        collection.get.side_effect = get_batch

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.pkl")
            with (
                patch.object(bm25, "get_collection", return_value=collection),
                patch.object(bm25, "BM25_DIR", directory),
                patch.object(bm25, "BM25_PATH", index_path),
                patch.object(bm25, "tokenize", side_effect=lambda text: text.split()),
            ):
                self.assertTrue(bm25.build_bm25_index())

            with open(index_path, "rb") as index_file:
                index = pickle.load(index_file)

        self.assertEqual(
            [call.kwargs["offset"] for call in collection.get.call_args_list],
            [0, 1000, 2000],
        )
        self.assertEqual(
            [call.kwargs["limit"] for call in collection.get.call_args_list],
            [1000, 1000, 501],
        )
        self.assertEqual(index["ids"], ids)
        self.assertEqual(index["documents"], documents)


if __name__ == "__main__":
    unittest.main()
