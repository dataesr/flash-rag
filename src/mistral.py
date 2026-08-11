import os
import base64
import json
from typing import List, Dict, Any
import numpy as np
from mistralai.client import Mistral
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction, Space
from chromadb.utils.embedding_functions import register_embedding_function
from dotenv import load_dotenv

MAX_DOCUMENTS_PER_BATCH = 8

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)


def batch_documents_for_embedding(input: Documents, max_documents_per_batch: int = MAX_DOCUMENTS_PER_BATCH) -> list[Documents]:
    """Split a document list into smaller batches to stay under Mistral's request limits."""
    if not input:
        return []

    if max_documents_per_batch <= 0:
        raise ValueError("max_documents_per_batch must be greater than zero")

    return [list(input[i : i + max_documents_per_batch]) for i in range(0, len(input), max_documents_per_batch)]


def encode_file(path: str) -> str | None:
    try:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    except Exception as error:
        print(f"[error] Error while encoding {path}: {error}")
        return None


def mistral_ocr(document_path: str, document_name: str) -> dict | None:
    encoded_file = encode_file(document_path)
    if not encoded_file:
        return None

    try:
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{encoded_file}",
                "document_name": document_name,
            },
            extract_footer=True,
            extract_header=True,
        )
        data = json.loads(response.model_dump_json())
        return data
    except json.JSONDecodeError as error:
        print(f"[error] Error while decoding json response: {error}")
        print(f"[debug] response: {response}")
        return None
    except Exception as error:
        print(f"[error] Error while processing {document_name}: {error}")
        return None


@register_embedding_function
class MistralEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self):
        """
        Initialize the MistralEmbeddingFunction.
        """
        self.model = "mistral-embed"
        if not client:
            raise ValueError("Mistral client not initialized")
        self.client = client

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.
        """
        if not all(isinstance(item, str) for item in input):
            raise ValueError("Mistral only supports text documents, not images")

        batch_inputs = batch_documents_for_embedding(input)
        embeddings: list[np.ndarray] = []
        for batch in batch_inputs:
            output = self.client.embeddings.create(
                model=self.model,
                inputs=batch,
            )
            embeddings.extend(np.array(data.embedding) for data in output.data)

        return embeddings

    @staticmethod
    def name() -> str:
        return "mistral"

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> List[Space]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction[Documents]":
        return MistralEmbeddingFunction()

    def get_config(self) -> Dict[str, Any]:
        return {
            "model": "mistral-embed",
        }
