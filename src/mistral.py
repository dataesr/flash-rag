import os
import base64
import json
import numpy as np
from typing import List, Dict, Any
from mistralai.client import Mistral
from mistralai.client.models import SystemMessage, UserMessage
from chromadb.api.types import Embeddings, Documents, EmbeddingFunction, Space
from chromadb.utils.embedding_functions import register_embedding_function
from dotenv import load_dotenv

MAX_DOCUMENTS_PER_BATCH = 8

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_RAG_MODEL = "mistral-small-latest"

client = Mistral(api_key=MISTRAL_API_KEY)


def batch_mistral_documents(input: Documents, max_documents_per_batch: int = MAX_DOCUMENTS_PER_BATCH) -> list[Documents]:
    """Split a document list into smaller batches to stay under Mistral's request limits."""
    if not input:
        return []

    if max_documents_per_batch <= 0:
        raise ValueError("max_documents_per_batch must be greater than zero")

    return [list(input[i : i + max_documents_per_batch]) for i in range(0, len(input), max_documents_per_batch)]


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

        batches = batch_mistral_documents(input)
        embeddings: list[np.ndarray] = []
        for batch in batches:
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


SYSTEM_PROMPT = """Tu es un assistant d'analyse de données. Tu réponds aux questions en te basant UNIQUEMENT sur les documents fournis.

Règles strictes :
1. Réponds directement à la question posée
2. Cite les sources [nom du fichier - titre du document - année] quand tu cites des données
3. Si les documents ne contiennent pas la réponse, dis-le explicitement
4. Pour les chiffres : sois précis, inclus les années et les unités
5. Si plusieurs documents contiennent des informations contradictoires, note-le
6. Sois concis : pas de phrases inutiles

Format : réponse courte et factuelle."""

USER_PROMPT = """Documents :
---
{docs}
---

Question : {query}

Réponse :"""


def build_user_prompt(query: str, documents: list[dict]) -> str:
    """
    Build rag user prompt from a query and a list of documents
    """

    # Format docs
    docs = "\n\n".join(
        [
            f"[{doc['metadata'].get('title', 'Sans titre')} ({doc['metadata'].get('publication_date', 'N/A')})]"
            f"\n{doc['document']}"
            for doc in documents
        ]
    )
    prompt = USER_PROMPT.format(query=query, docs=docs)
    return prompt


def mistral_rag_answer(query_text: str, documents: list[dict]) -> str:
    user_prompt = build_user_prompt(query_text, documents)
    print(f"[debug] User prompt:\n{user_prompt}")

    chat_response = client.chat.complete(
        model=MISTRAL_RAG_MODEL,
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),
            UserMessage(content=user_prompt),
        ],
        temperature=0.1,
        max_tokens=500,
    )

    answer: str = chat_response.choices[0].message.content
    return answer.strip()
