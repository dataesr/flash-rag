import os
import base64
import json
from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=MISTRAL_API_KEY)


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
