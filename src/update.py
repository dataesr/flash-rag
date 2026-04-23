from pydantic import BaseModel
from src.load import load
from src.extract import extract
from src.parse import parse
from src.populate import populate


class LoadRequest(BaseModel):
    load_skip_fetch: bool = True
    load_skip_download: bool = True
    load_force_download: bool = False


class ExtractRequest(BaseModel):
    extract_force: bool = False


class ParseRequest(BaseModel):
    parse_force: bool = False


class PopulateRequest(BaseModel):
    populate_override: bool = False
    populate_reset: bool = False


class UpdateRequest(LoadRequest, ExtractRequest, ParseRequest, PopulateRequest):
    pass


def run_load(payload: LoadRequest):
    # load new documents
    print(f"\n{'='*60}")
    print("=== Loading documents ===")
    print(f"{'='*60}")
    load(
        skip_fetch=payload.load_skip_fetch,
        skip_download=payload.load_skip_download,
        force_download=payload.load_force_download,
    )


def run_extract(payload: ExtractRequest):
    # extract documents (OCR)
    print(f"\n{'='*60}")
    print("=== Extracting documents ===")
    print(f"{'='*60}")
    extract(force_extract=payload.extract_force)


def run_parse(payload: ParseRequest):
    # parse documents
    print(f"\n{'='*60}")
    print("=== Parsing documents ===")
    print(f"{'='*60}")
    parse(force_parse=payload.parse_force)


def run_populate(payload: PopulateRequest):
    # populate collection
    print(f"\n{'='*60}")
    print("=== Populating collection ===")
    print(f"{'='*60}")
    populate(reset=payload.populate_reset, override=payload.populate_override)


def run_update(payload: UpdateRequest):
    run_load(LoadRequest(**payload.model_dump()))
    run_extract(ExtractRequest(**payload.model_dump()))
    run_parse(ParseRequest(**payload.model_dump()))
    run_populate(PopulateRequest(**payload.model_dump()))

    print(f"\n{'='*60}")
    print("=== Update Complete ===")
    print(f"{'='*60}\n")
