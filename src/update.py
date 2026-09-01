import argparse
from typing import Literal
from pydantic import BaseModel
from src.pipelines.load_ssmesr import load as load_ssmesr
from src.pipelines.load_eesr import load as load_eesr
from src.pipelines.extract_ssmesr import extract as extract_ssmesr
from src.pipelines.transform_ssmesr import transform as transform_ssmesr
from src.pipelines.transform_eesr import transform as transform_eesr
from src.populate import populate

SOURCES = ["ssmesr", "eesr"]
LOAD_FNC = {
    "ssmesr": load_ssmesr,
    "eesr": load_eesr,
}
EXTRACT_FNC = {
    "ssmesr": extract_ssmesr,
}
TRANSFORM_FNC = {
    "ssmesr": transform_ssmesr,
    "eesr": transform_eesr,
}


class UpdateRequest(BaseModel):
    task: Literal["all", "load", "extract", "transform", "populate"] = "all"
    source: Literal["all", "ssmesr", "eesr"] = "all"
    use_cache: bool = True
    force_download: bool = False
    db_override: bool = False
    db_reset: bool = False


def update(payload: UpdateRequest):
    sources = [payload.source] if payload.source != "all" else SOURCES
    for source in sources:
        load_fnc = LOAD_FNC.get(source)
        extract_fnc = EXTRACT_FNC.get(source)
        transform_fnc = TRANSFORM_FNC.get(source)

        if payload.task in ["all", "load"]:
            # load new documents
            if load_fnc:
                print(f"\n{'='*60}")
                print(f"=== Loading {source.upper()} documents ===")
                print(f"{'='*60}")
                load_fnc(use_cache=payload.use_cache, force_download=payload.force_download)

        if payload.task in ["all", "extract"]:
            # extract documents (OCR)
            if extract_fnc:
                print(f"\n{'='*60}")
                print(f"=== Extracting {source.upper()} documents ===")
                print(f"{'='*60}")
                extract_fnc(use_cache=payload.use_cache)

        if payload.task in ["all", "transform"]:
            # transform documents (chunking)
            if transform_fnc:
                print(f"\n{'='*60}")
                print(f"=== Chunking {source.upper()} documents ===")
                print(f"{'='*60}")
                transform_fnc(use_cache=payload.use_cache)

    if payload.task in ["all", "populate"]:
        # populate collection
        print(f"\n{'='*60}")
        print("=== Populating collection ===")
        print(f"{'='*60}")
        populate(source=payload.source, reset=payload.db_reset, override=payload.db_override)

    print(f"\n{'='*60}")
    print("=== Update Complete ===")
    print(f"{'='*60}\n")


def update_cli():
    parser = argparse.ArgumentParser(description="Update the database with new documents")
    parser.add_argument(
        "--task", choices=["all", "load", "extract", "transform", "populate"], default="all", help="Task to perform"
    )
    parser.add_argument("--source", choices=["all", "ssmesr", "eesr"], default="all", help="Source to update")
    parser.add_argument("--no-cache", action="store_true", help="Force reprocessing of documents")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of documents")
    parser.add_argument("--db-override", action="store_true", help="Override existing documents in the database")
    parser.add_argument("--db-reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()

    payload = UpdateRequest(
        task=args.task,
        source=args.source,
        use_cache=not args.no_cache,
        force_download=args.force_download,
        db_override=args.db_override,
        db_reset=args.db_reset,
    )
    update(payload)


if __name__ == "__main__":
    update_cli()
