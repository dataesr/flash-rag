import argparse
from typing import Literal
from pydantic import BaseModel
from src.pipelines.load_ssmesr import load as load_ssmesr
from src.pipelines.load_eesr import load as load_eesr
from src.pipelines.extract_ssmesr import extract as extract_ssmesr
from src.pipelines.transform_ssmesr import transform as transform_ssmesr
from src.pipelines.transform_eesr import transform as transform_eesr
from src.populate import populate

REFERENCES = ["ssmesr", "eesr"]
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
    reference: Literal["all", "ssmesr", "eesr"] = "all"
    use_cache: bool = True
    use_fetch: bool = True
    force_download: bool = False
    force_ocr: bool = False
    db_override: bool = False
    db_reset: bool = False


def update(payload: UpdateRequest):
    refs = [payload.reference] if payload.reference != "all" else REFERENCES
    for ref in refs:
        load_fnc = LOAD_FNC.get(ref)
        extract_fnc = EXTRACT_FNC.get(ref)
        transform_fnc = TRANSFORM_FNC.get(ref)

        if payload.task in ["all", "load"]:
            # load new documents
            if load_fnc:
                print(f"\n{'='*60}")
                print(f"=== Loading {ref.upper()} documents ===")
                print(f"{'='*60}")
                load_fnc(use_cache=payload.use_cache, use_fetch=payload.use_fetch, force_download=payload.force_download)

        if payload.task in ["all", "extract"]:
            # extract documents (OCR)
            if extract_fnc:
                print(f"\n{'='*60}")
                print(f"=== Extracting {ref.upper()} documents ===")
                print(f"{'='*60}")
                extract_fnc(use_cache=payload.use_cache, force_ocr=payload.force_ocr)

        if payload.task in ["transform"]:
            # transform documents (chunking)
            if transform_fnc:
                print(f"\n{'='*60}")
                print(f"=== Chunking {ref.upper()} documents ===")
                print(f"{'='*60}")
                transform_fnc(use_cache=payload.use_cache)

    if payload.task in ["all", "populate"]:
        # populate collection
        print(f"\n{'='*60}")
        print("=== Populating collection ===")
        print(f"{'='*60}")
        populate(
            reference=payload.reference,
            use_cache=payload.use_cache,
            reset=payload.db_reset,
            override=payload.db_override,
        )

    print(f"\n{'='*60}")
    print("=== Update Complete ===")
    print(f"{'='*60}\n")


def update_cli():
    parser = argparse.ArgumentParser(description="Update the database with new documents")
    parser.add_argument(
        "--task", choices=["all", "load", "extract", "transform", "populate"], default="all", help="Task to perform"
    )
    parser.add_argument("--reference", choices=["all", "ssmesr", "eesr"], default="all", help="Reference to update")
    parser.add_argument("--no-cache", action="store_true", help="Force reprocessing of documents")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of documents")
    parser.add_argument("--force-ocr", action="store_true", help="Force re-ocr of files")
    parser.add_argument("--db-override", action="store_true", help="Override existing documents in the database")
    parser.add_argument("--db-reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()

    payload = UpdateRequest(
        task=args.task,
        reference=args.reference,
        use_cache=not args.no_cache,
        force_download=args.force_download,
        force_ocr=args.force_ocr,
        db_override=args.db_override,
        db_reset=args.db_reset,
    )
    update(payload)


if __name__ == "__main__":
    update_cli()
