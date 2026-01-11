import logging
import os
import requests



ALLOWED_INPUT_EXTENSIONS = {".ipynb"}
ALLOWED_LEAK_TYPES = {"preproc", "overlap", "both"}

logger = logging.getLogger(__name__)


def _check_slicer_url(slicer_url: str) -> None:
    base_url = slicer_url.rstrip("/")
    logger.debug("Checking slicer URL reachability for %s", base_url)
    errors = []
    for candidate in (base_url, f"{base_url}/health"):
        try:
            response = requests.get(candidate, timeout=5)
        except requests.RequestException as exc:
            errors.append(f"{candidate} -> {exc}")
            continue

        if response.ok:
            return
        errors.append(f"{candidate} -> {response.status_code}")

    raise ConnectionError(f"Slicer URL is not reachable: {', '.join(errors)}")


def validate_pipeline_inputs(
    input_file: str,
    output_directory: str,
    slicer_url: str,
    leak_type: str,
) -> str:
    logger.debug("Validating pipeline inputs for %s", input_file)
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    extension = os.path.splitext(input_file)[1].lower()
    if extension != "ipynb":
        raise ValueError("Input file must be a Jupyter notebook with .ipynb extension.")

    normalized_leak_type = leak_type.strip().lower()
    if normalized_leak_type not in ALLOWED_LEAK_TYPES:
        raise ValueError("Leak type must be 'preproc', 'overlap', or 'both'.")

    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            raise NotADirectoryError(
                f"Output path exists but is not a directory: {output_directory}"
            )
    else:
        os.makedirs(output_directory, exist_ok=True)

    _check_slicer_url(slicer_url)
    logger.info("Pipeline input validation succeeded for %s", input_file)
    return normalized_leak_type
