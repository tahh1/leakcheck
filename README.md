# LeakCheck

LeakCheck is an LLM-based data leakage detection pipeline built on top of our ml-slicer. It converts notebooks to scripts, slices code into train/test snippets, and runs few-shot LLM checks for preprocessing and overlap leakage.

## Features

- End-to-end leakage detection pipeline for notebooks.
- Few-shot LLM detection for preprocessing and overlap leakage.
- Maps detections back to notebook cell and line coordinates.
- Optional logging helpers for pipeline visibility.

## Requirements

- Python 3.10+
- A running slicer service that exposes a `POST /slice` endpoint (and optionally `GET /health`).
- An LLM API key available as `api_key` in your environment.

## Installation

```bash
pip install -e .
```

If you do not already have ml-slicer installed, you can run it as docker container following this [[repo](https://github.com/tahh1/ml-slicer)] .

## Quickstart

```python
from leakcheck.logging_config import configure_logging
from leakcheck.api import detect_leaks

configure_logging()

result_path = detect_leaks(
    input_file="/path/to/notebook.ipynb",
    output_directory="/path/to/output",
    slicer_url="http://localhost:8000",
    leak_type="both",
)

print("Results saved to", result_path)
```

## Inputs and Outputs

- `input_file`: Jupyter notebook path (`.ipynb`).
- `output_directory`: created if it does not exist.
- `slicer_url`: base URL of the slicer service.
- `leak_type`: `preproc`, `overlap`, or `both`.

The pipeline writes a JSON report named `<notebook>_detection_results.json` in the output directory.

## Logging

LeakCheck uses the standard library `logging` module. Call `configure_logging()` once in your app to enable INFO-level logs:

```python
from leakcheck.logging_config import configure_logging

configure_logging()
```

## Notes

- The slicer is expected to return a ZIP of snippet data for the uploaded file.
