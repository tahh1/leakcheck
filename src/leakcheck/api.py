import logging

from leakcheck.pipeline import run_detection_pipeline
from leakcheck.checks import validate_pipeline_inputs

logger = logging.getLogger(__name__)

def detect_leaks(
    input_file: str,
    output_directory: str,
    slicer_url: str,
    leak_type: str = "both",
):
    """
    Public API for running leakage detection on a Jupyter notebook or script.
    """
    logger.info("Starting leak detection for %s", input_file)
    leak_type = validate_pipeline_inputs(
        input_file=input_file,
        output_directory=output_directory,
        slicer_url=slicer_url,
        leak_type=leak_type,
    )
    result_path = run_detection_pipeline(
        input_file=input_file,
        output_directory=output_directory,
        slicer_url=slicer_url,
        leak_type=leak_type,
    )
    logger.info("Leak detection results saved to %s", result_path)
    return result_path




    
    


            
    
