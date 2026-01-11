import logging
import os
import json
import requests
import pandas as pd
from leakcheck.mapping.mapping import map_to_original_jupyter
from leakcheck.detection.detector import detect_leakage
from leakcheck.mapping.convert_to_script import jupyter_to_script

logger = logging.getLogger(__name__)




def _get_base_name(input_file: str) -> str:
    return os.path.splitext(os.path.basename(input_file))[0]




def collect_input(dir_path: str):
    
    snippets = []
    model_infos = []

    for filename in os.listdir(dir_path):
        if not filename.endswith("_original.py"):
            continue

        base_name = filename[:-len("_original.py")]
        model_info_path = os.path.join(
            dir_path, f"{base_name}_snippet_model_info.csv"
        )

        if not os.path.exists(model_info_path):
            continue

        original_path = os.path.join(dir_path, filename)
        with open(original_path, "r", encoding="utf-8") as f:
            snippets.append(f.read())

        df = pd.read_csv(
            model_info_path,
            delimiter="\t",
            names=["train", "train_line", "test", "test_line"],
        )

        model_info = (
            f"[Training method: {df['train'][0]}, Training line: {df['train_line'][0]}]\n"
            f"[Testing method: {df['test'][0]}, Testing line: {df['test_line'][0]}]"
        )

        model_infos.append(model_info)

    logger.info("Collected %d snippets from %s", len(snippets), dir_path)
    return snippets, model_infos

        
def convert_jupyter_to_script(
    input_file: str,
    output_directory: str,
):
    
    logger.info("Converting notebook to script for %s", input_file)
    with open(input_file, "r") as infile:
        content = infile.read()
        
    content = json.loads(content.decode("utf-8", errors="replace"))
    
    src_code, script_to_jupyter_line_mapping = jupyter_to_script(content)

    output_path = os.path.join(
        output_directory,
        f"{_get_base_name(input_file)}.py",
    )
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(src_code)
        
    with open(f"{output_path[:-3]}.mapping.json", "w", encoding="utf-8") as outfile:
        outfile.write(json.dumps(script_to_jupyter_line_mapping))
    logger.info("Wrote script and mapping to %s", output_directory)
   
def collect_slices(
    input_file: str,
    output_directory: str,
):
    
    
    snippets_dir = os.path.join(
        output_directory,
        f"{_get_base_name(input_file)}_snippets",
    )
    
    if not os.path.exists(snippets_dir):
        raise FileNotFoundError(f"We could not find the _snippets folder for {os.path.basename(input_file)}")
    
     
    snippets,model_infos = collect_input(dir_path=snippets_dir)   
    
    analysis_record = [{"snippet":snippet,"model info":model_info} 
                       for snippet,model_info in zip(snippets,model_infos)]
    
    logger.info("Built analysis record with %d entries", len(analysis_record))
    return analysis_record

def run_slicer(slicer_url: str, input_file: str, output_directory: str):

    url = f"{slicer_url}/slice"
    logger.info("Running slicer at %s for %s", url, input_file)

    with open(input_file, "rb") as f:
        response = requests.post(
            url,
            files={"file": f}
        )

    output_path = os.path.join(
        output_directory,
        f"{_get_base_name(input_file)}_results.zip",
    )
    
    with open(output_path, "wb") as out:
        out.write(response.content)
        
    unzip_path = output_path[:-11] + "_snippets"
    os.system(f"unzip -o {output_path} -d {unzip_path}")
    logger.info("Slicer output extracted to %s", unzip_path)
        
def run_detection_pipeline(
    input_file: str,
    output_directory: str,
    slicer_url: str,
    leak_type: str,
    num_shots: int = 3,
):
    logger.info("Starting detection pipeline with leak_type=%s", leak_type)
    input_extension = os.path.splitext(input_file)[1].lower()

    # 1. Convert notebook to script
    if input_extension == ".ipynb":
        convert_jupyter_to_script(input_file, output_directory)

    # 2. Run slicer
    run_slicer(slicer_url, input_file, output_directory)

    # 3. Collect slices
    analysis_record = collect_slices(input_file, output_directory)

    # 4. Run detection
    if leak_type == "both":
        analysis_record = detect_leakage(analysis_record, "preproc", num_shots)
        analysis_record = detect_leakage(analysis_record, "overlap", num_shots)
    else:
        analysis_record = detect_leakage(analysis_record, leak_type, num_shots)

    # 5. Map results back to Jupyter
    analysis_record = map_to_original_jupyter(analysis_record, input_file)

    # 6. Persist results
    save_path = os.path.join(
        output_directory,
        f"{_get_base_name(input_file)}_detection_results.json",
    )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {"Pairs analysis results": analysis_record},
            f,
            indent=4
        )

    logger.info("Detection pipeline finished; results saved to %s", save_path)
    return save_path
