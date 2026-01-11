import json
import os
import re
from copy import deepcopy

def parse_loc_mapping(mapping_text):
    """Parse LOC Mapping text into a dictionary {snippet_loc: original_loc}."""
    mapping = {}
    lines = mapping_text.strip().split("\n")
    for line in lines:
        snippet_loc, original_loc = line.split("\t")
        mapping[int(snippet_loc)] = int(original_loc)
    return mapping


def script_line_to_notebook_coord(script_ln, line_mapping, jupyter_mapping):
    key = norm_key(script_ln)
    if key not in line_mapping:
        return None

    jupyter_key = line_mapping[key]
    entry = jupyter_mapping.get(str(jupyter_key - 1))
    if not entry:
        return None

    cell = entry.get("cell")
    line = entry.get("line")
    if cell is None or line is None:
        return None

    return f"{cell}:{line}"


def map_to_original_jupyter(analysis_record,file_path):
    
    for i,pair in enumerate(analysis_record):
 
        script_mapping_path = os.path.join(f"{file_path[:-6]}-fact/_snippets/{pair}_mapping.fact")
        jupyter_mapping_path = os.path.join(f"{file_path[:-6]}.mapping.json")

        with open(jupyter_mapping_path, "r") as infile:
            jupyter_mapping = json.load(infile)
        
        if os.path.exists(script_mapping_path):
            with open(script_mapping_path,"r") as f:
                mapping_raw = f.read()
        
        line_mapping = parse_loc_mapping(mapping_raw)
        
        
        output = deepcopy(pair)

        # 1️⃣ --- REMAP DETECTION PHASE FIELDS ---

        # --- Remap "snippet" block line prefixes ---
        # --- Remap "snippet" block line prefixes ---
        if "snippet" in output and isinstance(output["snippet"], str):
            remapped_snippet = []
            for line in output["snippet"].split("\n"):
                m = re.match(r"^(\d+)\s+(.*)", line)
                if m:
                    snippet_ln = m.group(1)
                    code = m.group(2)
                    key = norm_key(snippet_ln)
                    orig = "UNKNOWN_{snippet_ln}"
                    if key in line_mapping:
                        jupyter_key = line_mapping[key]
                        if str(jupyter_key-1) in jupyter_mapping:
                            jupyter_mapping_entry = jupyter_mapping[str(jupyter_key-1)]
                            cell = jupyter_mapping_entry.get("cell")
                            line = jupyter_mapping_entry.get("line")
                            orig = f"{cell}:{line }"
                    remapped_snippet.append(f"{orig} {code}")
                else:
                    remapped_snippet.append(line)
            output["snippet"] = "\n".join(remapped_snippet)

        # --- Remap model info references ---
        if "model info" in output and isinstance(output["model info"], str):
            def repl(m):
                script_ln = m.group(1)
                coord = script_line_to_notebook_coord(
                    script_ln, line_mapping, jupyter_mapping
                )
                return f"line: {coord}" if coord else f"line: {script_ln}"

            output["model info"] = re.sub(
                r"line:\s*(\d+)",
                repl,
                output["model info"]
            )


        # --- Remap leakage_lines ---
        if "preproc_leakage_lines" in output and isinstance(output["preproc_leakage_lines"], list):
            for item in output["preproc_leakage_lines"]:
                script_ln = item["line_number"]

                coord = script_line_to_notebook_coord(
                    script_ln, line_mapping, jupyter_mapping
                )

                if coord:
                    item["line_number"] = coord

                item["explanation"] = re.sub(
                    r"line\s+(\d+)",
                    lambda m: (
                        f"line {script_line_to_notebook_coord(m.group(1), line_mapping, jupyter_mapping)}"
                        if script_line_to_notebook_coord(m.group(1), line_mapping, jupyter_mapping)
                        else f"line {m.group(1)}"
                    ),
                    item["explanation"]
                )

        # --- Remap leakage_lines ---
        if "overlap_leakage_lines" in output and isinstance(output["overlap_leakage_lines"], list):
            for item in output["overlap_leakage_lines"]:
                script_ln = item["line_number"]

                coord = script_line_to_notebook_coord(
                    script_ln, line_mapping, jupyter_mapping
                )

                if coord:
                    item["line_number"] = coord

                item["explanation"] = re.sub(
                    r"line\s+(\d+)",
                    lambda m: (
                        f"line {script_line_to_notebook_coord(m.group(1), line_mapping, jupyter_mapping)}"
                        if script_line_to_notebook_coord(m.group(1), line_mapping, jupyter_mapping)
                        else f"line {m.group(1)}"
                    ),
                    item["explanation"]
                )



        analysis_record[i] = output
    return analysis_record

def norm_key(k):
    """Normalize keys used for indexing line_mapping: strip + int if possible, else return original."""
    try:
        return int(k.strip()) if isinstance(k, str) else int(k)
    except Exception:
        return k


    
    