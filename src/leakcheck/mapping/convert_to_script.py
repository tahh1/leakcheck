def normalize_source(source):
    """
    Ensure cell source is a list of strings.
    """
    if isinstance(source, list):
        return source
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return []


def transform_ipython_line(line):
    stripped = line.lstrip()

    # Assignment from shell
    if "=" in line and stripped.split("=", 1)[1].lstrip().startswith("!"):
        lhs, rhs = line.split("=", 1)
        cmd = rhs.lstrip()[1:].rstrip("\n")
        return f"{lhs.strip()} = get_ipython().getoutput({cmd!r})\n", True

    # Line magic
    if stripped.startswith("%") and not stripped.startswith("%%"):
        cmd = stripped[1:].rstrip("\n")
        if " " in cmd:
            name, args = cmd.split(" ", 1)
        else:
            name, args = cmd, ""
        return (
            f"get_ipython().run_line_magic({name!r}, {args!r})\n",
            False,
        )

    # Shell command
    if stripped.startswith("!"):
        cmd = stripped[1:].rstrip("\n")
        return f"get_ipython().system({cmd!r})\n", False

    # Help / introspection
    if stripped.rstrip().endswith("?"):
        return f"# {line.rstrip()}\n", False

    # Normal Python
    return line, True

def transform_cell(cell):
    source = normalize_source(cell["source"])
    if not source:
        return [], []

    first = source[0].lstrip()

    # Cell magic
    if first.startswith("%%"):
        header = first[2:].rstrip("\n")
        if " " in header:
            name, args = header.split(" ", 1)
        else:
            name, args = header, ""

        body = "".join(source[1:]).replace("'''", "\\'''")
        code = (
            f"get_ipython().run_cell_magic("
            f"{name!r}, {args!r}, '''{body}''')\n"
        )
        return code.splitlines(keepends=True), []  # no mapping

    # Normal cell
    output = []
    mapped_lines = []
    for j, line in enumerate(source):
        new_line, is_python = transform_ipython_line(line)
        
        output.append(new_line)
        if is_python:
            mapped_lines.append(j)

    return output, mapped_lines

def normalize_to_utf8(text):
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text.encode("utf-8", errors="replace").decode("utf-8")


def jupyter_to_script(jupyter_file):
    src_code = []
    script_to_jupyter_line_mapping = {}
    script_line_count = -1

    for i, cell in enumerate(jupyter_file["cells"]):
        if cell["cell_type"] != "code":
            continue
        
        
        src_code.append(f"# =====Cell {i} | type: code=====\n")
        script_line_count += 1
        lines, mapped_input_lines = transform_cell(cell)
        if lines == ['']:
            continue
        last_line = ""

        for out_idx, line in enumerate(lines):
            src_code.append(normalize_to_utf8(line))
            script_line_count += 1
            
            # Only map real Python lines
            if out_idx in mapped_input_lines:
                script_to_jupyter_line_mapping[script_line_count] = {
                    "cell": i,
                    "line": out_idx,
                    "code": normalize_to_utf8(line)
                }
                last_line = normalize_to_utf8(line)

        if last_line:
            src_code += ["\n\n"] if last_line[-1] != "\n" else ["\n"]
            script_line_count += 1

    return src_code, script_to_jupyter_line_mapping
  
    

