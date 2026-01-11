try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.8 fallback
    from importlib_resources import files
from pathlib import Path

class Config(object):
    def __init__(self,overlap_fs_path: str,preproc_fs_path:str,model:str):
        self.overlap_fs_path = overlap_fs_path
        self.preproc_fs_path = preproc_fs_path
        self.model = model

OVERLAP_INDEX = (
    files("leakcheck")
    .joinpath("data/In-Context Learning examples - Examples with numbering and explanations and model info - overlap.csv")
)
PREPROC_INDEX = (
    files("leakcheck")
    .joinpath("data/In-Context Learning examples - Examples with numbering and explanations and model info - preproc.csv")
)
configs = Config(Path(OVERLAP_INDEX),
                 Path(PREPROC_INDEX),
                 "o3-mini-2025-01-31")
    