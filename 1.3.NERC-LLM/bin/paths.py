import os, sys

HERE = os.path.abspath(os.path.dirname(__file__))  # location of this file

# Default local model directory (UPC scratch layout from course materials).
# Override on your cluster: export LLM_MODEL_ROOT=/path/to/hf/checkpoints
DEFAULT_LLM_MODEL_ROOT = "/scratch/nas/1/PDI/mml0/models"


def resolve_llm_model_path(model: str) -> str:
    """
    Resolve the HuggingFace / local path for a short name or hub id.

    - Absolute paths or explicit relative paths (./ ../) are returned normalized.
    - If ``LLM_MODEL_ROOT/<model>`` exists as a directory, use it (local snapshot).
    - If ``model`` contains a slash (e.g. ``meta-llama/Llama-3.2-3B-Instruct``),
      treat it as a HuggingFace hub id and return it unchanged for ``from_pretrained``.
    - Otherwise return ``LLM_MODEL_ROOT/<model>`` (legacy layout).
    """
    if not model:
        return model
    root = os.environ.get("LLM_MODEL_ROOT", DEFAULT_LLM_MODEL_ROOT).rstrip(os.sep)
    if os.path.isabs(model):
        return os.path.normpath(model)
    if model.startswith((".", "./", "../")):
        return os.path.normpath(os.path.join(os.getcwd(), model))
    candidate = os.path.join(root, model)
    if os.path.isdir(candidate):
        return candidate
    if "/" in model:
        return model
    return candidate


def model_slug(model: str) -> str:
    """Filesystem-safe token derived from CLI model name (hub ids, paths)."""
    if not model:
        return model
    s = model.replace(os.sep, "_").replace("/", "__")
    if s.startswith("."):
        s = "_" + s[1:]
    return s


# one level up, current classifier approach for this task
CLASSIFIER = os.path.dirname(HERE) 
# needed directories for this classifier 
PREPROCESS = os.path.join(CLASSIFIER,"preprocessed")
MODELS = os.path.join(CLASSIFIER,"models")
RESULTS = os.path.join(CLASSIFIER,"results")

# three levels up, main project dir
MAIN = os.path.dirname(os.path.dirname(os.path.dirname(HERE))) 
# useful project directories
DATA = os.path.join(MAIN,"data") # down to "data"
RESOURCES = os.path.join(MAIN,"resources") # down to "resources"
UTIL = os.path.join(MAIN,"util") # down to "util"
# some useful scripts there
sys.path.append(UTIL)

