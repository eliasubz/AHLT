#! /usr/bin/python3

import sys, os

import torch
from dataset import Dataset
from train import do_train
from predict import predict
from codemaps import DEFAULT_FEATURES

##########################################################
#
#  Experiment runner for NERC-NN (assignment 1.2)
#
#  Steps:
#    parse        — spaCy-parse XML files and save as .pck
#    train        — train all models in search_space (skips already trained)
#    predict_all  — run all trained models on devel (or test) and evaluate
#    predict      — run a single named model  (requires name=<modelname>)
#
#  Example full run:
#    python3 run.py parse
#    python3 run.py train epochs=10 batch_size=32
#    python3 run.py predict_all
#    python3 analyze_results.py
#
##########################################################

BINDIR  = os.path.abspath(os.path.dirname(__file__))
NERDIR  = os.path.dirname(BINDIR)
SOLDIR  = os.path.dirname(NERDIR)
MAINDIR = os.path.dirname(SOLDIR)
DATADIR = os.path.join(MAINDIR, "data")
UTILDIR = os.path.join(MAINDIR, "util")
sys.path.append(UTILDIR)
from evaluator import evaluate


# -----------------------------------------------------------------------
# Short codes used in model names to represent feature groups
# -----------------------------------------------------------------------
FEAT_SHORT = {
    'ortho':      'Or',
    'ext_full':   'EF',
    'ext_part':   'EP',
    'drug_morph': 'DM',
    'greek':      'Gk',
    'length':     'Ln',
    'spacy':      'Sp',
}


def get_model_name(p):
    """Unique name encoding all varied hyperparameters."""
    e_str = "-".join(map(str, p['emb_sizes']))
    f_str = "".join(FEAT_SHORT.get(g, g) for g in p.get('features', DEFAULT_FEATURES))
    s_str = str(p.get('suf_len', 5))
    return (f"model_h{p['hidden_size']}_l{p['num_layers']}"
            f"_d{p['dropout']}_e{e_str}_f{f_str}_s{s_str}")


# -----------------------------------------------------------------------
# EXPERIMENT SEARCH SPACES
# Run Phase 1 first, analyse, then uncomment Phase 2, etc.
# Already-trained models are skipped automatically.
# -----------------------------------------------------------------------

# ===== PHASE 1: Architecture sweep (default features, suf_len=5) =====
search_space = {
    "hidden_size": [64, 128, 200],
    "num_layers":  [1, 2],
    "dropout":     [0.1, 0.2],
    "emb_sizes":   [(100, 100, 50)],
    "features":    [['ortho', 'ext_full', 'ext_part']],
    "suf_len":     [5],
}

# ===== PHASE 2: Feature sweep =====
# Set BEST_ARCH to the best config found in Phase 1, then uncomment.
#
# BEST_ARCH = dict(hidden_size=128, num_layers=1, dropout=0.2,
#                  emb_sizes=(100, 100, 50))
# search_space = {
#     **{k: [v] for k, v in BEST_ARCH.items()},
#     "features": [
#         ['ortho', 'ext_full', 'ext_part'],                              # baseline
#         ['ortho', 'ext_full', 'ext_part', 'drug_morph'],               # +morphology
#         ['ortho', 'ext_full', 'ext_part', 'drug_morph', 'greek', 'length'],  # +all
#         ['ortho', 'ext_full', 'ext_part', 'drug_morph', 'greek', 'length', 'spacy'],  # +spacy
#     ],
#     "suf_len": [5],
# }

# ===== PHASE 3: Suffix length sweep =====
# Set BEST_FEATURES to the best feature set from Phase 2, then uncomment.
#
# BEST_FEATURES = ['ortho', 'ext_full', 'ext_part', 'drug_morph']
# search_space = {
#     **{k: [v] for k, v in BEST_ARCH.items()},
#     "features": [BEST_FEATURES],
#     "suf_len":  [3, 5, 7],
# }


# -----------------------------------------------------------------------
# Parse command-line params  (key=value pairs, flags)
# -----------------------------------------------------------------------
params = {}
for p in sys.argv[1:]:
    if "=" in p:
        par, val = p.split("=", 1)
        params[par] = val
        if par in ("batch_size", "max_len", "epochs"):
            params[par] = int(val)

if "name" not in params:
    params["name"] = "mymodel_000"


# -----------------------------------------------------------------------
# parse: spaCy-parse XML and save pickle files
# -----------------------------------------------------------------------
if "parse" in sys.argv[1:]:
    os.makedirs(os.path.join(NERDIR, "preprocessed"), exist_ok=True)
    if "test" in sys.argv[1:]:
        print("Creating parsed test pickle file...")
        ds = Dataset(os.path.join(DATADIR, "test.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed", "test.pck"))
    else:
        print("Creating parsed train pickle file...")
        ds = Dataset(os.path.join(DATADIR, "train.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed", "train.pck"))
        print("Creating parsed devel pickle file...")
        ds = Dataset(os.path.join(DATADIR, "devel.xml"))
        ds.save(os.path.join(NERDIR, "preprocessed", "devel.pck"))


# -----------------------------------------------------------------------
# train: grid-search over search_space, skip already-trained models
# -----------------------------------------------------------------------
if "train" in sys.argv[1:]:
    os.makedirs(os.path.join(NERDIR, "models"), exist_ok=True)

    for h in search_space["hidden_size"]:
      for l in search_space["num_layers"]:
        for d in search_space["dropout"]:
          for e in search_space["emb_sizes"]:
            for f in search_space["features"]:
              for s in search_space["suf_len"]:

                current_params = params.copy()
                current_params.update({
                    "hidden_size": h,
                    "num_layers":  l,
                    "dropout":     d,
                    "emb_sizes":   e,
                    "features":    f,
                    "suf_len":     s,
                })

                model_name = get_model_name(current_params)
                model_path = os.path.join(NERDIR, "models", model_name)

                if os.path.exists(os.path.join(model_path, "network.nn")):
                    print(f"Skipping {model_name} (already trained).")
                    continue

                print(f"\n{'='*60}\nTraining: {model_name}\n{'='*60}")
                try:
                    do_train(
                        os.path.join(NERDIR, "preprocessed", "train.pck"),
                        os.path.join(NERDIR, "preprocessed", "devel.pck"),
                        current_params,
                        model_path,
                    )
                    torch.cuda.empty_cache()
                except Exception as err:
                    print(f"Failed to train {model_name}: {err}")


# -----------------------------------------------------------------------
# predict: run one named model on devel (or test)
# -----------------------------------------------------------------------
if "predict" in sys.argv[1:]:
    os.makedirs(os.path.join(NERDIR, "results"), exist_ok=True)
    dataset_type = "test" if "test" in sys.argv[1:] else "devel"
    out_file  = os.path.join(NERDIR, "results",  f"{dataset_type}-{params['name']}.out")
    stat_file = os.path.join(NERDIR, "results",  f"{dataset_type}-{params['name']}.stats")
    predict(
        os.path.join(NERDIR, "models", params["name"]),
        os.path.join(NERDIR, "preprocessed", f"{dataset_type}.pck"),
        params, out_file,
    )
    evaluate("NER", os.path.join(DATADIR, f"{dataset_type}.xml"), out_file, stat_file)


# -----------------------------------------------------------------------
# predict_all: run every model in models/ that hasn't been predicted yet
# -----------------------------------------------------------------------
if "predict_all" in sys.argv[1:]:
    os.makedirs(os.path.join(NERDIR, "results"), exist_ok=True)
    model_dir    = os.path.join(NERDIR, "models")
    dataset_type = "test" if "test" in sys.argv[1:] else "devel"

    if os.path.exists(model_dir):
        for model_item in sorted(os.listdir(model_dir)):
            model_path = os.path.join(model_dir, model_item)
            if not os.path.isdir(model_path):
                continue
            out_file  = os.path.join(NERDIR, "results", f"{dataset_type}-{model_item}.out")
            stat_file = os.path.join(NERDIR, "results", f"{dataset_type}-{model_item}.stats")
            if os.path.exists(stat_file):
                print(f"Skipping {model_item} (already evaluated).")
                continue
            print(f"Predicting: {model_item}")
            try:
                predict(model_path,
                        os.path.join(NERDIR, "preprocessed", f"{dataset_type}.pck"),
                        params, out_file)
                evaluate("NER", os.path.join(DATADIR, f"{dataset_type}.xml"),
                         out_file, stat_file)
            except Exception as err:
                print(f"Failed to predict {model_item}: {err}")
