"""Package per-landmark early-warning artifacts for the inference API.

This utility reads the component estimators, stacking correlator, and
configuration for weeks 2, 4, and 6 from ``strict_early_warning_v2/week_N/``.
It accepts the current artifact filenames and falls back to the supported
legacy names when a correlator or configuration file uses an older name. The
loaded items are grouped by landmark and serialized, with compression, as
``unified_model.joblib`` beside this script.

Run ``python combine_models.py`` from the project directory after
``strict_time_aware_early_warning_v2.py`` has produced the week-specific
artifacts. The resulting bundle is the default model file loaded by
``api.py``; this script does not train models or change their parameters.
"""
import json
import joblib
from pathlib import Path

# ---------------------------------------------------------------------
# Artifact locations
# ---------------------------------------------------------------------

BASE = Path(__file__).resolve().parent / "strict_early_warning_v2"
OUTPUT = Path(__file__).resolve().parent / "unified_model.joblib"

# ---------------------------------------------------------------------
# Landmark bundle assembly
# ---------------------------------------------------------------------

bundle = {}
for week in (2, 4, 6):
    folder = BASE / f"week_{week}"
    
    component_models = joblib.load(folder / "component_models.joblib")
    
    correlator_path = folder / "learned_correlator.joblib"
    if not correlator_path.exists():
        correlator_path = folder / "stacking_correlator.joblib"
    correlator = joblib.load(correlator_path)
    
    config_path = folder / "model_config.json"
    if not config_path.exists():
        config_path = folder / "landmark_summary.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    bundle[week] = {
        "component_models": component_models,
        "correlator": correlator,
        "config": config,
    }
    print(f"Week {week}: loaded")

# ---------------------------------------------------------------------
# Bundle serialization
# ---------------------------------------------------------------------

joblib.dump(bundle, OUTPUT, compress=3)
print(f"\nSaved unified model: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / (1024*1024):.1f} MB")
