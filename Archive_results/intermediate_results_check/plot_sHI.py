'''
Loads a saved cross-validation ensemble model and plots its sHI predictions for one panel.
'''
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf

from ae_cross_validation_helper import ClipLayer, plot_ensemble_sHI
from CNN_AE import KSparse, ExpandLastDim, SqueezeLastDim
from create_datastores import prepare_datastores
from config import DEFAULT_FREQ_INDEX, ALL_BASE_PANELS, CROSS_VALIDATION_RESULTS_AE_DIR
from AE_train import monotonicity_loss

path_i = 0
freq_i = DEFAULT_FREQ_INDEX
panel_number = "105"  # panel to plot sHI for

# Load the saved ensemble model for the specified path
model_path = CROSS_VALIDATION_RESULTS_AE_DIR / f"path_{path_i}" / "ensemble_model.keras"
custom_objects = {
    "KSparse": KSparse,
    "ExpandLastDim": ExpandLastDim,
    "SqueezeLastDim": SqueezeLastDim,
    "ClipLayer": ClipLayer,
    "monotonicity_loss": monotonicity_loss,
}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)


train_ds_names = [p for p in ALL_BASE_PANELS if p != panel_number]

_, _, _, ds_dict, _, States_dict, norm_stats = prepare_datastores(
    path_i=path_i, freq_i=freq_i, base_batch_size=8, test_batch_size=1,
    train_ds_names=train_ds_names, val_ds_names=[panel_number], test_ds_names=[panel_number], include_benchmark=True,
)

save_path = str(Path(model_path).parent / f"ensemble_sHI_panel_{panel_number}.png")
plot_ensemble_sHI(model, ds_dict, norm_stats, States_dict, [panel_number], save_path=save_path)
print(f"Saved: {save_path}")
