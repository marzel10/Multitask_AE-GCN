'''
This file:
- defines extract_shi function 
- creates raw data files with the latent values that can be used as nodes features in the graph dataset
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

import os
import torch
import numpy as np
import tensorflow as tf

from CNN_AE import ExpandLastDim, KSparse, SqueezeLastDim
from ae_cross_validation_helper import ClipLayer, _CompatHeNormal
from AE_train import monotonicity_loss
from create_datastores import prepare_datastores
from config import  GRAPH_DATA_DIR, ALL_BASE_PANELS, TEST_RUN_DIR, CV_PANELS, CV_PANELS_INT, DEBUG_MODE

def extract_shi(folders, freq, dataset, fold,GAN_dir=str(TEST_RUN_DIR)):
    '''
    Extract sHI values from the trained autoencoder models for a given dataset and frequency.
    inputs:
    - folders: list of folder names within GAN_dir where the models are stored (models after bayesian optimization)
    - freq: frequency index to extract sHI for (e.g. 3)
    - dataset: dataset/panel number to extract sHI for (e.g. '104')
    - GAN_dir: base directory where the folders are located

    outputs:
    - latents_all: list of numpy arrays, each containing the sHI values for a path (shape: n_states x 1)
    - path_labels: list of path indices corresponding to each entry in latents_all
    - big_latent_all: list of numpy arrays, each containing the big latent values (the one used for reconstruction) for all paths and states

    '''
    folders = [os.path.join(GAN_dir, f) for f in folders]
    latents_all = []
    path_labels = []
    big_latent_all = []
    train_ds_names = [p for p in CV_PANELS if p != str(fold)]

    for folder in folders:

        # Extract path number from folder name
        last_part = folder.split("_")[-1]
        path_i = last_part.find("path")
        path = int(last_part[path_i + 4:]) if path_i != -1 else None


        for file in os.listdir(folder):

            # Only load the ensemble model file, skip other files
            if not file.endswith(f"val_{fold}.keras"):
                continue

            # Load the model with custom objects defined in other files
            CUSTOM_OBJECTS = {
                "KSparse": KSparse,
                "ExpandLastDim": ExpandLastDim,
                "SqueezeLastDim": SqueezeLastDim,
                "ClipLayer": ClipLayer,
                "monotonicity_loss": monotonicity_loss,
                "HeNormal": _CompatHeNormal,
            }
            model = tf.keras.models.load_model(os.path.join(folder, file), custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)
            model = tf.keras.Model(inputs=model.input, outputs=model.outputs + [model.get_layer("latent_space").output])

            _, _, _, ds_dict, *_rest = prepare_datastores(
                path_i=path, freq_i=freq, base_batch_size=16, test_batch_size=1,
                train_ds_names=train_ds_names, val_ds_names=[str(fold)], test_ds_names=[dataset],
                include_benchmark=True,
            )
            ds = ds_dict[dataset]
            # Get the sHI (shape: n_states x 1) and latent space (shape: n_states x latent_dim) values from the model
            sHI, _, latent_space = model.predict(ds, verbose=0)
           
            if sHI.ndim == 3:
                sHI = sHI.reshape(-1, sHI.shape[-1])

            latents_all.append(sHI)
            path_labels.append(path)
            big_latent_all.append(latent_space)

    return latents_all, path_labels, big_latent_all

def pre_compute_AE_output():
    counter = 0
    for freq in range(0,6):
        folders = [f"Multi_path_BO_fixed_freq{freq}\\Bayesian_CNN_AE_path{i}" for i in range(0, 28)]
        datasets = ALL_BASE_PANELS
    
        for dataset in datasets:
            for fold in CV_PANELS_INT:
            
                latents_all, path_labels, big_latent_all = extract_shi(folders, freq, dataset, fold)
                if DEBUG_MODE:
                    print(f"Collected latents for {len(latents_all)} paths.")
                    print(f"Shape of latents matrix: {np.array(latents_all).shape}")
                    print(f"Shape of the path labels: {np.array(path_labels).shape}")
                    print(f"Big latent shape: {np.array(big_latent_all).shape}")

                out_dict ={"shi": np.array(latents_all), "path_labels": np.array(path_labels), "big_latent": np.array(big_latent_all)}
                folder = GRAPH_DATA_DIR / "raw"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                torch.save(out_dict, os.path.join(folder, f"panel_{dataset}_shi_raw_{freq}_fold{fold}.pt"))
                counter += 1
                print(f"Done saving panel {dataset} frequency {freq} fold {fold}.({counter/(len(datasets)*len(CV_PANELS_INT)*6)*100:.2f}%)")



if __name__ == "__main__":
    pre_compute_AE_output()
