'''
This file:
- defines the Panel_GraphDataset class for creating graph datasets from the extracted sHI values and attention-based adjacency matrices
- the dataset is created for each state, where nodes represent paths and edges represent crossing paths weighted by the attention values
- includes a main block that demonstrates how to load the dataset and print some example graph data for a given panel and frequency
- by default, it can load either the sHI features or the big latent features extracted from the autoencoder, depending on the big_latent flag

If you do any changes in the process method, make sure to delete the processed files to trigger re-processing with the new code.
'''

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("data", "models", "tools", "training", "intermediate_results_check", "results_analysis"):
    _p = str(_PROJECT_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import warnings
# format error messages with panel and state information for easier debugging
warnings.formatwarning = lambda msg, *_, **__: f"Warning: {msg}\n"

from torch_geometric.data import Data, InMemoryDataset
import torch
from weight_matrix import find_crossings, adjencency_matrix, find_crossings_by_area
import numpy as np
from create_datastores import states
from config import BETA_CONSTANT, NR_SAVED_STATES, STATE_START_INDICES, mat_file_path,  PANEL_123_SUBPANELS

class Panel_GraphDataset(InMemoryDataset):
    def __init__(self, root, panel_number,freq, big_latent=True, type="peak", fold='ensemble', transform=None, pre_transform=None, beta_constant=BETA_CONSTANT):
        self.panel_number = panel_number
        self.freq = freq
        self.big_latent = big_latent # if True, the model is trained on the latent space of AE, otherwise on the sub-HI (one number per node)
        if type in ("peak_tff", "peak_fft","peak_tft"):
            self.type = "peak"  # treat all peak types the same for dataset creation
        else:
            self.type = type
        self.fold = fold
        self.beta_constant = beta_constant
        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.fold == 'ensemble':
            return [f"panel_{self.panel_number}_shi_raw_{self.freq}.pt"]
        else:
            return [f"panel_{self.panel_number}_shi_raw_{self.freq}_fold{self.fold}.pt"]


    @property
    def processed_file_names(self):
        if self.fold == 'ensemble':
            if self.big_latent:
                if self.type == "peak_and_area":
                    if self.beta_constant!=BETA_CONSTANT:
                        return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_area_beta{self.beta_constant}.pt"]
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak_and_area.pt"]
        
                elif self.type == "peak":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak.pt"]
                elif self.type == "peak_only":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak_only.pt"]
                elif self.type == "geometry_only":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_geometry_only.pt"]
                else:
                    raise ValueError(f"Unknown adjacency matrix type: {self.type}")

            else:
                if self.type in ("peak_and_area"):
                    if self.beta_constant!=BETA_CONSTANT:
                        return [f"panel_{self.panel_number}_processed_{self.freq}_area_beta{self.beta_constant}.pt"]
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak_and_area.pt"]
                elif self.type == "peak":
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak.pt"]
                elif self.type == "peak_only":
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak_only.pt"]
                elif self.type == "geometry_only":
                    return [f"panel_{self.panel_number}_processed_{self.freq}_geometry_only.pt"]
                else:
                    raise ValueError(f"Unknown adjacency matrix type: {self.type}")
        else:
            if self.big_latent:
                if self.type == "peak_and_area":
                    if self.beta_constant!=BETA_CONSTANT:
                        return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_area_beta{self.beta_constant}_fold{self.fold}.pt"]
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak_and_area_fold{self.fold}.pt"]
                elif self.type == "peak":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak_fold{self.fold}.pt"]
                elif self.type == "peak_only":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_peak_only_fold{self.fold}.pt"]
                elif self.type == "geometry_only":
                    return [f"panel_{self.panel_number}_processed_big_latent_{self.freq}_geometry_only_fold{self.fold}.pt"]
                else:
                    raise ValueError(f"Unknown adjacency matrix type: {self.type}")
            else:
                if self.type == "peak_and_area":
                    if self.beta_constant!=BETA_CONSTANT:
                        return [f"panel_{self.panel_number}_processed_{self.freq}_area_beta{self.beta_constant}_fold{self.fold}.pt"]
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak_and_area_fold{self.fold}.pt"]
                elif self.type in ("peak"):
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak_fold{self.fold}.pt"]
                elif self.type == "peak_only":
                    return [f"panel_{self.panel_number}_processed_{self.freq}_peak_only_fold{self.fold}.pt"]
                elif self.type == "geometry_only":
                    return [f"panel_{self.panel_number}_processed_{self.freq}_geometry_only_fold{self.fold}.pt"]    
                else:
                    raise ValueError(f"Unknown adjacency matrix type: {self.type}")


    def process(self):
        # Load sub_HI from AE
        raw_data = torch.load(self.raw_paths[0], weights_only=False)

        features = np.array(raw_data["shi"]) # shape: paths x states x 1
        big_latent = raw_data["big_latent"] # shape: paths x states x latent_dim

        is_123 = str(self.panel_number) == "123"
        # for "123", each subpanel's adjacency batch is computed once (on first
        # encounter) and cached; for single-file panels it's computed once up front.
        subpanel_states_cache = {} if is_123 else None
        subpanel_adj_cache = {} if is_123 else None
        if not is_123:
            current_state = states(str(mat_file_path(self.panel_number)))
            adj_matrix_all = adjencency_matrix(current_state, state_idx=":", freq_idx=self.freq, type=self.type, beta_constant=self.beta_constant)  # shape: num_states x paths x paths

        # Process raw_data into a list of Data objects
        data_list = []
        t_begin = time.perf_counter()

        if self.type in ( "peak_and_area"):
            connection_matrix = find_crossings_by_area(beta_constant=self.beta_constant)
        elif self.type =="peak_only":
            connection_matrix = np.ones((28, 28), dtype=float)  # all paths connected
        else:
            connection_matrix = find_crossings()

        # connection_matrix doesn't depend on state, so the edge topology is fixed for the whole panel
        edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)  # shape: 2 x num_edges
        edge_mask = connection_matrix != 0

        for state in range(features.shape[1]):
            if self.big_latent:
                x = torch.tensor(big_latent[:, state, :], dtype=torch.float)  # shape: (num_paths, latent_dim)
            else:
                x = torch.tensor(features[:, state, 0], dtype=torch.float).unsqueeze(1)  # shape: (num_paths, 1)

            if (x == 0).all():
                warnings.warn(f"Warning: All features are zero for state {state}. Check if the raw features are correct.")

            t_state_load_start = time.perf_counter()
            if is_123:
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state < end_idx:
                            subpanel = sp
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    if subpanel not in subpanel_states_cache:
                        subpanel_states_cache[subpanel] = states(str(mat_file_path(subpanel)))
                        subpanel_adj_cache[subpanel] = adjencency_matrix(subpanel_states_cache[subpanel], state_idx=":", freq_idx=self.freq, type=self.type, beta_constant=self.beta_constant)  # shape: sub_num_states x paths x paths
                    local_idx = state - STATE_START_INDICES[subpanel]
                    adj_matrix = subpanel_adj_cache[subpanel][local_idx]
            else:
                adj_matrix = adj_matrix_all[state]
            t_state_load_end = time.perf_counter()
        

            if (adj_matrix == 0).all():
                warnings.warn(f"Warning: Adjacency matrix for state {state} is all zeros. Check if the attention values are correct.")


            adj_matrix = adj_matrix[edge_mask]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)


            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long),
                                  panel=torch.tensor([self.panel_number], dtype=torch.long)))
            t_after_append = time.perf_counter()

        t_end = time.perf_counter()
        print(f"Processed data for panel {self.panel_number}, freq {self.freq}, fold {self.fold}, in {t_end - t_begin:.2f}s with {len(data_list)} states.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class features_GraphDataset(InMemoryDataset):
    # This dataset is used if the graph is trained on raw time-frequency features (no AE).
    # type selects the adjacency-matrix scheme, same as Panel_GraphDataset/config.GCN_TYPES;
    # defaults to "peak" so existing callers and already-cached panel_*_processed_raw_features_diff_<freq>.pt files keep working unchanged.
    def __init__(self, root, panel_number, freq, type="peak", transform=None, pre_transform=None, beta_constant=BETA_CONSTANT):
        self.panel_number = panel_number
        self.freq = freq
        if type in ("peak_tff", "peak_fft", "peak_tft"):
            self.type = "peak"  # treat all peak types the same for dataset creation
        else:
            self.type = type
        self.beta_constant = beta_constant

        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.panel_number == 123:
            return [str(f"States_{sp}_freq{self.freq}_all_paths_diff_full.npy") for sp in PANEL_123_SUBPANELS]
        else:
            return [str( f"States_{self.panel_number}_freq{self.freq}_all_paths_diff_full.npy")]

    @property
    def processed_file_names(self):
        if self.type == "peak":
            return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}.pt"]
        elif self.type == "peak_and_area":
            if self.beta_constant != BETA_CONSTANT:
                return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}_area_beta{self.beta_constant}.pt"]
            return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}_peak_and_area.pt"]
        elif self.type == "peak_only":
            return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}_peak_only.pt"]
        elif self.type == "geometry_only":
            return [f"panel_{self.panel_number}_processed_raw_features_diff_{self.freq}_geometry_only.pt"]
        else:
            raise ValueError(f"Unknown adjacency matrix type: {self.type}")

    def process(self):
        if self.panel_number == 123:
            features_list = []
            for file in self.raw_paths:
                features = np.load(file)
                features_list.append(features)
            features = np.concatenate(features_list, axis=1)  # shape: paths x states x features
            print(f"Concatenated features shape for panel 123: {features.shape}")
        else:
            features = np.load(self.raw_paths[0])  # shape: paths x states x features
            print(f"Loaded features shape for panel {self.panel_number}: {features.shape}")

        #
        is_123 = str(self.panel_number) == "123"
        subpanel_states_cache = {} if is_123 else None
        if not is_123:
            current_state = states(str(mat_file_path(self.panel_number)))

        if self.type == "peak_and_area":
            connection_matrix = find_crossings_by_area(beta_constant=self.beta_constant)
        elif self.type == "peak_only":
            connection_matrix = np.ones((28, 28), dtype=float)  # all paths connected
        else:
            connection_matrix = find_crossings()
        edge_index = torch.tensor(np.array(np.nonzero(connection_matrix)), dtype=torch.long)

        # Process features into a list of Data objects
        data_list = []

        for state in range(features.shape[1]):
            x = torch.tensor(features[:, state, :], dtype=torch.float)  # shape: (num_paths, num_features)

            if is_123:
                    # find which subpanel this state belongs to
                    subpanel = None
                    for sp in NR_SAVED_STATES.keys():
                        start_idx = STATE_START_INDICES[sp]
                        end_idx = start_idx + NR_SAVED_STATES[sp]
                        if start_idx <= state < end_idx:
                            subpanel = sp
                            break
                    if subpanel is None:
                        raise ValueError(f"State index {state} does not belong to any known subpanel.")
                    if subpanel not in subpanel_states_cache:
                        subpanel_states_cache[subpanel] = states(str(mat_file_path(subpanel)))
                    current_state = subpanel_states_cache[subpanel]

            adj_matrix = adjencency_matrix(current_state, state_idx=state, freq_idx=self.freq, type=self.type, beta_constant=self.beta_constant)  # shape: paths x paths
            adj_matrix = adj_matrix[connection_matrix != 0]  # shape: num_edges
            edge_weight = torch.tensor((np.array(adj_matrix)), dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                                  y=torch.tensor([state], dtype=torch.long),
                                  panel=torch.tensor([self.panel_number], dtype=torch.long)))
        data, slices = self.collate(data_list)

        # make sure the processed directory exists
        processed_dir = Path(self.processed_paths[0]).parent
        processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed data saved to {self.processed_paths[0]} with {len(data_list)} states.")


