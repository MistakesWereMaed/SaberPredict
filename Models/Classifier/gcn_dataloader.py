import torch

import yaml

import numpy as np
import pandas as pd

from tqdm import tqdm

from torch_geometric.data import Data, Dataset

class sabre_dataset(Dataset):

    def __init__(self, yaml_fl: str):
        super().__init__()

        self.yaml_fl = yaml_fl

        if ".yaml" in yaml_fl:

            self.loaded_data = self._load_yaml(yaml_fl)

        elif ".csv" in yaml_fl:
            self.loaded_data = self._load_csv(yaml_fl)

        self.graph_obj_1 = [
                            (0, 1), (1, 3),
                            (3, 5), (1, 2),
                            (0, 2), (2, 4),
                            (4, 6),
                            (5, 7), (7, 9),  # left arm
                            (6, 8), (8, 10), # right arm
                            (5, 6),          # shoulders
                            (11, 12),        # hips
                            (5, 11), (6, 12),# torso
                            (11, 13), (13, 15), # left leg
                            (12, 14), (14, 16)  # right leg
                        ]

        self.graph_obj_2 = [
                            (17, 1+17), (1+17, 3+17),
                            (3+17, 5+17), (1+17, 2+17),
                            (0+17, 2+17), (2+17, 4+17),
                            (4+17, 6+17),
                            (5+17, 7+17), (7+17, 9+17),  # left arm
                            (6+17, 8+17), (8+17, 10+17), # right arm
                            (5+17, 6+17),          # shoulders
                            (11+17, 12+17),        # hips
                            (5+17, 11+17), (6+17, 12+17),# torso
                            (11+17, 13+17), (13+17, 15+17), # left leg
                            (12+17, 14+17), (14+17, 16+17)  # right leg
                        ]

        self.data = self._format_all_data_to_lst(self.loaded_data)


    def _load_yaml(self, yaml_fl: str):

        with open(yaml_fl) as fl:
            out = yaml.full_load(fl)

        return out

    def _string_to_real_coords(self, txt_val):
        kpts = txt_val.split(", ")
        kp_1 = float(kpts[0][1:])
        kp_2 = float(kpts[1][:-2])
        
        return (kp_1, kp_2)

    def _load_csv(self, csv_fl: str):

        out = pd.read_csv(csv_fl)

        all_wanted_keys = ['kepoint_0', 'kepoint_1', 'kepoint_10', 'kepoint_11',
            'kepoint_12', 'kepoint_13', 'kepoint_14', 'kepoint_15', 'kepoint_16',
            'kepoint_2', 'kepoint_3', 'kepoint_4', 'kepoint_5', 'kepoint_6',
            'kepoint_7', 'kepoint_8', 'kepoint_9']

        wanted_dict = {}
        for k in out.keys():
            if k in all_wanted_keys:
                wanted_values = out[k].apply(self._string_to_real_coords)
                wanted_dict[k] = wanted_values.to_numpy().flatten()

            else:
                wanted_dict[k] = out[k].to_numpy().flatten()

        return wanted_dict

    def _find_fl_frame_pairing(self, dictionary: dict):
        
        fl_vals = dictionary["file"]
        frame_vals = dictionary["frame"]

        fl_frame_pairs = [f"{fl} {frame}" for (fl, frame) in zip(fl_vals, frame_vals)]

        fl_frame_pairs = np.unique(fl_frame_pairs)

        return fl_frame_pairs

    def _find_idx_by_fl_frame_pairing(self, dictionary, fl_frame_pair: str):

        fl_frame_lst = fl_frame_pair.split(" ")

        where_fl_is = set(np.argwhere(np.array(dictionary["file"]) == fl_frame_lst[0]).flatten())
        where_frame_is = set(np.argwhere(np.array(dictionary["frame"]) == int(fl_frame_lst[1])).flatten())

        shared_idx = where_fl_is & where_frame_is

        shared_idx = list(shared_idx)

        return shared_idx

    def _find_wanted_data_by_idx(self, idx: list, dictionary: dict):

        wanted_indices = [str(r) for r in range(17)]

        lst_of_vals_obj_1 = {k:[] for k in wanted_indices}
        lst_of_vals_obj_2 = {k:[] for k in wanted_indices}

        for k in dictionary.keys():
            if "_" in k:
                wanted_idx = k.split("_")[-1]
                if wanted_idx in wanted_indices:
                    val_1 = dictionary[k][idx[0]]
                    lst_of_vals_obj_1[wanted_idx].append((val_1, k))

                    if len(idx) == 2:
                        val_2 = dictionary[k][idx[1]]
                        lst_of_vals_obj_2[wanted_idx].append((val_2, k))

        return lst_of_vals_obj_1, lst_of_vals_obj_2

    def _partition_list_into_correct_format(self, value_lst: list):
        
        new_value_lst = []

        for e in value_lst:

            if "kepoint" in e[1]:
                new_value_lst.append((e[0][0], e[1] + "x"))
                new_value_lst.append((e[0][1], e[1] + "y"))

            else:
                new_value_lst.append(e)

        return new_value_lst

    def _format_to_data_obj(self, val_dict_1: dict, val_dict_2: dict):
        
        features = [] # keypoints, confidence

        graph = []

        for k in val_dict_1.keys():
            
            values_for_node_k = val_dict_1[k]

            values_for_node_k = self._partition_list_into_correct_format(values_for_node_k)

            values_for_node_k_sorted = sorted(values_for_node_k, key=lambda a: a[1])

            wanted_vals = [a[0] for a in values_for_node_k_sorted]

            features.append(wanted_vals)

        graph += self.graph_obj_1

        if val_dict_2 != None:

            for k in val_dict_1.keys():
            
                values_for_node_k = val_dict_1[k]
            
                values_for_node_k = self._partition_list_into_correct_format(values_for_node_k)

                values_for_node_k_sorted = sorted(values_for_node_k, key=lambda a: a[1])

                wanted_vals = [a[0] for a in values_for_node_k_sorted]

                features.append(wanted_vals)

            graph += self.graph_obj_2

        feature_arr = np.stack(features)

        return graph, feature_arr

    def _format_all_data_to_lst(self, dictionary_of_data: dict):
        
        fl_frame_lst = self._find_fl_frame_pairing(dictionary_of_data)

        lst_of_data_obj = []

        pbar = tqdm(total = len(fl_frame_lst), desc="Loading Files", leave=False)

        for fl_frame_str in fl_frame_lst:
            idx = self._find_idx_by_fl_frame_pairing(dictionary_of_data, fl_frame_str)

            lb = dictionary_of_data["label"][idx[0]]

            if len(idx) == 1:
                features_1,_ = self._find_wanted_data_by_idx(idx, dictionary_of_data)
                graph, feature_arr = self._format_to_data_obj(features_1, None)

            elif len(idx) == 2:
                features_1, feature_2 = self._find_wanted_data_by_idx(idx, dictionary_of_data)
                graph, feature_arr = self._format_to_data_obj(features_1, feature_2)


            graph = torch.tensor(list(zip(*graph)), dtype=torch.long)
            feature_tensor = torch.tensor(feature_arr)

            data_obj = Data(x=feature_tensor, edge_index=graph, y = lb)
            lst_of_data_obj.append(data_obj)

            pbar.update(1)

        pbar.close()

        return lst_of_data_obj

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]


def main():
    fl_for_yaml = "/Users/johannesbauer/Documents/Coding/SaberPredict/data/test_2.yaml"
    ds = sabre_dataset(fl_for_yaml)


if __name__ == "__main__":
    main()