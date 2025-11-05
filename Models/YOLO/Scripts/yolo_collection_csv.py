import os
import cv2
import yaml

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from ultralytics import YOLO
from Dataset.scripts.preprocessing.optical_flow import create_optical_flow_output

def load_model(model_name = "yolo11s-seg.pt"):
    model = YOLO(model_name)

    return model

def apply_yolo_to_output(md, fl_nm):
    
    out = md.predict(fl_nm, project = "/Users/johannesbauer/Documents/Coding/SaberPredict/Dataset/yolo_results", name="result_1")

    return out

def segment_video_for_frame(result):

    segmentation_mask_lst = []

    original_image = result.orig_img

    classes = result.boxes.cls.numpy()

    for idx in range(len(result.masks)):

        if classes[idx] < 0.5:

            mask = result.masks[idx].data.squeeze().cpu().numpy()

            final_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation = cv2.INTER_NEAREST)

            segmentation_mask_lst.append(final_mask)

    return segmentation_mask_lst

def return_correct_segmentation_map(optical_map, segmentation_map):
    
    max_val = [-1, -1]
    second_max_val = [-1, -1]

    counter = 0

    quarter_data = np.sum(np.ones(segmentation_map[0].shape))/30

    for seg_img in segmentation_map:
        #plt.imshow(optical_map*seg_img)
        #plt.show()

        if np.sum(seg_img) > quarter_data:
            continue

        else:
            sum_val = np.mean(optical_map*seg_img)

            if sum_val > max_val[0]:

                second_max_val[0] = max_val[0]
                second_max_val[1] = max_val[1]

                max_val[0] = sum_val
                max_val[1] = counter

            elif sum_val > second_max_val[0]:
                second_max_val[0] = sum_val
                second_max_val[1] = counter

            counter += 1

    #new_segmentation_map = segmentation_map[max_val[1]] + segmentation_map[second_max_val[1]]

    segmentation_map_1 = np.clip(segmentation_map[max_val[1]], 0, 1)
    segmentation_map_2 = np.clip(segmentation_map[second_max_val[1]], 0, 1)

    return segmentation_map_1, segmentation_map_2

def expand_segmentation(seg_map_lst):
    
    kernel = np.ones((7,7))
    
    expanded_lst = []

    for seg_map in seg_map_lst:
        val_lst = []
        for img in seg_map:
            new_dialated_image = cv2.dilate(img, kernel, iterations = 3)

            val_lst.append(new_dialated_image)

        expanded_lst.append(val_lst)

    return expanded_lst

def segment_out_images(original_img_lst, segmentation_map_lst):
    
    segmented_out_img_lst = []

    for img, seg_map in zip(original_img_lst, segmentation_map_lst):

        val_lst = []
        for seg_arr in seg_map:
            img_mask_3d = np.dstack([seg_arr, seg_arr, seg_arr])

            masked_output = img*img_mask_3d
            masked_output = masked_output.astype(np.uint8)
            val_lst.append(masked_output)
        segmented_out_img_lst.append(val_lst)

    return segmented_out_img_lst

def yolo_pose_collect(md, segmented_imgs_lst):
    
    pose_results_lst = []

    for img_lst in segmented_imgs_lst:
        val_lst = []
        for img in img_lst:
            pose_res = md.predict(img, verbose = False)
            val_lst.append(pose_res)

        pose_results_lst.append(val_lst)

    return pose_results_lst

def get_label(fl: str):

    lb = fl.split(os.sep)[-1].split(".")[0].split("_")[-1]

    return lb

def save_pose_estimates(results_lst, dictionary, fl):

    for idx, result_tuple in enumerate(results_lst):

        for idx_spec, res in enumerate(result_tuple):
            
            keypoints = np.squeeze(res[0].keypoints.xy.numpy())
            confidence = res[0].keypoints.conf.numpy().flatten()

            sved_fl = fl.split("clips")[-1]

            lb = get_label(fl)

            if len(confidence) == 0:
                continue

            dictionary["frame"].append(idx)
            dictionary["file"].append(sved_fl)
            dictionary["label"].append(lb)

            for final_idx, (k_pt, conf) in enumerate(zip(keypoints, confidence)):

                

                try:
                    k_pt = (float(k_pt[0]), float(k_pt[1]))
                    conf = float(conf)

                except:
                    res[0].show()
                    print(idx)
                    assert 1 == 0

                nm_of_conf = f"conf_{final_idx}"
                nm_of_k_pt = f"kepoint_{final_idx}"

                if nm_of_k_pt not in list(dictionary.keys()):
                    dictionary[nm_of_k_pt] = []
                    dictionary[nm_of_k_pt].append(k_pt)

                    dictionary[nm_of_conf] = []
                    dictionary[nm_of_conf].append(conf)

                else:
                    dictionary[nm_of_k_pt].append(k_pt)
                    dictionary[nm_of_conf].append(conf)

    return dictionary

def main_yolo_collection_method(mp4_file):
    
    md_seg = load_model("yolo11s-seg.pt")
    md_pose = load_model("yolo11n-pose.pt")


    out_optical = create_optical_flow_output(mp4_file)

    out_results = md_seg(mp4_file, verbose = False)

    original_img_lst = [res.orig_img for res in out_results]

    assert len(out_results) == len(out_optical), "There is not enought correct segmentation maps."

    correct_segmentations_lst = []

    dictioanry_imp = {
        "file": [],
        "frame": [],
        "label": [],
    }

    for idx, (result, opt) in enumerate(zip(out_results, out_optical)):
        
        out_return_segmentation = segment_video_for_frame(result)

        correct_segmentations = return_correct_segmentation_map(opt, out_return_segmentation)

        correct_segmentations_lst.append(correct_segmentations)

    expanded_segmentation_lst = expand_segmentation(correct_segmentations_lst)

    segmented_out_values = segment_out_images(original_img_lst, expanded_segmentation_lst)

    new_segmented_video = yolo_pose_collect(md_pose, segmented_out_values)

    out_dict = save_pose_estimates(new_segmented_video, dictioanry_imp, mp4_file)        

    return out_dict

def main():
    base_dir = "/Users/johannesbauer/Documents/Coding/SaberPredict/Dataset/clips/1/*"

    fls_to_analyze = glob(base_dir)
    fls_to_analyze = [fl for fl in fls_to_analyze if ".mp4" in fl]

    lst_of_dictionaries = []

    print()
    pbar = tqdm(total = len(fls_to_analyze), leave=False, desc = "Analyzing Images")

    for fl in fls_to_analyze:

        can_move_on = False
        counter = 0

        while can_move_on == False:
            #try:
            new_dict = main_yolo_collection_method(fl)
            can_move_on = True

            #except:
            #    counter += 1
            #    print(f"Failed, start attempt {counter}")
            #    can_move_on = True

            #    if counter == 10:
            #        break

        lst_of_dictionaries.append(new_dict)

        pbar.update(1)

    pbar.close()

    print("Yay - you got this far. Now, we are saving our results.")
    print()
    sv_fl = "/Users/johannesbauer/Documents/Coding/SaberPredict/data/test_2.yaml"

    final_dict = {}

    #for dict in lst_of_dictionaries:
    #    final_dict.update(dict)

    for d in lst_of_dictionaries:

        for k in d.keys():
            if k not in list(final_dict.keys()):
                final_dict[k] = d[k]
            else:
                final_dict[k] += d[k]


    with open(sv_fl, "w") as fl:
        yaml.dump(final_dict, fl)

    print("Done!!!")

if __name__ == "__main__":
    main()