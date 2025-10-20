import os
import cv2
import json

import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from optical_flow import create_optical_flow_output

def load_model(model_name = "yolo11s-seg.pt"):
    model = YOLO(model_name)

    return model

def apply_yolo_to_output(md, fl_nm):
    
    out = md.predict(fl_nm, project = "/Users/johannesbauer/Documents/Coding/SaberPredict/Dataset/yolo_results", name="result_1")

    return out

def segment_video_for_frame(result):

    segmentation_mask_lst = []

    original_image = result.orig_img

    for idx in range(len(result.masks)):

        mask = result.masks[idx].data.squeeze().cpu().numpy()

        final_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation = cv2.INTER_LINEAR)

        segmentation_mask_lst.append(final_mask)

    return segmentation_mask_lst

def return_correct_segmentation_map(optical_map, segmentation_map):
    
    max_val = [-1, -1]
    second_max_val = [-1, -1]

    counter = 0

    quarter_data = np.sum(np.ones(segmentation_map[0].shape))/3

    for seg_img in segmentation_map:
        #plt.imshow(optical_map*seg_img)
        #plt.show()

        if np.sum(seg_img) > quarter_data:
            continue

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
            pose_res = md.predict(img)
            val_lst.append(pose_res)

        pose_results_lst.append(val_lst)

    return pose_results_lst

def save_pose_estimates(results_lst, sv_path):

    for idx, result_tuple in enumerate(results_lst):
    
        new_dir = os.path.join(sv_path, f"frame_{idx}")

        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        for idx_spec, res in enumerate(result_tuple):
            
            sv_file_name = os.path.join(new_dir, f"person_{idx_spec}.json")

            out = res[0].to_json()

            with open(sv_file_name, "w") as fl:
                json.dump(out, fl)

    print("Done!")

def main_yolo_collection_method(mp4_file):
    
    md_seg = load_model("yolo11s-seg.pt")
    md_pose = load_model("yolo11n-pose.pt")


    out_optical = create_optical_flow_output(mp4_file)

    out_results = md_seg(mp4_file)

    original_img_lst = [res.orig_img for res in out_results]

    assert len(out_results) == len(out_optical), "There is not enought correct segmentation maps."

    correct_segmentations_lst = []

    for result, opt in zip(out_results, out_optical):
        
        out_return_segmentation = segment_video_for_frame(result)

        correct_segmentations = return_correct_segmentation_map(opt, out_return_segmentation)

        correct_segmentations_lst.append(correct_segmentations)

    expanded_segmentation_lst = expand_segmentation(correct_segmentations_lst)

    segmented_out_values = segment_out_images(original_img_lst, expanded_segmentation_lst)

    new_segmented_video = yolo_pose_collect(md_pose, segmented_out_values)

    save_pose_estimates(new_segmented_video, "/Users/johannesbauer/Documents/Coding/SaberPredict/data/all_poses")        


def main():
    fl = "/Users/johannesbauer/Documents/Coding/SaberPredict/Dataset/clips/1/1_Left.mp4"

    main_yolo_collection_method(fl)


if __name__ == "__main__":
    main()