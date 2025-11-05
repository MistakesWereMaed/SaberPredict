import os

import numpy as np
from ultralytics import YOLO

DEFAULT_AUG = {
    "epoch": 50,
    "img_sz": 640,
    "device": "cpu"
}


def load_ultralytics_model():

    md = YOLO("yolov8n-pose.yaml")

    return md

def evaluate_model(data_fl_val, md):
    results_all = md.val(data_fl_val)

    all_ap = [res.all_ap for res in results_all]

    mean_ap = np.mean(all_ap)

    return mean_ap

def train_yolo(data_fl_train, md, **kwargs):

    epochs = kwargs.get("epoch", DEFAULT_AUG["epoch"])
    img_size = kwargs.get("img_sz", DEFAULT_AUG["img_sz"])
    device = kwargs.get("device", DEFAULT_AUG["device"])

    trained_model = md.train(data = data_fl_train,
                      epochs = epochs,
                      imgsz=img_size,
                      device=device,
                      
                      )

    validation_score = md.val() #evaluate_model(data_fl_val, trained_model)


    return validation_score, trained_model

def main():
    fl_to_train_data = "/Users/johannesbauer/Downloads/BladeDetection-pose/data.yaml"

    md = load_ultralytics_model()

    validation_score, trained_model = train_yolo(fl_to_train_data, md)

    print(validation_score)


if __name__ == "__main__":
    main()