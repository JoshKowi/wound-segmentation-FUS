import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import argparse
from utils.io.data import get_png_filename_list
from utils.postprocessing.hole_filling import fill_holes
from utils.postprocessing.remove_small_noise import remove_small_areas

def post_process_predictions(pred_path, threshold):
    file_list = get_png_filename_list(pred_path)
    for img_name in tqdm(file_list):
        img = cv2.imread(pred_path + img_name)
        _, threshed = cv2.threshold(img, threshold, 255, type=cv2.THRESH_BINARY)
        ################################################################################################################
        # call image post processing functions
        mask = np.zeros((226, 226, 3))
        filled = fill_holes(threshed, threshold,0.1)
        denoised = remove_small_areas(filled, threshold, 0.05)
        ################################################################################################################
        os.makedirs(pred_path + 'filled', exist_ok=True)
        os.makedirs(pred_path + 'post_processed', exist_ok=True)
        cv2.imwrite(pred_path + 'filled/' + img_name, filled)
        cv2.imwrite(pred_path + 'post_processed/' + img_name, denoised)

def evaluate_threshold(pred_path, label_path, threshold, ):
    label_list = get_png_filename_list(label_path)

    false_positives = 0
    false_negatives = 0
    true_positives = 0

    for label_name in tqdm(label_list):
        label = cv2.imread(label_path + label_name,0)
        post_processed = cv2.imread(pred_path + 'post_processed/' + label_name, 0)
        xdim = label.shape[0]
        ydim = label.shape[1]
        for x in range(xdim):
            for y in range(ydim):
                if post_processed[x, y] and label[x, y] > threshold:
                    true_positives += 1
                if label[x, y] > threshold > post_processed[x, y]:
                    false_negatives += 1
                if label[x, y] < threshold < post_processed[x, y]:
                    false_positives += 1

    IOU = float(true_positives) / (true_positives + false_negatives + false_positives)
    Dice = 2*float(true_positives) / (2*true_positives + false_negatives + false_positives)

    print("--------------------------------------------------------")
    print("Weight file: ", pred_path.rsplit("/")[1])
    print("--------------------------------------------------------")
    print("Threshold: ", threshold)
    print("True  pos = " + str(true_positives))
    print("False neg = " + str(false_negatives))
    print("False pos = " + str(false_positives))
    print("IOU = " + str(IOU))
    print("Dice = " + str(Dice))

def exec_evaluate(pred_path, label_path, thresholds=[120]):
    """Post processes model-output images and evaluates final results"""
    # test your own thresholds
    file_list = get_png_filename_list(pred_path)
    label_list = get_png_filename_list(label_path)
    if not len(file_list) == len(label_list):
        raise ValueError(
            f"Number of labels does not fit number of images: #labels={len(label_list)}; #files={len(file_list)}")
    for threshold in thresholds:
        post_process_predictions(pred_path, threshold)
        evaluate_threshold(pred_path, label_path, threshold)

if __name__ == "__main__":
    # adapt as needed
    pred_dir = './data/fuseg_augmented/test/predictions/02-05-366389-short_train-14eps/'
    label_path = './data/fuseg_augmented/test/labels/'
    thresholds = [117]  # currentmax: 117
    exec_evaluate(pred_dir, label_path, thresholds=thresholds)