# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import SimpleITK as sitk
from dataset import load_dataset
import os
from dataset import convert_to_brats_seg
import paths


def convert_to_original_coord_system(seg_pred, pat_in_dataset):
    orig_shape = pat_in_dataset['orig_shp']
    # axis order is z x y
    brain_bbox_z = pat_in_dataset['bbox_z']
    brain_bbox_x = pat_in_dataset['bbox_x']
    brain_bbox_y = pat_in_dataset['bbox_y']
    new_seg = np.zeros(orig_shape, dtype=np.uint8)
    tmp_z = np.min((orig_shape[0], brain_bbox_z[0] + seg_pred.shape[0]))
    tmp_x = np.min((orig_shape[1], brain_bbox_x[0] + seg_pred.shape[1]))
    tmp_y = np.min((orig_shape[2], brain_bbox_y[0] + seg_pred.shape[2]))
    new_seg[brain_bbox_z[0]:tmp_z, brain_bbox_x[0]:tmp_x, brain_bbox_y[0]:tmp_y] = seg_pred[:tmp_z -brain_bbox_z[0],
                                                                                   :tmp_x - brain_bbox_x[0],
                                                                                   :tmp_y - brain_bbox_y[0]]
    return new_seg


def save_train_dataset_as_nifti(results_dir=os.path.join(paths.results_folder, "final"),
            out_dir=os.path.join(paths.results_folder, "training_set_results")):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    a = load_dataset()
    for fold in range(5):
        working_dir = os.path.join(results_dir, "fold%d"%fold, "validation")
        ids_in_fold = os.listdir(working_dir)
        ids_in_fold.sort()
        ids_in_fold = [i for i in ids_in_fold if os.path.isdir(os.path.join(working_dir, i))]
        ids_in_fold_as_int = [int(i) for i in ids_in_fold]
        for pat_id in ids_in_fold_as_int:
            pat_in_dataset = a[pat_id]
            seg_pred = np.load(os.path.join(working_dir, "%03.0d"%pat_id, "segs.npz"))['seg_pred']
            b = convert_to_original_coord_system(seg_pred, pat_in_dataset)
            sitk_img = sitk.GetImageFromArray(b)
            sitk_img.SetSpacing(pat_in_dataset['spacing'])
            sitk_img.SetDirection(pat_in_dataset['direction'])
            sitk_img.SetOrigin(pat_in_dataset['origin'])
            sitk.WriteImage(sitk_img, os.path.join(out_dir, pat_in_dataset['name'] + ".nii.gz"))


def save_val_dataset_as_nifti(results_dir=os.path.join(paths.results_folder, "final"),
              out_dir=os.path.join(paths.results_folder, "val_set_results_new")):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    a = load_dataset(folder=paths.preprocessed_validation_data_folder)
    for pat in a.keys():
        probs = []
        for fold in range(5):
            working_dir = os.path.join(results_dir, "fold%d"%fold, "pred_val_set")
            res = np.load(os.path.join(working_dir, "%03.0d"%pat, "segs.npz"))
            probs.append(res['softmax_ouput'][None])
        prediction = np.vstack(probs).mean(0).argmax(0)
        prediction_new = convert_to_brats_seg(prediction)
        np.savez_compressed(os.path.join(out_dir, "%03.0d.npz"%pat), seg=prediction)
        b = convert_to_original_coord_system(prediction_new, a[pat])
        sitk_img = sitk.GetImageFromArray(b)
        sitk_img.SetSpacing(a[pat]['spacing'])
        sitk_img.SetDirection(a[pat]['direction'])
        sitk_img.SetOrigin(a[pat]['origin'])
        sitk.WriteImage(sitk_img, os.path.join(out_dir, a[pat]['name'] + ".nii.gz"))


def save_test_set_as_nifti(results_dir=os.path.join(paths.results_folder, "final"),
               out_dir=os.path.join(paths.results_folder, "test_set_results")):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    a = load_dataset(folder=paths.preprocessed_testing_data_folder)
    for pat in a.keys():
        probs = []
        for fold in range(5):
            working_dir = os.path.join(results_dir, "fold%d"%fold, "pred_test_set")
            res = np.load(os.path.join(working_dir, "%03.0d"%pat, "segs.npz"))
            probs.append(res['softmax_ouput'][None])
        prediction = np.vstack(probs).mean(0).argmax(0)
        prediction_new = convert_to_brats_seg(prediction)
        np.savez_compressed(os.path.join(out_dir, "%03.0d.npz"%pat), seg=prediction)
        b = convert_to_original_coord_system(prediction_new, a[pat])
        sitk_img = sitk.GetImageFromArray(b)
        sitk_img.SetSpacing(a[pat]['spacing'])
        sitk_img.SetDirection(a[pat]['direction'])
        sitk_img.SetOrigin(a[pat]['origin'])
        sitk.WriteImage(sitk_img, os.path.join(out_dir, a[pat]['name'] + ".nii.gz"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-of", "--output_folder", help="where to save the results", type=str)
    parser.add_argument("-d", "--dataset", help="which dataset should be exported? (train/val/test)", type=str)
    args = parser.parse_args()
    if args.dataset == "train":
        save_train_dataset_as_nifti(os.path.join(paths.results_folder, "final"),
                                    args.output_folder)
    elif args.dataset == "val":
        save_val_dataset_as_nifti(os.path.join(paths.results_folder, "final"),
                                  args.output_folder)
    elif args.dataset == "test":
        save_test_set_as_nifti(os.path.join(paths.results_folder, "final"),
                               args.output_folder)
    else:
        raise ValueError("Unknown value for --dataset")
