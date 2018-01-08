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
from medpy import metric
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import reshape_by_padding_upper_coords
from scipy.ndimage import binary_fill_holes


def calculate_validation_metrics(image_pred, image_gt, do_resec=False):
    image_gt = np.array(image_gt)
    image_pred = np.array(image_pred)

    def calculate_metrics(mask1, mask2):
        try:
            true_positives = metric.obj_tpr(mask1, mask2)
            if mask2.sum()!=0:
                false_positives = metric.obj_fpr(mask1, mask2)
            else:
                false_positives = 0
            if mask1.sum() == 0 or mask2.sum()==0:
                hd = 999
                assd = 999
                asd = 999
            else:
                hd = 999#metric.hd(mask1, mask2)
                assd = 999#metric.assd(mask1, mask2)
                asd = 999#metric.asd(mask1, mask2)
            dc = metric.dc(mask1, mask2)
            precision = metric.precision(mask1, mask2)
            recall = metric.recall(mask1, mask2)
            ravd = metric.ravd(mask1, mask2)
            return true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd
        except:
            return 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999, 99999

    class_labels = {
        0: 'background',
        1: 'edema',
        2: 'enhancing tumor',
        3: 'necrosis'
    }
    classes = np.arange(4)
    if do_resec:
        class_labels[4] = 'resec'
        classes = np.arange(5)

    # determine valid classes (those that actually appear in image_gt). Some images may miss some classes
    # classes = [c for c in classes if np.sum(image_gt==c) != 0]
    assert image_gt.shape == image_pred.shape
    accuracy = np.sum(image_gt == image_pred) / float(image_pred.size)
    class_metrics = {}

    # complete tumor
    mask1 = (image_gt==1) | (image_gt==2) | (image_gt==3)
    mask2 = (image_pred==1) | (image_pred==2) | (image_pred==3)
    if mask1.sum()!=0:
        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = "complete tumor"
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd,
                                'Vol pred': np.sum(mask2),
                                'Vol gt': np.sum(mask1)}
    else:
        label = "complete tumor"
        class_metrics[label] = {'true_positives': 99999,
                                'false_positives': 99999,
                                'DICE\t\t': 99999,
                                'Hausdorff dist': 99999,
                                'precision\t': 99999,
                                'recall\t\t': 99999,
                                'rel abs vol diff': 99999,
                                'avg surf dist symm': 99999,
                                'avg surf dist\t': 99999,
                                'Vol pred': np.sum(mask2),
                                'Vol gt': np.sum(mask1)}
    # tumor core
    mask1 = (image_gt==3) | (image_gt==2)
    mask2 = (image_pred==3) | (image_pred==2)
    if mask1.sum()!=0:
        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = "tumor core"
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd,
                                'Vol pred': np.sum(mask2),
                                'Vol gt': np.sum(mask1)
                                }
    else:
        label = "tumor core"
        class_metrics[label] = {'true_positives': 99999,
                                'false_positives': 99999,
                                'DICE\t\t': 99999,
                                'Hausdorff dist': 99999,
                                'precision\t': 99999,
                                'recall\t\t': 99999,
                                'rel abs vol diff': 99999,
                                'avg surf dist symm': 99999,
                                'avg surf dist\t': 99999,
                                'Vol pred': np.sum(mask2),
                                'Vol gt': np.sum(mask1)}

    for i, c in enumerate(classes):
        mask1 = image_gt==c
        mask2 = image_pred==c

        true_positives, false_positives, dc, hd, precision, recall, ravd, assd, asd = calculate_metrics(mask1, mask2)
        label = c
        if class_labels is not None and c in class_labels.keys():
            label = class_labels[c]
        class_metrics[label] = {'true_positives': true_positives,
                                'false_positives': false_positives,
                                'DICE\t\t': dc,
                                'Hausdorff dist': hd,
                                'precision\t': precision,
                                'recall\t\t': recall,
                                'rel abs vol diff': ravd,
                                'avg surf dist symm': assd,
                                'avg surf dist\t': asd,
                                'Vol pred': np.sum(mask2),
                                'Vol gt': np.sum(mask1)
                                }
    return accuracy, class_metrics


def create_brain_masks(data):
    shp = list(data.shape)
    brain_mask = np.zeros(shp, dtype=np.float32)
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            this_mask = data[b, c] != 0
            this_mask = binary_fill_holes(this_mask)
            brain_mask[b, c] = this_mask
    return brain_mask


def run_validation_mirroring(pred_fn, results_out_folder, use_patients, write_images=True,
                   hasBrainMask=True, BATCH_SIZE=None, num_repeats=1, preprocess_fn=None, save_npy=True,
                             use_t1km_sub=False, save_proba=False):
    all_official_metrics = np.zeros((len(use_patients.keys()), 13))
    ctr = 0
    print "predicting image"
    cmap = ListedColormap([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0.3, 0.5, 1)])

    for patient_id in use_patients.keys():
        print patient_id
        output_folder = os.path.join(results_out_folder, "%03.0d" % patient_id)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        this_patient = use_patients[patient_id]
        shp = this_patient["data"].shape[1:]

        new_shp = (shp[0] + 16 - shp[0] % 16, shp[1] + 16 - shp[1] % 16, shp[2] + 16 - shp[2] % 16)

        t1_img = np.array(
            reshape_by_padding_upper_coords(this_patient["data"][0], new_shp,
                                                                          pad_value=None))
        t1km_img = np.array(
            reshape_by_padding_upper_coords(this_patient["data"][1], new_shp,
                                                                          pad_value=None))
        t2_img = np.array(
            reshape_by_padding_upper_coords(this_patient["data"][2], new_shp,
                                                                          pad_value=None))
        flair_img = np.array(
            reshape_by_padding_upper_coords(this_patient["data"][3], new_shp,
                                                                          pad_value=None))
        seg_combined = np.array(
            reshape_by_padding_upper_coords(this_patient["data"][5], new_shp,
                                                                          pad_value=0))


        if use_t1km_sub:
            t1km_sub_img = np.array(
                reshape_by_padding_upper_coords(this_patient["t1km_sub"], new_shp,
                                                                              pad_value=None))

        seg_new = np.zeros(seg_combined.shape, dtype=np.float32)
        seg_new[seg_combined == 2] = 1
        seg_new[seg_combined == 4] = 2
        seg_new[seg_combined == 1] = 3

        num_channels = 4

        if use_t1km_sub:
            num_channels = 5

        data = np.zeros(tuple([1] + [num_channels] + list(new_shp)), dtype=np.float32)
        data[0, 0] = t1_img
        data[0, 1] = t1km_img
        data[0, 2] = t2_img
        data[0, 3] = flair_img
        if use_t1km_sub:
            data[0, 4] = t1km_sub_img

        bet_mask = create_brain_masks(data)

        if preprocess_fn is not None:
            data = preprocess_fn(data, bet_mask)

        if BATCH_SIZE is not None:
            data = np.vstack([data] * BATCH_SIZE)
        all_preds = []
        for i in range(num_repeats):
            for m in range(8):
                data_for_net = np.array(data)
                if m == 0:
                    pass
                if m == 1:
                    data_for_net = data_for_net[:, :, :, :, ::-1]
                if m == 2:
                    data_for_net = data_for_net[:, :, :, ::-1, :]
                if m == 3:
                    data_for_net = data_for_net[:, :, :, ::-1, ::-1]
                if m == 4:
                    data_for_net = data_for_net[:, :, ::-1, :, :]
                if m == 5:
                    data_for_net = data_for_net[:, :, ::-1, :, ::-1]
                if m == 6:
                    data_for_net = data_for_net[:, :, ::-1, ::-1, :]
                if m == 7:
                    data_for_net = data_for_net[:, :, ::-1, ::-1, ::-1]

                p = pred_fn(data_for_net)

                if m == 0:
                    pass
                if m == 1:
                    p = p[:, :, :, :, ::-1]
                if m == 2:
                    p = p[:, :, :, ::-1, :]
                if m == 3:
                    p = p[:, :, :, ::-1, ::-1]
                if m == 4:
                    p = p[:, :, ::-1, :, :]
                if m == 5:
                    p = p[:, :, ::-1, :, ::-1]
                if m == 6:
                    p = p[:, :, ::-1, ::-1, :]
                if m == 7:
                    p = p[:, :, ::-1, ::-1, ::-1]

                all_preds.append(p)
        stacked = np.vstack(all_preds)
        softmax_output = stacked.mean(0)
        predicted_segmentation = softmax_output.argmax(0)
        uncertainty = stacked.var(0).mean(0)

        predicted_segmentation = predicted_segmentation[:shp[0], :shp[1], :shp[2]]
        uncertainty = uncertainty[:shp[0], :shp[1], :shp[2]]
        data = data[:, :, :shp[0], :shp[1], :shp[2]]
        seg_new = seg_new[:shp[0], :shp[1], :shp[2]]
        softmax_output = softmax_output[:, :shp[0], :shp[1], :shp[2]]

        print predicted_segmentation.shape

        print "post processing"
        # ToDO see if we really need this post processing
        #image_pred_postprocessed = post_process_prediction(predicted_segmentation, min_size=min_size)
        image_pred_postprocessed = predicted_segmentation
        if hasBrainMask:
            seg_combined[seg_combined == 0] = 1
            seg_combined -= 1

        pred_new = np.zeros(image_pred_postprocessed.shape, dtype=np.float32)
        pred_new[image_pred_postprocessed == 1] = 2
        pred_new[image_pred_postprocessed == 2] = 4
        pred_new[image_pred_postprocessed == 3] = 1

        seg_combined = seg_combined.astype(np.int32)
        if save_npy:
            np.savez_compressed(os.path.join(output_folder, "segs"), seg_gt=seg_combined[:shp[0], :shp[1], :shp[2]],
                                seg_pred=pred_new, softmax_ouput=softmax_output, uncertainty=uncertainty)
        if save_proba:
            np.save(os.path.join(output_folder, "seg_probas"),
                    np.vstack(all_preds).astype(np.float16)[:, :, :shp[0], :shp[1], :shp[2]])


        print "calculating metrics"

        acc, metrics_by_class = calculate_validation_metrics(image_pred_postprocessed, seg_new)
        with open(os.path.join(output_folder, "metrics.pkl"), 'w') as f:
            import cPickle
            cPickle.dump(metrics_by_class, f)

        with open(os.path.join(output_folder, "validation_metrics.txt"), 'w') as f:
            f.write("The overall accuracy on this dataset was: \t%f\n\n" % acc)
            for c in metrics_by_class.keys():
                f.write("Results for label: %s\n" % c)
                for metrc in metrics_by_class[c].keys():
                    f.write("%s: \t%f\n" % (metrc, metrics_by_class[c][metrc]))
                f.write("\n")

        all_official_metrics[ctr][0] = patient_id

        if "complete tumor" in metrics_by_class.keys():
            all_official_metrics[ctr][1] = metrics_by_class["complete tumor"]["DICE\t\t"]
            all_official_metrics[ctr][4] = metrics_by_class["complete tumor"]["precision\t"]
            all_official_metrics[ctr][7] = metrics_by_class["complete tumor"]["recall\t\t"]
            all_official_metrics[ctr][10] = metrics_by_class["complete tumor"]["Hausdorff dist"]
        else:
            all_official_metrics[ctr][1] = 999
            all_official_metrics[ctr][4] = 999
            all_official_metrics[ctr][7] = 999
            all_official_metrics[ctr][10] = 999
        if "tumor core" in metrics_by_class.keys():
            all_official_metrics[ctr][2] = metrics_by_class["tumor core"]["DICE\t\t"]
            all_official_metrics[ctr][5] = metrics_by_class["tumor core"]["precision\t"]
            all_official_metrics[ctr][8] = metrics_by_class["tumor core"]["recall\t\t"]
            all_official_metrics[ctr][11] = metrics_by_class["tumor core"]["Hausdorff dist"]
        else:
            all_official_metrics[ctr][2] = 999
            all_official_metrics[ctr][5] = 999
            all_official_metrics[ctr][8] = 999
            all_official_metrics[ctr][11] = 999
        if "enhancing tumor" in metrics_by_class.keys():
            all_official_metrics[ctr][3] = metrics_by_class["enhancing tumor"]["DICE\t\t"]
            all_official_metrics[ctr][6] = metrics_by_class["enhancing tumor"]["precision\t"]
            all_official_metrics[ctr][9] = metrics_by_class["enhancing tumor"]["recall\t\t"]
            all_official_metrics[ctr][12] = metrics_by_class["enhancing tumor"]["Hausdorff dist"]
        else:
            all_official_metrics[ctr][3] = 999
            all_official_metrics[ctr][6] = 999
            all_official_metrics[ctr][9] = 999
            all_official_metrics[ctr][12] = 999

        output_folder_images = os.path.join(output_folder, "segmentation_slices")
        if not os.path.isdir(output_folder_images):
            os.mkdir(output_folder_images)

        if write_images:
            t1_img, t1km_img, flair_img, t2_img = data[0]
            n_rot = 2
            uncertainty[:, 0, 0:2] = (np.min(uncertainty), np.max(uncertainty))
            print "writing segmentation images"
            for i in range(0, image_pred_postprocessed.shape[0]):
                image_pred_postprocessed[i][0, 0:6] = [0, 1, 2, 3, 4, 5]
                seg_new[i][0, 0:6] = [0, 1, 2, 3, 4, 5]
                errors = seg_new[i] == image_pred_postprocessed[i]
                errors[0, 0:2] = [True, False]
                plt.figure(figsize=(18, 10))
                plt_ctr = 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(t1_img[i], n_rot), cmap="gray")
                plt.title("t1")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(t1km_img[i], n_rot), cmap="gray")
                plt.title("t1km")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(flair_img[i], n_rot), cmap="gray")
                plt.title("flair")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(t2_img[i], n_rot), cmap="gray")
                plt.title("t2")
                plt_ctr += 1

                if use_t1km_sub:
                    plt.subplot(2, 5, plt_ctr)
                    plt.imshow(np.rot90(t1km_sub_img[i], n_rot), cmap="gray")
                    plt.title("t1km_sub")
                    plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(image_pred_postprocessed[i], n_rot), cmap=cmap)
                plt.title("prediction")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(seg_new[i], n_rot), cmap=cmap)
                plt.title("gt")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(errors, n_rot), cmap="gray")
                plt.title("errors")
                plt_ctr += 1

                img_for_overlay = np.copy(np.rot90(t1km_img[i], n_rot))
                img_for_overlay -= img_for_overlay.min()
                img_for_overlay /= img_for_overlay.max()* (1/0.8)
                seg_for_overlay = np.copy(np.rot90(image_pred_postprocessed[i], n_rot))
                gt_for_overlay = np.copy(np.rot90(seg_new[i], n_rot))
                img_for_overlay_gt = np.vstack([np.copy(img_for_overlay)[None]]*3)
                img_for_overlay = np.vstack([np.copy(img_for_overlay)[None]]*3)
                img_for_overlay_gt[2][gt_for_overlay==1] += 0.3
                img_for_overlay_gt[1][gt_for_overlay==2] += 0.3
                img_for_overlay_gt[0][gt_for_overlay==3] += 0.3

                img_for_overlay[2][seg_for_overlay==1] += 0.3
                img_for_overlay[1][seg_for_overlay==2] += 0.3
                img_for_overlay[0][seg_for_overlay==3] += 0.3

                np.clip(img_for_overlay_gt, 0, 0.999)
                np.clip(img_for_overlay_gt, 0, 0.999)

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(img_for_overlay_gt.transpose(1, 2, 0))
                plt.title("overlay gt")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(img_for_overlay.transpose(1, 2, 0))
                plt.title("overlay pred")
                plt_ctr += 1

                plt.subplot(2, 5, plt_ctr)
                plt.imshow(np.rot90(uncertainty[i], n_rot), cmap="gray")
                plt.title("uncertainty")
                plt_ctr += 1

                plt.tight_layout()

                plt.savefig(
                    os.path.join(output_folder_images, "patient%d_segWholeDataset_z%03.0f" % (patient_id, i)))
                plt.close()
        ctr += 1

    np.save(os.path.join(results_out_folder, "evaluation_metrics.npy"), all_official_metrics)
    np.savetxt(os.path.join(results_out_folder, "evaluation_metrics.txt"), all_official_metrics)
    metrics = np.load(os.path.join(results_out_folder, "evaluation_metrics.npy"))
    averages = np.zeros(metrics.shape[1])
    for i in range(1, metrics.shape[1]):
        # hausdorff is set to 999 if it cannot be computed
        averages[i] = np.mean(
            metrics[:, i][(metrics[:, i] != 999) & (metrics[:, i] != 99999)])
    np.savetxt(os.path.join(results_out_folder, "evaluation_metrics_averages.txt"), averages.reshape((1, -1)))
    with open(os.path.join(results_out_folder, 'all_metrics.pkl'), 'w') as f:
        import cPickle
        cPickle.dump(metrics_by_class, f)
