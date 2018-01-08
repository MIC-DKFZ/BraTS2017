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


import theano.tensor as T
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import theano


def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape)==2:
            pad_value = image[0,0]
        elif len(shape)==3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0+int(shape[0]), 0:0+int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0+int(shape[0]), 0:0+int(shape[1]), 0:0+int(shape[2])] = image
    return res


def softmax_helper(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def printLosses(all_training_losses, all_training_accs, all_validation_losses, all_valid_accur, fname,
                samplesPerEpoch=10, val_dice_scores=None, val_dice_scores_labels=None, ylim_score=None):
    try:
        from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    except:
        from lasagne.layers import Conv2DLayer
    fig, ax1 = plt.subplots(figsize=(16, 12))
    trainLoss_x_values = np.arange(1/float(samplesPerEpoch), len(all_training_losses)/float(samplesPerEpoch)+0.000001,
                                   1/float(samplesPerEpoch))
    val_x_values = np.arange(1, len(all_validation_losses)+0.001, 1)
    ax1.plot(trainLoss_x_values, all_training_losses, 'b--', linewidth=2)
    ax1.plot(val_x_values, all_validation_losses, color='b', linewidth=2)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    if ylim_score is not None:
        ax1.set_ylim(ylim_score)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax2 = ax1.twinx()
    ax2.plot(trainLoss_x_values, all_training_accs, 'r--', linewidth=2)
    ax2.plot(val_x_values, all_valid_accur, color='r', linewidth=2)
    ax2.set_ylabel('accuracy')
    for t2 in ax2.get_yticklabels():
        t2.set_color('r')
    ax2_legend_text = ['trainAcc', 'validAcc']

    if val_dice_scores is not None:
        assert len(val_dice_scores) == len(all_validation_losses)
        num_auc_scores_per_timestep = val_dice_scores.shape[1]
        for auc_id in xrange(num_auc_scores_per_timestep):
            ax2.plot(val_x_values, val_dice_scores[:, auc_id], linestyle=":", linewidth=4, markersize=10)
            ax2_legend_text.append(val_dice_scores_labels[auc_id])

    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax1.legend(['trainLoss', 'validLoss'], loc="center right", bbox_to_anchor=(1.3, 0.4))
    ax2.legend(ax2_legend_text, loc="center right", bbox_to_anchor=(1.3, 0.6))
    plt.savefig(fname)
    plt.close()


def soft_dice_per_img_in_batch(y_pred, y_true, BATCH_SIZE):
    num_pixels_per_sample = y_true.shape[0] // BATCH_SIZE
    dice_scores = T.zeros((BATCH_SIZE, y_pred.shape[1]), dtype=theano.config.floatX)
    y_pred = y_pred.reshape((BATCH_SIZE, num_pixels_per_sample, y_pred.shape[1]))
    y_true = y_true.reshape((BATCH_SIZE, num_pixels_per_sample, y_true.shape[1]))
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    for i in range(BATCH_SIZE):
        intersect = T.sum(y_pred[i] * y_true[i], 0)
        denominator = T.sum(y_pred[i], 0) + T.sum(y_true[i], 0)
        dice_scores = T.set_subtensor(dice_scores[i], T.constant(2) * intersect / (denominator + T.constant(1e-6)))
    return dice_scores


def hard_dice_per_img_in_batch(y_pred, y_true, n_classes, BATCH_SIZE):
    num_pixels_per_sample = y_true.shape[0] // BATCH_SIZE
    # y_true must be label map, not one hot encoding
    y_true = T.flatten(y_true)
    y_pred = T.argmax(y_pred, axis=1)

    dice = T.zeros((BATCH_SIZE, n_classes))
    y_pred = y_pred.reshape((BATCH_SIZE, num_pixels_per_sample))
    y_true = y_true.reshape((BATCH_SIZE, num_pixels_per_sample))

    for b in range(BATCH_SIZE):
        for i in range(n_classes):
            i_val = T.constant(i)
            y_true_i = T.eq(y_true[b], i_val)
            y_pred_i = T.eq(y_pred[b], i_val)
            dice = T.set_subtensor(dice[b, i], (T.constant(2.) * T.sum(y_true_i * y_pred_i) + T.constant(1e-7)) /
                                   (T.sum(y_true_i) + T.sum(y_pred_i) + T.constant(1e-7)))
    return dice