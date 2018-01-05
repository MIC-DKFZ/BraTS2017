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
import matplotlib
matplotlib.use('Agg')
import cPickle
import lasagne
import os
import sys
import theano.tensor
from utils_validation import run_validation_mirroring
from Utils.general_utils import softmax_helper
from network_architecture import build_net
import theano.tensor as T
from sklearn.cross_validation import KFold
from dataset import load_dataset
import paths


def preprocess(data, bet_mask):
    brain_mask = bet_mask
    data = np.clip(data, -5, 5)
    for c in range(data[0].shape[0]):
        data[0][c] -= data[0][c][brain_mask[0, c] != 0].min()
        data[0][c] /= data[0][c][brain_mask[0, c] != 0].max()
        data[0][c, brain_mask[0, c] == 0] = 0
    return data


def run(fold=0):
    print fold
    I_AM_FOLD = fold

    all_data = load_dataset()
    keys_sorted = np.sort(all_data.keys())

    crossval_folds = KFold(len(all_data.keys()), n_folds=5, shuffle=True, random_state=123456)

    ctr = 0
    for train_idx, test_idx in crossval_folds:
        print len(train_idx), len(test_idx)
        if ctr == I_AM_FOLD:
            test_keys = [keys_sorted[i] for i in test_idx]
            break
        ctr += 1

    validation_data = {i:all_data[i] for i in test_keys}


    use_patients = validation_data
    EXPERIMENT_NAME = "final"
    results_folder = os.path.join(paths.results_folder,
                               EXPERIMENT_NAME,  "fold%d" % I_AM_FOLD)
    write_images = False
    save_npy = True

    INPUT_PATCH_SIZE =(None, None, None)
    BATCH_SIZE = 2
    n_repeats=2
    num_classes=4

    x_sym = T.tensor5()

    net, seg_layer = build_net(x_sym, INPUT_PATCH_SIZE, num_classes, 4, 16, batch_size=BATCH_SIZE,
                                           do_instance_norm=True)
    output_layer = seg_layer

    results_out_folder = os.path.join(results_folder, "validation")
    if not os.path.isdir(results_out_folder):
        os.mkdir(results_out_folder)

    with open(os.path.join(results_folder, "%s_Params.pkl" % (EXPERIMENT_NAME)), 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    print "compiling theano functions"
    output = softmax_helper(lasagne.layers.get_output(output_layer, x_sym, deterministic=False,
                                                      batch_norm_update_averages=False, batch_norm_use_averages=False))
    pred_fn = theano.function([x_sym], output)
    _ = pred_fn(np.random.random((BATCH_SIZE, 4, 176, 192, 176)).astype(np.float32))  # preallocate memory on GPU

    run_validation_mirroring(pred_fn, results_out_folder, use_patients, write_images=write_images, hasBrainMask=False,
                             BATCH_SIZE=BATCH_SIZE, num_repeats=n_repeats, preprocess_fn=preprocess, save_npy=save_npy)

if __name__ == "__main__":
    fold = int(sys.argv[1])
    run(fold)
