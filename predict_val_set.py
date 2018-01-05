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
from utils import softmax_helper
from network_architecture import build_net
import theano.tensor as T
from dataset import load_dataset
import paths
from validate_network import preprocess


def run(fold=0):
    print fold
    I_AM_FOLD = fold
    all_data = load_dataset(folder=paths.preprocessed_validation_data_folder)

    use_patients = all_data
    experiment_name = "final"
    results_folder = os.path.join(paths.results_folder, experiment_name,
                                  "fold%d"%I_AM_FOLD)
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

    results_out_folder = os.path.join(results_folder, "pred_val_set")
    if not os.path.isdir(results_out_folder):
        os.mkdir(results_out_folder)

    with open(os.path.join(results_folder, "%s_Params.pkl" % (experiment_name)), 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)

    print "compiling theano functions"
    output = softmax_helper(lasagne.layers.get_output(output_layer, x_sym, deterministic=False,
                                                      batch_norm_update_averages=False, batch_norm_use_averages=False))
    pred_fn = theano.function([x_sym], output)
    _ = pred_fn(np.random.random((BATCH_SIZE, 4, 176, 192, 176)).astype(np.float32))

    run_validation_mirroring(pred_fn, results_out_folder, use_patients, write_images=write_images, hasBrainMask=False,
                             BATCH_SIZE=BATCH_SIZE, num_repeats=n_repeats, preprocess_fn=preprocess, save_npy=save_npy,
                             save_proba=False)

if __name__ == "__main__":
    fold = int(sys.argv[1])
    run(fold)
