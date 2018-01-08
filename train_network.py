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


import matplotlib
matplotlib.use('Agg')
import lasagne
import theano.tensor as T
import numpy as np
import theano
import sys
import os
import cPickle
from network_architecture import build_net
from utils import printLosses
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    Mirror, GammaTransform, ConvertSegToOnehotTransform, ContrastAugmentationTransform, BrightnessTransform, Compose
from batchgenerators.dataloading import MultiThreadedAugmenter
import time
from utils import soft_dice_per_img_in_batch, hard_dice_per_img_in_batch
from sklearn.cross_validation import KFold # this is deprecated -> bad, but need to keep it here for reproducibility
from dataset import load_dataset, BatchGenerator3D_random_sampling
from custom_transforms import BrainMaskAwareStretchZeroOneTransform, GenerateBrainMaskTransform
import paths


def create_data_gen_train(patient_data_train, INPUT_PATCH_SIZE, num_classes, BATCH_SIZE, contrast_range=(0.75, 1.5),
                          gamma_range = (0.6, 2),
                                  num_workers=5, num_cached_per_worker=3,
                                  do_elastic_transform=False, alpha=(0., 1300.), sigma=(10., 13.),
                                  do_rotation=False, a_x=(0., 2*np.pi), a_y=(0., 2*np.pi), a_z=(0., 2*np.pi),
                                  do_scale=True, scale_range=(0.75, 1.25), seeds=None):
    if seeds is None:
        seeds = [None]*num_workers
    elif seeds == 'range':
        seeds = range(num_workers)
    else:
        assert len(seeds) == num_workers
    data_gen_train = BatchGenerator3D_random_sampling(patient_data_train, BATCH_SIZE, num_batches=None, seed=False,
                                                          patch_size=(160, 192, 160), convert_labels=True)
    tr_transforms = []
    tr_transforms.append(DataChannelSelectionTransform([0, 1, 2, 3]))
    tr_transforms.append(GenerateBrainMaskTransform())
    tr_transforms.append(Mirror())
    tr_transforms.append(SpatialTransform(INPUT_PATCH_SIZE, list(np.array(INPUT_PATCH_SIZE)//2.),
                                       do_elastic_deform=do_elastic_transform, alpha=alpha, sigma=sigma,
                                       do_rotation=do_rotation, angle_x=a_x, angle_y=a_y, angle_z=a_z,
                                       do_scale=do_scale, scale=scale_range, border_mode_data='nearest',
                                       border_cval_data=0, order_data=3, border_mode_seg='constant', border_cval_seg=0,
                                       order_seg=0, random_crop=True))
    tr_transforms.append(BrainMaskAwareStretchZeroOneTransform((-5, 5), True))
    tr_transforms.append(ContrastAugmentationTransform(contrast_range, True))
    tr_transforms.append(GammaTransform(gamma_range, False))
    tr_transforms.append(BrainMaskAwareStretchZeroOneTransform(per_channel=True))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True))
    tr_transforms.append(SegChannelSelectionTransform([0]))
    tr_transforms.append(ConvertSegToOnehotTransform(range(num_classes), 0, "seg_onehot"))

    gen_train = MultiThreadedAugmenter(data_gen_train, Compose(tr_transforms), num_workers, num_cached_per_worker,
                                       seeds)
    gen_train.restart()
    return gen_train


def run(fold=0):
    print fold
    # =================================================================================================================
    I_AM_FOLD = fold
    np.random.seed(65432)
    lasagne.random.set_rng(np.random.RandomState(98765))
    sys.setrecursionlimit(2000)
    BATCH_SIZE = 2
    INPUT_PATCH_SIZE =(128, 128, 128)
    num_classes=4

    EXPERIMENT_NAME = "final"
    results_dir = os.path.join(paths.results_folder)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, EXPERIMENT_NAME)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, "fold%d"%I_AM_FOLD)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    n_epochs = 300
    lr_decay = np.float32(0.985)
    base_lr = np.float32(0.0005)
    n_batches_per_epoch = 100
    n_test_batches = 10
    n_feedbacks_per_epoch = 10.
    num_workers = 6
    workers_seeds = [123, 1234, 12345, 123456, 1234567, 12345678]

    # =================================================================================================================

    all_data = load_dataset()
    keys_sorted = np.sort(all_data.keys())

    crossval_folds = KFold(len(all_data.keys()), n_folds=5, shuffle=True, random_state=123456)

    ctr = 0
    for train_idx, test_idx in crossval_folds:
        print len(train_idx), len(test_idx)
        if ctr == I_AM_FOLD:
            train_keys = [keys_sorted[i] for i in train_idx]
            test_keys = [keys_sorted[i] for i in test_idx]
            break
        ctr += 1

    train_data = {i:all_data[i] for i in train_keys}
    test_data = {i:all_data[i] for i in test_keys}

    data_gen_train = create_data_gen_train(train_data, INPUT_PATCH_SIZE, num_classes, BATCH_SIZE,
                                           contrast_range=(0.75, 1.5), gamma_range = (0.8, 1.5),
                                           num_workers=num_workers, num_cached_per_worker=2,
                                           do_elastic_transform=True, alpha=(0., 1300.), sigma=(10., 13.),
                                           do_rotation=True, a_x=(0., 2*np.pi), a_y=(0., 2*np.pi), a_z=(0., 2*np.pi),
                                           do_scale=True, scale_range=(0.75, 1.25), seeds=workers_seeds)

    data_gen_validation = BatchGenerator3D_random_sampling(test_data, BATCH_SIZE, num_batches=None, seed=False,
                                                           patch_size=INPUT_PATCH_SIZE, convert_labels=True)
    val_transforms = []
    val_transforms.append(GenerateBrainMaskTransform())
    val_transforms.append(BrainMaskAwareStretchZeroOneTransform(clip_range=(-5, 5), per_channel=True))
    val_transforms.append(SegChannelSelectionTransform([0]))
    val_transforms.append(ConvertSegToOnehotTransform(range(4), 0, "seg_onehot"))
    val_transforms.append(DataChannelSelectionTransform([0, 1, 2, 3]))
    data_gen_validation = MultiThreadedAugmenter(data_gen_validation, Compose(val_transforms), 2, 2)

    x_sym = T.tensor5()
    seg_sym = T.matrix()

    net, seg_layer = build_net(x_sym, INPUT_PATCH_SIZE, num_classes, 4, 16, batch_size=BATCH_SIZE,
                               do_instance_norm=True)
    output_layer_for_loss = net

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-5

    # the distinction between prediction_train and test is important only if we enable dropout (batch norm/inst norm
    # does not use or save moving averages)
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False,
                                                 batch_norm_update_averages=False, batch_norm_use_averages=False)

    loss_vec = - soft_dice_per_img_in_batch(prediction_train, seg_sym, BATCH_SIZE)[:, 1:]

    loss = loss_vec.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True,
                                                batch_norm_update_averages=False, batch_norm_use_averages=False)
    loss_val = - soft_dice_per_img_in_batch(prediction_test, seg_sym, BATCH_SIZE)[:, 1:]

    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    # learning rate has to be a shared variable because we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(base_lr)
    updates = lasagne.updates.adam(T.grad(loss, params), params, learning_rate=learning_rate, beta1=0.9, beta2=0.999)

    dc = hard_dice_per_img_in_batch(prediction_test, seg_sym.argmax(1), num_classes, BATCH_SIZE).mean(0)

    train_fn = theano.function([x_sym, seg_sym], [loss, acc_train, loss_vec], updates=updates)
    val_fn = theano.function([x_sym, seg_sym], [loss_val, acc, dc])

    all_val_dice_scores=None

    all_training_losses = []
    all_validation_losses = []
    all_validation_accuracies = []
    all_training_accuracies = []
    val_dice_scores = []
    epoch = 0

    while epoch < n_epochs:
        if epoch == 100:
            data_gen_train = create_data_gen_train(train_data, INPUT_PATCH_SIZE, num_classes, BATCH_SIZE,
                                                   contrast_range=(0.85, 1.25), gamma_range = (0.8, 1.5),
                                                   num_workers=6, num_cached_per_worker=2,
                                                   do_elastic_transform=True, alpha=(0., 1000.), sigma=(10., 13.),
                                                   do_rotation=True, a_x=(0., 2*np.pi), a_y=(-np.pi/8., np.pi/8.),
                                                   a_z=(-np.pi/8., np.pi/8.), do_scale=True, scale_range=(0.85, 1.15),
                                                   seeds=workers_seeds)

        if epoch == 175:
            data_gen_train = create_data_gen_train(train_data, INPUT_PATCH_SIZE, num_classes, BATCH_SIZE,
                                                   contrast_range=(0.9, 1.1), gamma_range = (0.85, 1.3),
                                                   num_workers=6, num_cached_per_worker=2,
                                                   do_elastic_transform=True, alpha=(0., 750.), sigma=(10., 13.),
                                                   do_rotation=True, a_x=(0., 2*np.pi), a_y=(-0.00001, 0.00001),
                                                   a_z=(-0.00001, 0.00001), do_scale=True, scale_range=(0.85, 1.15),
                                                   seeds=workers_seeds)

        epoch_start_time = time.time()
        learning_rate.set_value(np.float32(base_lr* lr_decay**(epoch)))
        print "epoch: ", epoch, " learning rate: ", learning_rate.get_value()
        train_loss = 0
        train_acc_tmp = 0
        train_loss_tmp = 0
        batch_ctr = 0
        for data_dict in data_gen_train:
            data = data_dict["data"].astype(np.float32)
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 4, 1).reshape((-1, num_classes))
            if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)) == 0:
                print "number of batches: ", batch_ctr, "/", n_batches_per_epoch
                print "training_loss since last update: ", \
                    train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch), " train accuracy: ", \
                    train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch)
                all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
                all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
                train_loss_tmp = 0
                train_acc_tmp = 0
                if len(val_dice_scores) > 0:
                    all_val_dice_scores = np.concatenate(val_dice_scores, axis=0).reshape((-1, num_classes))
                try:
                    printLosses(all_training_losses, all_training_accuracies, all_validation_losses,
                                all_validation_accuracies, os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME),
                                n_feedbacks_per_epoch, val_dice_scores=all_val_dice_scores,
                                val_dice_scores_labels=["brain", "1", "2", "3", "4", "5"])
                except:
                    pass
            loss_vec, acc, l = train_fn(data, seg)

            loss = loss_vec.mean()
            train_loss += loss
            train_loss_tmp += loss
            train_acc_tmp += acc
            batch_ctr += 1
            if batch_ctr >= n_batches_per_epoch:
                break
        all_training_losses.append(train_loss_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
        all_training_accuracies.append(train_acc_tmp/np.floor(n_batches_per_epoch/n_feedbacks_per_epoch))
        train_loss /= n_batches_per_epoch
        print "training loss average on epoch: ", train_loss

        val_loss = 0
        accuracies = []
        valid_batch_ctr = 0
        all_dice = []
        for data_dict in data_gen_validation:
            data = data_dict["data"].astype(np.float32)
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 4, 1).reshape((-1, num_classes))
            w = np.zeros(num_classes, dtype=np.float32)
            w[np.unique(seg.argmax(-1))] = 1
            loss, acc, dice = val_fn(data, seg)
            dice[w==0] = 2
            all_dice.append(dice)
            val_loss += loss
            accuracies.append(acc)
            valid_batch_ctr += 1
            if valid_batch_ctr >= n_test_batches:
                break
        all_dice = np.vstack(all_dice)
        dice_means = np.zeros(num_classes)
        for i in range(num_classes):
            dice_means[i] = all_dice[all_dice[:, i]!=2, i].mean()
        val_loss /= n_test_batches
        print "val loss: ", val_loss
        print "val acc: ", np.mean(accuracies), "\n"
        print "val dice: ", dice_means
        print "This epoch took %f sec" % (time.time()-epoch_start_time)
        val_dice_scores.append(dice_means)
        all_validation_losses.append(val_loss)
        all_validation_accuracies.append(np.mean(accuracies))
        all_val_dice_scores = np.concatenate(val_dice_scores, axis=0).reshape((-1, num_classes))
        try:
            printLosses(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies,
                        os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME), n_feedbacks_per_epoch,
                        val_dice_scores=all_val_dice_scores, val_dice_scores_labels=["brain", "1", "2", "3", "4", "5"])
        except:
            pass
        with open(os.path.join(results_dir, "%s_Params.pkl" % (EXPERIMENT_NAME)), 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
        with open(os.path.join(results_dir, "%s_allLossesNAccur.pkl"% (EXPERIMENT_NAME)), 'w') as f:
            cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses,
                          all_validation_accuracies, val_dice_scores], f)
        epoch += 1


if __name__ == "__main__":
    fold = int(sys.argv[1])
    run(fold)
