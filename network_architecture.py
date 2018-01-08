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


from lasagne.layers import InputLayer, DimshuffleLayer, ReshapeLayer, ConcatLayer, Upscale3DLayer, NonlinearityLayer, \
    BatchNormLayer, ElemwiseSumLayer, DropoutLayer, Conv3DLayer
from lasagne.init import HeNormal
from lasagne.nonlinearities import linear
import lasagne


def build_net(input_var=None, input_shape=(128, 128, 128), num_output_classes=4, num_input_channels=4,
              base_n_filter=8, do_instance_norm=True, batch_size=None, dropout_p=0.3, do_norm=True):
    nonlin = lasagne.nonlinearities.leaky_rectify
    if do_instance_norm:
        axes = (2, 3, 4)
    else:
        axes = 'auto'

    def conv_norm_lrelu(l_in, feat_out):
        l = Conv3DLayer(l_in, feat_out, 3, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
        if do_norm:
            l = BatchNormLayer(l, axes=axes)
        return NonlinearityLayer(l, nonlin)

    def norm_lrelu_conv(l_in, feat_out, stride=1, filter_size=3):
        if do_norm:
            l_in = BatchNormLayer(l_in, axes=axes)
        l = NonlinearityLayer(l_in, nonlin)
        return Conv3DLayer(l, feat_out, filter_size, stride, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))

    def lrelu_conv(l_in, feat_out, stride=1, filter_size=3):
        l = NonlinearityLayer(l_in, nonlin)
        return Conv3DLayer(l, feat_out, filter_size, stride, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))

    def norm_lrelu_upscale_conv_norm_lrelu(l_in, feat_out):
        if do_norm:
            l_in = BatchNormLayer(l_in, axes=axes)
        l = NonlinearityLayer(l_in, nonlin)
        l = Upscale3DLayer(l, 2)
        l = Conv3DLayer(l, feat_out, 3, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
        if do_norm:
            l = BatchNormLayer(l, axes=axes)
        l = NonlinearityLayer(l, nonlin)
        return l

    l_in = InputLayer(shape=(batch_size, num_input_channels, input_shape[0], input_shape[1], input_shape[2]),
                      input_var=input_var)

    l = r = Conv3DLayer(l_in, num_filters=base_n_filter, filter_size=3, stride=1, nonlinearity=linear, pad='same',
                        W=HeNormal(gain='relu'))
    l = NonlinearityLayer(l, nonlin)
    l = Conv3DLayer(l, num_filters=base_n_filter, filter_size=3, stride=1, nonlinearity=linear, pad='same',
                    W=HeNormal(gain='relu'))
    l = DropoutLayer(l, dropout_p)
    l = lrelu_conv(l, base_n_filter, 1, 3)
    l = ElemwiseSumLayer((l, r))
    skip1 = NonlinearityLayer(l, nonlin)
    if do_norm:
        l = BatchNormLayer(l, axes=axes)
    l = NonlinearityLayer(l, nonlin)

    l = r = Conv3DLayer(l, base_n_filter*2, 3, 2, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_conv(l, base_n_filter*2)
    l = DropoutLayer(l, dropout_p)
    l = norm_lrelu_conv(l, base_n_filter*2)
    l = ElemwiseSumLayer((l, r))
    if do_norm:
        l = BatchNormLayer(l, axes=axes)
    l = skip2 = NonlinearityLayer(l, nonlin)

    l = r = Conv3DLayer(l, base_n_filter*4, 3, 2, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_conv(l, base_n_filter*4)
    l = DropoutLayer(l, dropout_p)
    l = norm_lrelu_conv(l, base_n_filter*4)
    l = ElemwiseSumLayer((l, r))
    if do_norm:
        l = BatchNormLayer(l, axes=axes)
    l = skip3 = NonlinearityLayer(l, nonlin)

    l = r = Conv3DLayer(l, base_n_filter*8, 3, 2, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_conv(l, base_n_filter*8)
    l = DropoutLayer(l, dropout_p)
    l = norm_lrelu_conv(l, base_n_filter*8)
    l = ElemwiseSumLayer((l, r))
    if do_norm:
        l = BatchNormLayer(l, axes=axes)
    l = skip4 = NonlinearityLayer(l, nonlin)

    l = r = Conv3DLayer(l, base_n_filter*16, 3, 2, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_conv(l, base_n_filter*16)
    l = DropoutLayer(l, dropout_p)
    l = norm_lrelu_conv(l, base_n_filter*16)
    l = ElemwiseSumLayer((l, r))
    l = norm_lrelu_upscale_conv_norm_lrelu(l, base_n_filter*8)

    l = Conv3DLayer(l, base_n_filter*8, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    if do_norm:
        l = BatchNormLayer(l, axes=axes)
    l = NonlinearityLayer(l, nonlin)

    l = ConcatLayer((skip4, l), cropping=[None, None, 'center', 'center'])
    l = conv_norm_lrelu(l, base_n_filter*16)
    l = Conv3DLayer(l, base_n_filter*8, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_upscale_conv_norm_lrelu(l, base_n_filter*4)

    l = ConcatLayer((skip3, l), cropping=[None, None, 'center', 'center'])
    l = ds2 = conv_norm_lrelu(l, base_n_filter*8)
    l = Conv3DLayer(l, base_n_filter*4, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_upscale_conv_norm_lrelu(l, base_n_filter*2)

    l = ConcatLayer((skip2, l), cropping=[None, None, 'center', 'center'])
    l = ds3 = conv_norm_lrelu(l, base_n_filter*4)
    l = Conv3DLayer(l, base_n_filter*2, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    l = norm_lrelu_upscale_conv_norm_lrelu(l, base_n_filter)

    l = ConcatLayer((skip1, l), cropping=[None, None, 'center', 'center'])
    l = conv_norm_lrelu(l, base_n_filter*2)
    l_pred = Conv3DLayer(l, num_output_classes, 1, pad='same', nonlinearity=None)

    ds2_1x1_conv = Conv3DLayer(ds2, num_output_classes, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale = Upscale3DLayer(ds2_1x1_conv, 2)
    ds3_1x1_conv = Conv3DLayer(ds3, num_output_classes, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale_ds3_sum = ElemwiseSumLayer((ds1_ds2_sum_upscale, ds3_1x1_conv))
    ds1_ds2_sum_upscale_ds3_sum_upscale = Upscale3DLayer(ds1_ds2_sum_upscale_ds3_sum, 2)

    l = seg_layer = ElemwiseSumLayer((l_pred, ds1_ds2_sum_upscale_ds3_sum_upscale))
    l = DimshuffleLayer(l, (0, 2, 3, 4, 1))
    batch_size, n_rows, n_cols, n_z, _ = lasagne.layers.get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols * n_z, num_output_classes))
    l = NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.softmax)
    return l, seg_layer