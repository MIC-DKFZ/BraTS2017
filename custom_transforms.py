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
from scipy.ndimage import binary_fill_holes
from batchgenerators.transforms import AbstractTransform


def brain_mask_aware_stretch_zero_one(data, seg, clip_range=None, per_channel=False):
    data_new = np.copy(data)
    for sample in range(data.shape[0]):
        tmp = np.array(data[sample])
        if clip_range is not None:
            tmp = np.clip(tmp, clip_range[0], clip_range[1])
        if not per_channel:
            brain_mask = seg[sample][1:] != 0
            tmp -= tmp[brain_mask!=0].min()
            tmp /= tmp[brain_mask!=0].max()
            tmp[brain_mask == 0] = 0
        else:
            brain_mask = seg[sample][1:] != 0
            for c in range(tmp.shape[0]):
                tmp[c] -= tmp[c][brain_mask[c] != 0].min()
                tmp[c] /= tmp[c][brain_mask[c] != 0].max()
                tmp[c, brain_mask[c] == 0] = 0
        data_new[sample] = tmp
    return data_new


def create_brain_masks(data, seg):
    shp = list(data.shape)
    num_seg = seg.shape[1]
    shp[1] += num_seg
    seg_with_brain_mask = np.zeros(shp, dtype=np.float32)
    seg_with_brain_mask[:, :num_seg] = seg
    for b in range(data.shape[0]):
        for c in range(data.shape[1]):
            this_mask = data[b, c] != 0
            this_mask = binary_fill_holes(this_mask)
            seg_with_brain_mask[b, c + num_seg] = this_mask
    return seg_with_brain_mask


class BrainMaskAwareStretchZeroOneTransform(AbstractTransform):
    def __init__(self, clip_range=None, per_channel=False):
        self.per_channel = per_channel
        self.clip_range = clip_range

    def __call__(self, **data_dict):
        data_dict['data'] = brain_mask_aware_stretch_zero_one(data_dict['data'], data_dict['seg'], self.clip_range,
                                                              self.per_channel)
        return data_dict


class GenerateBrainMaskTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        data_dict['seg'] = create_brain_masks(data_dict['data'], data_dict['seg'])
        return data_dict
