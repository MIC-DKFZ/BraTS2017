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
import os.path
import os
import SimpleITK as sitk
from multiprocessing import Pool
import os.path as path
from utils import reshape_by_padding_upper_coords
from scipy.ndimage.morphology import binary_fill_holes
import cPickle
from batchgenerators.dataloading import DataLoaderBase
from batchgenerators.augmentations.utils import resize_image_by_padding_batched, random_crop_3D_image_batched
import paths


def extract_brain_region(image, segmentation, outside_value=0):
    brain_voxels = np.where(segmentation != outside_value)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    return image[resizer], [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]


def cut_off_values_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    if mask is None:
        mask = image != image[0,0,0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
    return image


def run(folder, out_folder, id, name, return_if_no_seg=True):
    print id
    if not path.isfile(path.join(folder, "%s_flair.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_t1.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_seg.nii.gz" % name)):
        if return_if_no_seg:
            return
    if not path.isfile(path.join(folder, "%s_t1ce.nii.gz" % name)):
        return
    if not path.isfile(path.join(folder, "%s_t2.nii.gz" % name)):
        return

    t1_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_t1.nii.gz" % name))).astype(np.float32)
    t1_img_sitk = sitk.ReadImage(path.join(folder, "%s_t1.nii.gz" % name))
    t1c_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_t1ce.nii.gz" % name))).astype(np.float32)
    t2_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_t2.nii.gz" % name))).astype(np.float32)
    flair_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_flair.nii.gz" % name))).astype(np.float32)
    try:
        seg_img = sitk.GetArrayFromImage(sitk.ReadImage(path.join(folder, "%s_seg.nii.gz" % name))).astype(np.float32)
    except RuntimeError:
        seg_img = np.zeros(t1_img.shape)
    except IOError:
        seg_img = np.zeros(t1_img.shape)

    original_shape = t1_img.shape

    brain_mask = (t1_img != t1_img[0, 0, 0]) & (t1c_img != t1c_img[0, 0, 0]) & (t2_img != t2_img[0, 0, 0]) & (
    flair_img != flair_img[0, 0, 0])

    # compute bbox of brain, This is now actually also returned when calling extract_brain_region, but was not at the
    # time this code was initially written. In order to not break anything we will keep it like it was
    brain_voxels = np.where(brain_mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    with open(os.path.join(out_folder, "%03.0d.pkl" % id), 'w') as f:
        dp = {}
        dp['orig_shp'] = original_shape
        dp['bbox_z'] = [minZidx, maxZidx]
        dp['bbox_x'] = [minXidx, maxXidx]
        dp['bbox_y'] = [minYidx, maxYidx]
        dp['spacing'] = t1_img_sitk.GetSpacing()
        dp['direction'] = t1_img_sitk.GetDirection()
        dp['origin'] = t1_img_sitk.GetOrigin()
        cPickle.dump(dp, f)

    t1km_sub = t1c_img - t1_img
    tmp = (t1c_img != 0) & (t1_img != 0)
    tmp = binary_fill_holes(tmp.astype(int))
    t1km_sub[~tmp.astype(bool)] = 0

    t1_img, bbox = extract_brain_region(t1_img, brain_mask, 0)
    t1c_img, bbox = extract_brain_region(t1c_img, brain_mask, 0)
    t2_img, bbox = extract_brain_region(t2_img, brain_mask, 0)
    flair_img, bbox = extract_brain_region(flair_img, brain_mask, 0)
    t1km_sub, bbox = extract_brain_region(t1km_sub, brain_mask, 0)
    seg_img, bbox = extract_brain_region(seg_img, brain_mask, 0)

    assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape
    msk = t1_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1_img, msk, 2., 98.)
    t1_img[msk] = (t1_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t1c_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1c_img, msk, 2., 98.)
    t1c_img[msk] = (t1c_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t2_img != 0
    tmp = cut_off_values_upper_lower_percentile(t2_img, msk, 2., 98.)
    t2_img[msk] = (t2_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = flair_img != 0
    tmp = cut_off_values_upper_lower_percentile(flair_img, msk, 2., 98.)
    flair_img[msk] = (flair_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t1km_sub != 0
    tmp = cut_off_values_upper_lower_percentile(t1km_sub, msk, 2., 98.)
    t1km_sub[msk] = (t1km_sub[msk] - tmp[msk].mean()) / tmp[msk].std()

    shp = t1_img.shape
    pad_size = np.max(np.vstack((np.array([128, 128, 128]), np.array(shp))), 0)
    t1_img = reshape_by_padding_upper_coords(t1_img, pad_size, 0)
    t1c_img = reshape_by_padding_upper_coords(t1c_img, pad_size, 0)
    t2_img = reshape_by_padding_upper_coords(t2_img, pad_size, 0)
    flair_img = reshape_by_padding_upper_coords(flair_img, pad_size, 0)
    t1km_sub = reshape_by_padding_upper_coords(t1km_sub, pad_size, 0)
    seg_img = reshape_by_padding_upper_coords(seg_img, pad_size, 0)

    all_data = np.zeros([6] + list(t1_img.shape), dtype=np.float32)
    all_data[0] = t1_img
    all_data[1] = t1c_img
    all_data[2] = t2_img
    all_data[3] = flair_img
    all_data[4] = t1km_sub
    all_data[5] = seg_img
    np.save(os.path.join(out_folder, "%03.0d" % id), all_data)


def run_star(args):
    return run(*args)


def run_preprocessing_BraTS2017_trainSet(base_folder=paths.raw_training_data_folder,
                                folder_out=paths.preprocessed_training_data_folder):
    ctr = 0
    id_name_conversion = []
    for f in ("HGG", "LGG"):
        fld = os.path.join(base_folder, f)
        patients = os.listdir(fld)
        patients.sort()
        fldrs = [os.path.join(fld, pt) for pt in patients]
        p = Pool(8)
        p.map(run_star, zip(fldrs,
                            [folder_out] * len(patients),
                            range(ctr, ctr + len(patients)),
                            patients))
        p.close()
        p.join()
        for i, j in zip(patients, range(ctr, ctr + len(patients))):
            id_name_conversion.append([i, j, f])
        ctr += (ctr + len(patients))
    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
    from shutil import copyfile
    copyfile(os.path.join(base_folder, "survival_data.csv"), os.path.join(folder_out, "survival_data.csv"))


def run_preprocessing_BraTS2017_valOrTestSet(base_folder=paths.raw_validation_data_folder,
                                       folder_out=paths.preprocessed_validation_data_folder):
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    ctr = 0
    id_name_conversion = []
    fld = os.path.join(base_folder)
    patients = os.listdir(fld)
    patients.sort()
    fldrs = [os.path.join(fld, pt) for pt in patients]
    p = Pool(8)
    p.map(run_star, zip(fldrs,
                        [folder_out] * len(patients),
                        range(ctr, ctr + len(patients)),
                        patients, len(patients) * [False]))
    p.close()
    p.join()
    for i, j in zip(patients, range(ctr, ctr + len(patients))):
        id_name_conversion.append([i, j, 'unknown'])  # not known whether HGG or LGG
    ctr += (ctr + len(patients))
    id_name_conversion = np.vstack(id_name_conversion)
    np.savetxt(os.path.join(folder_out, "id_name_conversion.txt"), id_name_conversion, fmt="%s")
    from shutil import copyfile
    copyfile(os.path.join(base_folder, "survival_evaluation.csv"), os.path.join(folder_out, "survival_evaluation.csv"))


def load_dataset(pat_ids=range(300), folder=paths.preprocessed_training_data_folder):
    id_name_conversion = np.loadtxt(os.path.join(folder, "id_name_conversion.txt"), dtype="str")
    idxs = id_name_conversion[:, 1].astype(int)
    dataset = {}
    for p in pat_ids:
        if os.path.isfile(os.path.join(folder, "%03.0d.npy" % p)):
            dataset[p] = {}
            dataset[p]['data'] = np.load(os.path.join(folder, "%03.0d.npy" % p), mmap_mode='r')
            dataset[p]['idx'] = p
            dataset[p]['name'] = id_name_conversion[np.where(idxs == p)[0][0], 0]
            dataset[p]['type'] = id_name_conversion[np.where(idxs == p)[0][0], 2]

            with open(os.path.join(folder, "%03.0d.pkl" % p), 'r') as f:
                dp = cPickle.load(f)

            dataset[p]['orig_shp'] = dp['orig_shp']
            dataset[p]['bbox_z'] = dp['bbox_z']
            dataset[p]['bbox_x'] = dp['bbox_x']
            dataset[p]['bbox_y'] = dp['bbox_y']
            dataset[p]['spacing'] = dp['spacing']
            dataset[p]['direction'] = dp['direction']
            dataset[p]['origin'] = dp['origin']
    return dataset



class BatchGenerator3D_random_sampling(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE, num_batches, seed, patch_size=(128, 128, 128), convert_labels=False):
        self.convert_labels = convert_labels
        self._patch_size = patch_size
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches, seed)

    def generate_train_batch(self):
        ids = np.random.choice(self._data.keys(), self.BATCH_SIZE)
        data = np.zeros((self.BATCH_SIZE, 5, self._patch_size[0], self._patch_size[1], self._patch_size[2]),
                        dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self._patch_size[0], self._patch_size[1], self._patch_size[2]),
                       dtype=np.float32)
        types = []
        patient_names = []
        identifiers = []
        ages = []
        survivals = []
        for j, i in enumerate(ids):
            types.append(self._data[i]['type'])
            patient_names.append(self._data[i]['name'])
            identifiers.append(self._data[i]['idx'])
            # construct a batch, not very efficient
            data_all = self._data[i]['data'][None]
            if np.any(np.array(data_all.shape[2:]) - np.array(self._patch_size) < 0):
                new_shp = np.max(np.vstack((np.array(data_all.shape[2:])[None], np.array(self._patch_size)[None])), 0)
                data_all = resize_image_by_padding_batched(data_all, new_shp, 0)
            data_all = random_crop_3D_image_batched(data_all, self._patch_size)
            data[j, :] = data_all[0, :5]
            if self.convert_labels:
                seg[j, 0] = convert_brats_seg(data_all[0, 5])
            else:
                seg[j, 0] = data_all[0, 5]
            if 'survival' in self._data[i].keys():
                survivals.append(self._data[i]['survival'])
            else:
                survivals.append(np.nan)
            if 'age' in self._data[i].keys():
                ages.append(self._data[i]['age'])
            else:
                ages.append(np.nan)
        return {'data': data, 'seg': seg, "idx": ids, "grades": types, "identifiers": identifiers,
                "patient_names": patient_names, 'survival':survivals, 'age':ages}


def convert_brats_seg(seg):
    new_seg = np.zeros(seg.shape, seg.dtype)
    new_seg[seg == 1] = 3
    new_seg[seg == 2] = 1
    new_seg[seg == 4] = 2
    return new_seg


def convert_to_brats_seg(seg):
    new_seg = np.zeros(seg.shape, seg.dtype)
    new_seg[seg == 1] = 2
    new_seg[seg == 2] = 4
    new_seg[seg == 3] = 1
    return new_seg

