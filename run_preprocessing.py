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


import argparse
from dataset import run_preprocessing_BraTS2017_trainSet, run_preprocessing_BraTS2017_valOrTestSet
import paths


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", help="can be train or testval. Use testval for validation and test datasets",
                    type=str)
args = parser.parse_args()

if args.mode == "val":
    run_preprocessing_BraTS2017_valOrTestSet(paths.raw_validation_data_folder, paths.preprocessed_validation_data_folder)
elif args.mode == "train":
    run_preprocessing_BraTS2017_trainSet(paths.raw_training_data_folder, paths.preprocessed_training_data_folder)
elif args.mode == "test":
    run_preprocessing_BraTS2017_trainSet(paths.raw_testing_data_folder, paths.preprocessed_testing_data_folder)
else:
    raise ValueError("Unknown value for --mode. Use \"train\", \"test\" or \"val\"")
