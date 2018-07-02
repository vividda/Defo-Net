# Created on Tue Oct 17 01:24:46 2017
#
# @author: Zhihua Wang 

import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor



class VoxelDataGenerator(object): 

    def __init__(self, csv_file, data_dir, datatype, mode, batch_size, shuffle,
            preprocess_conditions=False, buffer_size=1000):

        self.preprocess_conditions = preprocess_conditions
        self.data_dir = data_dir
        self.datatype = datatype
        self.condition_dir = os.path.join(self.data_dir, csv_file)
        self.condition_names = list(datatype['names'][1:]) 
        self.mode = mode 

        self._read_csv_file()
        
        self.data_size_train = len(self.train_filename)
        self.data_size_test = len(self.test_filename)

        if shuffle:
            self._shuffle_lists()


        self.train_x = ["../Postprocess/x_train/" + s for s in self.train_filename]
        self.train_y = ["../Postprocess/y_train/" + s for s in self.train_filename]
        self.train_gen = ["../Postprocess/gen_train/" + s for s in self.train_filename]
        self.train_dis = ["../Postprocess/dis_train/" + s for s in self.train_filename]


        self.test_x = ["../Postprocess/x_test/" + s for s in self.test_filename]
        self.test_y = ["../Postprocess/y_test/" + s for s in self.test_filename]
        self.test_gen = ["../Postprocess/gen_test/" + s for s in self.test_filename]
        self.test_dis = ["../Postprocess/dis_test/" + s for s in self.test_filename]


        # convert lists to TF tensor
        self.train_x = convert_to_tensor(self.train_x, dtype=dtypes.string)
        self.train_y = convert_to_tensor(self.train_y, dtype=dtypes.string)
        self.train_gen = convert_to_tensor(self.train_gen, dtype=dtypes.string)
        self.train_dis = convert_to_tensor(self.train_dis, dtype=dtypes.string)
        self.train_filename = convert_to_tensor(self.train_filename, dtype=dtypes.string)

        self.test_x = convert_to_tensor(self.test_x, dtype=dtypes.string)
        self.test_y = convert_to_tensor(self.test_y, dtype=dtypes.string)
        self.test_gen = convert_to_tensor(self.test_gen, dtype=dtypes.string)
        self.test_dis = convert_to_tensor(self.test_dis, dtype=dtypes.string)
        self.test_filename = convert_to_tensor(self.test_filename, dtype=dtypes.string)



        # create dataset
        train_data = Dataset.from_tensor_slices((self.train_x, self.train_y, self.train_gen,
                                                 self.train_dis,
                                            self.train_filename))
        
        test_data = Dataset.from_tensor_slices((self.test_x, self.test_y, self.test_gen, self.test_dis,
                                            self.test_filename))


        if mode == 'training':
            data = train_data.map(self._parse_function_train, num_threads=2,
                      output_buffer_size=2*batch_size)

        elif mode == 'inference':
            data = test_data.map(self._parse_function_inference, num_threads=2,
                      output_buffer_size=2*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        data = data.batch(batch_size)

        self.data = data



    def _read_csv_file(self):
        self.train_namelist = []
        self.test_namelist = []
        self.conditions = []
        output_dir_vox = "../Postprocess"
        input_dir_vox = "../Preprocess_Voxel"

        input_dir = os.path.join(output_dir_vox, "x_train")
        for src_path in sorted(os.listdir(input_dir)):
            name, _ = os.path.splitext(os.path.basename(src_path))
            self.train_namelist.append(name)
        self.train_filename = [s + ".txt" for s in self.train_namelist]


        input_dir = os.path.join(output_dir_vox, "x_test")
        for src_path in sorted(os.listdir(input_dir)):
            name, _ = os.path.splitext(os.path.basename(src_path))
            self.test_namelist.append(name)
        self.test_filename = [s + ".txt" for s in self.test_namelist]



    def _shuffle_lists(self):
        train_filename = self.train_filename
        test_filename = self.test_filename

        conditions = self.conditions
        permutation_train = np.random.permutation(self.data_size_train)
        permutation_test = np.random.permutation(self.data_size_test)

        self.train_filename = []
        self.test_filename = []

        for i in permutation_train:
            self.train_filename.append(train_filename[i])
        for i in permutation_test:
            self.test_filename.append(test_filename[i])



    def _parse_function_train(self, vox_X, vox_Y, condition_gen, condition_dis, filename):

        print "Current Directory is", os.getcwd()
        vox_X = tf.read_file(vox_X)
        vox_X = tf.decode_raw(vox_X, tf.float64)
        vox_X = tf.reshape(vox_X, [64, 64, 64, 1])
        

        vox_Y = tf.read_file(vox_Y)
        vox_Y = tf.decode_raw(vox_Y, tf.float64)
        vox_Y = tf.reshape(vox_Y, [64, 64, 64, 1])

        condition_gen = tf.read_file(condition_gen)
        condition_gen = tf.decode_raw(condition_gen, tf.float64)
        
        condition_dis = tf.read_file(condition_dis)
        condition_dis = tf.decode_raw(condition_dis, tf.float64)
        condition_dis = tf.reshape(condition_dis, [32, 32, 32, 1])

        return vox_X, vox_Y, condition_gen, condition_dis, filename


    def _parse_function_inference(self, vox_X, vox_Y, condition_gen, condition_dis, filename):

        print "Current Directory is", os.getcwd()
        vox_X = tf.read_file(vox_X)
        vox_X = tf.decode_raw(vox_X, tf.float64)
        vox_X = tf.reshape(vox_X, [64, 64, 64, 1])
        

        vox_Y = tf.read_file(vox_Y)
        vox_Y = tf.decode_raw(vox_Y, tf.float64)
        vox_Y = tf.reshape(vox_Y, [64, 64, 64, 1])

        condition_gen = tf.read_file(condition_gen)
        condition_gen = tf.decode_raw(condition_gen, tf.float64)
        
        condition_dis = tf.read_file(condition_dis)
        condition_dis = tf.decode_raw(condition_dis, tf.float64)
        condition_dis = tf.reshape(condition_dis, [32, 32, 32, 1])

        return vox_X, vox_Y, condition_gen, condition_dis, filename




