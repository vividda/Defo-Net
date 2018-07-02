import os
import threading
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datatype = {'names': ('file_name', 'force_strength', 'force_location'), 'formats': ('S12', 'f4', 'f4')}



complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

sys.path.append('..')

# pcl supposed to have shape: 480 preprocess_x 640 preprocess_x 3 = 921600
pcl_PATH = 'data/'
normalization_label = True

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_csv(filename):
    data_csv = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=datatype)
    label_type_list = list(datatype['names'][1:])
    if normalization_label:
        data_csv = normalization_label(label_type_list, data_csv)

    all_filenames = data_csv[datatype['names'][0]]
    labels = data_csv[label_type_list]
    print "condition csv file loaded"
    return all_filenames, labels

def normalization_label(label_type_list, data_csv):
    for label_type in label_type_list:
        raw_sensor_data = data_csv[label_type]
        max = np.ndarray.max(raw_sensor_data)
        min = np.ndarray.min(raw_sensor_data)
        if max != min:
            range = max - min
            one_bit = 1 / range
            new_sensor_data = (raw_sensor_data - min) * one_bit
            new_sensor_data = np.ndarray.round(new_sensor_data,decimals=4)
        else:
            new_sensor_data = raw_sensor_data/max # 1
        data_csv[label_type] = new_sensor_data
    return data_csv

def get_txt_binary(filename):
    """ You can read in the pcl using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    pcl = np.loadtxt(filename)
    pcl = np.asarray(pcl, np.uint8)
    shape = np.array(pcl.shape, np.int32)
    return shape.tobytes(), pcl.tobytes() # convert pcl to raw data bytes in the array.

def get_csv_binary(filename):
    label = np.asarray(filename, np.float32)
    shape= np.array(filename.shape, np.int32)
    return shape.tobytes(), label.tobytes()

def write_tfrecord(label, pcl1_files, pcl2_files, tfrecord_file):
    shapel, label = get_csv_binary(label)
    options = tf.python_io.TFRecordOptions(compression_type=1)
    writer = tf.python_io.TFRecordWriter(tfrecord_file, options=options)

    for pcl1_file, pcl2_file in zip(pcl1_files, pcl2_files):

        shape1, binary_pcl1 = get_txt_binary(pcl1_file)
        shape2, binary_pcl2 = get_txt_binary(pcl2_file)

        # write label, shape, and pcl content to the TFRecord file
        example = tf.train.Example(features=tf.train.Features(feature={
            'shapel': _bytes_feature(shapel),
            'label': _bytes_feature(label),
            'shape1': _bytes_feature(shape1),
            'pcl1': _bytes_feature(binary_pcl1),
            'shape2': _bytes_feature(shape2),
            'pcl2': _bytes_feature(binary_pcl2)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(filenames):
    dataset = tf.data.TFRecordDataset(filenames,compression_type="ZLIB")


    def parser(record):
        keys_to_features = {
                    'shapel': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string),
                    'shape1': tf.FixedLenFeature([], tf.string),
                    'pcl1': tf.FixedLenFeature([], tf.string),
                    'shape2': tf.FixedLenFeature([], tf.string),
                    'pcl2': tf.FixedLenFeature([], tf.string),}
        parsed = tf.parse_single_example(record, keys_to_features)
        # pcl was saved as uint8, so we have to decode as uint8.
        pcl1 = tf.decode_raw(parsed['pcl1'], tf.uint8)
        shape1 = tf.decode_raw(parsed['shape1'], tf.int32)
        pcl2 = tf.decode_raw(parsed['pcl2'], tf.uint8)
        shape2 = tf.decode_raw(parsed['shape2'], tf.int32)
        label = tf.decode_raw(parsed['label'], tf.float32)
        shapel = tf.decode_raw(parsed['shapel'], tf.int32)

        # the pcl tensor is flattened out, so we have to reconstruct the shape
        reshape_pcl1 = tf.reshape(pcl1, shape1)
        reshape_pcl2 = tf.reshape(pcl2, shape2)
        reshape_label = tf.reshape(label, shapel)

        # return reshape_pcl1, reshape_pcl2, reshape_label
        return reshape_pcl1, reshape_pcl2, reshape_label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=2)
    # dataset = dataset.batch(2)
    dataset = dataset.padded_batch(2, padded_shapes=([None, 3],[None, 3],[2]))

    dataset = dataset.repeat(2)

    return dataset



    # plot(pcl1, pcl2)

def write_data(all_filenames, labels, pcl_file_name, dst_path):
    src_paths_x = ["preprocess_x/" + _  for _ in pcl_file_name]
    src_paths_y = ["preprocess_y/" + _  for _ in pcl_file_name]
    options = tf.python_io.TFRecordOptions(compression_type=1)
    writer = tf.python_io.TFRecordWriter(dst_path, options=options)

    global total
    total = len(pcl_file_name)
    print "processing %d files" % total

    global start
    start = time.time()


    for src_path_x, src_path_y, name_list in zip(src_paths_x, src_paths_y, pcl_file_name):
        shape1, binary_pcl1 = get_txt_binary(src_path_x)
        shape2, binary_pcl2 = get_txt_binary(src_path_y)
        name_list, _ = os.path.splitext(name_list)
        label = find_csv_label(all_filenames, labels, name_list)
        shapel, label = get_csv_binary(label)

        # write label, shape, and pcl content to the TFRecord file
        example = tf.train.Example(features=tf.train.Features(feature={
            'shapel': _bytes_feature(shapel),
            'label': _bytes_feature(label),
            'shape1': _bytes_feature(shape1),
            'pcl1': _bytes_feature(binary_pcl1),
            'shape2': _bytes_feature(shape2),
            'pcl2': _bytes_feature(binary_pcl2)
        }))
        writer.write(example.SerializeToString())
        complete()
    global num_complete
    num_complete = 0

    writer.close()

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))
        last_complete = now


def find_csv_label(all_filenames, labels, name_list):
    index = np.where(all_filenames == name_list)
    index = np.asscalar(index[0])
    label = labels[index]
    label = np.asarray(list(label))
    return label


def main():
    preprocess_x = "preprocess_x"
    preprocess_y = "preprocess_y"
    label_dir = "Preprocess_Con.csv"


    all_x = sorted(os.listdir(preprocess_x))
    all_y = sorted(os.listdir(preprocess_y))

    if set(all_x) == set(all_y):
        print('input dataset match with output dataset')
    else:
        raise NameError('input output do not match')

    split = 0.7
    split_index = int(np.floor(len(all_x) * split))
    training = all_x[:split_index]
    testing = all_x[split_index:]
    all_filenames, labels = load_csv(label_dir)
    tfrecord_file_tr = 'train.tfrecord'
    tfrecord_file_val = 'test.tfrecord'

    write_data(all_filenames, labels, training, tfrecord_file_tr)
    write_data(all_filenames, labels, testing, tfrecord_file_val)



if __name__ == '__main__':
    main()
