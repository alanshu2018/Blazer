#coding: utf-8

import os
import sys
import tensorflow as tf

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

cols = ["ip", "app", "device", "os", "channel", "click_time"]

def _to_sequence_example(record,label):
    context = tf.train.Features(feature={
    })
    feature_lists = tf.train.FeatureLists(feature_list={
        "features": _int64_feature_list(record),
        "label": _int64_feature_list([label]),
    })
    sequence_example = tf.train.SequenceExample(
        context= context, feature_lists = feature_lists
    )

    return sequence_example

def parse_click_time(click_time):
    click_time = click_time[8:]
    day = int(click_time[:2])
    hour = int(click_time[3:5])
    min = int(click_time[6:8])
    #print(day, hour, min)
    return [day, hour, int(min/10), int(hour/4)]

#click_time="2017-11-06 14:32:21"
#parse_click_time(click_time)
#sys.exit(-1)

data_dir="data"
file_name="train.tf"
output_file = os.path.join(data_dir,file_name)
writer = tf.python_io.TFRecordWriter(output_file)
train_file = "data/train.csv"
counter = 0
idx = 0
for line in open(train_file):
    idx +=1
    if idx == 1:
        # skip header
        continue
    flds = line.strip().split(",")
    if len(flds)!=8:
        continue
    click_time = flds[5]

    day, hour, min_10, hour_4 = parse_click_time(click_time)
    record = (int(flds[0]),int(flds[1]),int(flds[2]),int(flds[3]),
              int(flds[4]),day, hour, min_10, hour_4)

    sequence_example = _to_sequence_example(record,int(flds[7]))
    if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        counter += 1

        if not counter % 1000:
            print("Processed %d"%(counter))

writer.close()
