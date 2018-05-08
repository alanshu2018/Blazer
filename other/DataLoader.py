#coding: utf-8

import pandas as pd
import gc
from sklearn.utils import shuffle

class FileObject(object):
    def __init__(self,filename, total_len):
        self.filename = filename
        self.next_read_pos = 0
        self.total_len = total_len

        self.read_start = 0
        self.read_len = 0
        self.data = []

    def read_some_data(self, do_shuffle):
        print("Read:read_len={},total_len={}".format(self.read_len,self.total_len))
        if self.read_len >0 and self.read_len >= self.total_len:
            self.read_start = 0
            return

        del self.data
        gc.collect()
        print("Load data: read_pos={}".format(self.next_read_pos))
        self.data = pd.read_csv(self.filename,skiprows=self.next_read_pos,nrows=10000000)
        if do_shuffle:
            self.data = shuffle(self.data)
        self.read_len = len(self.data)

        self.read_start = 0
        self.next_read_pos = self.next_read_pos + self.read_len
        print("Load data: read_len={}".format(self.read_len))

    def reset(self):
        print("Reset:read_len={},total_len={}".format(self.read_len,self.total_len))
        if self.read_len >0 and self.read_len >= self.total_len:
            self.read_start = 0
            return

        self.read_start = 0
        self.read_len = 0
        self.next_read_pos = 0

    def get_minibatch(self,batch_size, do_shuffle):
        if self.read_start >= self.read_len:
            # read file
            if self.next_read_pos +1 >= self.total_len:
                return None

            self.read_some_data(do_shuffle)

        #print("{}\t{}\t{}\t{}".format(self.read_start,self.read_len,self.next_read_pos,self.total_len))
        start = self.read_start
        end = start + batch_size
        if end > self.read_len:
            end = self.read_len

        self.read_start = end
        batch = self.data[start:end].values
        return (batch[:,1:],batch[:,0])

class DataLoader(object):
    def __init__(self,files):
        self.files = files
        self.file_num = len(self.files)
        self.file_objs = [ FileObject(file_name, file_lines) for (file_name,file_lines) in self.files]
        self.file_idx = 0

    def get_minibatch(self, batch_size=20000, epochs=1, do_shuffle=True):
        for _ in range(epochs):
            for file_obj in self.file_objs:
                file_obj.reset()
                while True:
                    batch = file_obj.get_minibatch(batch_size, do_shuffle)
                    if batch is None:
                        break
                    yield batch

if __name__ == "__main__":
    data_loader = DataLoader(files=[
        ("data/dev1_feature_norm.simple.csv",2000868),
        #("data/dev2_feature_norm.simple.csv",1999132),
        #("data/test_feature_norm.simple.csv",18790469),
        #("data/train_feature_norm.simple.csv",180903890),
    ])
    for batch_x,batch_y in data_loader.get_minibatch(20000,epochs=1,do_shuffle=False):
        print(batch_x.shape)
        print(batch_y.shape)
        print(batch_x[:5,:8])
        print(batch_y[:5])
        break
