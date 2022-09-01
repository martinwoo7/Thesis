import pandas as pd
import numpy as np


TRAIN_PATH = "..."
TEST_PATH = "..."


class DataPKG:
    _train_x = []
    _train_y = []
    _test_x = []
    _test_y = []
    _epoch_count = 0
    _epochs_completed = 0
    _trainset_size = 0


    def __init__(self,train_path,test_path,class_list):
        
        
        # create train set
        input_trainset = pd.read_csv(train_path, header = None)
        trainset = input_trainset.to_numpy()
        train_row, train_col = trainset.shape
        self._trainset_size = train_row
        
        
        for i in range(train_row):
            # check if label is valid
            label = input_trainset.iloc[i, 0]
            
            if label in class_list:

                #print('label is ',label)
                # append sensor values
                self._train_x.append(input_trainset.iloc[i, 1:973])
                # create test_y
                temp_train_index = class_list.index(label)
                y = [0 for _ in range(len(class_list))]
                y[temp_train_index] = 1
                self._train_y.append(y)
        

        # create test set
        input_testset = pd.read_csv(test_path, header = None)
        testset = input_testset.to_numpy()
        test_row, test_col = testset.shape
        #print('trainset.shape: ', testset.shape)
        #print('test_row: ', test_row)
        for j in range(test_row):
            # check if label is valid
            label = input_testset.iloc[j, 0]
            if label not in class_list:
                continue
            # append sensor values
            self._test_x.append(input_testset.iloc[j, 1:973])
            # create test_y
            temp_test_index = class_list.index(label)
            y = [0 for _ in range(len(class_list))]
            y[temp_test_index] = 1
            self._test_y.append(y)


    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    @property
    def test_x(self):
        return self._test_x

    @property
    def test_y(self):
        return self._test_y

    @property
    def epoch_count(self):
        return self._epoch_count

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._epoch_count
        if start + batch_size > self._trainset_size:
            # Finished
            self._epochs_completed += 1
            remaining_trainset = self._trainset_size - start
            if remaining_trainset != 0:
                x_rest_part = self.train_x[start:self._trainset_size]
                y_rest_part = self.train_y[start:self._trainset_size]
                # Start next epoch
                start = 0
                self._epoch_count = batch_size - remaining_trainset
                end = self._epoch_count
                x_new_part = self.train_x[start:end]
                y_new_part = self.train_y[start:end]
                #print('x_rest_part: ', x_rest_part)
                #print('y_new_part: ', y_new_part)
                batch_x,batch_y = np.concatenate(
                    (x_rest_part, x_new_part), axis=0), np.concatenate(
                    (y_rest_part, y_new_part), axis=0)

            else:
                # Start next epoch
                start = 0
                self._epoch_count = batch_size - remaining_trainset
                end = self._epoch_count
                batch_x = self.train_x[start:end]
                batch_y = self.train_y[start:end]
        else:
            
            self._epoch_count += batch_size
            end = self.epoch_count
            batch_x = self.train_x[start:end]
            batch_y = self.train_y[start:end]
        
        return np.array(batch_x),np.array(batch_y)



    def get_test_data(self):
        x = self.test_x
        y = self.test_y
        return np.array(x), np.array(y)

    def get_train_data(self):
        x = self.train_x
        y = self.train_y
        return np.array(x), np.array(y)
