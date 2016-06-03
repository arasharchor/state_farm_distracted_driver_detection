import os
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
import pandas as pd
import datetime
import h5py

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

def restore_train_data(path):
    hf = h5py.File(path, 'r')
    train_data = hf['train_data']
    train_target = hf['train_target']
    driver_id = hf['driver_id']
    unique_drivers = hf['unique_drivers']
    train_data_copy = np.copy(train_data)
    train_target_copy = np.copy(train_target)
    driver_id_copy = np.copy(driver_id)
    unique_drivers_copy = np.copy(unique_drivers)
    hf.close()
    return train_data_copy, train_target_copy, driver_id_copy, unique_drivers_copy

def restore_test_data(path):
    hf = h5py.File(path, 'r')
    test_data = hf['test_data']
    test_id = hf['test_id']
    test_data_copy = np.copy(test_data)
    test_id_copy = np.copy(test_id)
    hf.close()
    return test_data_copy, test_id_copy

def restore_training_data(img_rows, img_cols, color_type):
    print('Restore train from cache!')
    train_cache_path = os.path.join('..', 'cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(
        color_type) + '.h5')
    (train_data, train_target, driver_id, unique_drivers) = restore_train_data(train_cache_path)
    return train_data, train_target, driver_id, unique_drivers

def restore_testing_data(img_rows, img_cols, color_type):
    print('Restore test from cache!')
    test_cache_path = os.path.join('..', 'cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(
        color_type) + '.h5')
    (test_data, test_id) = restore_test_data(test_cache_path)
    return test_data, test_id

def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index

def split_training_and_validation(img_rows, img_cols, color_type):
    train_data, train_target, driver_id, unique_drivers = restore_training_data(img_rows, img_cols, color_type)
    driver_list = np.array(['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075', 'p081']) # 26 drivers;
    valid_number = 3
    random_validset_index = random.sample(range(0,25), valid_number)
    unique_list_valid = []
    random_validset_name = []

    for n in range(valid_number):
        random_validset_name = np.append(random_validset_name, driver_list[random_validset_index[n]])
        unique_list_valid = np.append(unique_list_valid, driver_list[random_validset_index[n]])

    unique_list_train = np.setdiff1d(driver_list, random_validset_name)
    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)
    return X_train, Y_train, train_index, X_valid, Y_valid, test_index

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir('../cache'):
        os.mkdir('../cache')
    open(os.path.join('..', 'cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join('..', 'cache', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join('..', 'cache', 'architecture.json')).read())
    model.load_weights(os.path.join('..', 'cache', 'model_weights.h5'))
    return model

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('../subm'):
        os.mkdir('../subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('..', 'subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

def create_model_vgg(img_rows, img_cols, color_type = 1, weight_load = False):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(color_type,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weight_load == True:
        model.load_weights('../cache/vgg16_weights.h5')

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model

def run_train():
    img_rows = 224
    img_cols = 224
    color_type = 1
    batch_size = 32
    nb_epoch = 10
    weight_load = False

    (X_train, Y_train, train_index, X_valid, Y_valid, test_index) = split_training_and_validation(img_rows, img_cols, color_type)
    #(test_data, test_id) = restore_testing_data(img_rows, img_cols, color_type)
    model = create_model_vgg(img_rows, img_cols, color_type, weight_load)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))

    save_model(model)

    # yfull_train = dict()
    # yfull_test = []
    #
    # predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)
    # score = log_loss(Y_valid, predictions_valid)
    # print('Score log_loss: ', score)
    #
    # # Store valid predictions
    # for i in range(len(test_index)):
    #     yfull_train[test_index[i]] = predictions_valid[i]
    #
    # # Store test predictions
    # test_prediction = model.predict(test_data, batch_size=128, verbose=1)
    # yfull_test.append(test_prediction)
    #
    # print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))
    # info_string = 'loss_' + str(score) \
    #               + '_r_' + str(img_rows) \
    #               + '_c_' + str(img_cols) \
    #               + '_ep_' + str(nb_epoch)
    #
    # test_res = merge_several_folds_mean(yfull_test, 1)
    # create_submission(test_res, test_id, info_string)

run_train()

