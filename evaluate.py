# -*- coding: utf-8 -*-
"""
@author: Terada
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import random as rn
import shutil
import datetime
import csv
import glob
import pandas as pd
import re
from configparser import ConfigParser, ExtendedInterpolation

from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf

import read_image as imread_mod
import read_text as txtread_mod

def init_random_seed(_seed = 0):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(_seed)
    rn.seed(_seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(_seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

class DatasetGeneration():
    def __init__(self):
        self.filename1_str = 'Image'
        self.filename2_str = 'land'

    def common_filename_check(self, _fname1, _fname2, _fname1_str='Image', _fname2_str='land', _fname1_ext='jpg', _fname2_ext='txt'):
        import re

        _1to2name = [re.sub(_fname1_str, _fname2_str, _fname1[i].split(_fname1_ext)[0]) for i in range(len(_fname1))]
        _com_name = []

        for i in range(len(_1to2name)):
            for j in range(len(_fname2)):
                if _1to2name[i] in _fname2[j]:
                    _com_name.append(_1to2name[i].split(_fname2_str)[1])
                    break

        iext = np.array([_fname1_ext] * len(_com_name))
        _com_fname1 = np.core.defchararray.add(_com_name, iext)
        _com_fname1 = [re.sub('^', _fname1_str, _com_fname1[i]) for i in range(len(_com_fname1))]

        text = np.array([_fname2_ext] * len(_com_name))
        _com_fname2 = np.core.defchararray.add(_com_name, text)
        _com_fname2 = [re.sub('^', _fname2_str, _com_fname2[i]) for i in range(len(_com_fname2))]
        return _com_fname1, _com_fname2

    def load_data(self, _image_path, _label_path, _io_fname, _img_size, _input_size, data_check=True):
        X, iname = imread_mod.main(_image_path, _input_size) # read image
        y, tname = txtread_mod.main(_label_path, _io_fname, _img_size, False) # read text

        if data_check:
            iname, tname = self.common_filename_check(iname, tname, self.filename1_str, self.filename2_str)
            print("Common Files : {0}".format(len(iname)))
            _iname_path = []
            _tname_path = []
            for i in range(len(iname)):
                _iname_path.append(os.path.dirname(_image_path) + "//" + iname[i])
                _tname_path.append(os.path.dirname(_label_path) + "//" + tname[i])
            X, iname = imread_mod.main(_iname_path, _input_size)
            y, tname = txtread_mod.main(_tname_path, _io_fname, _img_size, False)

        self.filename = tname
        return X, y

    def load_label(self, _path):

        fp = open("data/attribute.csv")
        data = fp.read()
        fp.close()
        gen_dict = {}
        age_dict = {}
        lines = data.split('\n')
        for i, line in enumerate(lines):
            if i == 0:  # header
                continue
            if line == "":
                break
            vals = line.split(',')
            id = int(vals[0].split('R')[1])
            gen_dict[id] = int(vals[1]) - 1
            age_dict[id] = int(vals[2]) - 2

        gen_list = []
        age_list = []
        pattern = '([+-]?[0-9]+\.?[0-9]*)'
        for i in range(len(self.filename)):
            num = re.findall(pattern, os.path.basename(self.filename[i]))[0]
            num = int(num.split('.')[0])
            gen_list.append(gen_dict[num])
            age_list.append(age_dict[num])
        gen_list.insert(0, 1)  # max value
        age_list.insert(0, 4)  # max value
        one_hot_y1 = np_utils.to_categorical(gen_list)
        one_hot_y2 = np_utils.to_categorical(age_list)
        one_hot_y1 = np.delete(one_hot_y1, 0, 0)  # delete max value
        one_hot_y2 = np.delete(one_hot_y2, 0, 0)  # delete max value

        with open(_path + "gender_label.csv", 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [genderlabel_dict[str(i)] for i in range(len(genderlabel_dict))]
            writer.writerow(header)
            for row in one_hot_y1:
                writer.writerow(row)
        with open(_path + "age_label.csv", 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [agelabel_dict[str(i)] for i in range(len(agelabel_dict))]
            writer.writerow(header)
            for row in one_hot_y2:
                writer.writerow(row)

        return  one_hot_y1, one_hot_y2

class Evaluate():
    def __init__(self):
        self.gd = DatasetGeneration()
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.flag_of_start = True

    def calcDiffLabel(self, _label_csv, _predict, _path):
        name = _path.split('\\')[len(_path.split('\\')) - 1]
        # get label data
        label_df = pd.read_csv(_label_csv, index_col=0)
        label_index = label_df.index
        label_column = label_df.columns
        # set column and index from label
        pred_df = pd.DataFrame(_predict, columns=label_column, index=label_index)
        # calc diff (abs)
        diff_df = abs(label_df - pred_df)
        diff_df.to_csv(self.result_path + "diff_{0}.csv".format(name))
        # calc cols mean
        #plt.figure()
        diff_df_mean = diff_df.mean()
        diff_df_mean.to_csv(self.result_path + "col_mean_{0}.csv".format(name))
        #diff_df_mean.plot.bar()
        #plt.savefig(self.result_path + "fig_cols_mean_{0}.png".format(name))
        # calc rows mean
        #plt.figure()
        diffT_df_mean = diff_df.T.mean()
        diffT_df_mean.to_csv(self.result_path + "row_mean_{0}.csv".format(name))
        #diffT_df_mean.plot.bar()
        #plt.savefig(self.result_path + "fig_rows_mean_{0}.png".format(name))

    def prediction_multi(self, _model, _X_test, _path):
        y_predict_all = _model.predict(_X_test)
        y_predict = y_predict_all[0]
        y1_predict = y_predict_all[1]
        y2_predict = y_predict_all[2]
        half_size = self.input_size.max() / 2
        y_predict[0::2] = (y_predict[0::2] * half_size + half_size) * self.size_ratio[1] + 0.5
        y_predict[1::2] = (y_predict[1::2] * half_size + half_size) * self.size_ratio[0] + 0.5
        y_predict = y_predict.astype(np.int)
        dirname = _path.split('\\')[len(_path.split('\\')) - 1]
        with open(self.result_path + "predict_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [landmark_dict[str(i)] for i in range(len(landmark_dict))]
            writer.writerow(header)
            for row in y_predict:
                writer.writerow(row)
        with open(self.result_path + "predict1_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [genderlabel_dict[str(i)] for i in range(len(genderlabel_dict))]
            writer.writerow(header)
            for row in y1_predict:
                writer.writerow(row)
        with open(self.result_path + "predict2_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            header = [agelabel_dict[str(i)] for i in range(len(agelabel_dict))]
            writer.writerow(header)
            for row in y2_predict:
                writer.writerow(row)

        return y_predict

    def prediction(self, _model, _X_test, _path):
        y_predict = _model.predict(_X_test)
        half_size = self.input_size.max() / 2
        y_predict[0::2] = (y_predict[0::2] * half_size + half_size) * self.size_ratio[1] + 0.5
        y_predict[1::2] = (y_predict[1::2] * half_size + half_size) * self.size_ratio[0] + 0.5
        y_predict = y_predict.astype(np.int)
        dirname = _path.split('\\')[len(_path.split('\\')) - 1]
        with open(self.result_path + "predict_{0}.csv".format(dirname), 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            lm_label = [landmark_dict[str(i)] for i in range(len(landmark_dict))]
            writer.writerow(lm_label)
            for row in y_predict:
                writer.writerow(row)

        return y_predict

    def calcImageRatio(self):
        self.size_ratio = np.array([self.img_size[0] / self.input_size[0], self.img_size[1] / self.input_size[1]])

    def model_evaluation(self, _model, _X_test, _y_test, _path):
        score = _model.evaluate(_X_test, _y_test, batch_size=self.batch_size, verbose=1)
        dirname = _path.split('\\')[len(_path.split('\\'))-1]
        with open(self.save_score, 'a', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            if self.flag_of_start:
                self.firstWrite = False
                writer.writerows([["",_model.metrics_names[0], _model.metrics_names[1]]])
            writer.writerow(list([dirname, score[0], score[1]]))

    def run(self):
        ## Read Data
        print("Read File ...")
        self.calcImageRatio()
        X_test, y_test = self.gd.load_data(self.val_image_path, self.val_label_path, self.save_labelset, self.img_size, self.input_size)
        y1_test, y2_test = self.gd.load_label(self.result_path)

        models_dir = glob.glob(self.model_path)
        for model_dir in models_dir:
            model_dirname = os.path.dirname(model_dir)

            ## Load Model
            print("Load Model : {0}".format(model_dirname))
            json_name = '{0}/{1}'.format(model_dirname, self.load_architecture)
            weights_name = '{0}/{1}'.format(model_dirname, self.load_weights)
            if not os.path.exists(weights_name):
                weights_name = '{0}/{1}'.format(model_dirname, self.load_weights_checkpoint)
            model = model_from_json(open(json_name).read())
            model.load_weights(weights_name)
            model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

            ## Predict TestData
            print("Predict TestData ...")
            if not self.isMultiTask:
                self.model_evaluation(model, X_test, y_test, model_dirname)
                y_pre = self.prediction(model, X_test, model_dirname)
            else:
                self.model_evaluation(model, X_test, [y_test, y1_test, y2_test], model_dirname)
                y_pre = self.prediction_multi(model, X_test, model_dirname)

            ## Evaluate Model
            print("Evaluate Model ...")
            self.calcDiffLabel(self.save_labelset, y_pre, model_dirname)

    def setParameter(self, _conf_file):
        self.config.read(_conf_file)

        ## Image Data
        conf_io = self.config['IO']
        self.img_size = np.array([conf_io.getint('image_height'),conf_io.getint('image_width')])
        self.input_size = np.array([conf_io.getint('input_height'),conf_io.getint('input_width')])
        self.input_size = self.input_size.astype(np.int)

        ## Dataset
        conf_d = self.config['Dataset']
        testName = conf_d['base_test']
        testOther = conf_d['base_test_plus']
        self.val_image_path = '{0}//{1}{2}//*.jpg'.format(conf_d['image_path'], testName, testOther)
        self.val_label_path = '{0}//{1}{2}//*.txt'.format(conf_d['label_path'], testName, testOther)
        comment=conf_d['comment']
        targetPath = conf_d['target_path']

        ## Result
        now = datetime.datetime.now()
        time = "{0:02d}{1:02d}{2:02d}{3:02d}".format(now.month, now.day, now.hour, now.minute)
        self.result_path = './result/{0}/{1}_{2}_{3}/'.format(targetPath, testName, comment, time)
        os.makedirs(self.result_path, exist_ok=True)

        ## Parameter
        conf_p = self.config['Parameter']
        self.batch_size = conf_p.getint('batch_size') #128

        ## Malti-Task
        conf_m = self.config['MaltiTask']
        self.isMultiTask = conf_m.getboolean('multi_task_on')

        ## Filename
        conf_s = self.config['File']
        self.load_architecture = conf_s['architecture'] #'architecture.json'
        self.load_weights = conf_s['weights'] #'weights.h5'
        self.load_weights_checkpoint = conf_s['weights_checkpoint'] #'weights_checkpoint.h5'
        self.save_labelset = self.result_path + conf_s['labelset'] #'label.csv'
        self.save_score = self.result_path + conf_s['score'] #'score.csv'
        self.model_path = conf_s['model_path']

landmark_dict = {'0':'left_eye_outer_corner_x',
                 '1':'left_eye_outer_corner_y',
                 '2':'left_eye_inner_corner_x',
                 '3':'left_eye_inner_corner_y',
                 '4':'right_eye_inner_corner_x',
                 '5':'right_eye_inner_corner_y',
                 '6': 'right_eye_outer_corner_x',
                 '7': 'right_eye_outer_corner_y',
                 '8': 'left_nose_top_x',
                 '9': 'left_nose_top_y',
                 '10': 'left_nose_bottom_x',
                 '11': 'left_nose_bottom_y',
                 '12': 'right_nose_top_x',
                 '13': 'right_nose_top_y',
                 '14': 'right_nose_bottom_x',
                 '15': 'right_nose_bottom_y',
                 '16': 'nose_root_x',
                 '17': 'nose_root_y',
                 '18': 'mouth_center_top_lip_x',
                 '19': 'mouth_center_top_lip_y',
                 '20': 'mouth_left_corner_x',
                 '21': 'mouth_left_corner_y',
                 '22': 'mouth_center_bottom_lip_x',
                 '23': 'mouth_center_bottom_lip_y',
                 '24': 'mouth_right_corner_x',
                 '25': 'mouth_right_corner_y',
                 '26': 'mouth_center_lip_x',
                 '27': 'mouth_center_lip_y'}

genderlabel_dict = {'0':'man',
                    '1':'woman'}

agelabel_dict = {'0':'20',
                 '1':'30',
                 '2':'40',
                 '3':'50',
                 '4':'60'}

if __name__ == "__main__":
    ev = Evaluate()
    conf = ["./param//evaluate.ini"]
    ev.setParameter(conf)
    ev.run()
