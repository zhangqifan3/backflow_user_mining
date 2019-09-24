# -*- coding: utf-8 -*-
# @Author: zhangqifan
# @Date  : 2019/6/18
# ==============================================================================

import os
from os.path import dirname, abspath
import sys
import configparser

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

BASE_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'conf')

SCHEMA_CONF_FILE = 'schema_conf.ini'
FEATURE_CONF_FILE = 'feature_conf.ini'
MODEL_CONF_FILE = 'model_conf.ini'
CROSS_FEAT_CONF_FILE = 'cross_feat_conf.ini'

class Config(object):
    """Config class
    读取配置文件
    """
    def __init__(self,
                 schema_conf_file=SCHEMA_CONF_FILE,
                 model_conf_file=MODEL_CONF_FILE,
                 feature_conf_file=FEATURE_CONF_FILE,
                 cross_feat_conf_file = CROSS_FEAT_CONF_FILE):
        self._schema_conf_file = os.path.join(BASE_DIR, schema_conf_file)
        self._model_conf_file = os.path.join(BASE_DIR, model_conf_file)
        self._feature_conf_file = os.path.join(BASE_DIR, feature_conf_file)
        self._cross_feat_conf_file = os.path.join(BASE_DIR, cross_feat_conf_file)

    def read_schema_conf(self):
        '''
        读取schema_conf内容
        :return:
            All_features：{'1'：'feature name', ...}
        '''
        conf = configparser.ConfigParser()
        conf.read(self._schema_conf_file)
        secs = conf.sections()
        All_features = conf.items(secs[0])
        All_features = dict(All_features)
        return All_features

    def read_feature_conf(self):
        '''
        读取feature_conf内容
        :return:
            feature_conf_dic：{'feature name'：{}，...}
            secs: ['feature name', ...]
        '''
        conf = configparser.ConfigParser()
        conf.read(self._feature_conf_file)
        secs = conf.sections()
        feature_conf_dic = {}
        for i in range(len(secs)):
            opt = conf.items(secs[i])
            opt = dict(opt)
            feature_conf_dic[secs[i]] = opt
        return feature_conf_dic, secs

    def read_model_conf(self):
        '''
        读取model_conf内容
        :return:
            model_conf_dic
        '''
        conf = configparser.ConfigParser()
        conf.read(self._model_conf_file)
        secs = conf.sections()
        model_conf_dic = {}
        for i in range(len(secs)):
            opt = conf.items(secs[i])
            opt = dict(opt)
            model_conf_dic[secs[i]] = opt
        return model_conf_dic

    def read_cross_feature_conf(self):
        conf = configparser.ConfigParser()
        conf.read(self._cross_feat_conf_file)
        secs = conf.sections()
        cross_feature_conf_dic = {}
        for i in range(len(secs)):
            opt = conf.items(secs[i])
            opt = dict(opt)
            cross_feature_conf_dic[secs[i]] = opt
        conf_list = []
        for features, conf in cross_feature_conf_dic.items():
            features = [f.strip() for f in features.split('&')]
            hash_bucket_size = int(conf["hash_bucket_size"])
            is_deep = conf["is_deep"] if conf["is_deep"] is not None else 1
            conf_list.append((features, hash_bucket_size, is_deep))
        return conf_list
