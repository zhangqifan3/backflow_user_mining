# -*- coding: utf-8 -*-
# @Author: zhangqifan
# @Date  : 2019/6/18
# ==============================================================================
import tensorflow as tf
import os
import sys
import json

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)
from os.path import dirname, abspath
from lib.build_feature_columns import build_model_columns
from lib.read_conf import Config

MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')
print(MODEL_DIR)
CONF = Config()
#def _build_distribution():
#    """Build distribution configuration variable TF_CONFIG in tf.estimator API"""
#    distribution = 1
#    TF_CONFIG = {'is_distribution':1,'cluster':{'ps':['180.169.142.187:9001'],'chief':['180.169.142.187:9002'],'worker':['180.169.142.187:9003', '180.169.142.187:9004']},
#            'job_name':'woker','task_index':0}
#    if distribution == 1:
#        cluster_spec = {'ps':['180.169.142.187:9101'],'chief':['180.169.142.187:9102'],'worker':['180.169.142.187:9103', '180.169.142.187:9104']}
#        job_name = 'worker'
#        task_index = 1
#        os.environ['TF_CONFIG'] = json.dumps(
#            {'cluster': cluster_spec,
#            'task': {'type': job_name, 'index': task_index}})
#        run_config = tf.estimator.RunConfig()
#    return run_config

def build_estimator():

    """Build an estimator using official tf.estimator API.
        Args:
            model_dir: model save base directory
            model_type: one of {`wide`, `deep`, `wide_deep`}
        Returns:
            model instance of tf.estimator.Estimator class
      """
    wide_columns, deep_columns = build_model_columns()  # 确定模型的特征输入
    model_type = CONF.read_model_conf()['model_conf']['model_type']
 #   session_config = tf.ConfigProto(device_count={'GPU': 0,'GPU':1,'GPU':2})
 #   session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
 #   session_config.gpu_options.allow_growth = True
 #   session_config.gpu_options.allocator_type = 'BFC'
 #   run_config = tf.estimator.RunConfig().replace(session_config=session_config)
 #   run_config = _build_distribution()
    print(model_type)
    if model_type == 'wide_deep':
        return tf.estimator.DNNLinearCombinedClassifier(
                              model_dir=MODEL_DIR,
                              linear_feature_columns=wide_columns,
                              linear_optimizer=tf.train.FtrlOptimizer(
                                  learning_rate=0.001,
                                  l1_regularization_strength=0.1,
                                  l2_regularization_strength=1),
                              dnn_feature_columns=deep_columns,
                              dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                  learning_rate=0.001,
                                  l1_regularization_strength=0.01,
                                  l2_regularization_strength=0.01),
                              dnn_hidden_units=[100, 100, 100],
                              dnn_activation_fn=tf.nn.relu,
                              #    dnn_dropout= ,
                              n_classes=2,
                              #    weight_column=weight_column,
                              label_vocabulary=None,
                              input_layer_partitioner=None)
  #                            config=run_config)
    elif model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir=MODEL_DIR,
            feature_columns=wide_columns,
           # weight_column=weight_column,
            optimizer=tf.train.FtrlOptimizer(  # can not read from conf
                learning_rate=0.01,
                l1_regularization_strength=0.5,
                l2_regularization_strength=1),
            partitioner=None)
    else:
        return tf.estimator.DNNClassifier(
            model_dir=MODEL_DIR,
            feature_columns=deep_columns,
            hidden_units=[100,100,100],
            optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.005,
                l1_regularization_strength=0.01,
                l2_regularization_strength=0.01),  # {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}
            activation_fn=tf.nn.relu,  
           # dropout=,
           # weight_column=,
            input_layer_partitioner=None)


if __name__=="__main__":
    pass
    build_estimator()
