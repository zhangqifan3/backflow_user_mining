#/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zhangqifan
# @Date  : 2019/8/29
"""Wide and Deep Model Prediction
Not support for custom classifier, cause use different variable name scope, key not found in checkpoint"""

#import argparse
import os
import sys
import time
import pandas as pd

import tensorflow as tf

from lib.read_conf import Config
from lib.tf_dataset import input_fn
from lib.build_estimator import build_estimator




def main():
    CONFIG = Config()
    model_conf = CONFIG.read_model_conf()['model_conf']
    model = build_estimator()
    predictions = model.predict(input_fn=lambda: input_fn('/home/leadtek/zhangqifan/reflux_user_pro/data/pred_data/all_data.csv','pred'),
                                predict_keys=None,
                                hooks=None,
                                checkpoint_path=None)  # defaults None to use latest_checkpoint
    res = []
    for pred_dict in predictions:  # dict{probabilities, classes, class_ids}
        opt = []
        class_id = pred_dict['class_ids'][0]
        opt.append(class_id)
        probability = pred_dict['probabilities']
        opt.append(probability[1])
        res.append(opt)
        # print('class_id:',class_id,'probability:',probability)
    res_df = pd.DataFrame(res, columns=['class_id','probability'])
    x = res_df[res_df['class_id'].isin([1])]
    sample = pd.read_csv("/home/leadtek/zhangqifan/reflux_user_pro/data/opt_all_data.csv",sep=' ')
    res_sample = pd.concat([sample,res_df],axis=1)
    res_sample.to_csv(r"/home/leadtek/zhangqifan/reflux_user_pro/res.csv", header=True, index=False,
                                    sep=' ')

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
