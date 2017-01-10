import tensorflow as tf
import numpy as np

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-i", "--input_data", help="specify the input data (default=data/test.json)")
args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    from toefl_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

if args.input_data is not None:
    config.test_file = args.input_data

config.strong_supervision = False

config.train_mode = False

print 'Testing DMN '

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from toefl_plus import DMN_PLUS
        model = DMN_PLUS(config)

print '==> initializing variables'
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print '==> restoring weights'
    saver.restore(session, 'weights/TOEFL.weights')

    print '==> running DMN'
    pred_list = model.run_test_epoch(session, model.test)
    answer_file = open("answer.txt","w")
    for pred in pred_list:
        pred = np.reshape(pred, (-1,))
        answer_file.write(str(np.argmax(pred)))
        answer_file.write("\n")

