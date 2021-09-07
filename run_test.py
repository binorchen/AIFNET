'''
    This file is used to run all the testing sets in one command. 
    Just run: 'python run_test.py --ts', where --ts is the datasets directories
    
    However, one can always run 'test.py' directly.
    e.g.  'python test.py -d LFDOF -r 688 -c 1008 -p ./test_set/LFDOF/input -gtp ./test_set/LFDOF/ground_truth -gt 1'

    Author: Bin Chen, Lingyan Ruan
    Email: binorchen@gmail.com
    Date: Sep 7, 2021
'''

import os, argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-ts', '--test_sets', type = str, default = './test_set', help = 'the path of all testing sets')
args = parser.parse_args()

test_sets = os.listdir(args.test_sets)

for ts in test_sets:
    if ts == 'LFDOF':
        input_path = os.path.join(args.test_sets, ts, 'input')
        gt_path = os.path.join(args.test_sets, ts, 'ground_truth')
        row = 688
        col = 1008
        cmd = 'python test.py' + ' -d ' + ts + ' -r ' + str(row) + ' -c ' + str(col) + ' -p ' + input_path + ' -gtp ' + gt_path + ' -gt ' + '1'
        print('Running Command: ', cmd)
        os.system(cmd)
    elif ts == 'DPD':
        input_path = os.path.join(args.test_sets, ts)
        gt_path = 'None'
        row = 560
        col = 832
        cmd = 'python test.py' + ' -d ' + ts + ' -r ' + str(row) + ' -c ' + str(col) + ' -p ' + input_path + ' -gtp ' + gt_path + ' -gt ' + '0'
        print('Running Command: ', cmd)
        os.system(cmd)
    elif ts == 'RTF':
        input_path = os.path.join(args.test_sets, ts)
        gt_path = 'None'
        row = 352
        col = 336
        cmd = 'python test.py' + ' -d ' + ts + ' -r ' + str(row) + ' -c ' + str(col) + ' -p ' + input_path + ' -gtp ' + gt_path + ' -gt ' + '0'
        print('Running Command: ', cmd)
        os.system(cmd)
    else:
        input_path = os.path.join(args.test_sets, ts)
        img_names = list(os.path.join(input_path, name) for name in os.listdir(input_path))
        gt_path = 'None'
        for img_name in img_names:
            img = scipy.misc.imread(img_name, mode='RGB')
            row = img.shape[0]
            col = img.shape[1]
            cmd = 'python test.py' + ' -d ' + ts + ' -r ' + str(row) + ' -c ' + str(col) + ' -p ' + img_name + ' -gtp ' + gt_path + ' -gt ' + '0'
            print('Running Command: ', cmd)
            os.system(cmd)
