##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester


##########################################################################################
# parameters

size = 20

env_params = {
    'problem_type': 'VRPTW',
    'problem_size': size,
    'pomo_size': size,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result',  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 64,
    'test_batch_size': 64,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 8,
    'test_data_load': {
        'enable': True,
        'filename': ''
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_vrptw100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    global index
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(tester.result_folder)

    list_all = []

    route_all = []

    time_list = []  

    import time

    for i in range(1,4):
        start = time.time()
        tester_params['test_data_load']['filename'] = f'/content/MTNCO/Test_instances/mock_data/data_VRPTW_{list[index]}.pt'
        result, route = tester.run()
        list_all.append(result)
        route_all.append(route)

        run_time = time.time() - start
        time_list.append(run_time)


    print("All results:", list_all)
    print("All routes:", route_all)

    print("Average results:", sum(list_all)/len(list_all))
    print("Average time:", sum(time_list)/len(time_list))


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
