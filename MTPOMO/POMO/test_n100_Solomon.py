import os
import sys
import torch
import logging

# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from utils.utils import create_logger, copy_all_src
from VRPTester import VRPTester as Tester

# Parameters
env_params = {
    'problem_type': "VRPTW", # test problem type
    'problem_size': 100,
    'pomo_size': 100,
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

list = [
    'c101', 'c102', 'c103', 'c104', 'c105', 'c106', 'c107', 'c108', 'c109',
    'c201', 'c202', 'c203', 'c204', 'c205', 'c206', 'c207', 'c208',
    'r101', 'r102', 'r103', 'r104', 'r105', 'r106', 'r107', 'r108', 'r109', 'r110', 'r111', 'r112',
    'r201', 'r202', 'r203', 'r204', 'r205', 'r206', 'r207', 'r208', 'r209', 'r210', 'r211',
    'rc101', 'rc102', 'rc103', 'rc104', 'rc105', 'rc106', 'rc107', 'rc108',
    'rc201', 'rc202', 'rc203', 'rc204', 'rc205', 'rc206', 'rc207', 'rc208'
]


# list = ['c101', 'c102', 'c103', 'c104', 'c105', 'c106', 'c107', 'c108', 'c109']
list = ['c201', 'c202', 'c203', 'c204', 'c205', 'c206', 'c207', 'c208']

# list = ['c101']

# list = ['gen0', 'gen1', 'gen2', 'gen3', 'gen4', 'gen5', 'gen6', 'gen7', 'gen8', 'gen9']

index = 0

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../../Trained_models/100',  # directory path of pre-trained model and log files saved.
        'epoch': 10000,  # epoch version of pre-trained model to load.
    },
    'test_episodes': 1000,
    'test_batch_size': 500,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 500,
    'test_data_load': {
        'enable': True,
        'filename': ''  # Placeholder, will be set in main()
    },
}

if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test_' + env_params['problem_type'] + '_n' + str(env_params['problem_size']) + '_with_instNorm',
        'filename': 'run_log'
    }
}

def main():
    global index  # Declare index as global
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    copy_all_src(tester.result_folder)

    list_all = []

    route_all = []

    for i in range(len(list)):
        tester_params['test_data_load']['filename'] = f'/content/MTNCO/Test_instances/Solomon100/data_VRPTW_{list[index]}.pt'
        result, route = tester.run()
        list_all.append(result)
        route_all.append(route)
        index += 1

        # print(route[0][0])

    print("All results:", list_all)

    print("All routes:", route_all)

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    main()
