import argparse
import random 
import numpy as np
import os
import gtimer as gt
from collections import OrderedDict
import torch

from rlkit.core import logger, eval_util

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times

def log_stats(epoch, policy, paths):
    logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

    """
    Trainer
    """
    logger.record_dict(policy.get_diagnostics(), prefix='trainer/')

    """
    Evaluation
    """
    logger.record_dict(
        eval_util.get_generic_path_information(paths),
        prefix="evaluation/",
    )

    """
    Misc
    """
    gt.stamp('logging')
    logger.record_dict(get_epoch_timings())
    logger.record_tabular('Epoch', epoch)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

def enable_gpus(gpu_str):
#     if (gpu_str is not ""):
    if (gpu_str != ""): 
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return