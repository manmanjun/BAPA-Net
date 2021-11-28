import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader, cityscapesLoader_16
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet


def get_loader(name, num = 19):
    """get_loader
    :param name:
    """
    if num == 19:
        return {
            "cityscapes": cityscapesLoader,
            "gta": GTA5DataSet,
            "syn": SynthiaDataSet
        }[name]
    else:
        return {
            "cityscapes": cityscapesLoader_16,
            "gta": GTA5DataSet,
            "syn": SynthiaDataSet
        }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '/opt/data/private/dataset/Cityscapes/data'
    if name == 'gta' or name == 'gtaUniform':
        return '/opt/data/private/dataset/GTA5'
    if name == 'syn':
        return '/opt/data/private/dataset/RAND_CITYSCAPES'
