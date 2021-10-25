# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict

from torch import nn


def re_copy_model(model_src: Dict, model_des: nn.Module, /) -> None:
    model_des.load_state_dict(model_src)
    model_des.eval()
