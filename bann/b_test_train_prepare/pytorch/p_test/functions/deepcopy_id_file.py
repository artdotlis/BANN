# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy

from pan.public.constants.net_tree_id_constants import ANNTreeIdType


def deepcopy_id_file(id_file: ANNTreeIdType, mod_str: str, /) -> ANNTreeIdType:
    copy_id = deepcopy(id_file)
    if mod_str:
        copy_id.add_modifier(mod_str)
    return copy_id
