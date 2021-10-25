# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimInterfaceArgs


def check_dicts_consistency(args: HyperOptimInterfaceArgs, /) -> None:
    if len(set(len(elem) for elem in args.__dict__.values())) != 1:
        raise KnownHyperOptimError(
            "The given HyperOptimInterfaceArgs have different length in their attributes!"
        )
    for main_key, main_val in args.__dict__.items():
        for comp_key, comp_val in args.__dict__.items():
            for dict_key in main_val.keys():
                if dict_key not in comp_val:
                    raise KnownHyperOptimError(
                        f"Could not find the key {dict_key} from {main_key} in {comp_key}!"
                    )
