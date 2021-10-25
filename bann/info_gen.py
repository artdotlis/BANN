# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from bann.b_pan_integration.net_interface_lib import net_interface_config
from bann.b_container.states.framework.framework_lib import get_all_framework_names, \
    get_framework_states
from bann.b_container.states.general.general_lib import get_general_states


def _print_all_frameworks() -> None:
    fram_lib = get_all_framework_names()
    print(f"\nAll available frameworks:\n\t{' '.join(fram_lib)}")
    frame_states_lib = get_framework_states()
    for frame_name, frame_states in frame_states_lib.items():
        print(f"\nFramework {frame_name}:")
        for key_val, el_val in frame_states.__dict__.items():
            print(f"\t{key_val}: {','.join(el_val)}")
        print(
            f"\tmachine: "
            + f"{','.join(key_v for key_v in net_interface_config()[frame_name].net_dict.keys())}"
        )


def _print_all_gen_states_libs() -> None:
    gen_lib = get_general_states()
    for key_val, el_val in gen_lib.items():
        print(f"Gen {key_val}: {','.join(el_val)}")


def info_gen() -> None:
    _print_all_gen_states_libs()
    _print_all_frameworks()


if __name__ == '__main__':
    info_gen()
