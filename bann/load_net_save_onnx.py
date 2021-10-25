# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

from bann.b_frameworks.b_onnx.onnx_net_interface import BOnnxInterface
from bann.b_pan_integration.net_interface_lib import net_interface_config

from pan.public.functions.load_net_function import net_interface_load_net


def _load_save_net(path_arg: Tuple[Path, Path], class_name: str, framework: str,
                   t_sizes: Tuple[int, ...], prep_args: Tuple[int, ...], /) -> None:
    in_file, out_folder = path_arg
    interface_lib = net_interface_config()
    if framework not in interface_lib:
        print(f"Could not find framework: {framework}\n{','.join(interface_lib)}")
        sys.exit()
    class_lib = interface_lib[framework].net_dict
    if class_name not in class_lib:
        print(f"Could not find framework: {class_name}\n{','.join(class_lib)}")
        sys.exit()
    if not (out_folder.exists() and out_folder.is_dir()):
        out_folder.mkdir()
    with in_file.open('rb') as net_h:
        buffer_load = net_interface_load_net(class_lib[class_name].net_type, net_h.read())
        model_net = buffer_load.ann_container.get_net_com
        if not isinstance(model_net, BOnnxInterface):
            print("The model does not support onnx interface")
            sys.exit()
        model_net.save_onnx_output(out_folder, t_sizes, prep_args)


def load_net_save_onnx() -> None:
    arg_parser = argparse.ArgumentParser(
        prog='bann_lnso',
        description="A script for loading a network and saving its onnx representation"
    )
    arg_parser.add_argument(
        '-o', '--out',
        action='store',
        type=str,
        required=True,
        help='the output folder',
        dest='out_folder',
        metavar='str'
    )
    arg_parser.add_argument(
        '-i', '--input',
        action='store',
        type=str,
        required=True,
        help='the input net-file',
        dest='in_file',
        metavar='str'
    )
    arg_parser.add_argument(
        '-c', '--class',
        action='store',
        type=str,
        required=True,
        help='the class name of the saved network',
        dest='class_name',
        metavar='str'
    )
    arg_parser.add_argument(
        '-f', '--framework',
        action='store',
        type=str,
        required=True,
        help='the framework name of the saved network',
        dest='framework_name',
        metavar='str'
    )
    arg_parser.add_argument(
        '-s', '--sizes',
        nargs='+',
        action='store',
        type=int,
        required=False,
        default=[],
        help="the tensor sizes, which should be used in the dummy tensor-tuple",
        dest='sizes',
        metavar='int'
    )
    arg_parser.add_argument(
        '-p', '--prep_args',
        nargs='+',
        action='store',
        type=int,
        required=False,
        default=[],
        help="the preparation arguments, which should be used during before tracing",
        dest='prep_args',
        metavar='int'
    )
    args = arg_parser.parse_args()
    _load_save_net(
        (Path(args.in_file), Path(args.out_folder)), args.class_name, args.framework_name,
        tuple(args.sizes), tuple(args.prep_args)
    )


if __name__ == '__main__':
    load_net_save_onnx()
