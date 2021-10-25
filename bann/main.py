# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import argparse
from pathlib import Path
from typing import Dict, Callable, Final

from bann.version import BANN_VERSION
from bann.b_pan_integration.config_container_lib import create_config_container
from pan.public.functions.ann_builder import ann_builder
from pan.public.functions.ann_config_dict import create_test_ann_config, create_train_ann_config, \
    create_check_test_ann_config, create_check_train_ann_config

_SWITCH: Final[Dict[str, Callable[[str, str, int, int, bool], Dict]]] = {
    'test':
        lambda conf_dir, out_dir, threads, dump_size, keep_dump: create_test_ann_config(
            conf_dir, out_dir
        ),
    'train': create_train_ann_config
}
_SWITCH_CHECK: Final[Dict[str, Callable[[str], Dict]]] = {
    'test': create_check_test_ann_config,
    'train': create_check_train_ann_config
}


def main() -> None:
    configs_dict = create_config_container()
    arg_parser = argparse.ArgumentParser(
        prog='bann',
        description="A library for building artificial neural networks."
    )
    arg_parser.add_argument(
        action='store',
        choices=['test', 'train'],
        type=str,
        help='indicates if you want to start a test- or a train-session',
        dest='test_train',
        metavar='test/train',
    )
    arg_parser.add_argument(
        '-l', '--log',
        action='store',
        type=str,
        required=True,
        help='the directory for the log files',
        dest='log_dir',
        metavar='str'
    )
    arg_parser.add_argument(
        '-c', '--conf',
        action='store',
        type=str,
        required=True,
        help='the directory of the configuration files',
        dest='conf_dir',
        metavar='str'
    )
    arg_parser.add_argument(
        '-o', '--out',
        action='store',
        type=str,
        required=True,
        help='the output directory',
        dest='out_dir',
        metavar='str'
    )
    arg_parser.add_argument(
        '-s', '--size',
        action='store',
        type=int,
        required=False,
        default=0,
        help="""
        the dump size for the train results,
        can be set to 0 for tests or if the results should not be dumped
        """,
        dest='dump_size',
        metavar='int'
    )
    arg_parser.add_argument(
        '-t', '--threads',
        action='store',
        type=int,
        required=False,
        default=1,
        help="""
        the number of processes used for working with graphs
        """,
        dest='threads',
        metavar='int'
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
        required=False,
        help="if set, only checks the given configuration files",
        dest='check'
    )
    arg_parser.add_argument(
        '--keep_dump',
        action='store_true',
        required=False,
        help="if set, keeps the dumped data, otherwise dumped data will be deleted",
        dest='keep_dump'
    )
    arg_parser.add_argument('--version', action='version', version=BANN_VERSION)
    args = arg_parser.parse_args()
    if args.check:
        conf_net = _SWITCH_CHECK.get(args.test_train, create_check_train_ann_config)(args.conf_dir)
    else:
        conf_net = _SWITCH.get(args.test_train, create_train_ann_config)(
            args.conf_dir, args.out_dir, args.threads, args.dump_size, args.keep_dump
        )
    ann_builder(configs_dict, Path(args.log_dir), conf_net)


if __name__ == '__main__':
    main()
