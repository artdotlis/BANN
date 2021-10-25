# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import argparse
from typing import Dict

from bann.b_container.states.general.general_lib import get_general_states_param_names
from bann.b_container.constants.gen_strings import GenStPaName
from bann.b_container.states.framework.framework_lib import get_framework_states_param_names
from bann.b_container.constants.fr_string import FrStPName


def _print_state(param_name: str, container: Dict[str, str], /) -> None:
    if container:
        print(f"The state {param_name} accepts the following arguments:")
        for name_value, type_value in container.items():
            print(f"\t{name_value}: {type_value}")
    else:
        print(f"The state {param_name} could not be found!")


def info_spec() -> None:
    arg_parser = argparse.ArgumentParser(
        prog='bann_info',
        description="""
        Specific information about bann, a library for building artificial neural networks.
        """
    )
    arg_parser.add_argument(
        action='store',
        type=str,
        help='the framework you want to be informed about',
        dest='framework',
        metavar='str',
    )
    arg_parser.add_argument(
        '-lr_sch',
        action='store',
        type=str,
        required=False,
        help='name of the lr-scheduler',
        dest='lr_sch',
        metavar='str'
    )
    arg_parser.add_argument(
        '-optim',
        action='store',
        type=str,
        required=False,
        help='name of the optim algorithm',
        dest='optim',
        metavar='str'
    )
    arg_parser.add_argument(
        '-crit',
        action='store',
        type=str,
        required=False,
        help='name of the loss-function',
        dest='criterion',
        metavar='str'
    )
    arg_parser.add_argument(
        '-tr',
        action='store',
        type=str,
        required=False,
        help='name of the trainer algorithm',
        dest='train',
        metavar='str'
    )
    arg_parser.add_argument(
        '-prep',
        action='store',
        type=str,
        required=False,
        help='name of the prepare algorithm',
        dest='prep',
        metavar='str'
    )
    arg_parser.add_argument(
        '-te',
        action='store',
        type=str,
        required=False,
        help='name of the tester algorithm',
        dest='test',
        metavar='str'
    )
    arg_parser.add_argument(
        '-net',
        action='store',
        type=str,
        required=False,
        help='name of the network',
        dest='net',
        metavar='str'
    )
    arg_parser.add_argument(
        '-hyper',
        action='store',
        type=str,
        required=False,
        help='name of the hyper-optimization algorithm',
        dest='hyper',
        metavar='str'
    )
    arg_parser.add_argument(
        '-init',
        action='store',
        type=str,
        required=False,
        help='name of the initializer',
        dest='init',
        metavar='str'
    )
    args = arg_parser.parse_args()
    if args.lr_sch is not None:
        _print_state(
            FrStPName.LRSCH.value,
            get_framework_states_param_names(args.framework, FrStPName.LRSCH.value, args.lr_sch)
        )
    if args.optim is not None:
        _print_state(
            FrStPName.OPTIM.value,
            get_framework_states_param_names(args.framework, FrStPName.OPTIM.value, args.optim)
        )
    if args.criterion is not None:
        _print_state(
            FrStPName.CRIT.value, get_framework_states_param_names(
                args.framework, FrStPName.CRIT.value, args.criterion
            )
        )
    if args.prep is not None:
        _print_state(
            FrStPName.PR.value,
            get_framework_states_param_names(args.framework, FrStPName.PR.value, args.prep)
        )
    if args.train is not None:
        _print_state(
            FrStPName.TR.value,
            get_framework_states_param_names(args.framework, FrStPName.TR.value, args.train)
        )
    if args.test is not None:
        _print_state(
            FrStPName.TE.value,
            get_framework_states_param_names(args.framework, FrStPName.TE.value, args.test)
        )

    if args.net is not None:
        _print_state(
            GenStPaName.NET.value,
            get_general_states_param_names(GenStPaName.NET.value, args.net)
        )

    if args.hyper is not None:
        _print_state(
            GenStPaName.HYPER.value,
            get_general_states_param_names(GenStPaName.HYPER.value, args.hyper)
        )

    if args.init is not None:
        _print_state(
            GenStPaName.INIT.value,
            get_general_states_param_names(GenStPaName.INIT.value, args.init)
        )


if __name__ == '__main__':
    info_spec()
