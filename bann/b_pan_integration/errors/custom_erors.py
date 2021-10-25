# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from pan.public.errors.custom_errors import KnownErrorPubPan


class KnownErrorBannPan(KnownErrorPubPan):
    pass


@final
class NetInterfaceError(KnownErrorBannPan):
    pass


@final
class NetConnectionError(KnownErrorBannPan):
    pass


@final
class NetPlotterError(KnownErrorBannPan):
    pass
