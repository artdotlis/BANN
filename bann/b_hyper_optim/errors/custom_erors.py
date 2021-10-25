# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from pan.public.errors.custom_errors import KnownErrorPubPan


class KnownErrorHyperImp(KnownErrorPubPan):
    pass


@final
class KnownHyperError(KnownErrorHyperImp):
    pass


@final
class KnownRandomError(KnownErrorHyperImp):
    pass


@final
class KnownPSOError(KnownErrorHyperImp):
    pass


@final
class KnownEGAError(KnownErrorHyperImp):
    pass


@final
class KnownSimplyTrainError(KnownErrorHyperImp):
    pass


@final
class KnownDPError(KnownErrorHyperImp):
    pass
