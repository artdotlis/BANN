# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from pan.public.errors.custom_errors import KnownErrorPubPan


class KnownErrorBannTT(KnownErrorPubPan):
    pass


@final
class KnownTrainerError(KnownErrorBannTT):
    pass


@final
class KnownTesterError(KnownErrorBannTT):
    pass


@final
class KnownSplitterError(KnownErrorBannTT):
    pass


@final
class KnownPrepareError(KnownErrorBannTT):
    pass
