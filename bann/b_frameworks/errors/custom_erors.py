# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from pan.public.errors.custom_errors import KnownErrorPubPan


class KnownErrorBannFramework(KnownErrorPubPan):
    pass


@final
class KnownSimpleAnnError(KnownErrorBannFramework):
    pass


@final
class KnownLibError(KnownErrorBannFramework):
    pass


class KnownExNetError(KnownErrorBannFramework):
    pass


@final
class KnownSPPError(KnownErrorBannFramework):
    pass


@final
class KnownCriterionError(KnownErrorBannFramework):
    pass
