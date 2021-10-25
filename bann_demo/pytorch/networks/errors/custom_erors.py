# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from bann.b_frameworks.errors.custom_erors import KnownErrorBannFramework


@final
class KnownSimpleCheckError(KnownErrorBannFramework):
    pass


@final
class KnownSimpleDemoError(KnownErrorBannFramework):
    pass


@final
class KnownComplexDemoError(KnownErrorBannFramework):
    pass


@final
class KnownRBMError(KnownErrorBannFramework):
    pass


@final
class KnownAutoEncoderError(KnownErrorBannFramework):
    pass
