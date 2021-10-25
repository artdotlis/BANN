# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

from pan.public.errors.custom_errors import KnownErrorPubPan


class KnownErrorContainer(KnownErrorPubPan):
    pass


@final
class KnownCheckComError(KnownErrorContainer):
    pass


@final
class KnownExtendError(KnownErrorContainer):
    pass


@final
class KnownPrintPError(KnownErrorContainer):
    pass


@final
class KnownOptimStateError(KnownErrorContainer):
    pass


@final
class KnownInitStateError(KnownErrorContainer):
    pass


@final
class KnownTrainStateError(KnownErrorContainer):
    pass


@final
class KnownLRSchedulerStateError(KnownErrorContainer):
    pass


@final
class KnownNetStateError(KnownErrorContainer):
    pass


@final
class KnownFrameworkError(KnownErrorContainer):
    pass


@final
class KnownHyperOptimError(KnownErrorContainer):
    pass


@final
class KnownPrepareError(KnownErrorContainer):
    pass


@final
class KnownCriterionStateError(KnownErrorContainer):
    pass


@final
class KnownTestStateError(KnownErrorContainer):
    pass


@final
class KnownMinMaxError(KnownErrorContainer):
    pass


@final
class KnownTruthFunError(KnownErrorContainer):
    pass


@final
class KnownActivationFunctionError(KnownErrorContainer):
    pass
