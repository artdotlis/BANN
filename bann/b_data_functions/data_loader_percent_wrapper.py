# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import atexit
import io
import os
import re
from typing import Pattern, TextIO, Final, final
from multiprocessing import current_process

from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_PERCENT_GRABBER: Final[Pattern[str]] = re.compile(r'(\d+\.?\d*)%')


@final
class RedirectWriteToLoggerPercent(io.TextIOWrapper):

    def __init__(self, sync_out: SyncStdoutInterface, old_error: TextIO, /) -> None:
        dev_null = open(os.devnull, 'wb')
        atexit.register(dev_null.close)
        super().__init__(dev_null)
        self.__sync_out = sync_out
        self.__stderr_out = old_error
        self.__previous_msg = ""

    def write(self, s: str) -> int:
        msg = s
        found = _PERCENT_GRABBER.search(msg)
        if found is not None and float(found.group(1)) >= 0:
            if float(found.group(1)) % 10 == 0 and self.__previous_msg != f"{found.group(1)}%":
                self.__previous_msg = f"{found.group(1)}%"
                logger_print_to_console(
                    self.__sync_out,
                    f"{current_process().name}"
                    + f" ({current_process().pid}): {self.__previous_msg}"
                )
        else:
            self.__stderr_out.write(f"The message\n{msg}\ncould not been redirected\n")
            self.__stderr_out.flush()
        return len(msg)
