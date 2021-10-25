# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Any


@dataclass(init=False)
class LegoContInit:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()
