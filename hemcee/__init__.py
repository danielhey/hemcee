# -*- coding: utf-8 -*-

# Copyright 2018 Dan Foreman-Mackey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division, print_function

__version__ = "0.0.0"

try:
    __HEMCEE_SETUP__
except NameError:
    __HEMCEE_SETUP__ = False

if not __HEMCEE_SETUP__:
    __all__ = [
        "step_size", "metric", "autocorr",
        "NoUTurnSampler",
    ]

    from . import step_size, metric, autocorr
    from .nuts import NoUTurnSampler
