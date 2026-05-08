# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


class Installable(ABC):
    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...

    def _unsupported_result(self, operation: str) -> "InstallStatusResult":
        return InstallStatusResult(
            False,
            f"Unsupported installable operation '{operation}' for item type: {self.__class__.__name__}",
        )

    def install(self, installer: "BaseInstaller") -> "InstallStatusResult":
        return self._unsupported_result("install")

    def uninstall(self, installer: "BaseInstaller") -> "InstallStatusResult":
        return self._unsupported_result("uninstall")

    def is_installed(self, installer: "BaseInstaller") -> "InstallStatusResult":
        return self._unsupported_result("is_installed")

    def mark_as_installed(self, installer: "BaseInstaller") -> "InstallStatusResult":
        return self._unsupported_result("mark_as_installed")


class InstallStatusResult:
    """
    Class representing the result of an installation, uninstallation, or status check.

    Attributes
        success (bool): Indicates whether the operation was successful.
        message (str): A message providing additional information about the result.
        details (Optional[Dict[str, str]]): A dictionary containing details about the result for each test template.
    """

    def __init__(self, success: bool, message: str = "", details: dict[Installable, Self] | None = None):
        self.success = success
        self.message = message
        self.details = details if details else {}

    def __bool__(self):
        return self.success

    def __str__(self):
        details_str = "\n".join(f"  - {key}: {value}" for key, value in self.details.items())
        return f"{self.message}\n{details_str}" if self.details else self.message
