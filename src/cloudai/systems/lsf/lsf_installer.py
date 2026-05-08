# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

from cloudai.core import (
    BaseInstaller,
    DockerImage,
    Installable,
    InstallStatusResult,
)
from cloudai.systems.slurm import SlurmInstaller, SlurmSystem

from .lsf_system import LSFSystem


class LSFInstaller(BaseInstaller):
    """Installer for LSF systems."""

    PREREQUISITES = ("bsub", "bjobs", "bhosts", "lsid", "lsload")

    def __init__(self, system: LSFSystem):
        super().__init__(system)
        self.system = system

    @property
    def slurm_installer(self):
        if not hasattr(self, "_slurm_installer"):
            if not isinstance(self.system, SlurmSystem):
                raise TypeError("The system must be of type SlurmSystem to use SlurmInstaller.")
            self._slurm_installer = SlurmInstaller(self.system)
        return self._slurm_installer

    def _check_prerequisites(self) -> InstallStatusResult:
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            return InstallStatusResult(False, base_prerequisites_result.message)

        try:
            self._check_required_binaries()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

    def install_one(self, item: Installable) -> InstallStatusResult:
        logging.debug(f"Attempt to install {item}")

        if type(item) is DockerImage:
            logging.info(f"Skipping installation of Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")

        return super().install_one(item)

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        logging.debug(f"Attempt to uninstall {item!r}")
        return super().uninstall_one(item)

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if type(item) is DockerImage:
            logging.info(f"Skipping installation check for Docker image {item} in LSF system.")
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")

        return super().is_installed_one(item)

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        logging.debug(f"Marking {item!r} as installed.")
        if type(item) is DockerImage:
            return InstallStatusResult(True, "Docker image installation skipped for LSF system.")
        return super().mark_as_installed_one(item)
