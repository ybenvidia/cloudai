# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    HFModel,
    Installable,
    InstallStatusResult,
)
from cloudai.util.hf_model_manager import HFModelManager

from .runai_system import RunAISystem


class RunAIInstaller(BaseInstaller):
    """Installer for RunAI systems."""

    def __init__(self, system: RunAISystem):
        super().__init__(system)
        self.hf_model_manager = HFModelManager(system.hf_home_path)

    def _check_prerequisites(self) -> InstallStatusResult:
        logging.info("Checking prerequisites for RunAI installation.")
        return InstallStatusResult(True)

    def install_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, HFModel):
            return self.hf_model_manager.download_model(item)
        return super().install_one(item)

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, HFModel):
            return self.hf_model_manager.remove_model(item)
        return super().uninstall_one(item)

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, HFModel):
            return self.hf_model_manager.is_model_downloaded(item)
        return super().is_installed_one(item)

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, HFModel):
            item.installed_path = self.system.hf_home_path  # fake path is OK here as the whole HF home will be mounted
            return InstallStatusResult(True)
        return super().mark_as_installed_one(item)
