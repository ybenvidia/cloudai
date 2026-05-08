# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import subprocess
from pathlib import Path

from cloudai.core import (
    BaseInstaller,
    DockerImage,
    HFModel,
    Installable,
    InstallStatusResult,
)
from cloudai.util.hf_model_manager import HFModelManager

from .docker_image_cache_manager import DockerImageCacheManager, DockerImageCacheResult
from .slurm_system import SlurmSystem


class SlurmInstaller(BaseInstaller):
    """Installer for Slurm systems."""

    PREREQUISITES = ("git", "sbatch", "sinfo", "squeue", "srun", "scancel", "sacct")
    REQUIRED_SRUN_OPTIONS = (
        "--mpi",
        "--gpus-per-node",
        "--ntasks-per-node",
        "--container-image",
        "--container-mounts",
    )

    def __init__(self, system: SlurmSystem):
        super().__init__(system)
        self.system = system
        self.docker_image_cache_manager = DockerImageCacheManager(system)
        self.hf_model_manager = HFModelManager(system.hf_home_path)

    def _check_prerequisites(self) -> InstallStatusResult:
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            return InstallStatusResult(False, base_prerequisites_result.message)

        try:
            self._check_required_binaries()
            self._check_srun_options()
            return InstallStatusResult(True)
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self) -> None:
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")

    def _check_srun_options(self) -> None:
        """
        Check for the presence of specific srun options.

        Calls `srun --help` and verifying the options. Raises an exception if any required options are missing.
        """
        try:
            result = subprocess.run(["srun", "--help"], text=True, capture_output=True, check=True)
            help_output = result.stdout
        except subprocess.CalledProcessError as e:
            raise EnvironmentError(f"Failed to execute 'srun --help': {e}") from e

        missing_options = [option for option in self.REQUIRED_SRUN_OPTIONS if option not in help_output]
        if missing_options:
            missing_options_str = ", ".join(missing_options)
            raise EnvironmentError(f"Required srun options missing: {missing_options_str}")

    def install_one(self, item: Installable) -> InstallStatusResult:
        logging.debug(f"Attempt to install {item}")
        if self.is_installable_type(item, DockerImage):
            res = self._install_docker_image(item)
            return InstallStatusResult(res.success, res.message)
        elif self.is_installable_type(item, HFModel):
            if not self._is_hf_home_accessible():
                item.installed_path = self.system.hf_home_path
                return InstallStatusResult(
                    True,
                    f"HF home path '{self.system.hf_home_path}' is not accessible locally, "
                    f"skipping download of {item.model_name}. "
                    "Ensure the model is available on compute nodes.",
                )
            return self.hf_model_manager.download_model(item)

        return super().install_one(item)

    def uninstall_one(self, item: Installable) -> InstallStatusResult:
        logging.debug(f"Attempt to uninstall {item!r}")
        if self.is_installable_type(item, DockerImage):
            res = self._uninstall_docker_image(item)
            return InstallStatusResult(res.success, res.message)
        elif self.is_installable_type(item, HFModel):
            return self.hf_model_manager.remove_model(item)

        return super().uninstall_one(item)

    def is_installed_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, DockerImage):
            res = self.docker_image_cache_manager.check_docker_image_exists(item.url, item.cache_filename)
            if res.success and res.docker_image_path:
                item.installed_path = res.docker_image_path
            return InstallStatusResult(res.success, res.message)
        elif self.is_installable_type(item, HFModel):
            if not self._is_hf_home_accessible():
                item.installed_path = self.system.hf_home_path
                return InstallStatusResult(True)
            return self.hf_model_manager.is_model_downloaded(item)

        return super().is_installed_one(item)

    def mark_as_installed_one(self, item: Installable) -> InstallStatusResult:
        if self.is_installable_type(item, DockerImage):
            if self.system.cache_docker_images_locally and not isinstance(item.installed_path, Path):
                item.installed_path = self.system.install_path / item.cache_filename
            return InstallStatusResult(True)
        elif self.is_installable_type(item, HFModel):
            item.installed_path = self.system.hf_home_path  # fake path is OK here as the whole HF home will be mounted
            return InstallStatusResult(True)

        return super().mark_as_installed_one(item)

    def _is_hf_home_accessible(self) -> bool:
        """Check if hf_home_path is accessible locally (parent directory exists and is writable)."""
        try:
            parent = self.system.hf_home_path.resolve().parent
            return parent.exists() and parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)

        except (OSError, RuntimeError):
            return False

    def _install_docker_image(self, item: DockerImage) -> DockerImageCacheResult:
        res = self.docker_image_cache_manager.ensure_docker_image(item.url, item.cache_filename)
        if res.success and res.docker_image_path:
            item.installed_path = res.docker_image_path
        return res

    def _uninstall_docker_image(self, item: DockerImage) -> DockerImageCacheResult:
        res = self.docker_image_cache_manager.uninstall_cached_image(item.cache_filename)
        if res.success:
            item.installed_path = item.url
        return res
