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

from __future__ import annotations

import logging

from cloudai.core import BaseInstaller, InstallStatusResult
from cloudai.util.lazy_imports import lazy


class KubernetesInstaller(BaseInstaller):
    """Installer for Kubernetes systems."""

    def _check_prerequisites(self) -> InstallStatusResult:
        """
        Check for the presence of required binaries and Kubernetes configurations.

        This ensures the system environment is properly set up before proceeding with the installation
        or uninstallation processes.

        Returns
            InstallStatusResult: Result containing the status of the prerequisite check and any error message.
        """
        # Check base prerequisites using the parent class method
        base_prerequisites_result = super()._check_prerequisites()
        if not base_prerequisites_result.success:
            logging.error(f"Prerequisite check failed in base installer: {base_prerequisites_result.message}")
            return InstallStatusResult(False, base_prerequisites_result.message)

        # Load Kubernetes configuration
        try:
            lazy.k8s.config.load_kube_config()
        except Exception as e:
            message = (
                f"Installation failed during prerequisite checking stage because Kubernetes configuration could not "
                f"be loaded. Please ensure that your Kubernetes configuration is properly set up. Original error: {e!r}"
            )
            logging.error(message)
            return InstallStatusResult(False, message)

        logging.info("All prerequisites are met. Proceeding with installation.")
        return InstallStatusResult(True)
