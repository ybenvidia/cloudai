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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .base import Installable, InstallStatusResult

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


@dataclass
class DockerImage(Installable):
    """Docker image object."""

    url: str
    _installed_path: Optional[Union[str, Path]] = field(default=None, repr=False)

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, DockerImage) and other.url == self.url

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash(self.url)

    def __str__(self) -> str:
        """Return the string representation of the docker image."""
        return f"DockerImage(url={self.url})"

    @property
    def cache_filename(self) -> str:
        """Return the cache filename for the docker image."""
        tag, wo_prefix = "notag", self.url
        is_local = wo_prefix.startswith("/") or wo_prefix.startswith(".")
        if "://" in wo_prefix:
            wo_prefix = self.url.split("://", maxsplit=1)[1]
        if ":" in wo_prefix:
            tag = wo_prefix.rsplit(":", maxsplit=1)[1]
        wo_tag = wo_prefix.rsplit(":", maxsplit=1)[0]
        if is_local:
            img_name = wo_tag.rsplit("/", maxsplit=1)[1]
        else:
            parts = wo_tag.split("/")
            img_name = "_".join(parts[:-1]) + "__" + parts[-1]

        # Replace # with _ in img_name to avoid filesystem issues
        img_name = img_name.replace("#", "_")

        path = f"{img_name}__{tag}.sqsh"
        return path.replace("/", "_").replace("#", "_").strip("_")

    @property
    def installed_path(self) -> Union[str, Path]:
        """Return the cached path or URL of the docker image."""
        if self._installed_path:
            return self._installed_path.absolute() if isinstance(self._installed_path, Path) else self._installed_path
        local_image_path = Path(self.url)
        if local_image_path.is_absolute() or self.url.startswith("."):
            return local_image_path.absolute()
        return self.url

    @installed_path.setter
    def installed_path(self, value: Union[str, Path]) -> None:
        self._installed_path = value

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        return InstallStatusResult(True, f"Docker image {self} does not require installation.")

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        return InstallStatusResult(True, f"Docker image {self} does not require uninstallation.")

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        return InstallStatusResult(True, f"Docker image {self} is available through the container runtime.")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        return InstallStatusResult(True, f"Docker image {self} marked as installed.")
