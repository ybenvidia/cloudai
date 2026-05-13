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

from pathlib import Path

import pytest

from cloudai.core import BaseInstaller, DockerImage


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://fake_url/img", "fake_url__img__notag.sqsh"),
        ("nvcr.io#nvidia/pytorch:24.02-py3", "nvcr.io_nvidia__pytorch__24.02-py3.sqsh"),
        ("/local/disk/file", "file__notag.sqsh"),
        ("/local/disk/file:tag", "file__tag.sqsh"),
        ("./local/disk/file:tag", "file__tag.sqsh"),
        ("ditdab.com#org/team/image:latest", "ditdab.com_org_team__image__latest.sqsh"),
        ("registry.invalid.com:5000#group/project", "registry.invalid.com__5000_group_project.sqsh"),
        ("registry.invalid:5050#group/project:latest", "registry.invalid:5050_group__project__latest.sqsh"),
        ("registry.invalid:5051#team/proj/image.v1", "registry.invalid__5051_team_proj_image.v1.sqsh"),
        ("nvcr.io/nvidia#nemo:24.07", "nvcr.io__nvidia_nemo__24.07.sqsh"),
    ],
)
def test_docker_cache_filename(url: str, expected: str):
    assert DockerImage(url).cache_filename == expected, f"Input: {url}"


def test_docker_image_installed_path():
    docker_image = DockerImage("fake_url/img")

    string_path = "fake_url/img"
    docker_image._installed_path = string_path
    assert docker_image.installed_path == "fake_url/img"

    path_obj = Path("/another/path")
    docker_image._installed_path = path_obj
    assert isinstance(docker_image.installed_path, Path)
    assert docker_image.installed_path == path_obj.absolute()


def test_default_operations_are_noop(slurm_system):
    installer = BaseInstaller(slurm_system)
    image = DockerImage("fake_url/img")

    results = [
        installer.install_one(image),
        installer.is_installed_one(image),
        installer.uninstall_one(image),
        installer.mark_as_installed_one(image),
    ]

    assert all(result.success for result in results)
    assert image.installed_path == image.url
