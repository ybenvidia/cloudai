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
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import pytest

from cloudai.core import GitRepo, Installable, InstallContext, InstallStatusResult


@pytest.fixture
def git() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


class CustomInstallable(Installable):
    def __init__(self):
        self.context = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CustomInstallable)

    def __hash__(self) -> int:
        return hash("CustomInstallable")

    def install(self, context: InstallContext) -> InstallStatusResult:
        self.context = context
        return InstallStatusResult(True, "custom installed")


def test_install_context_exposes_system_paths_and_capabilities(slurm_system):
    context = InstallContext(
        installer=Mock(),
        install_dir=slurm_system.install_path,
        hf_home_dir=slurm_system.hf_home_path,
    )

    assert context.install_dir == slurm_system.install_path
    assert context.hf_home_dir == slurm_system.hf_home_path
    assert context.installer is not None


def test_default_installable_operations_return_unsupported(slurm_system):
    item = CustomInstallable()
    context = InstallContext(
        installer=Mock(),
        install_dir=slurm_system.install_path,
        hf_home_dir=slurm_system.hf_home_path,
    )

    for operation in [item.uninstall, item.is_installed, item.mark_as_installed]:
        result = operation(context)
        assert not result.success
        assert "Unsupported installable operation" in result.message
        assert type(item).__name__ in result.message


def test_custom_installable_can_implement_install_api(slurm_system):
    item = CustomInstallable()
    context = InstallContext(
        installer=Mock(),
        install_dir=slurm_system.install_path,
        hf_home_dir=slurm_system.hf_home_path,
    )

    result = item.install(context)

    assert result.success
    assert result.message == "custom installed"
    assert item.context is context


class TestGitRepoSubmodules:
    @pytest.mark.parametrize("init_submodules", [True, False])
    def test_check_submodules_state_no_submodules(self, git: GitRepo, init_submodules: bool):
        git.init_submodules = init_submodules

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            result, message = git.check_submodules_state(Path("/repo"))

        assert result is True
        assert message == ""

    @pytest.mark.parametrize(
        ("init_submodules", "stdout", "expected_result", "expected_message"),
        [
            (True, " 0123456789abcdef path/to/submodule\n", True, ""),
            (True, "-0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
            (True, "+0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
            (True, "U0123456789abcdef path/to/submodule\n", False, "Cloned repo has not all submodules initialized."),
            (False, "-0123456789abcdef path/to/submodule\n", True, ""),
            (
                False,
                " 0123456789abcdef path/to/submodule\n",
                False,
                "Cloned repo has some submodules initialized but requires none to be.",
            ),
            (
                False,
                "+0123456789abcdef path/to/submodule\n",
                False,
                "Cloned repo has some submodules initialized but requires none to be.",
            ),
            (
                False,
                "U0123456789abcdef path/to/submodule\n",
                False,
                "Cloned repo has some submodules initialized but requires none to be.",
            ),
        ],
    )
    def test_check_submodules_state(
        self,
        git: GitRepo,
        init_submodules: bool,
        stdout: str,
        expected_result: bool,
        expected_message: str,
    ):
        git.init_submodules = init_submodules

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
            result, message = git.check_submodules_state(Path("/repo"))

        assert result is expected_result
        assert message == expected_message

    def test_check_submodules_state_status_failure(self, git: GitRepo):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
            result, message = git.check_submodules_state(Path("/repo"))

        assert result is False
        assert message == "Failed to get submodule status: err"

    @pytest.mark.parametrize(
        ("init_submodules", "stdout", "expected_command"),
        [
            (True, "-0123456789abcdef path/to/submodule\n", ["git", "submodule", "update", "--init", "--recursive"]),
            (False, " 0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
            (False, "+0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
            (False, "U0123456789abcdef path/to/submodule\n", ["git", "submodule", "deinit", "--all", "--force"]),
        ],
    )
    def test_ensure_submodules_state_reconciles(
        self, git: GitRepo, init_submodules: bool, stdout: str, expected_command: list[str]
    ):
        git.init_submodules = init_submodules

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
                CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            ]
            result, message = git.ensure_submodules_state(Path("/repo"))

        assert result is True
        assert message == ""
        assert mock_run.call_args_list[1].args[0] == expected_command

    @pytest.mark.parametrize("init_submodules", [True, False])
    def test_ensure_submodules_state_noop_when_matching(self, git: GitRepo, init_submodules: bool):
        git.init_submodules = init_submodules
        stdout = " 0123456789abcdef path/to/submodule\n" if init_submodules else "-0123456789abcdef path/to/submodule\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
            result, message = git.ensure_submodules_state(Path("/repo"))

        assert result is True
        assert message == ""
        assert mock_run.call_count == 1

    @pytest.mark.parametrize(
        ("init_submodules", "stdout", "expected_message"),
        [
            (True, "-0123456789abcdef path/to/submodule\n", "Failed to initialize submodules: err"),
            (False, " 0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
            (False, "+0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
            (False, "U0123456789abcdef path/to/submodule\n", "Failed to deinitialize submodules: err"),
        ],
    )
    def test_ensure_submodules_state_reconcile_failure(
        self,
        git: GitRepo,
        init_submodules: bool,
        stdout: str,
        expected_message: str,
    ):
        git.init_submodules = init_submodules

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=""),
                CompletedProcess(args=[], returncode=1, stdout="", stderr="err"),
            ]
            result, message = git.ensure_submodules_state(Path("/repo"))

        assert result is False
        assert message == expected_message

    def test_ensure_submodules_state_status_fails(self, git: GitRepo):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [CompletedProcess(args=[], returncode=1, stdout="bla", stderr="bla")]
            result, message = git.ensure_submodules_state(Path("/repo"))

        assert result is False
        assert "bla" in message
        assert mock_run.call_count == 1
