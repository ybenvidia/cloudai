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
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, File, GitRepo, InstallStatusResult, PythonExecutable


@pytest.fixture
def git() -> GitRepo:
    return GitRepo(url="./git_url", commit="commit_hash")


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


class TestFile:
    @pytest.fixture
    def installer(self, slurm_system):
        installer = BaseInstaller(slurm_system)
        installer.system.install_path.mkdir(parents=True)
        return installer

    @pytest.fixture
    def f(self, tmp_path: Path) -> File:
        f = tmp_path / "file"
        f.write_text("content")
        return File(f)

    def test_no_dst(self, installer: BaseInstaller, f: File):
        res = f.install(installer)
        assert res.success
        assert f.installed_path == installer.system.install_path / f.src.name
        assert f.installed_path.exists()
        assert f.installed_path.read_bytes() == f.src.read_bytes()

    def test_file_exists_but_overriden(self, installer: BaseInstaller, f: File):
        f.installed_path = installer.system.install_path / f.src.name
        f.installed_path.touch()
        res = f.install(installer)
        assert res.success
        assert f.src.read_bytes() == f.installed_path.read_bytes()

    def test_is_installed_checks_content(self, installer: BaseInstaller, f: File):
        f.installed_path = installer.system.install_path / f.src.name
        f.installed_path.touch()
        f.src.write_text("new content")

        res = f.is_installed(installer)
        assert not res.success


class TestPythonExecutable:
    @pytest.fixture
    def installer(self, slurm_system):
        installer = BaseInstaller(slurm_system)
        installer.system.install_path.mkdir(parents=True)
        installer._check_low_thread_environment = lambda threshold=None: False
        return installer

    @pytest.fixture
    def setup_repo(self, installer: BaseInstaller, git: GitRepo):
        repo_dir = installer.system.install_path / git.repo_name
        subdir = repo_dir / "subdir"

        repo_dir.mkdir(parents=True, exist_ok=True)
        subdir.mkdir(parents=True, exist_ok=True)

        pyproject_file = subdir / "pyproject.toml"
        requirements_file = subdir / "requirements.txt"

        pyproject_file.touch()
        requirements_file.touch()

        return repo_dir, subdir, pyproject_file, requirements_file

    def test_venv_created(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        with (
            patch.object(PythonExecutable, "_install_dependencies", return_value=InstallStatusResult(True)),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = py._create_venv(installer)
        assert res.success
        mock_run.assert_called_once_with(["python", "-m", "venv", str(venv_path)], capture_output=True, text=True)

    @pytest.mark.parametrize("failure_on_venv_creation,reqs_install_failure", [(True, False), (False, True)])
    def test_error_creating_venv(
        self, installer: BaseInstaller, git: GitRepo, failure_on_venv_creation: bool, reqs_install_failure: bool
    ):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name

        def mock_run(*args, **kwargs):
            venv_path.mkdir()
            if failure_on_venv_creation and "venv" in args[0]:
                return CompletedProcess(args=args, returncode=1, stderr="err")
            return CompletedProcess(args=args, returncode=0)

        dependencies_result = InstallStatusResult(False, "err") if reqs_install_failure else InstallStatusResult(True)
        with (
            patch.object(PythonExecutable, "_install_dependencies", return_value=dependencies_result),
            patch("subprocess.run", side_effect=mock_run),
        ):
            res = py._create_venv(installer)
        assert not res.success
        if failure_on_venv_creation:
            assert res.message == "Failed to create venv:\nSTDOUT:\nNone\nSTDERR:\nerr"
        else:
            assert res.message == "err"
        assert not venv_path.exists(), "venv folder wasn't removed after unsuccessful installation"

    def test_venv_already_exists(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        venv_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = py._create_venv(installer)
        assert mock_run.call_count == 0
        assert res.success
        assert res.message == f"Virtual environment already exists at {venv_path}."

    def test_requirements_no_file(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        venv_path = installer.system.install_path / py.venv_name
        venv_path.mkdir()
        res = py._install_requirements(venv_path, installer.system.install_path / "requirements.txt")
        assert not res.success
        assert (
            res.message
            == f"Requirements file is invalid or does not exist: {installer.system.install_path / 'requirements.txt'}"
        )

    def test_requirements_installed(self, installer: BaseInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        venv_path = installer.system.install_path / "venv"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
                venv_path, requirements_file
            )
        assert res.success
        mock_run.assert_called_once_with(
            [str(venv_path / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
        )

    def test_requirements_not_installed(self, installer: BaseInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = PythonExecutable(GitRepo(url="./git_url", commit="commit_hash"))._install_requirements(
                installer.system.install_path, requirements_file
            )
        assert not res.success
        assert res.message == "Failed to install dependencies from requirements.txt: err"

    def test_all_good_flow(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        py.git_repo.installed_path = installer.system.install_path / py.git_repo.repo_name

        repo_dir = py.git_repo.installed_path
        repo_dir.mkdir(parents=True, exist_ok=True)
        pyproject_file = repo_dir / "pyproject.toml"
        pyproject_file.write_text("[tool.poetry]\nname = 'dummy_project'")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0, stdout=f"{git.commit}\n", stderr="")
            res = py.install(installer)

        assert res.success
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert py.venv_path == installer.system.install_path / py.venv_name

    def test_is_installed_no_repo(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        res = py.is_installed(installer)
        assert not res.success
        assert res.message == f"Git repository {py.git_repo.url} not cloned"
        assert not (installer.system.install_path / py.git_repo.repo_name).exists()
        assert not py.git_repo.installed_path
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_is_installed_no_venv(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.git_repo.repo_name).mkdir()
        res = py.is_installed(installer)
        assert not res.success
        assert res.message == f"Virtual environment not created for {py.git_repo.url}"
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert (installer.system.install_path / py.git_repo.repo_name).exists()
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_is_installed_ok(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.git_repo.repo_name).mkdir()
        (installer.system.install_path / py.venv_name).mkdir()
        res = py.is_installed(installer)
        assert res.success
        assert res.message == "Python executable installed"
        assert py.git_repo.installed_path == installer.system.install_path / py.git_repo.repo_name
        assert (installer.system.install_path / py.git_repo.repo_name).exists()
        assert py.venv_path == installer.system.install_path / py.venv_name
        assert py.venv_path

    def test_uninstall_no_venv(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        py.venv_path = installer.system.install_path / py.venv_name
        res = py.uninstall(installer)
        assert res.success
        assert res.message == f"Virtual environment {py.venv_name} is not created."

    def test_uninstall_venv_removed_ok(self, installer: BaseInstaller, git: GitRepo):
        py = PythonExecutable(git)
        (installer.system.install_path / py.venv_name).mkdir()
        (installer.system.install_path / py.venv_name / "file").touch()
        py.venv_path = installer.system.install_path / py.venv_name
        res = py.uninstall(installer)
        assert res.success
        assert not (installer.system.install_path / py.venv_name).exists()
        assert not py.venv_path

    def test_install_python_executable_prefers_pyproject_toml(self, installer: BaseInstaller, git: GitRepo, setup_repo):
        repo_dir, subdir, _, _ = setup_repo

        py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=True)
        py.git_repo.installed_path = repo_dir

        with (
            patch.object(PythonExecutable, "_install_pyproject", return_value=InstallStatusResult(True)) as pyproject,
            patch.object(PythonExecutable, "_install_requirements", return_value=InstallStatusResult(True)) as reqs,
        ):
            res = py._install_dependencies(installer)

        assert res.success
        pyproject.assert_called_once_with(installer.system.install_path / py.venv_name, subdir)
        reqs.assert_not_called()

    def test_install_python_executable_prefers_requirements_txt(
        self, installer: BaseInstaller, git: GitRepo, setup_repo
    ):
        repo_dir, *_ = setup_repo

        py = PythonExecutable(git, project_subpath=Path("subdir"), dependencies_from_pyproject=False)
        py.git_repo.installed_path = repo_dir

        with (
            patch.object(PythonExecutable, "_install_requirements", return_value=InstallStatusResult(True)) as reqs,
            patch.object(PythonExecutable, "_install_pyproject", return_value=InstallStatusResult(True)) as pyproject,
        ):
            res = py._install_dependencies(installer)

        assert res.success
        pyproject.assert_not_called()
        reqs.assert_called_once()
