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
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

if TYPE_CHECKING:
    from .base_installer import BaseInstaller


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


class GitRepo(Installable, BaseModel):
    """Git repository object."""

    model_config = ConfigDict(extra="forbid")

    url: str
    commit: str
    init_submodules: bool = False
    installed_path: Optional[Path] = None
    mount_as: Optional[str] = None

    def __repr__(self) -> str:
        return f"GitRepo(url={self.url}, commit={self.commit})"

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, GitRepo) and other.url == self.url and other.commit == self.commit

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash((self.url, self.commit))

    @property
    def repo_name(self) -> str:
        repo_name = self.url.rsplit("/", maxsplit=1)[1].replace(".git", "")
        return f"{repo_name}__{self.commit}"

    @property
    def container_mount(self) -> str:
        return self.mount_as or f"/git/{self.repo_name}"

    def check_submodules_state(self, repo_path: Path) -> tuple[bool, str]:
        """Check if submodules state in the cloned repo matches self.init_submodules."""
        result = subprocess.run(
            ["git", "submodule", "status", "--recursive"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Failed to get submodule status: {result.stderr}"
        output = [line for line in result.stdout.splitlines() if line.strip()]

        has_submodules = bool(output)
        if not has_submodules:
            return True, ""

        status_prefixes = [line[0] for line in output]
        if self.init_submodules and not all(prefix == " " for prefix in status_prefixes):
            return False, "Cloned repo has not all submodules initialized."
        if not self.init_submodules and not all(prefix == "-" for prefix in status_prefixes):
            return False, "Cloned repo has some submodules initialized but requires none to be."

        return True, ""

    def ensure_submodules_state(self, repo_path: Path) -> tuple[bool, str]:
        """Ensure submodules state in the cloned repo matches self.init_submodules (install or deinstall them)."""
        submodules_are_ok, submodules_are_ok_msg = self.check_submodules_state(repo_path)
        if submodules_are_ok:
            return True, ""
        if not submodules_are_ok and "Failed to get submodule status" in submodules_are_ok_msg:
            return False, submodules_are_ok_msg

        cmd = ["update", "--init", "--recursive"] if self.init_submodules else ["deinit", "--all", "--force"]
        result = subprocess.run(["git", "submodule", *cmd], cwd=str(repo_path), capture_output=True, text=True)
        if result.returncode != 0:
            action = "initialize" if self.init_submodules else "deinitialize"
            return False, f"Failed to {action} submodules: {result.stderr}"

        return True, ""

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = installer.system.install_path / self.repo_name
        if repo_path.exists():
            verify_res = self._verify_commit(self.commit, repo_path)
            if not verify_res.success:
                return verify_res
            submodules_res, submodules_msg = self.ensure_submodules_state(repo_path)
            if not submodules_res:
                return InstallStatusResult(False, submodules_msg)
            self.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_and_setup_repo(installer, repo_path)
        if not res.success:
            return res

        self.installed_path = repo_path
        return InstallStatusResult(True)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        logging.debug(f"Uninstalling git repository at {self.installed_path=}")
        repo_path = self.installed_path if self.installed_path else installer.system.install_path / self.repo_name
        if not repo_path.exists():
            return InstallStatusResult(True, f"Repository {self.url} is not cloned.")

        logging.debug(f"Removing folder {repo_path}")
        shutil.rmtree(repo_path)
        self.installed_path = None

        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = installer.system.install_path / self.repo_name
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {self.url} not cloned")
        verify_res = self._verify_commit(self.commit, repo_path)
        if not verify_res.success:
            return verify_res

        verify_submodules, msg_submodules = self.check_submodules_state(repo_path)
        if not verify_submodules:
            return InstallStatusResult(False, msg_submodules)

        self.installed_path = repo_path
        return InstallStatusResult(True)

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.repo_name
        return InstallStatusResult(True)

    def _clone_and_setup_repo(self, installer: "BaseInstaller", repo_path: Path) -> InstallStatusResult:
        res = self._clone_repository(installer, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(self.commit, repo_path)
        if not res.success:
            logging.error(f"Checkout failed, removing cloned repository at {repo_path}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return res

        submodules_res, submodules_msg = self.ensure_submodules_state(repo_path)
        if not submodules_res:
            logging.error(f"Submodule setup failed with `{submodules_msg}`, removing cloned repository at {repo_path}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return InstallStatusResult(False, submodules_msg)

        return InstallStatusResult(True)

    def _clone_repository(self, installer: "BaseInstaller", path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {self.url} into {path}")
        clone_cmd = ["git", "clone"]

        if installer.is_low_thread_environment:
            clone_cmd.extend(["-c", "pack.threads=4"])

        clone_cmd.extend([self.url, str(path)])

        logging.debug(f"Running git clone command: {' '.join(clone_cmd)}")
        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to clone repository: {result.stderr}")
        return InstallStatusResult(True)

    def _checkout_commit(self, commit_hash: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Checking out specific commit in {path}: {commit_hash}")
        result = subprocess.run(["git", "checkout", commit_hash], cwd=str(path), capture_output=True, text=True)
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to checkout commit {commit_hash}: {result.stderr}")
        return InstallStatusResult(True)

    def _verify_commit(self, ref: str, path: Path) -> InstallStatusResult:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {result.stderr}")
        actual_commit = result.stdout.strip()

        try:
            commit_resolved = subprocess.run(
                ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        if commit_resolved.returncode != 0:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {commit_resolved.stderr}")
        expected_commit = commit_resolved.stdout.strip()

        try:
            branch_resolved = subprocess.run(
                ["git", "symbolic-ref", "--short", "-q", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        actual_branch = branch_resolved.stdout.strip() if branch_resolved.returncode == 0 else ""

        if actual_commit == expected_commit or ref == actual_branch:
            return InstallStatusResult(True)

        return InstallStatusResult(
            success=False,
            message=(
                f"Failed to verify commit in {path}: {actual_commit=}, {actual_branch=}, expected was {ref} or "
                f"{expected_commit=}"
            ),
        )


@dataclass
class PythonExecutable(Installable):
    """Python executable object."""

    git_repo: GitRepo
    venv_path: Optional[Path] = None
    project_subpath: Optional[Path] = None
    dependencies_from_pyproject: bool = True

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return (
            isinstance(other, PythonExecutable)
            and other.git_repo.url == self.git_repo.url
            and other.git_repo.commit == self.git_repo.commit
        )

    def __hash__(self) -> int:
        """Hash the installable object."""
        return self.git_repo.__hash__()

    def __str__(self) -> str:
        """Return the string representation of the python executable."""
        return f"PythonExecutable(git_url={self.git_repo.url}, commit_hash={self.git_repo.commit})"

    @property
    def venv_name(self) -> str:
        return f"{self.git_repo.repo_name}-venv"

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        res = self.git_repo.install(installer)
        if not res.success:
            return res

        return self._create_venv(installer)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        res = self.git_repo.uninstall(installer)
        if not res.success:
            return res

        logging.debug(f"Uninstalling virtual environment at {self.venv_path=}")
        venv_path = self.venv_path if self.venv_path else installer.system.install_path / self.venv_name
        if not venv_path.exists():
            return InstallStatusResult(True, f"Virtual environment {self.venv_name} is not created.")

        logging.debug(f"Removing folder {venv_path}")
        shutil.rmtree(venv_path)
        self.venv_path = None

        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = (
            self.git_repo.installed_path
            if self.git_repo.installed_path
            else installer.system.install_path / self.git_repo.repo_name
        )
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {self.git_repo.url} not cloned")
        self.git_repo.installed_path = repo_path

        venv_path = self.venv_path if self.venv_path else installer.system.install_path / self.venv_name
        if not venv_path.exists():
            return InstallStatusResult(False, f"Virtual environment not created for {self.git_repo.url}")
        self.venv_path = venv_path

        return InstallStatusResult(True, "Python executable installed")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.git_repo.installed_path = installer.system.install_path / self.git_repo.repo_name
        self.venv_path = installer.system.install_path / self.venv_name
        return InstallStatusResult(True)

    def _create_venv(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = installer.system.install_path / self.venv_name
        logging.debug(f"Creating virtual environment in {venv_path}")
        if venv_path.exists():
            msg = f"Virtual environment already exists at {venv_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        cmd = ["python", "-m", "venv", str(venv_path)]
        logging.debug(f"Creating venv using cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        logging.debug(f"venv creation STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        if result.returncode != 0:
            if venv_path.exists():
                shutil.rmtree(venv_path)
            return InstallStatusResult(
                False, f"Failed to create venv:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        res = self._install_dependencies(installer)
        if not res.success:
            if venv_path.exists():
                shutil.rmtree(venv_path)
            return res

        self.venv_path = installer.system.install_path / self.venv_name

        return InstallStatusResult(True)

    def _install_dependencies(self, installer: "BaseInstaller") -> InstallStatusResult:
        venv_path = installer.system.install_path / self.venv_name

        if not self.git_repo.installed_path:
            return InstallStatusResult(False, "Git repository must be installed before creating virtual environment.")

        project_dir = self.git_repo.installed_path

        if self.project_subpath:
            project_dir = project_dir / self.project_subpath

        pyproject_toml = project_dir / "pyproject.toml"
        requirements_txt = project_dir / "requirements.txt"

        if pyproject_toml.exists() and requirements_txt.exists():
            if self.dependencies_from_pyproject:
                return self._install_pyproject(venv_path, project_dir)
            return self._install_requirements(venv_path, requirements_txt)
        if pyproject_toml.exists():
            return self._install_pyproject(venv_path, project_dir)
        if requirements_txt.exists():
            return self._install_requirements(venv_path, requirements_txt)

        return InstallStatusResult(False, "No pyproject.toml or requirements.txt found for installation.")

    def _install_pyproject(self, venv_dir: Path, project_dir: Path) -> InstallStatusResult:
        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", str(project_dir)]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install {project_dir} using pip: {result.stderr}")

        return InstallStatusResult(True)

    def _install_requirements(self, venv_dir: Path, requirements_txt: Path) -> InstallStatusResult:
        if not requirements_txt.is_file():
            return InstallStatusResult(False, f"Requirements file is invalid or does not exist: {requirements_txt}")

        install_cmd = [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_txt)]
        logging.debug(f"Installing dependencies using: {' '.join(install_cmd)}")
        result = subprocess.run(install_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to install dependencies from requirements.txt: {result.stderr}")

        return InstallStatusResult(True)


@dataclass
class File(Installable):
    """File object."""

    src: Path
    _installed_path: Optional[Path] = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        return self._installed_path or self.src

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, File) and other.src == self.src

    def __hash__(self) -> int:
        return hash(self.src)

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.src.name
        shutil.copyfile(self.src, self.installed_path, follow_symlinks=False)
        return InstallStatusResult(True)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        if self.installed_path != self.src:
            self.installed_path.unlink()
            self._installed_path = None
            return InstallStatusResult(True)
        logging.debug(f"File {self.installed_path} does not exist.")
        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        installed_path = installer.system.install_path / self.src.name
        if installed_path.exists() and installed_path.read_text() == self.src.read_text():
            self.installed_path = installed_path
            return InstallStatusResult(True)
        return InstallStatusResult(False, f"File {installed_path} does not exist")

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.src.name
        return InstallStatusResult(True)


@dataclass
class HFModel(Installable):
    """HuggingFace Model object."""

    model_name: str
    _installed_path: Path | None = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        if self._installed_path:
            return self._installed_path
        return Path("hub") / self.model_name

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HFModel) and other.model_name == self.model_name

    def __hash__(self) -> int:
        return hash(self.model_name)
