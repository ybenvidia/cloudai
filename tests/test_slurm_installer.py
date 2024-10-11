from subprocess import CompletedProcess
from unittest.mock import Mock, patch

import pytest
from cloudai import InstallStatusResult
from cloudai.installer.installables import DockerImage, PythonExecutable
from cloudai.installer.slurm_installer import SlurmInstaller
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.docker_image_cache_manager import DockerImageCacheResult


class TestInstallOneDocker:
    @pytest.fixture
    def installer(self, slurm_system: SlurmSystem):
        si = SlurmInstaller(slurm_system)
        si.install_path.mkdir()
        return si

    def test_image_is_local(self, installer: SlurmInstaller):
        cached_file = installer.system.install_path / "some_image"
        d = DockerImage(str(cached_file))
        cached_file.touch()
        res = installer._install_docker_image(d)

        assert res.success
        assert res.message == f"Docker image file path is valid: {cached_file}."
        assert d.installed_path == cached_file

    def test_image_is_already_cached(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer._install_docker_image(d)

        assert res.success
        assert res.message == f"Cached Docker image already exists at {cached_file}."
        assert d.installed_path == cached_file

    def test_uninstall_docker_image(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer._uninstall_docker_image(d)

        assert res.success
        assert res.message == f"Cached Docker image removed successfully from {cached_file}."
        assert d.installed_path == d.url

    def test_is_installed_when_cache_exists(self, installer: SlurmInstaller):
        d = DockerImage("fake_url/img")
        cached_file = installer.system.install_path / d.cache_filename
        cached_file.touch()

        res = installer.is_installed_one(d)

        assert res.success
        assert res.message == f"Cached Docker image already exists at {cached_file}."
        assert d.installed_path == cached_file


class TestInstallOnePythonExecutable:
    @pytest.fixture
    def installer(self, slurm_system: SlurmSystem):
        si = SlurmInstaller(slurm_system)
        si.install_path.mkdir()
        return si

    def test_repo_exists(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        repo_path.mkdir()
        res = installer._install_python_executable(py)
        assert res.success
        assert res.message == f"Python executable repository already exists at {repo_path}."

    def test_repo_cloned(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._clone_repository(py.git_url, repo_path)
        assert res.success
        mock_run.assert_called_once_with(["git", "clone", py.git_url, str(repo_path)], capture_output=True, text=True)

    def test_error_cloning_repo(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._clone_repository(py.git_url, repo_path)
        assert not res.success
        assert res.message == "Failed to clone repository: err"

    def test_commit_checked_out(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._checkout_commit(py.commit_hash, repo_path)
        assert res.success
        mock_run.assert_called_once_with(
            ["git", "checkout", py.commit_hash], cwd=str(repo_path), capture_output=True, text=True
        )

    def test_error_checking_out_commit(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        repo_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._checkout_commit(py.commit_hash, repo_path)
        assert not res.success
        assert res.message == "Failed to checkout commit: err"

    def test_venv_created(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        venv_path = installer.system.install_path / py.venv_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._create_venv(venv_path)
        assert res.success
        mock_run.assert_called_once_with(["python", "-m", "venv", str(venv_path)], capture_output=True, text=True)

    def test_error_creating_venv(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        venv_path = installer.system.install_path / py.venv_name
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._create_venv(venv_path)
        assert not res.success
        assert res.message == "Failed to create venv: err"

    def test_venv_already_exists(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        venv_path = installer.system.install_path / py.venv_name
        venv_path.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._create_venv(venv_path)
        assert mock_run.call_count == 0
        assert res.success
        assert res.message == f"Virtual environment already exists at {venv_path}."

    def test_requiretements_no_file(self, installer: SlurmInstaller):
        res = installer._install_requirements(
            installer.system.install_path, installer.system.install_path / "requirements.txt"
        )
        assert not res.success
        assert (
            res.message
            == f"Requirements file is invalid or does not exist: {installer.system.install_path / 'requirements.txt'}"
        )

    def test_requirements_installed(self, installer: SlurmInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=0)
            res = installer._install_requirements(installer.system.install_path, requirements_file)
        assert res.success
        mock_run.assert_called_once_with(
            [
                installer.system.install_path / "bin" / "python",
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_file),
            ],
            capture_output=True,
            text=True,
        )

    def test_requirements_not_installed(self, installer: SlurmInstaller):
        requirements_file = installer.system.install_path / "requirements.txt"
        requirements_file.touch()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args=[], returncode=1, stderr="err")
            res = installer._install_requirements(installer.system.install_path, requirements_file)
        assert not res.success
        assert res.message == "Failed to install requirements: err"

    def test_repo_is_prepared(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        installer._clone_repository = Mock(return_value=InstallStatusResult(True))
        installer._checkout_commit = Mock(return_value=InstallStatusResult(True))
        installer._create_venv = Mock(return_value=InstallStatusResult(True))
        installer._install_requirements = Mock(return_value=InstallStatusResult(True))
        res = installer._install_python_executable(py)
        assert res.success
        assert py.installed_path == installer.system.install_path / py.repo_name
        assert py.venv_path == installer.system.install_path / py.venv_name

    def test_uninstall_no_repo(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        res = installer._uninstall_python_executable(py)
        assert res.success
        assert res.message == f"Repository {py.git_url} is not cloned."

    def test_uninstall_repo_removed(self, installer: SlurmInstaller):
        py = PythonExecutable("./git_url", "commit_hash")
        repo_path = installer.system.install_path / py.repo_name
        repo_path.mkdir()
        (repo_path / "file").touch()  # test with non-empty directory
        py.installed_path = repo_path
        res = installer._uninstall_python_executable(py)
        assert res.success
        assert py.installed_path is None
        assert not repo_path.exists()


def test_check_supported(slurm_system: SlurmSystem):
    installer = SlurmInstaller(slurm_system)
    installer._install_docker_image = lambda item: DockerImageCacheResult(True)
    installer._install_python_executable = lambda item: InstallStatusResult(True)
    installer._uninstall_docker_image = lambda item: DockerImageCacheResult(True)
    installer._uninstall_python_executable = lambda item: InstallStatusResult(True)

    items = [DockerImage("fake_url/img"), PythonExecutable("git_url", "commit_hash")]
    for item in items:
        res = installer.install_one(item)
        assert res.success

        res = installer.uninstall_one(item)
        assert res.success
