#!/usr/bin/env python3

# ------------------------------------------------------------------------------
# This file is part of solidity.
#
# solidity is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# solidity is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with solidity.  If not, see <http://www.gnu.org/licenses/>
#
# (c) 2023 solidity contributors.
# ------------------------------------------------------------------------------

import os
import re
import subprocess
import sys
from abc import abstractmethod, ABCMeta
from argparse import ArgumentParser
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from textwrap import dedent
from typing import List, Set

# Our scripts/ is not a proper Python package so we need to modify PYTHONPATH to import from it
# pragma pylint: disable=import-error,wrong-import-position
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, f"{PROJECT_ROOT}/scripts/common")

from git_helpers import git_commit_hash

SOLC_FULL_VERSION_REGEX = re.compile(r"^[a-zA-Z: ]*(.*)$")
SOLC_SHORT_VERSION_REGEX = re.compile(r"^([0-9.]+).*\+|\-$")

evm_version = os.environ.get("DEFAULT_EVM")
CURRENT_EVM_VERSION: str = evm_version if evm_version is not None else "shanghai"


class SettingsPreset(Enum):
    LEGACY_NO_OPTIMIZE = 'legacy-no-optimize'
    IR_NO_OPTIMIZE = 'ir-no-optimize'
    LEGACY_OPTIMIZE_EVM_ONLY = 'legacy-optimize-evm-only'
    IR_OPTIMIZE_EVM_ONLY = 'ir-optimize-evm-only'
    LEGACY_OPTIMIZE_EVM_YUL = 'legacy-optimize-evm+yul'
    IR_OPTIMIZE_EVM_YUL = 'ir-optimize-evm+yul'


@dataclass
class TestConfig:
    name: str
    repo_url: str
    ref_type: str
    ref: str
    build_dependency: str = field(default="nodejs")
    compile_only_presets: List[SettingsPreset] = field(default_factory=list)
    settings_presets: List[SettingsPreset] = field(default_factory=lambda: list(SettingsPreset))
    evm_version: str = field(default=CURRENT_EVM_VERSION)

    def selected_presets(self) -> Set[SettingsPreset]:
        return set(self.compile_only_presets + self.settings_presets)


class TestRunner(metaclass=ABCMeta):
    config: TestConfig
    solc_binary_type: str
    solc_binary_path: Path

    def __init__(self, argv, config: TestConfig):
        args = parse_command_line(f"{config.name} external tests", argv)
        self.config = config
        self.solc_binary_type = args.solc_binary_type
        self.solc_binary_path = args.solc_binary_path
        self.env = os.environ.copy()
        self.tmp_dir = mkdtemp(prefix=f"ext-test-{config.name}-")
        self.test_dir = Path(self.tmp_dir) / "ext"

    def setup_solc(self) -> str:
        if self.solc_binary_type == "solcjs":
            # TODO: add support to solc-js # pylint: disable=fixme
            raise NotImplementedError()
        print("Setting up solc...")
        solc_version_output = subprocess.getoutput(f"{self.solc_binary_path} --version").split(":")[1]
        return parse_solc_version(solc_version_output)

    @staticmethod
    def on_local_test_dir(fn):
        """Run a function inside the test directory"""

        def f(self, *args, **kwargs):
            assert self.test_dir is not None
            os.chdir(self.test_dir)
            return fn(self, *args, **kwargs)

        return f

    def setup_environment(self):
        """Configure the project build environment"""
        print("Configuring Runner building environment...")
        replace_version_pragmas(self.test_dir)

    @on_local_test_dir
    def clean(self):
        """Clean temporary directories"""
        rmtree(self.tmp_dir)

    @on_local_test_dir
    @abstractmethod
    def configure(self, presets: List[SettingsPreset]):
        # TODO: default to hardhat # pylint: disable=fixme
        raise NotImplementedError()

    @on_local_test_dir
    @abstractmethod
    def compile(self, preset: SettingsPreset):
        # TODO: default to hardhat # pylint: disable=fixme
        raise NotImplementedError()

    @on_local_test_dir
    @abstractmethod
    def run_test(self):
        # TODO: default to hardhat # pylint: disable=fixme
        raise NotImplementedError()


# Helper functions
def compiler_settings(evm_version: str, via_ir: bool = False, optimizer: bool = False, yul: bool = False) -> dict:
    return {
        "optimizer": {"enabled": optimizer, "details": {"yul": yul}},
        "evmVersion": evm_version,
        "viaIR": via_ir,
    }


def settings_from_preset(preset: SettingsPreset, evm_version: str) -> dict:
    return {
        SettingsPreset.LEGACY_NO_OPTIMIZE:       compiler_settings(evm_version),
        SettingsPreset.IR_NO_OPTIMIZE:           compiler_settings(evm_version, via_ir=True),
        SettingsPreset.LEGACY_OPTIMIZE_EVM_ONLY: compiler_settings(evm_version, optimizer=True),
        SettingsPreset.IR_OPTIMIZE_EVM_ONLY:     compiler_settings(evm_version, via_ir=True, optimizer=True),
        SettingsPreset.LEGACY_OPTIMIZE_EVM_YUL:  compiler_settings(evm_version, optimizer=True, yul=True),
        SettingsPreset.IR_OPTIMIZE_EVM_YUL:      compiler_settings(evm_version, via_ir=True, optimizer=True, yul=True),
    }[preset]


def parse_command_line(description: str, args: List[str]):
    arg_parser = ArgumentParser(description)
    arg_parser.add_argument(
        "solc_binary_type",
        metavar="solc-binary-type",
        type=str,
        default="native",
        choices=["native", "solcjs"],
        help="""Solidity compiler binary type""",
    )
    arg_parser.add_argument(
        "solc_binary_path",
        metavar="solc-binary-path",
        type=Path,
        default=Path("/usr/local/bin/solc"),
        help="""Path to solc binary""",
    )
    return arg_parser.parse_args(args)


def download_project(test_dir: Path, repo_url: str, ref_type: str = "branch", ref: str = "master"):
    assert ref_type in ("commit", "branch", "tag")

    print(f"Cloning {ref_type} {ref} of {repo_url}...")
    if ref_type == "commit":
        os.mkdir(test_dir)
        os.chdir(test_dir)
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", ref], check=True)
        subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], check=True)
    else:
        os.chdir(test_dir.parent)
        subprocess.run(["git", "clone", "--no-progress", "--depth", "1", repo_url, "-b", ref, test_dir.resolve()], check=True)
        if not test_dir.exists():
            raise RuntimeError("Failed to clone the project.")
        os.chdir(test_dir)

    if (test_dir / ".gitmodules").exists():
        subprocess.run(["git", "submodule", "update", "--init"], check=True)

    print(f"Current commit hash: {git_commit_hash()}")


def parse_solc_version(solc_version_string: str) -> str:
    solc_version_match = re.search(SOLC_FULL_VERSION_REGEX, solc_version_string)
    if solc_version_match is None:
        raise RuntimeError(f"Solc version could not be found in: {solc_version_string}.")
    return solc_version_match.group(1)


def get_solc_short_version(solc_full_version: str) -> str:
    solc_short_version_match = re.search(SOLC_SHORT_VERSION_REGEX, solc_full_version)
    if solc_short_version_match is None:
        raise RuntimeError(f"Error extracting short version string from: {solc_full_version}.")
    return solc_short_version_match.group(1)


def store_benchmark_report(self):
    raise NotImplementedError()


def replace_version_pragmas(test_dir: Path):
    """
    Replace fixed-version pragmas (part of Consensys best practice).
    Include all directories to also cover node dependencies.
    """
    print("Replacing fixed-version pragmas...")
    for source in test_dir.glob("**/*.sol"):
        content = source.read_text(encoding="utf-8")
        content = re.sub(r"pragma solidity [^;]+;", r"pragma solidity >=0.0;", content)
        with open(source, "w", encoding="utf-8") as f:
            f.write(content)


def run_test(runner: TestRunner):
    print(f"Testing {runner.config.name}...\n===========================")

    presets = runner.config.selected_presets()
    print(f"Selected settings presets: {' '.join(p.value for p in presets)}")

    # Configure solc compiler
    solc_version = runner.setup_solc()
    print(f"Using compiler version {solc_version}")

    # Download project
    download_project(runner.test_dir, runner.config.repo_url, runner.config.ref_type, runner.config.ref)

    # Configure run environment
    runner.setup_environment()

    # Configure TestRunner instance
    print(dedent(f"""\
        Configuring runner's profiles with:
        -------------------------------------
        Binary type: {runner.solc_binary_type}
        Compiler path: {runner.solc_binary_path}
        -------------------------------------
    """))
    runner.configure(presets)
    for preset in runner.config.selected_presets():
        print("Running compile function...")
        settings = settings_from_preset(preset, runner.config.evm_version)
        print(dedent(f"""\
            -------------------------------------
            Settings preset: {preset.value}
            Settings: {settings}
            EVM version: {runner.config.evm_version}
            Compiler version: {get_solc_short_version(solc_version)}
            Compiler version (full): {solc_version}
            -------------------------------------
        """))
        runner.compile(preset)
        # TODO: COMPILE_ONLY should be a command-line option # pylint: disable=fixme
        if os.environ.get("COMPILE_ONLY") == "1" or preset in runner.config.compile_only_presets:
            print("Skipping test function...")
        else:
            print("Running test function...")
            runner.run_test()
        # TODO: store_benchmark_report # pylint: disable=fixme
    runner.clean()
    print("Done.")
