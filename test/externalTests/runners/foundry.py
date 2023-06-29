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
from shutil import which
from textwrap import dedent
from typing import Optional, List

from exttest.common import settings_from_preset, TestRunner


def run_forge_command(command: str, env: Optional[dict] = None):
    subprocess.run(
        command.split(),
        env=env if env is not None else os.environ.copy(),
        check=True
    )


class FoundryRunner(TestRunner):
    """Configure and run Foundry-based projects"""

    foundry_config_file = "foundry.toml"

    def setup_environment(self):
        super().setup_environment()
        if which("forge") is None:
            raise RuntimeError("Forge not found.")

    @staticmethod
    def profile_name(preset: str):
        """Returns foundry profile name"""
        # Replace - or + by underscore to avoid invalid toml syntax
        return re.sub(r"(\-|\+)+", "_", preset)

    @staticmethod
    def profile_section(profile_fields: dict) -> str:
        return dedent(
            """\
            [profile.{name}]
            gas_reports = ["*"]
            auto_detect_solc = false
            solc = "{solc}"
            evm_version = "{evm_version}"
            optimizer = {optimizer}
            via_ir = {via_ir}

            [profile.{name}.optimizer_details]
            yul = {yul}
            """
        ).format(
            name=profile_fields["name"],
            solc=profile_fields["solc"],
            evm_version=profile_fields["evm_version"],
            optimizer=profile_fields["optimizer"],
            via_ir=profile_fields["via_ir"],
            yul=profile_fields["yul"],
        )

    @TestRunner.on_local_test_dir
    def compiler_settings(self, presets: List[str]):
        """Configure forge tests profiles"""

        profiles = []
        for preset in presets:
            settings = settings_from_preset(preset, self.config.evm_version)
            profiles.append(
                self.profile_section(
                    {
                        "name": self.profile_name(preset),
                        "solc": self.solc_binary_path,
                        "evm_version": self.config.evm_version,
                        "optimizer": settings["optimizer"]["enabled"],
                        "via_ir": settings["viaIR"],
                        "yul": settings["optimizer"]["details"]["yul"],
                    }
                )
            )

        with open(
            file=self.test_dir / self.foundry_config_file,
            mode="a",
            encoding="utf-8",
        ) as f:
            for profile in profiles:
                f.write(profile)

        run_forge_command("forge install", self.env)

    @TestRunner.on_local_test_dir
    def compile(self, preset: str):
        """Compile project"""

        # Set the Foundry profile environment variable
        self.env.update({"FOUNDRY_PROFILE": self.profile_name(preset)})
        run_forge_command("forge build", self.env)

    @TestRunner.on_local_test_dir
    def run_test(self):
        """Run project tests"""

        run_forge_command("forge test --gas-report", self.env)
