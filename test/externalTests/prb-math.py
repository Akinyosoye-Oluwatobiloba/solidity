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

import sys

from exttest.common import run_test, TestConfig
from runners.foundry import FoundryRunner

test_config = TestConfig(
    name="PRBMath",
    repo_url="https://github.com/PaulRBerg/prb-math.git",
    ref_type="branch",
    ref="main",
    build_dependency="rust",
    compile_only_presets=[
        #"ir-no-optimize", # Error: Yul exception:Variable expr_15841_address is 2 slot(s) too deep inside the stack. Stack too deep.
        #"ir-optimize-evm-only", # Error: Yul exception:Variable expr_15841_address is 2 slot(s) too deep inside the stack. Stack too deep.
    ],
    settings_presets=[
        "ir-optimize-evm+yul",
        "legacy-optimize-evm-only",
        "legacy-optimize-evm+yul",
        "legacy-no-optimize",
    ],
)

sys.exit(
    run_test(FoundryRunner(argv=sys.argv[1:], config=test_config))
)
