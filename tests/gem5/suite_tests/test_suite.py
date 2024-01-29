# Copyright (c) 2022 The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
from typing import (
    List,
    Optional,
    Tuple,
)

from testlib import *

if config.bin_path:
    resource_path = config.bin_path
else:
    resource_path = os.path.join(
        joinpath(absdirpath(__file__), "..", "resources")
    )


def test_suite(
    id: str,
    isa: str,
    is_fs: bool,
    version: Optional[str] = None,
    to_tick: Optional[int] = None,
):
    name = f"suite-{id}_{isa}_suite_test"

    verifiers = []

    config_args = [
        "--suite-id",
        id,
        "--isa",
        isa,
        "--resource-directory",
        resource_path,
    ]

    if is_fs:
        config_args.append("--fs-sim")
    if version:
        config_args.extend(["--version", version])
        name = f"suite-{id}_{version}_{isa}_suite_test"

    if to_tick:
        name += "_to-tick"
        exit_regex = re.compile(
            f"Exiting @ tick {str(to_tick)} because simulate\\(\\) limit reached"
        )
        verifiers.append(verifier.MatchRegex(exit_regex))
        config_args += ["--tick-exit", str(to_tick)]
        gem5_verify_config(
            name=name,
            fixtures=(),
            verifiers=verifiers,
            config=joinpath(
                config.base_dir,
                "tests",
                "gem5",
                "suite_tests",
                "configs",
                "suite_run_workload.py",
            ),
            config_args=config_args,
            valid_isas=(constants.all_compiled_tag,),
            valid_hosts=constants.supported_hosts,
        )


test_suite(
    id="riscv-vertical-microbenchmarks",
    isa="riscv",
    to_tick=10000000000,
    is_fs=False,
)

test_suite(
    id="npb-benchmark-suite",
    isa="x86",
    to_tick=10000000000,
    is_fs=True,
)

test_suite(
    id="gapbs-benchmark-suite",
    isa="x86",
    to_tick=10000000000,
    is_fs=True,
)
