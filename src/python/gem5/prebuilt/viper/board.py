# Copyright (c) 2023 The Regents of the University of California
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

from typing import List

from m5.util import warn, panic

from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.single_channel import SingleChannelDDR4_2400

# from m5.objects import HBM_1000_4H_1x128
from gem5.components.memory.hbm import HBM2Stack
from gem5.components.processors.simple_processor import SimpleProcessor
from cpu_cache_hierarchy import (
    ViperCPUCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gpus import MI100GPU
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.utils.requires import requires
from gem5.utils.override import overrides
from gem5.components.boards.kernel_disk_workload import KernelDiskWorkload

from m5.objects import (
    Addr,
    AddrRange,
    AMDGPUDevice,
    NoncoherentXBar,
)


class X86ViperBoard(X86Board):
    """


    Example
    -------

    An example of using the X86ViperBoard can be found in
    `<>`.

    To run:

    ```
    scons build/VEGA_X86/gem5.opt -j`nproc`
    ./build/VEGA_X86/gem5.opt <>
    ```

    """

    def __init__(self, gpu_model: str):
        """
        GPU model can be one of the following: Vega10, MI200
        """

        requires(
            isa_required=ISA.X86,
            coherence_protocol_required=CoherenceProtocol.GPU_VIPER,
        )

        memory = SingleChannelDDR4_2400(size="3GB")

        # Supported CPUs: Atomic and KVM
        processor = SimpleProcessor(
            cpu_type=CPUTypes.ATOMIC, isa=ISA.X86, num_cores=1
        )
        cache_hierarchy = ViperCPUCacheHierarchy(
            l1d_size="32kB",
            l1d_assoc=8,
            l1i_size="32kB",
            l1i_assoc=8,
            l2_size="1MB",
            l2_assoc=16,
        )
        # cache_hierarchy = NoCache()

        print(cache_hierarchy.get_ruby())

        super().__init__(
            clk_freq="3GHz",
            processor=processor,
            memory=memory,
            cache_hierarchy=cache_hierarchy,
        )

        # Anything added here will be run after

        self._gpu_mem_size = "16GiB"
        # Create AMDGPU and attach to south bridge
        gpu_device = AMDGPUDevice(pci_func=0, pci_dev=8, pci_bus=0)

        self.pc.south_bridge.gpu = gpu_device
        self.pc.south_bridge.gpu.memory = HBM2Stack(self._gpu_mem_size)
        self.pc.south_bridge.gpu.memories = (
            self.pc.south_bridge.gpu.memory.get_memory_interfaces()
        )

        mmio_trace = "/home/lyy/gem5-gpu/gem5/src/python/gem5/prebuilt/viper.vega_mmio.log"
        rom = ""  # This isn't really used, rom is on the disk
        if gpu_model == "MI100":
            self.gpu = MI100GPU(8, gpu_device, mmio_trace, rom)
        self.gpu.setup_device(gpu_device)
        self.pc.south_bridge.gpu.cp = self.gpu.gpu_cmd_proc
        # GPU interrupt handler
        self.pc.south_bridge.gpu.device_name = gpu_model
        self.pc.south_bridge.gpu.device_ih = gpu_device.device_ih
        self.pc.south_bridge.gpu.sdmas = gpu_device.sdmas
        self.pc.south_bridge.gpu.pm4_pkt_procs = gpu_device.pm4_pkt_procs
        self.pc.south_bridge.gpu.memory_manager = gpu_device.memory_manager

        xbar = NoncoherentXBar(
            frontend_latency=0,
            forward_latency=0,
            response_latency=0,
            header_latency=0,
            width=64,
        )
        self.pc.membus = xbar
        self.gpu.connect_iobus(xbar)  # a BaseXBar object parsed in

        self.gpu.set_cpu_pointer(
            processor.get_cores()[0].get_simobject()
        )  # a cpu pointer parsed in

        # connect GPU comes from connectGPU() in amdgpu.py
        self.pc.south_bridge.gpu.trace_file = mmio_trace
        self.pc.south_bridge.gpu.rom_binary = rom
        self.pc.south_bridge.gpu.DeviceID = gpu_device.DeviceID
        self.pc.south_bridge.gpu.SubsystemVendorID = (
            gpu_device.SubsystemVendorID
        )
        self.pc.south_bridge.gpu.SubsystemID = gpu_device.SubsystemID

        # Use the gem5 default of 0x280 OR'd  with 0x10 which tells Linux there is
        # a PCI capabilities list to travse.
        self.pc.south_bridge.gpu.Status = 0x0290

        # The PCI capabilities are like a linked list. The list has a memory
        # offset and a capability type ID read by the OS. Make the first
        # capability at 0x80 and set the PXCAP (PCI express) capability to
        # that address. Mark the type ID as PCI express.
        # We leave the next ID of PXCAP blank to end the list.
        self.pc.south_bridge.gpu.PXCAPBaseOffset = 0x80
        self.pc.south_bridge.gpu.CapabilityPtr = 0x80
        self.pc.south_bridge.gpu.PXCAPCapId = 0x10

        # Set bits 7 and 8 in the second PCIe device capabilities register which
        # reports support for PCIe atomics for 32 and 64 bits respectively.
        # Bit 9 for 128-bit compare and swap is not set because the amdgpu driver
        # does not check this.
        self.pc.south_bridge.gpu.PXCAPDevCap2 = 0x00000180

        # Set bit 6 to enable atomic requestor, meaning this device can request
        # atomics from other PCI devices.
        self.pc.south_bridge.gpu.PXCAPDevCtrl2 = 0x00000040

        self._setup_gpu_memory()

    # I think this isn't needed?  Not good with python classes
    # The shadow rom is no longer necessary which is what was here
    def _setup_gpu_memory(self) -> None:
        base_gpu_addr = min(self.mem_ranges[-1].end, 0x100000000)
        self.mem_ranges.append(
            AddrRange(start=base_gpu_addr, size=self._gpu_mem_size)
        )
        self.pc.south_bridge.gpu.memory.set_memory_range([self.mem_ranges[-1]])

    def _set_mmio_file(self, file_name: str):
        mmio_md5 = hashlib.md5(open(file_name, "rb").read()).hexdigest()
        if mmio_md5 != "c4ff3326ae8a036e329b8b595c83bd6d":
            panic("MMIO file does not match gem5 resources")

    @overrides(KernelDiskWorkload)
    def get_default_kernel_args(self) -> List[str]:
        return [
            "earlyprintk=ttyS0",
            "console=ttyS0,9600",
            "lpj=7999923",
            "root=/dev/sda1",
            "drm_kms_helper.fbdev_emulation=0",
            "modprobe.blacklist=amdgpu",
            "modprobe.blacklist=psmouse",
        ]
