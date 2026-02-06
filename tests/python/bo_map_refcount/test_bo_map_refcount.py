#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Minimal reproducer for bo.map() use-after-free bug.
#
# bo.map() returns a memoryview via py::memoryview::from_memory() which
# does NOT hold a reference to the parent bo. When the bo is deleted,
# the underlying DMA buffer is freed, but the memoryview (and any numpy
# array created from it) still points to the freed memory, causing a
# segfault on access.
#
# Usage:
#   source /opt/xilinx/xrt/setup.sh
#   python3 test_bo_map_refcount.py
#
# Expected (before fix): Segmentation fault
# Expected (after fix):  PASS

import gc
import sys
import numpy as np

import pyxrt as xrt


def test_memoryview_outlives_bo():
    """memoryview from bo.map() must remain valid after bo is deleted."""
    dev = xrt.device(0)
    bo = xrt.bo(dev, 40, xrt.bo.host_only, 0)

    mv = bo.map()
    assert isinstance(mv, memoryview), f"Expected memoryview, got {type(mv)}"

    # Write a known pattern through the memoryview
    arr = np.frombuffer(mv, dtype=np.int32)
    arr[:] = np.arange(10, dtype=np.int32) + 100

    # Delete the bo — this frees the underlying buffer
    del bo
    gc.collect()

    expected = np.arange(10, dtype=np.int32) + 100
    if not np.array_equal(arr, expected):
        print(f"FAIL: data corrupted after bo deleted")
        print(f"  expected: {expected}")
        print(f"  got:      {arr}")
        return False

    return True


if __name__ == "__main__":
    results = []

    print("test_memoryview_outlives_bo ... ", end="", flush=True)
    ok = test_memoryview_outlives_bo()
    print("PASS" if ok else "FAIL")

    if ok:
        sys.exit(0)
    else:
        sys.exit(1)
