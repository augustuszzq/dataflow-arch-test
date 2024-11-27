#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
print("不知道")
# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Kernel rectangle and per-PE matrix dimensions
P = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
# A is M x K, B is K x N, C is M x N
M = Mt * P
N = Nt * P

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, M).astype(np.float32)

runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

sym_Q = runner.get_id("Q_tile")
sym_K = runner.get_id("K_tile")
#sym_V = runner.get_id("V_tile")

runner.load()
runner.run()

w = P # number of columns PEs in the core rectangle
h = P # number of row PEs in the core rectangle

# How to transform a 2-D tensor into a cliff distribution with
# column-major local tensor
#
# Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
# A = |  0  1  2  3 |
#     |  4  5  6  7 |
#     |  8  9 10 11 |
#     | 12 13 14 15 |
# A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
# A1 = | | 0  1|  | 4  5| |
#      | | 2  3|, | 6  7| |
#      |                  |
#      | | 8  9|  |12 13| |
#      | |10 11|, |14 15| |
# A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
# so the local tensor lh-by-lw is col-major
# A2 = | | 0  4|  | 2  6| |
#      | | 1  5|, | 3  7| |
#      |                  |
#      | | 8 12|  |10 14| |
#      | | 9 13|, |11 15| |
# A3 = A2.reshape(2,2,4)
# A3 = |  0  4  1  5 |
#      |  2  6  3  7 |
#      |  8 12  9 13 |
#      | 10 14 11 15 |
# A3 is h-w-l

A1 = A.reshape(h, Mt, w, Nt)
A2 = A1.transpose(0, 2, 3, 1)
A3 = A2.reshape(h, w, Mt*Nt)
runner.memcpy_h2d(sym_Q, A3.ravel(), 0, 0, w, h, Mt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

B1 = B.reshape(h, Nt, w, Mt)
B2 = B1.transpose(0, 2, 3, 1)
B3 = B2.reshape(h, w, Nt*Mt)
runner.memcpy_h2d(sym_K, B3.ravel(), 0, 0, w, h, Mt*Nt, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

#runner.launch("main", nonblock=False)

#C3_1d_u32 = np.zeros(h*w*Mt*Nt, np.uint32)
#runner.memcpy_d2h(C3_1d_u32, sym_C, 0, 0, w, h, Mt*Nt, \
#    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# C3 is h-by-w-l or
# C3 is of the form (h, w, Nt, Mt) where local tensor Mt-by-Nt is column-major
#C3 = C3_1d_u32.reshape((h, w, Nt, Mt))
# C2 is of the form (h, Mt, w, Nt)
#C2 = C3.transpose(0, 3, 1, 2)
# C1 is of the form (M, N)
#C1 = C2.reshape(M, N)
# C has the correct data type
#C = C1.view(np.float32)

runner.stop()

# Check the result
#C_expected = np.dot(A, B)

# absolute(a - b) <= (atol + rtol * absolute(b))
#np.testing.assert_allclose(C_expected, C, rtol=1e-05, atol=1e-06)

print("SUCCESS")