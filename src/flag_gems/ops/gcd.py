import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def gcd_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.int64)
    b = tl.load(b_ptr + offsets, mask=mask, other=0).to(tl.int64)

    a = tl.abs(a)
    b = tl.abs(b)

    # Euclidean algorithm
    for _ in range(64):
        cond = b != 0
        r = a % tl.maximum(b, 1)
        a = tl.where(cond, b, a)
        b = tl.where(cond, r, b)

    tl.store(out_ptr + offsets, a, mask=mask)


def gcd(A, B):
    logger.debug("GEMS GCD")
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(A.device):
        gcd_kernel[grid](A, B, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
