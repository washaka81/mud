import ctypes
import os
import torch
import multiprocessing
import sys

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

_lib = None
_vulkan_available = False


@torch._dynamo.disable
def _load_lib():
    global _lib, _vulkan_available
    if _lib is not None:
        return
    lib_path = os.path.join(os.path.dirname(__file__), "..", "target", "release", "libforge_llm.so")
    lib_path = os.path.abspath(lib_path)
    if not os.path.exists(lib_path):
        lib_path = os.path.join(os.path.dirname(__file__), "..", "target", "debug", "libforge_llm.so")
    lib_path = os.path.abspath(lib_path)
    if not os.path.exists(lib_path):
        raise RuntimeError(f"libforge_llm.so not found at {lib_path}. Build with: cargo build --release --lib")

    print(f"[VulkanBackend] Loading library from: {lib_path}")
    _lib = ctypes.CDLL(lib_path)

    for name, argtypes, restype in [
        ("vb_gemm_forward",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
          ctypes.c_float, ctypes.c_uint8], ctypes.c_int),
        ("vb_gemm_backward_input",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32], ctypes.c_int),
        ("vb_gemm_outer_product",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32], ctypes.c_int),
        ("vb_quantize",
         [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int),
        ("vb_init_vulkan", [], ctypes.c_int),
        ("vb_clear_caches", [], None),
    ]:
        fn = getattr(_lib, name)
        fn.argtypes = argtypes
        fn.restype = restype

    res = _lib.vb_init_vulkan()
    print(f"[VulkanBackend] vb_init_vulkan returned: {res}")
    if res == 0:
        _vulkan_available = True
        print(f"[VulkanBackend] Vulkan is available.")
    else:
        print(f"[VulkanBackend] Vulkan is NOT available.")


class _PackedCache:
    def __init__(self):
        self._cache = {}
        self._version = {}

    @torch._dynamo.disable
    def get(self, weight):
        key = id(weight)
        ver = weight._version  # PyTorch tensor version increments on mutation
        if key in self._cache and self._version.get(key) == ver:
            return self._cache[key]
        w_flat = weight.detach().contiguous().view(-1)
        n = w_flat.numel()
        packed = torch.empty((n + 15) // 16, dtype=torch.int32, device="cpu")
        _lib.vb_quantize(
            ctypes.cast(w_flat.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n),
            ctypes.cast(packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
        )
        self._cache[key] = packed
        self._version[key] = ver
        return packed

    @torch._dynamo.disable
    def clear(self):
        self._cache.clear()
        self._version.clear()

    def __len__(self):
        return len(self._cache)


_packed_cache = _PackedCache()


@torch._dynamo.disable
def pack_ternary(weight):
    return _packed_cache.get(weight)


@torch._dynamo.disable
def clear_caches():
    """Limpia todas las caches (llamar después de cada paso de training)."""
    _packed_cache.clear()
    if _lib is not None:
        _lib.vb_clear_caches()


@torch._dynamo.disable
def gemm_forward(x, w_packed, n_in, n_out, scale):
    batch = x.size(0)
    x_cont = x.contiguous()
    y = torch.zeros(batch, n_out, dtype=x.dtype, device="cpu")
    _lib.vb_gemm_forward(
        ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(batch), ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
        ctypes.c_float(scale),
        ctypes.c_uint8(1 if _vulkan_available else 0),
    )
    return y


@torch._dynamo.disable
def gemm_backward_input(dy, w_packed, n_in, n_out):
    batch = dy.size(0)
    dy_cont = dy.contiguous()
    dx = torch.zeros(batch, n_in, dtype=dy.dtype, device="cpu")
    _lib.vb_gemm_backward_input(
        ctypes.cast(dy_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(dx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(batch), ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
    )
    return dx


@torch._dynamo.disable
def gemm_outer_product(dy, x, n_out, n_in):
    batch = dy.size(0)
    dy_cont = dy.contiguous()
    x_cont = x.contiguous()
    dw = torch.zeros(n_out, n_in, dtype=dy.dtype, device="cpu")
    _lib.vb_gemm_outer_product(
        ctypes.cast(dy_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dw.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(batch), ctypes.c_uint32(n_out), ctypes.c_uint32(n_in),
    )
    return dw


class TernaryLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_fp, scale):
        n_in = weight_fp.size(1)
        n_out = weight_fp.size(0)
        w_packed = pack_ternary(weight_fp)
        ctx.save_for_backward(x, w_packed)
        ctx.n_in = n_in
        ctx.n_out = n_out
        ctx.scale = scale

        x_reshaped = x.reshape(-1, n_in)
        out = gemm_forward(x_reshaped, w_packed, n_in, n_out, scale)
        return out.view(*x.shape[:-1], n_out)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_packed = ctx.saved_tensors
        n_in, n_out, scale = ctx.n_in, ctx.n_out, ctx.scale

        grad_reshaped = grad_output.reshape(-1, n_out)
        x_reshaped = x.reshape(-1, n_in)

        grad_x = gemm_backward_input(grad_reshaped, w_packed, n_in, n_out)
        grad_w = gemm_outer_product(grad_reshaped, x_reshaped, n_out, n_in)
        
        return (grad_x * scale).view(*x.shape[:-1], n_in), grad_w * scale, None
