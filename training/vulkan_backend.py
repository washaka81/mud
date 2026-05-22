import ctypes
import os
import torch
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

_lib = None
_vulkan_available = False


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
        ("vb_gemv_forward",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32,
          ctypes.c_float, ctypes.c_uint8], ctypes.c_int),
        ("vb_gemv_backward_input",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32], ctypes.c_int),
        ("vb_outer_product",
         [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
          ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32], ctypes.c_int),
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

    def clear(self):
        self._cache.clear()
        self._version.clear()

    def __len__(self):
        return len(self._cache)


_packed_cache = _PackedCache()


def pack_ternary(weight):
    return _packed_cache.get(weight)


def clear_caches():
    """Limpia todas las caches (llamar después de cada paso de training)."""
    _packed_cache.clear()
    if _lib is not None:
        _lib.vb_clear_caches()


def gemv_forward(x, w_packed, n_in, n_out, scale):
    x_cont = x.contiguous()
    y = torch.zeros(n_out, dtype=x.dtype, device="cpu")
    _lib.vb_gemv_forward(
        ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
        ctypes.c_float(scale),
        ctypes.c_uint8(1 if _vulkan_available else 0),
    )
    return y


def gemv_backward_input(dy, w_packed, n_in, n_out):
    dy_cont = dy.contiguous()
    dx = torch.zeros(n_in, dtype=dy.dtype, device="cpu")
    _lib.vb_gemv_backward_input(
        ctypes.cast(dy_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(dx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
    )
    return dx


def outer_product(dy, x, n_out, n_in):
    dy_cont = dy.contiguous()
    x_cont = x.contiguous()
    dw = torch.zeros(n_out * n_in, dtype=dy.dtype, device="cpu")
    _lib.vb_outer_product(
        ctypes.cast(dy_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_cont.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(dw.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(n_out), ctypes.c_uint32(n_in),
    )
    return dw.view(n_out, n_in)


def _gemv_loop(x_flat, w_packed, n_in, n_out, scale):
    batch = x_flat.size(0)
    out = torch.zeros(batch, n_out)
    for b in range(batch):
        _lib.vb_gemv_forward(
            ctypes.cast(x_flat[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(out[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
            ctypes.c_float(scale),
            ctypes.c_uint8(1 if _vulkan_available else 0),
        )
    return out


def _backward_loop(grad_flat, x_flat, w_packed, n_in, n_out, scale):
    batch = grad_flat.size(0)
    grad_x = torch.zeros(batch, n_in)
    grad_w = torch.zeros(n_out, n_in)
    for b in range(batch):
        _lib.vb_gemv_backward_input(
            ctypes.cast(grad_flat[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(w_packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
            ctypes.cast(grad_x[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n_in), ctypes.c_uint32(n_out),
        )
        _lib.vb_outer_product(
            ctypes.cast(grad_flat[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(x_flat[b].data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(grad_w.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_uint32(n_out), ctypes.c_uint32(n_in),
        )
    return grad_x * scale, grad_w * scale


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

        x_flat = x.reshape(-1, n_in)
        if x_flat.size(0) == 1:
            out = gemv_forward(x_flat[0], w_packed, n_in, n_out, scale)
            return out.view(*x.shape[:-1], n_out)
        out = _gemv_loop(x_flat, w_packed, n_in, n_out, scale)
        return out.view(*x.shape[:-1], n_out)

    @staticmethod
    def backward(ctx, grad_output):
        x, w_packed = ctx.saved_tensors
        n_in, n_out, scale = ctx.n_in, ctx.n_out, ctx.scale

        grad_flat = grad_output.reshape(-1, n_out)
        x_flat = x.reshape(-1, n_in)

        if grad_flat.size(0) == 1:
            grad_x = gemv_backward_input(grad_flat[0], w_packed, n_in, n_out)
            grad_w = outer_product(grad_flat[0], x_flat[0], n_out, n_in)
            return (grad_x * scale).view(*x.shape[:-1], n_in), grad_w * scale, None

        grad_x, grad_w = _backward_loop(grad_flat, x_flat, w_packed, n_in, n_out, scale)
        return grad_x.view(*x.shape[:-1], n_in), grad_w, None
