"""Environment self-check for the Emo-CT / SEED experiments (Stage 0).

Verifies that the installed PyTorch can actually execute kernels on this GPU
(RTX 5060, sm_120 / Blackwell) *before* any GPU-hours are spent.

Run with the project environment:
    python emoct_seed/env_check.py
"""
import os
import sys


def main() -> int:
    ok = True
    print("python   :", sys.version.replace("\n", " "))

    try:
        import torch
    except Exception as exc:
        print("FATAL: cannot import torch:", exc)
        return 1

    print("torch    :", torch.__version__)
    print("cuda     :", torch.version.cuda)
    print("arch list:", torch.cuda.get_arch_list())

    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() is False")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    print("device   :", name)
    print("capability:", cap, "(expect (12, 0) for sm_120 / Blackwell)")
    print("vram     : %.2f GiB total, %.2f GiB free" % (props.total_memory / 1024 ** 3, free))

    if cap != (12, 0):
        print("WARNING: capability is not (12,0) - the wheel may not target this GPU")
        ok = False

    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        print("bf16     : ok (sum=%.3f)" % (a @ a).sum().item())
    except Exception as exc:
        print("FATAL: bf16 matmul failed:", exc)
        ok = False

    # The real input shape for the SEED runs: (batch, 1, 62 channels, 10 s @ 200 Hz).
    try:
        x = torch.zeros(4, 1, 62, 2000, device="cuda")
        conv = torch.nn.Conv2d(1, 8, (1, 50), padding="same").cuda()
        y = conv(x)
        print("conv     : ok", tuple(x.shape), "->", tuple(y.shape))
    except Exception as exc:
        print("FATAL: conv on (4,1,62,2000) failed:", exc)
        ok = False

    if free < 4.0:
        print("WARNING: only %.2f GiB free - close desktop apps before a long run" % free)

    for mod in ("numpy", "scipy", "pandas", "sklearn", "einops", "matplotlib"):
        try:
            __import__(mod)
            print("import   : %-10s ok" % mod)
        except Exception as exc:
            print("import   : %-10s MISSING (%s)" % (mod, exc))
            ok = False

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
