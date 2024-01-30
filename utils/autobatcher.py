import time
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def profile(input, ops, n=10, device=None):
    """ YOLOv3 speed/memory/FLOPs profiler
    Usage:
        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    def select_device():
        if torch.cuda.is_available() is True:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(f"- 🔃 - {'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s} ----")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x, ), verbose=False)[0] / 1E9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f'---- {p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s} ----')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

def check_train_batch_size(model, imagesize=640, amp=True):
    # Check training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imagesize)  # compute optimal batch size

def autobatch(model, imagesize=640, fraction=0.8, batch_size=16):
    # Automatically estimate best YOLOv3 batch size to use `fraction` of available CUDA memory

    # Check device

    print(f'- ↻ - Computing optimal batch size for imagesize: {imagesize} ----')
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        print(f'- ⚠️ - CUDA not detected, using default CPU batch-size {batch_size} ----')
        return batch_size
    if torch.backends.cudnn.benchmark:
        print(f'- ⚠️ - Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size} ----')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    print(f'- ↩ - {d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free ----')

    # Profile batch sizes
    if t <= 6:
        batch_sizes = [1, 2, 4, 8]
    elif t <= 16:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    try:
        img = [torch.empty(b, 3, imagesize, imagesize) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        print(f'{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        print(f'- WARNING ⚠️ - CUDA anomaly detected, recommend restart environment and retry command. ----')

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    print(f'- ✅ - Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ----')
    return b, t