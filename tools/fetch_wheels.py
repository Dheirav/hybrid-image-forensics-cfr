#!/usr/bin/env python3
"""Segmented, resumable wheel downloader.

Why: single connections to pypi.nvidia.com and download.pytorch.org run at ~100 KB/s on this
machine, while several parallel range requests add up almost linearly, and uv restarts a wheel
from zero when a connection resets. Wheels land in wheelhouse/, then:
    uv pip install --python .venv/bin/python --find-links wheelhouse -r requirements.txt
Progress (bytes done / total, measured rate, ETA) is printed by `--status`.
"""
import json, os, re, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = "wheelhouse"
SEG = 8 * 1024 * 1024
PYPI = {  # pinned to what `uv pip install --dry-run` resolved
    "nvidia-cublas-cu12": "12.8.4.1", "nvidia-cuda-cupti-cu12": "12.8.90", "nvidia-cuda-nvrtc-cu12": "12.8.93",
    "nvidia-cuda-runtime-cu12": "12.8.90", "nvidia-cudnn-cu12": "9.10.2.21", "nvidia-cufft-cu12": "11.3.3.83",
    "nvidia-cufile-cu12": "1.13.1.3", "nvidia-curand-cu12": "10.3.9.90", "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-cusparse-cu12": "12.5.8.93", "nvidia-cusparselt-cu12": "0.7.1", "nvidia-nccl-cu12": "2.27.5",
    "nvidia-nvjitlink-cu12": "12.8.93", "nvidia-nvshmem-cu12": "3.4.5", "nvidia-nvtx-cu12": "12.8.90",
    "triton": "3.6.0", "scipy": "1.18.1", "opencv-python-headless": "5.0.0.93",
}
TORCH_INDEX = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128"}


def pypi_url(name, ver):
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=30))
    for f in d["urls"]:
        fn = f["filename"]
        if fn.endswith(".whl") and ("cp312" in fn or "py3-none" in fn or "abi3" in fn) and \
                ("manylinux" in fn and "x86_64" in fn or "none-any" in fn):
            return f["url"], fn
    raise SystemExit(f"no wheel for {name} {ver}")


def torch_url(name, ver):
    html = urllib.request.urlopen(f"https://download.pytorch.org/whl/cu128/{name}/", timeout=30).read().decode()
    want = ver.replace("+", "%2B")
    for m in re.finditer(r'href="([^"#]+)', html):
        h = m.group(1)
        if want in h and "cp312-cp312" in h and "x86_64" in h:
            url = h if h.startswith("http") else "https://download.pytorch.org" + h
            return url, os.path.basename(url.split("#")[0]).replace("%2B", "+")
    raise SystemExit(f"no wheel for {name} {ver}")


def fetch(url, fn, pool):
    path = os.path.join(OUT, fn)
    size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30).headers["Content-Length"])
    if os.path.exists(path) and os.path.getsize(path) == size and not os.path.exists(path + ".part"):
        return
    part, done = path + ".part", path + ".done"
    state = set(json.load(open(done))) if os.path.exists(done) else set()
    with open(part, "ab"):
        pass
    os.truncate(part, size)
    lock = threading.Lock()

    def seg(i):
        lo, hi = i * SEG, min(size, (i + 1) * SEG) - 1
        for attempt in range(20):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
                data = urllib.request.urlopen(req, timeout=60).read()
                if len(data) != hi - lo + 1:
                    raise IOError("short read")
                with lock, open(part, "r+b") as f:
                    f.seek(lo); f.write(data)
                    state.add(i); json.dump(sorted(state), open(done, "w"))
                    log(len(data))
                return
            except Exception:
                time.sleep(2)
        raise SystemExit(f"segment {i} of {fn} failed 20 times")

    list(pool.map(seg, [i for i in range((size + SEG - 1) // SEG) if i not in state]))
    os.replace(part, path); os.remove(done)


def log(n):
    with open(os.path.join(OUT, "_progress.json"), "r+") as f:
        s = json.load(f); s["done"] += n; f.seek(0); f.truncate(); json.dump(s, f)


def status():
    s = json.load(open(os.path.join(OUT, "_progress.json")))
    el = time.time() - s["t0"]
    # bytes already on disk when this run began are not counted as rate
    rate = (s["done"] - s["base"]) / el if el > 0 else 0
    left = s["total"] - s["done"]
    eta = f"ETA ~{int(left / rate)}s" if rate > 0 and s["done"] > s["base"] else "no ETA yet"
    print(f"{s['done']/2**20:.0f}/{s['total']/2**20:.0f} MiB  rate {rate/2**20:.2f} MiB/s  elapsed {int(el)}s  {eta}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if "--status" in sys.argv:
        status(); sys.exit()
    jobs = [pypi_url(n, v) for n, v in PYPI.items()] + [torch_url(n, v) for n, v in TORCH_INDEX.items()]
    todo, total, base = [], 0, 0
    for url, fn in jobs:
        size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30).headers["Content-Length"])
        total += size
        p = os.path.join(OUT, fn)
        if os.path.exists(p) and os.path.getsize(p) == size:
            base += size
        else:
            todo.append((url, fn))
            d = p + ".done"
            if os.path.exists(d):
                base += len(json.load(open(d))) * SEG
    json.dump({"total": total, "done": base, "base": base, "t0": time.time()}, open(os.path.join(OUT, "_progress.json"), "w"))
    print(f"{len(todo)} wheels to fetch, {total/2**20:.0f} MiB total, {base/2**20:.0f} MiB already present", flush=True)
    with ThreadPoolExecutor(16) as pool:
        for url, fn in todo:
            fetch(url, fn, pool)
    print("all wheels present")
