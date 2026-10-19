#!/usr/bin/env python3
"""Fetch a small, deterministic COCO val2017 subset over a slow link.

Only instances_val2017.json is taken from the 241 MB annotations zip (HTTP range requests on the
zip directory), then only the selected images are downloaded one by one, resumable.
Licences: images carry per-image Flickr Creative Commons licences (kept in the manifest);
annotations are CC BY 4.0. Terms: https://cocodataset.org/#termsofuse
usage: fetch_coco_subset.py [--n 2000] [--status]
"""
import io, json, os, random, sys, threading, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = "data/coco_raw"
ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG = "http://images.cocodataset.org/val2017/{:012d}.jpg"
PROG = os.path.join(ROOT, "_progress.json")


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, so zipfile can open a remote zip."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for _ in range(10):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
                data = urllib.request.urlopen(req, timeout=120).read()
                self.pos += len(data)
                return data
            except Exception:
                time.sleep(3)
        raise IOError("range read failed")


def annotations():
    p = os.path.join(ROOT, "instances_val2017.json")
    if not os.path.exists(p):
        with zipfile.ZipFile(RangeFile(ANN_ZIP)) as z:
            open(p + ".tmp", "wb").write(z.read("annotations/instances_val2017.json"))
        os.replace(p + ".tmp", p)
    return json.load(open(p))


def select(ann, n, crop=384):
    """Images with a usable non-crowd polygon instance that fits fully inside a crop x crop window."""
    by_img = {}
    for a in ann["annotations"]:
        x, y, w, h = a["bbox"]
        if (not a["iscrowd"] and isinstance(a["segmentation"], list) and a["segmentation"]
                and 4000 <= a["area"] <= 0.2 * crop * crop and w <= crop - 40 and h <= crop - 40):
            by_img.setdefault(a["image_id"], []).append(a)
    imgs = {i["id"]: i for i in ann["images"]}
    ids = sorted(i for i in by_img if min(imgs[i]["width"], imgs[i]["height"]) >= crop)
    random.Random(0).shuffle(ids)
    keep = []
    for i in ids[:n]:
        a = max(by_img[i], key=lambda a: a["area"])
        keep.append({"image_id": i, "file": f"{i:012d}.jpg", "license": imgs[i]["license"],
                     "flickr_url": imgs[i]["flickr_url"], "width": imgs[i]["width"], "height": imgs[i]["height"],
                     "ann_id": a["id"], "bbox": a["bbox"], "area": a["area"], "segmentation": a["segmentation"]})
    return keep


def status():
    s = json.load(open(PROG))
    el = time.time() - s["t0"]
    done = len([f for f in os.listdir(os.path.join(ROOT, "images")) if f.endswith(".jpg")])
    new = done - s["base"]
    rate = new / el if el > 0 and new > 0 else 0
    eta = f"ETA ~{int((s['total'] - done) / rate)}s" if rate else "no ETA yet"
    print(f"{done}/{s['total']} images  rate {rate:.2f}/s  elapsed {int(el)}s  {eta}")


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2000
    os.makedirs(os.path.join(ROOT, "images"), exist_ok=True)
    keep = select(annotations(), n)
    json.dump(keep, open(os.path.join(ROOT, "selection.json"), "w"))
    have = {f for f in os.listdir(os.path.join(ROOT, "images")) if f.endswith(".jpg")}
    json.dump({"total": len(keep), "base": len(have), "t0": time.time()}, open(PROG, "w"))
    print(f"selected {len(keep)} images, {len(have)} already present", flush=True)

    def get(k):
        dst = os.path.join(ROOT, "images", k["file"])
        if os.path.exists(dst):
            return
        for _ in range(10):
            try:
                data = urllib.request.urlopen(IMG.format(k["image_id"]), timeout=60).read()
                open(dst + ".tmp", "wb").write(data)
                os.replace(dst + ".tmp", dst)
                return
            except Exception:
                time.sleep(3)
        print("failed", k["file"], flush=True)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, keep))
    print("done", flush=True)


if __name__ == "__main__":
    status() if "--status" in sys.argv else main()
