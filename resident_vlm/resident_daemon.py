#!/usr/bin/env python3
"""Resident VLM daemon: loads VILA once, evicts weights to the SD while asleep,
streams them back on wake. Speaks fire_watcher's resident wake protocol.

Protocol on the unix socket, one command per connection, JSON line reply:
    WAKE   -> reload weights from the card, open the camera, first verdict
    SLEEP  -> close the camera, evict weights, fadvise the fused file away
    STATS  -> current state without changing it
    QUIT   -> exit
"""
import argparse
import gc
import hashlib
import json
import os
import socket
import threading
import time

import torch
from jetson_utils import videoSource
from nano_llm import NanoLLM, ChatHistory


def diskstats(dev):
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return {"sectors_read": int(p[5]), "sectors_written": int(p[9])}
    raise KeyError(dev)


def gpu_mb():
    return round(torch.cuda.memory_allocated() / 1e6)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024)


def meta_params(module):
    return sum(1 for p in module.parameters() if p.device.type == "meta")


class Resident:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.state = "LOADING"
        self.camera = None
        self.loop_thread = None
        self.verdicts = 0
        self.fuse_calls = 0
        self._patch_fuse()

        t0 = time.monotonic()
        self.model = NanoLLM.from_pretrained(
            args.model, api="awq", quantization=args.quantization, vision_api="hf",
        )
        self.load_seconds = round(time.monotonic() - t0, 2)
        assert self.model.has_vision, "model has no vision tower"
        self.stamp = self._stamp()
        self._ensure_fused()
        self.state = "AWAKE"

    def _patch_fuse(self):
        import nano_llm.models.awq as awq
        orig = awq.make_quant_attn

        def counting(*a, **k):
            self.fuse_calls += 1
            return orig(*a, **k)

        awq.make_quant_attn = counting

    def _stamp(self):
        st = os.stat(self.args.quantization)
        raw = f"{self.args.model}|{os.path.basename(self.args.quantization)}|{st.st_size}|{torch.__version__}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _ensure_fused(self):
        meta_path = self.args.fused + ".meta.json"
        if os.path.exists(self.args.fused) and os.path.exists(meta_path):
            with open(meta_path) as f:
                have = json.load(f)
            if have.get("stamp") == self.stamp:
                self.fused_save_seconds = None
                return
            raise SystemExit(
                f"fused checkpoint stamp mismatch: file={have.get('stamp')} expected={self.stamp}\n"
                f"delete {self.args.fused} to rebuild it"
            )
        os.makedirs(os.path.dirname(self.args.fused), exist_ok=True)
        t0 = time.monotonic()
        torch.save({k: v.cpu() for k, v in self.model.model.state_dict().items()}, self.args.fused)
        self.fused_save_seconds = round(time.monotonic() - t0, 2)
        with open(meta_path, "w") as f:
            json.dump({"stamp": self.stamp, "model": self.args.model}, f)

    def _paced_preread(self, pace_mbps):
        budget = pace_mbps * 1e6
        chunk = self.args.chunk_mb << 20
        t0 = time.monotonic()
        total = 0
        with open(self.args.fused, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                total += len(b)
                target = t0 + total / budget
                now = time.monotonic()
                if target > now:
                    time.sleep(target - now)
        return round(time.monotonic() - t0, 3), total

    def _prefetch(self, prefetch_mb):
        chunk = prefetch_mb << 20
        t0 = time.monotonic()
        total = 0
        with open(self.args.fused, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                total += len(b)
        self.prefetch_stats = (round(time.monotonic() - t0, 3), total)

    def _load_from_card(self):
        sd = torch.load(self.args.fused, map_location="cpu", mmap=True)
        report = self.model.model.load_state_dict(sd, assign=True, strict=False)
        self.model.model.to("cuda")
        torch.cuda.synchronize()
        del sd
        gc.collect()
        return len(report.missing_keys), len(report.unexpected_keys)

    def _open_camera(self):
        opts = {"width": self.args.width, "height": self.args.height, "codec": self.args.codec}
        self.camera = videoSource(self.args.video_input, options=opts)

    def _close_camera(self):
        if self.camera is not None:
            self.camera.Close()
            self.camera = None

    def _infer_once(self):
        image = self.camera.Capture(format="rgb8", timeout=self.args.capture_timeout)
        if image is None:
            return None
        chat = ChatHistory(self.model)
        chat.append(role="user", image=image)
        chat.append(role="user", text=self.args.prompt)
        reply = self.model.generate(
            chat.embed_chat()[0], streaming=False, max_new_tokens=self.args.max_new_tokens,
        )
        return str(reply).strip()

    def _awake_loop(self):
        while self.state == "AWAKE":
            t0 = time.monotonic()
            try:
                text = self._infer_once()
            except Exception as exc:
                self.log("infer_error", error=repr(exc)[:200])
                break
            if text is None:
                continue
            self.verdicts += 1
            self._emit(text)
            self.log("verdict", seconds=round(time.monotonic() - t0, 2),
                     n=self.verdicts, text=text[:200])

    def _emit(self, text):
        with open(self.args.text_out, "a") as f:
            f.write(text.replace("\n", " ") + "\n")

    def log(self, event, **fields):
        rec = {"wall": time.time(), "state": self.state, "event": event, **fields}
        with open(self.args.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def wake(self, method=None):
        methods = {None: (self.args.pace_mbps, self.args.prefetch_mb),
                   "BURST": (0.0, self.args.chunk_mb),
                   "CONTINUOUS": (float("inf"), 0)}
        if method not in methods:
            return {"error": f"unknown wake method {method!r}"}
        pace_mbps, prefetch_mb = methods[method]
        with self.lock:
            if self.state == "AWAKE":
                return {"error": "already awake"}
            d0 = diskstats(self.args.dev)
            t0 = time.monotonic()
            preread_s, preread_b = (self._paced_preread(pace_mbps)
                                    if pace_mbps > 0 else (None, None))
            self.prefetch_stats = (None, None)
            pf = None
            if prefetch_mb > 0:
                pf = threading.Thread(target=self._prefetch, args=(prefetch_mb,), daemon=True)
                pf.start()
            t_load = time.monotonic()
            missing, unexpected = self._load_from_card()
            if pf is not None:
                pf.join(timeout=120)
            sd_read = round(time.monotonic() - t_load, 3)
            t1 = time.monotonic()
            self.state = "AWAKE"
            try:
                self._open_camera()
            except Exception:
                self.log("camera_open_failed")
                raise
            cam_open = round(time.monotonic() - t1, 3)
            t2 = time.monotonic()
            text = self._infer_once()
            first_verdict = round(time.monotonic() - t2, 3)
            if text is not None:
                self.verdicts += 1
                self._emit(text)
            d1 = diskstats(self.args.dev)
            out = {
                "method": method,
                "preread_seconds": preread_s,
                "preread_bytes": preread_b,
                "prefetch_seconds": self.prefetch_stats[0],
                "prefetch_bytes": self.prefetch_stats[1],
                "sd_read_seconds": sd_read,
                "camera_open_seconds": cam_open,
                "first_verdict_seconds": first_verdict,
                "wake_seconds": round(time.monotonic() - t0, 3),
                "bytes_read": (d1["sectors_read"] - d0["sectors_read"]) * 512,
                "missing_keys": missing, "unexpected_keys": unexpected,
                "fuse_calls_total": self.fuse_calls,
                "params_still_meta": meta_params(self.model.model),
                "gpu_mb": gpu_mb(), "rss_mb": rss_mb(),
                "verdict": (text or "")[:200],
            }
            self.log("awake", **out)
            self.loop_thread = threading.Thread(target=self._awake_loop, daemon=True)
            self.loop_thread.start()
            return out

    def sleep(self):
        with self.lock:
            if self.state == "SLEEPING":
                return {"error": "already asleep"}
            self.state = "SLEEPING"
            if self.loop_thread is not None:
                self.loop_thread.join(timeout=30)
                self.loop_thread = None
            t0 = time.monotonic()
            self._close_camera()
            cam_close = round(time.monotonic() - t0, 3)
            gpu_before, rss_before = gpu_mb(), rss_mb()
            self.model.model.to("meta")
            gc.collect()
            torch.cuda.empty_cache()
            os.sync()
            fd = os.open(self.args.fused, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            out = {
                "camera_close_seconds": cam_close,
                "sleep_seconds": round(time.monotonic() - t0, 3),
                "gpu_mb_freed": gpu_before - gpu_mb(),
                "rss_mb_freed": rss_before - rss_mb(),
                "gpu_mb_asleep": gpu_mb(), "rss_mb_asleep": rss_mb(),
                "params_evicted": meta_params(self.model.model),
            }
            self.log("slept", **out)
            return out

    def stats(self):
        return {"state": self.state, "gpu_mb": gpu_mb(), "rss_mb": rss_mb(),
                "verdicts": self.verdicts, "fuse_calls_total": self.fuse_calls,
                "load_seconds": self.load_seconds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sock", required=True)
    p.add_argument("--fused", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--model", default="Efficient-Large-Model/VILA1.5-3b-AWQ")
    p.add_argument("--quantization", default=os.environ.get("QPATH"))
    p.add_argument("--video-input", default="/dev/video0")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--codec", default="mjpeg")
    p.add_argument("--capture-timeout", type=int, default=5000)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--dev", default="nvme0n1")
    p.add_argument("--text-out", default=os.environ.get("TEXT_OUT_PATH", "/data/out_txt"))
    p.add_argument("--prompt", default="describe the scene in detail, and what is happening in it. "
                                       "do you see any signs of smoke? is smoke present?")
    p.add_argument("--pace-mbps", type=float, default=0.0)
    p.add_argument("--prefetch-mb", type=int, default=0)
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument("--sleep-on-start", action="store_true")
    args = p.parse_args()

    res = Resident(args)
    res.log("loaded", load_seconds=res.load_seconds, gpu_mb=gpu_mb(), rss_mb=rss_mb(),
            fused_save_seconds=res.fused_save_seconds, stamp=res.stamp)
    if args.sleep_on_start:
        res.sleep()

    if os.path.exists(args.sock):
        os.unlink(args.sock)
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(args.sock)
    os.chmod(args.sock, 0o777)
    srv.listen(4)
    print(f"[resident] ready on {args.sock} state={res.state}", flush=True)

    handlers = {"WAKE": res.wake, "SLEEP": res.sleep, "STATS": res.stats}
    while True:
        conn, _ = srv.accept()
        try:
            parts = conn.makefile().readline().strip().upper().split()
            cmd = parts[0] if parts else ""
            if cmd == "QUIT":
                conn.sendall(b'{"ok": true}\n')
                break
            fn = handlers.get(cmd)
            try:
                reply = fn(*parts[1:]) if fn else {"error": f"unknown command {cmd!r}"}
            except Exception as exc:
                reply = {"error": repr(exc)[:300]}
                res.log("command_failed", command=cmd, error=reply["error"])
            conn.sendall(json.dumps(reply).encode() + b"\n")
        finally:
            conn.close()
    srv.close()


if __name__ == "__main__":
    main()
