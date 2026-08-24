#!/usr/bin/env python3
import os
import time
import logging
import socket
import threading
import collections
from datetime import datetime, timedelta

import cv2
from termcolor import cprint
from jetson_utils import cudaMemcpy, cudaToNumpy, cudaFont

from nano_llm import NanoLLM, ChatHistory, remove_special_tokens
from nano_llm.utils import ArgParser, load_prompts, wrap_text
from nano_llm.plugins import VideoSource, VideoOutput

VIDEO_UDP_IP   = os.getenv("VIDEO_UDP_IP", "127.0.0.1")
VIDEO_UDP_PORT = int(os.getenv("VIDEO_UDP_PORT", "5005"))
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"[vision] will UDP-send to {VIDEO_UDP_IP}:{VIDEO_UDP_PORT}", flush=True)

# -------------------- Config (env + defaults) --------------------
ALERT_UDP_IP      = os.getenv("ALERT_UDP_IP", "127.0.0.1")      # classifier broadcast IP
ALERT_UDP_PORT    = int(os.getenv("ALERT_UDP_PORT", "5006"))    # classifier broadcast port
ALERT_KEYWORD     = os.getenv("ALERT_KEYWORD", "WILDFIRE")          # substring to match (case-insensitive)
ALERT_THRESHOLD   = int(os.getenv("ALERT_THRESHOLD", "1"))      # e.g., require 3 alerts
ALERT_WINDOW_SEC  = float(os.getenv("ALERT_WINDOW_SEC", "30"))  # count alerts within this window

CLIP_SECONDS      = float(os.getenv("CLIP_SECONDS", "1"))      # how much history to save
CLIP_DIR          = os.getenv("CLIP_DIR", "/out/clips")         # output directory for clips

# recording format (fallbacks if not inferred)
REC_FPS           = float(os.getenv("REC_FPS", "10"))
REC_WIDTH         = int(os.getenv("REC_WIDTH", "1280"))
REC_HEIGHT        = int(os.getenv("REC_HEIGHT", "720"))
REC_FOURCC        = os.getenv("REC_FOURCC", "mp4v")             # container/codec via OpenCV
# ----------------------------------------------------------------
# ---- Extra env for cooldown ----
ALERT_COOLDOWN_SEC = float(os.getenv("ALERT_COOLDOWN_SEC", "30"))
###NEW
SEND_INTERVAL_SEC = float(os.getenv("CLASSIFIER_SEND_INTERVAL", "3.0"))  # seconds between messages
LAST_SENT = 0

# parse args and set some defaults (original)
parser = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input', 'video_output'])
parser.add_argument("--max-images", type=int, default=8, help="the number of video frames to keep in the history")
args = parser.parse_args()

prompts = load_prompts(args.prompt) or ["What changes occurred in the video?"]
if not args.model:
    args.model = "Efficient-Large-Model/VILA1.5-3b"

print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model,
    api=args.api,
    quantization=args.quantization,
    max_context_len=args.max_context_len,
    vision_api=args.vision_api,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling,
)
assert model.has_vision

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# warm-up model
chat_history.append(role='user', text='What is 2+2?')
logging.info(f"Warmup response:  '{model.generate(chat_history.embed_chat()[0], streaming=False)}'".replace('\n','\\n'))
chat_history.reset()


class NullRecorder:
    def set_dims_if_needed(self, *args, **kwargs): pass
    def note_frame_interval(self, *args, **kwargs): pass
    def push(self, *args, **kwargs): pass
    def save_last(self, *args, **kwargs): return False


# -------------------- Circular Buffer + Recorder --------------------
class CircularFrameBuffer:
    """Stores (ts, frame_bgr) tuples; can dump last T seconds to a file."""
    def __init__(self, seconds=5.0, fps=15.0, width=1280, height=720, fourcc='mp4v'):
        self.seconds = float(seconds)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        # generously sized: seconds * fps * 1.5 to reduce drops if FPS fluctuates
        self.buffer = collections.deque(maxlen=int(self.seconds * self.fps * 2))
        self.lock = threading.Lock()
        self._fps_smoother = collections.deque(maxlen=60)  # rough fps estimation

    def push(self, ts, bgr):
        with self.lock:
            self.buffer.append((ts, bgr))
            # keep only last `seconds`
            cutoff = ts - self.seconds
            while self.buffer and self.buffer[0][0] < cutoff:
                self.buffer.popleft()

    def set_dims_if_needed(self, bgr):
        h, w = bgr.shape[:2]
        # first frame defines default dims/fps if env not set
        global REC_WIDTH, REC_HEIGHT
        if REC_WIDTH <= 0 or REC_HEIGHT <= 0:
            REC_WIDTH, REC_HEIGHT = w, h

    def note_frame_interval(self, dt):
        if dt > 0:
            self._fps_smoother.append(1.0 / dt)

    def current_fps(self):
        return sum(self._fps_smoother) / len(self._fps_smoother) if self._fps_smoother else self.fps

    def save_last(self, out_path, seconds=None):
        seconds = self.seconds if seconds is None else float(seconds)
        with self.lock:
            if not self.buffer:
                return False
            # gather frames within last `seconds`
            now_ts = self.buffer[-1][0]
            cutoff = now_ts - seconds
            frames = [f for (ts, f) in self.buffer if ts >= cutoff]
        if not frames:
            return False

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # resize if needed
        resized = []
        for f in frames:
            if (f.shape[1], f.shape[0]) != (self.width, self.height):
                f = cv2.resize(f, (self.width, self.height), interpolation=cv2.INTER_AREA)
            resized.append(f)

        fps = max(1.0, self.current_fps())
        writer = cv2.VideoWriter(out_path, self.fourcc, fps, (self.width, self.height))
        for f in resized:
            writer.write(f)
        writer.release()
        return True

# -------------------- UDP Alert Listener --------------------
class AlertCounter:
    """Counts alerts within a time window; triggers when threshold met."""
    def __init__(self, keyword="WILDFIRE", threshold=1, window_sec=30.0):
        self.keyword = keyword.lower()
        self.threshold = int(threshold)
        self.window = float(window_sec)
        self.ts = collections.deque()  # timestamps of recent matches
        self.lock = threading.Lock()
        self._trigger = False

    def feed(self, msg: bytes):
        text = msg.decode("utf-8", errors="ignore").lower().strip()
        if self.keyword in text:
            now = time.time()
            with self.lock:
                self.ts.append(now)
                cutoff = now - self.window
                while self.ts and self.ts[0] < cutoff:
                    self.ts.popleft()
                if len(self.ts) >= self.threshold:
                    self._trigger = True

    def consume_trigger(self):
        with self.lock:
            t = self._trigger
            self._trigger = False
            return t

def udp_listener(counter: AlertCounter, ip: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    print(f"[udp] listening for alerts on {ip}:{port}  keyword='{counter.keyword}'  threshold={counter.threshold}/{counter.window}s")
    while True:
        data, _ = sock.recvfrom(2048)
        counter.feed(data)

# -------------------- Video I/O + LLM overlay (original flow) --------------------
num_images = 0
last_image = None
last_text = ''

prev_push_ts = None

def on_video(image):

    global last_image,prev_push_ts

    ts = time.time()

     # ---- Convert current frame to numpy BGR for recorder ----
    try:
        np_rgba = cudaToNumpy(image)   # use *image*, not last_image
    except Exception as e:
        print(f"[on_video] cudaToNumpy failed: {e}", flush=True)
        return  # don't kill the whole pipeline

    frame_bgr = cv2.cvtColor(np_rgba, cv2.COLOR_RGBA2BGR)

    # set dimensions + FPS tracking
    recorder.set_dims_if_needed(frame_bgr)
    if prev_push_ts is not None:
        recorder.note_frame_interval(ts - prev_push_ts)
    prev_push_ts = ts

    recorder.push(ts, frame_bgr)

    # ---- Keep latest CUDA frame for the LLM loop ----
    last_image = cudaMemcpy(image)

    if last_text:
        font_text = remove_special_tokens(last_text)
        wrap_text(font, image, text=font_text, x=5, y=5, color=(120,215,21), background=font.Gray50)
    video_output(image)


video_source = VideoSource(**vars(args), cuda_stream=0)
video_source.add(on_video, threaded=False)
video_source.start()

video_output = VideoOutput(**vars(args))
video_output.start()

font = cudaFont()

# Recorder & alert machinery
#recorder = CircularFrameBuffer(seconds=CLIP_SECONDS, fps=REC_FPS, width=REC_WIDTH, height=REC_HEIGHT, fourcc=REC_FOURCC)

recorder = NullRecorder()
alert_counter = AlertCounter(keyword=ALERT_KEYWORD, threshold=ALERT_THRESHOLD, window_sec=ALERT_WINDOW_SEC)
threading.Thread(target=udp_listener, args=(alert_counter, ALERT_UDP_IP, ALERT_UDP_PORT), daemon=True).start()

# NEW: track last save time to avoid too many clips
_last_save_ts = 0.0

# For writing text lines (as in your original)
TEXT_OUT_PATH = os.getenv("TEXT_OUT_PATH", "data/out_txt")

# -------------------- Main loop --------------------
while True:
    if last_image is None:
        time.sleep(0.05)   # don't spin the CPU
        continue
    # # ---- push frame to circular buffer ----
    # ts = time.time()
    # frame_bgr = cv2.cvtColor(cudaToNumpy(last_image), cv2.COLOR_RGBA2BGR)
    # recorder.set_dims_if_needed(frame_bgr)

    # if prev_push_ts is not None:
    #     recorder.note_frame_interval(ts - prev_push_ts)
    # prev_push_ts = ts

    # recorder.push(ts, frame_bgr)

    # ---- LLM inference & overlay (original) ----
    chat_history.append('user', text=f'Image {num_images + 1}:')
    chat_history.append('user', image=last_image)

    last_image = None
    num_images += 3 ### NEW
 


    for prompt in prompts:
    	
        now = time.time()
        chat_history.append('user', prompt)
        embedding, _ = chat_history.embed_chat()

        print('>>', prompt)

        reply = model.generate(
            embedding,
            kv_cache=chat_history.kv_cache,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for token in reply:
            cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)
            if len(reply.tokens) == 1:
                last_text = token
            else:
                last_text = last_text + token

        chat_history.append('bot', reply)
        chat_history.pop(2)

        ###NEW
        
        SEND_EVERY = SEND_INTERVAL_SEC # send text every 2 seconds
        now = time.time()
        
        # --- SEND OVER UDP ---
        msg = (last_text or "").strip()
        if msg and now - LAST_SENT >= SEND_EVERY:
            try:
                sent = udp_sock.sendto(msg.encode("utf-8", errors="ignore"), (VIDEO_UDP_IP, VIDEO_UDP_PORT))
                print(f"[vision] UDP sent {sent} bytes to {VIDEO_UDP_IP}:{VIDEO_UDP_PORT}: {msg[:80]!r}", flush=True)
                LAST_SENT = now
            except Exception as e:
                print(f"[vision] UDP send error to {VIDEO_UDP_IP}:{VIDEO_UDP_PORT}: {e}", flush=True)

        # --- ALSO WRITE TO FILE (optional) ---
        try:
            os.makedirs(os.path.dirname(TEXT_OUT_PATH), exist_ok=True)
            with open(TEXT_OUT_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"[warn] could not append to {TEXT_OUT_PATH}: {e}")

    if num_images >= args.max_images:
        chat_history.reset()
        num_images = 0

    # ---- Did we reach alert threshold? Save clip (with cooldown). ----
    if alert_counter.consume_trigger():
        now = time.time()   # guaranteed defined here
        if now - _last_save_ts >= ALERT_COOLDOWN_SEC:
            _last_save_ts = now
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(CLIP_DIR, f"FIRE_{dt}.mp4")

            def _save_clip():
                ok = recorder.save_last(out_path, seconds=CLIP_SECONDS)
                print(f"[recorder] saved={ok} path={out_path}")

            threading.Thread(target=_save_clip, daemon=True).start()
        else:
            remaining = ALERT_COOLDOWN_SEC - (now - _last_save_ts)
            print(f"[recorder] cooldown active; skipping save ({remaining:.1f}s left)")

    if video_source.eos:
        video_output.stream.Close()
        break
