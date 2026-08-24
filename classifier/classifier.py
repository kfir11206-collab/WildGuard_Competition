#!/usr/bin/env python3
import os, sys, json, time, socket

# keep your original HF backend disables
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

sys.modules["triton"] = None

from transformers import pipeline

# ---- config (env) ----
UDP_IP       = os.environ.get("CLASSIFIER_UDP_IP", "0.0.0.0")   # listen
UDP_PORT     = int(os.environ.get("CLASSIFIER_UDP_PORT", "5005"))
OUTPUT_FILE  = os.environ.get("OUTPUT_FILE", "/data/classifier_results.jsonl")
THRESHOLD    = float(os.environ.get("THRESHOLD", "0.5"))

# FIRE broadcast over UDP (optional)
ALERT_ENABLE = os.environ.get("CLASSIFIER_ALERT_UDP_ENABLE", "0") == "1"
ALERT_IP     = os.environ.get("CLASSIFIER_ALERT_UDP_IP", "127.0.0.1")
ALERT_PORT   = int(os.environ.get("CLASSIFIER_ALERT_UDP_PORT", "5006"))

# hypotheses (unchanged)

FIRE_SENTENCES = ["A fast-spreading wildfire is burning through dry vegetation in an open outdoor area.",
"Flames are moving uncontrollably across a forest or grassland, consuming natural fuel.",
"A large outdoor fire is advancing with high heat and strong winds, typical of a wildfire.",
"Smoke plumes are rising from a wildfire spreading across trees and brush.",
"A natural landscape is ignited and expanding rapidly, consistent with a wildfire event.",
"Uncontrolled flames are traveling across wildland terrain, indicating a wildfire.",
"A wildfire is actively burning in a rural or forested region, with vegetation as fuel.",
"A spreading blaze is engulfing brush, scrub, or forest, characteristic of a wildfire.",
"High-intensity flames are sweeping across a natural area in a typical wildfire pattern.",
"An uncontrolled outdoor fire is spreading laterally through dry vegetation, indicating a wildfire."]


WILDFIRE_LABELS = [
    "a wildfire",
    "a forest fire",
    "a brush fire",
    "a grassland wildfire",
    "a plume of smoke"
]


NO_FIRE_LABELS = [
    "no fire",
    "no flames",
    "no wildfire",
    "clouds"
    "fog"
]

ALL_LABELS = WILDFIRE_LABELS + NO_FIRE_LABELS

H_TEMPLATE = "There is {} in this scene."



H_FIRE   = "This sentence indicates there is fire or smoke."
H_NOFIRE = "This sentence indicates there is no fire or smoke."

# --- extra alert options ---
ALERT_PLAIN  = os.environ.get("CLASSIFIER_ALERT_PLAIN", "1") == "1"

# global alert socket (lazy-created)
alert_sock = None
ALERT = "WILDFIRE"


def send_alert_if_fire(result):
    """Send JSON (and optional 'FIRE') over UDP if label == FIRE and alerts enabled."""
    if result.get("label") != ALERT:
        return
    global alert_sock

    # lazily create the socket once
    if alert_sock is None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # enable broadcast automatically when dest looks like broadcast
        if ALERT_IP.endswith(".255") or ALERT_IP == "255.255.255.255":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        alert_sock = s
        print(f"[wildfire_classifier] FIRE alerts -> {ALERT_IP}:{ALERT_PORT} "
              f"(broadcast={'yes' if s.getsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST) else 'no'})",
              flush=True)

    try:
        payload = json.dumps(result).encode("utf-8")
        alert_sock.sendto(payload, (ALERT_IP, ALERT_PORT))
        if ALERT_PLAIN:
            alert_sock.sendto(b"WILDFIRE", (ALERT_IP, ALERT_PORT))
        # optional: log a one-liner
        # print(f"[alert] sent FIRE to {ALERT_IP}:{ALERT_PORT}", flush=True)
    except Exception as e:
        print(f"[wildfire_classifier] UDP alert send error to {ALERT_IP}:{ALERT_PORT}: {e}", flush=True)



def load_model():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        framework="pt",
        device=-1  # CPU; keep as in your original
    )

def classify_wildfire(ppl, text, T_WILDFIRE=0.5):
    res = ppl(text, candidate_labels=ALL_LABELS, hypothesis_template=H_TEMPLATE)
    labels = res["labels"]
    scores = res["scores"]
    m = dict(zip(labels, scores))

    wildfire_score = max(m[l] for l in WILDFIRE_LABELS)
    no_fire_score = max(m[l] for l in NO_FIRE_LABELS)


    is_wildfire = (wildfire_score >= T_WILDFIRE) 
    return {
        "text": text,
        "wildfire_score": wildfire_score,
        "no_fire_score": no_fire_score,
        "label": "WILDFIRE" if is_wildfire else "NO_WILDFIRE",
        "timestamp": time.time(),
    }

def classify(ppl, text):
    res = ppl(text, candidate_labels=[H_FIRE, H_NOFIRE], hypothesis_template="{}")
    labels = res["labels"]; scores = res["scores"]
    m = dict(zip(labels, scores))
    fire_score   = m.get(H_FIRE, 0.0)
    nofire_score = m.get(H_NOFIRE, 0.0)
    fire = fire_score >= THRESHOLD and fire_score >= nofire_score
    return {
        "text": text,
        "fire_score": fire_score,
        "no_fire_score": nofire_score,
        "label": "FIRE" if fire else "NO_FIRE",
        "timestamp": time.time()
    }

def ensure_out_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def main():
    ppl = None
    ensure_out_dir(OUTPUT_FILE)

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind((UDP_IP, UDP_PORT))
    print(f"[wildfire_classifier] UDP listening on {UDP_IP}:{UDP_PORT}", flush=True)

   

    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfp:
        while True:
            data, addr = in_sock.recvfrom(4096)
            try:
                text = data.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not text:
                continue

            if ppl is None:
                ppl = load_model()

            result = classify_wildfire(ppl, text)
            # console one-liner (same style)
            #print(f"[{result['label']}] {result['text']} (p_fire={result['wildfire_score']:.2f})", flush=True)
            print(
            f"[{result['label']}] {result['text']} "
            f"(p_wildfire={result['wildfire_score']:.2f}, "
            f"p_nofire={result['no_fire_score']:.2f})",
            flush=True
            )
            # persist JSONL
            outfp.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfp.flush()
            send_alert_if_fire(result)

if __name__ == "__main__":
    try:
        print(f"[config] listen={UDP_IP}:{UDP_PORT}  out_file={OUTPUT_FILE}  "
              f"threshold={THRESHOLD}  alert_enable={ALERT_ENABLE}  "
              f"alert_dest={ALERT_IP}:{ALERT_PORT}  alert_plain={ALERT_PLAIN}", flush=True)
        main()
    except KeyboardInterrupt:
        pass