from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import os, time, requests

app = Flask(__name__)
g = Gauge("app_startup_seconds", "App startup time in seconds",
          ["app","cpus","memory", "heap"])

APP_NAME = os.getenv("APP_NAME", "demo")
TARGET = os.getenv("HEALTH_URL", "http://app:9080/healthz")
CPUS = os.getenv("CPUS", "")
MEM = os.getenv("MEMORY", "")
HEAP = os.getenv("HEAP", "")

measured = False
start_ts = time.time()

def is_ready_status(code: int) -> bool:
    """
    Decide whether the status code means "the app is up".
    Treat any non-5xx as up, so apps that return 401/403 for
    unauthenticated users are still considered ready.
    """
    return code < 500

def measure_once():
    global measured
    if measured:
        return
    try:
        r = requests.get(TARGET, timeout=0.5)
        if is_ready_status(r.status_code):
            elapsed = time.time() - start_ts
            g.labels(APP_NAME, CPUS, MEM, HEAP).set(elapsed)
            measured = True
    except Exception:
        pass


@app.route("/metrics")
def metrics():
    measure_once()
    return Response(generate_latest(REGISTRY), mimetype="text/plain")

@app.route("/")
def ok():
    return "sidecar ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9100)
