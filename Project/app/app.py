from flask import Flask
import time, os

app = Flask(__name__)
time.sleep(float(os.getenv("STARTUP_DELAY","3")))  # simulate startup work

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/")
def root():
    return "demo app", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
