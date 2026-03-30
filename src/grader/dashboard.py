"""Web dashboard for the Robot Grader system.

Serves live MJPEG camera feeds and a settings editor.  Run with::

    python -m grader.dashboard
    # or
    python scripts/run_dashboard.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template, request

from grader.camera import CameraManager
from grader.config import load_config

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
_CAMERA_NAMES = ("overhead", "arm_head", "bottom")


def create_app(config_dir: str | None = None) -> Flask:
    cfg = load_config(config_dir=config_dir)
    mock = cfg.get("system", {}).get("mock_mode", True)

    cam = CameraManager(cfg, mock_mode=mock)
    cam.start()

    # Lock for thread-safe frame reads with real cameras.
    frame_lock = threading.Lock()

    app = Flask(__name__)

    # -- MJPEG streaming -------------------------------------------------------

    def gen_frames(camera_name: str):
        fps = cam._cameras_cfg[camera_name].get("fps", 30)
        interval = 1.0 / fps
        while True:
            with frame_lock:
                frame = cam._read_frame(camera_name)
            if frame is None:
                time.sleep(0.1)
                continue
            _, buf = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
            time.sleep(interval)

    @app.route("/")
    def index():
        return render_template("index.html", cameras=_CAMERA_NAMES)

    @app.route("/video_feed/<camera_name>")
    def video_feed(camera_name: str):
        if camera_name not in _CAMERA_NAMES:
            return "Unknown camera", 404
        return Response(
            gen_frames(camera_name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # -- Config API ------------------------------------------------------------

    @app.route("/api/config", methods=["GET"])
    def get_config():
        with open(_CONFIG_PATH) as f:
            raw = f.read()
        return Response(raw, mimetype="text/yaml")

    @app.route("/api/config", methods=["PUT"])
    def put_config():
        body = request.get_data(as_text=True)
        try:
            parsed = yaml.safe_load(body)
        except yaml.YAMLError as e:
            return jsonify({"error": f"Invalid YAML: {e}"}), 400

        if not isinstance(parsed, dict):
            return jsonify({"error": "Config must be a YAML mapping"}), 400

        with open(_CONFIG_PATH, "w") as f:
            yaml.safe_dump(parsed, f, default_flow_style=False, sort_keys=False)

        return jsonify({"status": "saved"})

    @app.route("/api/config/field", methods=["PUT"])
    def put_config_field():
        """Update a single dotted config field, e.g. {"key": "conveyor.speed_mm_per_sec", "value": 75}."""
        data = request.get_json(silent=True)
        if not data or "key" not in data or "value" not in data:
            return jsonify({"error": "Need {key, value}"}), 400

        with open(_CONFIG_PATH) as f:
            cfg_data = yaml.safe_load(f)

        keys = data["key"].split(".")
        target = cfg_data
        for k in keys[:-1]:
            if not isinstance(target, dict) or k not in target:
                return jsonify({"error": f"Key path not found: {data['key']}"}), 400
            target = target[k]
        target[keys[-1]] = data["value"]

        with open(_CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg_data, f, default_flow_style=False, sort_keys=False)

        return jsonify({"status": "saved", "key": data["key"], "value": data["value"]})

    return app


def main() -> None:
    app = create_app()
    print("Dashboard running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
