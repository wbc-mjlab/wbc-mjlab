"""Layout for policy export under ``<run_dir>/params/``.

Training checkpoints and ``wbc-mjlab-play`` write:

- ``latest.onnx`` — policy-only graph
- ``wbc_tracking_params.yaml`` — deploy interface (joint PD, obs layout, tracking metadata)
"""

PLAY_PARAMS_SUBDIR = "params"
PLAY_ONNX_LATEST_NAME = "latest.onnx"
PLAY_TRACKING_PARAMS_NAME = "wbc_tracking_params.yaml"
