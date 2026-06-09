"""Layout for policy export under ``<run_dir>/params/``.

Training checkpoints and ``wbc-mjlab-play`` write:

- ``policy.onnx`` — policy-only graph
- ``config.yaml`` — deploy interface (joint PD, obs layout, tracking metadata)
"""

PLAY_PARAMS_SUBDIR = "params"
PLAY_POLICY_ONNX_NAME = "policy.onnx"
PLAY_CONFIG_NAME = "config.yaml"
