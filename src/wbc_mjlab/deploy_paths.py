"""Layout for policy export under ``<run_dir>/params/``.

Training checkpoints and ``wbc-mjlab-play`` write:

- ``policy.onnx`` — policy-only graph
- ``config.yaml`` — deploy interface (joint PD, obs layout, tracking metadata)
- ``motion_library.yaml`` — dataset name + clip list (built from loaded bundle on play)
- ``rsi_bin_stats.npz`` — optional adaptive RSI bin levels (when enabled)
"""

PLAY_PARAMS_SUBDIR = "params"
PLAY_POLICY_ONNX_NAME = "policy.onnx"
PLAY_CONFIG_NAME = "config.yaml"
PLAY_RSI_BIN_STATS_NAME = "rsi_bin_stats.npz"
PLAY_MOTION_LIBRARY_MANIFEST = "motion_library.yaml"
