"""Layout for ONNX export from ``wbc_mjlab.scripts.play``.

For whichever task you pass on the CLI, policy ONNX is written to
``<run_dir>/params/latest.onnx`` right after the checkpoint is loaded, before the
play viewer opens.
"""

PLAY_PARAMS_SUBDIR = "params"
PLAY_ONNX_LATEST_NAME = "latest.onnx"
