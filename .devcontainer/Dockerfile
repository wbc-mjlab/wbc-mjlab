FROM ghcr.io/astral-sh/uv:0.11.25 AS uv
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl ffmpeg git less make openssh-client \
    patchelf procps python3 python3-dev python3-venv sudo \
    libegl1 libgl1 libgles2 libglvnd0 libglx0 libopengl0 libosmesa6 \
    libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/

ARG USER_UID=1000
ARG USER_GID=1000
# Ubuntu's image may already reserve UID 1000 for its unused ubuntu account.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && test "$USER_UID" -gt 0 && test "$USER_GID" -gt 0 \
    && (getent group "$USER_GID" || groupadd --gid "$USER_GID" developer) \
    && useradd --uid "$USER_UID" --gid "$USER_GID" --create-home --shell /bin/bash developer \
    && echo 'developer ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer \
    && mkdir -p /opt/venv /home/developer/.cache /home/developer/.config/wbc-mjlab \
    && chown -R "$USER_UID:$USER_GID" /opt/venv /home/developer

# Keep the container interpreter separate from an existing host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TORCHINDUCTOR_CACHE_DIR=/home/developer/.cache/torch/inductor \
    TRITON_CACHE_DIR=/home/developer/.cache/triton \
    CUDA_CACHE_PATH=/home/developer/.cache/nvidia \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl
USER developer
WORKDIR /workspaces/wbc-mjlab
CMD ["sleep", "infinity"]
