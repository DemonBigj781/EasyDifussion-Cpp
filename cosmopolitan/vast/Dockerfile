# Build with --platform linux/amd64 and a resolved Ubuntu manifest digest.
# NVIDIA's runtime mounts the host's matching driver; never install a driver
# in this image. graphics is necessary for Vulkan driver injection:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
# X11 client libraries satisfy the default libGLX_nvidia Vulkan ICD even when
# no display server is used; NVIDIA documents this dependency here:
# https://download.nvidia.com/XFree86/Linux-x86_64/580.126.09/README/installedcomponents.html
ARG UBUNTU_IMAGE
FROM ${UBUNTU_IMAGE} AS runtime-base
ARG UBUNTU_IMAGE
LABEL org.opencontainers.image.title="Easy Diffusion Cosmopolitan native Vulkan validation"
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONDONTWRITEBYTECODE=1
RUN test "$(uname -m)" = x86_64 && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 libvulkan1 ca-certificates libgcc-s1 libstdc++6 libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

FROM runtime-base AS helper-builder
ARG UBUNTU_IMAGE
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*
COPY payload/ /opt/edcpp-test/build/
COPY prepare_helper.py /opt/edcpp-test/prepare_helper.py
RUN /usr/bin/python3 /opt/edcpp-test/prepare_helper.py \
      --build-dir /opt/edcpp-test/build --base-image "$UBUNTU_IMAGE"

FROM runtime-base AS runtime-image
COPY --from=helper-builder /opt/edcpp-test/build/ /opt/edcpp-test/build/
COPY run_test.py /opt/edcpp-test/run_test.py
ENV TMPDIR=/opt/edcpp-test/build/native-tmp
WORKDIR /opt/edcpp-test

# This free CI-only target tests the exact payload/cache against host Mesa.
# It is not the image published for physical NVIDIA hardware.
FROM runtime-image AS preflight
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers && \
    rm -rf /var/lib/apt/lists/* && \
    /usr/bin/python3 -c 'import glob,os; p=glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"); assert len(p)==1,p; os.symlink(p[0], "/opt/edcpp-test/lavapipe-icd.json")'

# The default/published target contains neither Mesa's ICD nor a compiler.
FROM runtime-image AS runtime
