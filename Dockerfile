ARG CUDA_VERSION="12.4.1"
ARG OS_VERSION="22.04"
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${OS_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------- packages -------------------------------- #

SHELL ["/bin/bash", "-c"]
ARG PYTHON_VERSION="3.11"
ARG PYTHON_MAJOR="3"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        vim \
        curl \
        htop \
        iotop \
        git \
        git-lfs \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_MAJOR}-pip \
        python${PYTHON_MAJOR}-setuptools \
        python${PYTHON_MAJOR}-wheel \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------- nvidia-container-toolkit ---------------------------------- #
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN apt-get update
RUN apt-get install -y nvidia-container-toolkit
# ---------------------------------- nvtop ---------------------------------- #

ARG OS_VERSION
RUN if [[ ${OS_VERSION} > "18.04" ]] ; then \
    apt-get update && \
    apt-get install -y --no-install-recommends nvtop ; else \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        libncurses5-dev \
        libncursesw5-dev && \
    git clone https://github.com/Syllo/nvtop.git && \
    mkdir -p nvtop/build && \
    cd nvtop/build && \
    cmake .. && \
    make && \
    make install && \
    cd ../../ && \
    rm -rf nvtop ; fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ------------------------------ python checks ------------------------------ #

ENV PYTHONUNBUFFERED=1
RUN python3 --version
RUN pip3 --version

# ------------------------------- user & group ------------------------------ #

RUN mkdir -p /package
COPY requirements.txt /package/requirements.txt

ARG USER_ID
ARG GROUP_ID
ARG NAME

# Create user and group only if they don’t exist
RUN groupadd --gid ${GROUP_ID} ${NAME} || true
RUN useradd \
    --no-log-init \
    --create-home \
    --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    -s /bin/bash ${NAME}

# Change permissions of workspace and package directory
ARG WORKDIR_PATH="/workspace"
RUN mkdir -p ${WORKDIR_PATH} && \
    chown -R ${USER_ID}:${GROUP_ID} ${WORKDIR_PATH} && \
    chown -R ${USER_ID}:${GROUP_ID} /package

USER ${NAME}
WORKDIR ${WORKDIR_PATH}

# ------------------------------- requirements ------------------------------ #

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -r /package/requirements.txt

# # Install the package itself (if applicable, otherwise remove this line)
# RUN python3 -m pip install --no-cache-dir ${WORKDIR_PATH}
