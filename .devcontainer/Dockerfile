# This uses multistage builds to customize performance for development and deployment.
# Any operating system that uses apt should be supported.
ARG OS=ubuntu:latest

# The base stage contains any dependency needed to both run and develop.
FROM ${OS} AS base
COPY requirements.txt /tmp/pip-tmp/
RUN apt update \
  && DEBIAN_FRONTEND=noninteractive \
  apt install -y --no-install-recommends \
  libboost-all-dev \
  # This is an optional dependency of GTSAM
  libtbb-dev \
  # Add Python for bindings.
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
  && rm -rf /tmp/pip-tmp

# The dev stage is used for development and includes all the compilers, git, and any dependency built from source.
FROM base AS dev
RUN apt update \
  && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  build-essential \
  clang-tidy \
  cmake \
  doxygen \
  gcovr \
  gdb \
  git \
  graphviz \
  texlive-latex-base \
  texlive-latex-extra \
  && rm -rf /var/lib/apt/lists/*
# Install GTSAM
WORKDIR /opt
ARG GTSAM_VERSION=develop
RUN git clone --branch ${GTSAM_VERSION} --depth 1 https://github.com/borglab/gtsam.git \
  # Make sure to use the newest compiler.
  && cmake -S gtsam -B gtsam/build \
  # See https://github.com/borglab/gtsam/blob/develop/INSTALL.md for configuration details
  -DCMAKE_BUILD_TYPE=Debug \
  -DGTSAM_BUILD_CONVENIENCE_LIBRARIES:OPTION=OFF \
  -DGTSAM_BUILD_UNSTABLE:OPTION=OFF \
  -DGTSAM_BUILD_PYTHON=OFF \
  && cmake --build gtsam/build --target install -- -j$(expr $(nproc) - 1) \
  && ldconfig
# Install OpenCV
ARG OPENCV_VERSION=4.x
RUN git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git \
  && cmake -S opencv -B opencv/build \
  # There are a ton of options listed at https://docs.opencv.org/4.5.5/db/d05/tutorial_config_reference.html
  # Anything not explicitly specified here is set to the default value.
  -DCMAKE_BUILD_TYPE=Debug \
  # We don't need all of OpenCV
  -DBUILD_LIST=core,features2d,flann \
  -DOPENCV_ENABLE_NONFREE=OFF \
  -DBUILD_opencv_python3=ON \
  # Build tests, samples, and applications
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  # Use CUDA, if available
  -DWITH_CUDA=OFF \
  # Enable OpenMP
  -DWITH_PTHREADS_PF=OFF \
  -DWITH_OPENMP=ON \
  && cmake --build opencv/build --target install -- -j$(expr $(nproc) - 1) \
  && ldconfig
WORKDIR /root
CMD [ "bash" ]

# The test stage compiles the source code and runs unit tests
FROM dev AS test
COPY . /opt/ground_texture_slam
WORKDIR /opt/ground_texture_slam/build
RUN cmake -S /opt/ground_texture_slam -B /opt/ground_texture_slam/build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  && make -j$(expr $(nproc) - 1)
# Add the build folder to the Python path so it can find the Python bindings.
ENV PYTHONPATH=/opt/ground_texture_slam/build
CMD [ "ctest", "-VV" ]

# The build stage compiles everything for deployment and installs it. It also compiles everything for Release mode.
FROM dev AS build
COPY . /opt/ground_texture_slam
# Uninstall GTSAM and install the Release version
RUN xargs rm -rf < /opt/gtsam/build/install_manifest.txt \
  && cmake -S /opt/gtsam -B /opt/gtsam/build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build /opt/gtsam/build --target install -- -j$(expr $(nproc) - 1)
# Uninstall OpenCV and install the Release version
RUN xargs rm -rf < /opt/opencv/build/install_manifest.txt \
  && cmake -S /opt/opencv -B /opt/opencv/build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build /opt/opencv/build --target install -- -j$(expr $(nproc) - 1)
# Lastly, install the compiled version of this package.
RUN cmake -S /opt/ground_texture_slam -B /opt/ground_texture_slam/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  && cmake --build /opt/ground_texture_slam/build --target install -- -j$(expr $(nproc) - 1)
CMD [ "bash" ]

# The final stage is for deployment. It is a minimal system based only on the dependencies and compiled code.
FROM base AS deploy
COPY --from=build /usr /usr
RUN ldconfig
# Make a local user to prevent security issues with admin rights.
RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user
# Copy in the example Python script for demonstration
COPY --from=build /opt/ground_texture_slam/src/simple_example.py /home/user/simple_example.py
CMD [ "simple_example" ]
