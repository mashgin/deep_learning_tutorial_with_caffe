#https://hub.docker.com/r/ruimashita/caffe-cpu/ is the original file. Some small modifications, also because the pull doesnt work.

FROM ruimashita/scipy

MAINTAINER takuya.wakisaka@moldweorp.com

ENV PYTHONPATH /opt/caffe/python
ENV PATH $PATH:/opt/caffe/.build_release/tools
ENV CAFFE_VERSION=master
WORKDIR /opt/caffe

RUN apt-get update && apt-get install -y --no-install-recommends \
  # for caffe
  libprotobuf-dev \
  libleveldb-dev \
  libsnappy-dev \
  libopencv-dev \
  libhdf5-serial-dev \
  protobuf-compiler \
  libatlas-base-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  liblmdb-dev \
  libboost-all-dev \
  python-opencv \

  # for /opt/caffe/python/draw_net.py
  python-pydot \

  # for /opt/caffe/scripts/download_model_from_gist.sh
  unzip \

  # for /opt/caffe/tools/extra/parse_log.sh
  bsdmainutils \

  && apt-get clean \
  && rm -rf /var/lib/apt/lists/


RUN cd /opt && git clone https://github.com/BVLC/caffe.git && cd caffe && git checkout $CAFFE_VERSION && \

    cp Makefile.config.example Makefile.config && \
    echo "DEBUG := 1"  >> Makefile.config && \
    echo "WITH_PYTHON_LAYER := 1" >> Makefile.config && \
    sed -i -e 's|/usr/lib/python2.7/dist-packages/numpy/core/include|/usr/local/lib/python2.7/dist-packages/numpy/core/include|' Makefile.config && \
    echo "CPU_ONLY := 1" >> Makefile.config && \

    # Build Caffe core
    make -j"$(nproc)" all && \

    # Install python requirements
    for req in $(cat python/requirements.txt); do pip install --no-cache-dir $req; done && \

    # Build Caffe python
    make -j"$(nproc)" pycaffe && \

    # test
    make -j"$(nproc)" test && \

    # run test
    make -j"$(nproc)" runtest && \

    # pytest
    make -j"$(nproc)" pytest

ENV GLOG_minloglevel=2

#trying to fix the '(Image:30): Gtk-WARNING **: cannot open display:' warning
#ENV DISPLAY=172.17.0.1:0

WORKDIR /

# for bug "libdc1394 error: Failed to initialize libdc1394"
# RUN sudo ln /dev/null /dev/raw1394

