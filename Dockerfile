#
#
#       This Dockerfile build a docker image with all grmonty compilation
#       dependencies:
#           - CUDA (v9.2)
#           - GCC (v5.4)
#           - PGI-CE (v18.4)
#           - GSL
#
#       NOTE: This Dockerfile expects a pgilinux-2018-184-x86-64.tar.gz to
#       be present at the same directory, in order for the image to be build. You can
#       download it from here: https://www.pgroup.com/products/community.htm
#
#
# This file was adapted from https://github.com/Hopobcn/pgi-docker (last visited in
# 24/08/18), which is under the MIT License:
#
# MIT License
# Copyright (c) 2017-2018 Pau Farré
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM nvidia/cuda:9.2-devel-ubuntu16.04

ENV PGI_VERSION 18.4
ENV PGI_FILE pgilinux-2018-184-x86-64
ENV PGI_INSTALL_DIR /opt/pgi
ENV PGI_HOME ${PGI_INSTALL_DIR}/linux86-64/${PGI_VERSION}
ENV PGI_BIN_DIR ${PGI_HOME}/bin
ENV PGI_LIB_DIR ${PGI_HOME}/lib

ENV PGI_SILENT true
ENV PGI_ACCEPT_EULA accept
ENV PGI_INSTALL_TYPE "single"
ENV PGI_INSTALL_NVIDIA false
ENV PGI_INSTALL_AMD false
ENV PGI_INSTALL_JAVA false
ENV PGI_INSTALL_MPI false
ENV PGI_MPI_GPU_SUPPORT false

ADD ${PGI_FILE}.tar.gz /tmp
RUN /tmp/${PGI_FILE}/install && rm -rf /tmp/*
RUN apt-get update && apt-get install -y libgsl2 libgsl-dev

ENV PATH ${PGI_BIN_DIR}:${PATH}
ENV LD_LIBRARY_PATH ${PGI_LIB_DIR}:${LD_LIBRARY_PATH}
ENV CUDA_HOME /usr/local/cuda

WORKDIR /grmonty

CMD ["bash"]
