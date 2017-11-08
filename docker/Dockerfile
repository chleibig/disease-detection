FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh && \
    echo "bd1655b4b313f7b2a1f2e15b7b925d03  miniconda.sh" | md5sum -c && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

COPY . /disease-detection
RUN $CONDA_DIR/bin/conda env update -n root -f /disease-detection/environment.yml     

COPY ./docker/.theanorc /root/.theanorc
COPY ./docker/keras.json /root/.keras/keras.json

WORKDIR /disease-detection

EXPOSE 8888
CMD jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
