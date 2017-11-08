FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

ENV HOME /home/user
ENV CONDA_DIR $HOME/conda

RUN mkdir -p $HOME && \
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh && \
    echo "bd1655b4b313f7b2a1f2e15b7b925d03  miniconda.sh" | md5sum -c && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

COPY . $HOME/disease-detection
RUN $CONDA_DIR/bin/conda env update -n root -f $HOME/disease-detection/environment.yml     

COPY ./docker/.theanorc $HOME/.theanorc
COPY ./docker/keras.json $HOME/.keras/keras.json

RUN groupadd -r user && useradd --no-log-init -r -g user user && \
    chown -R user $HOME 

USER user
WORKDIR $HOME/disease-detection
ENV PATH $CONDA_DIR/bin:$PATH

EXPOSE 8888
CMD jupyter notebook --ip=0.0.0.0 --no-browser 
