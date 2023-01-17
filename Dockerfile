# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt-get update
RUN apt-get -y install nmap
RUN apt-get -y install traceroute


# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

#RUN pip install --no-cache-dir networkx scipy
#RUN pip install geopandas babypandas
RUN pip install --upgrade pip
COPY full_requirements.txt .
RUN pip install -r full_requirements.txt

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]