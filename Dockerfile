FROM ubuntu:bionic-20190515

#################################################
# install system packages
RUN apt-get update \
    && apt-get install -y curl git openjdk-8-jre python3 python3-pip unzip vim

#################################################
# copy spojitr installation and demo
COPY docker_install_3rd_party.sh requirements.txt /data/tmp/
COPY spojitr_install /data/spojitr_install
COPY demo /data/demo

#################################################
# additional python dependencies
RUN pip3 install -r /data/tmp/requirements.txt

#################################################
# 3rd party: weka, spojit, git setup, etc
WORKDIR /data/tmp
RUN /data/tmp/docker_install_3rd_party.sh

#################################################
# install spojitr
RUN /data/spojitr_install/install.py

WORKDIR /root
