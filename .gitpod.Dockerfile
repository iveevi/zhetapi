FROM gitpod/workspace-full

USER root

RUN apt-get -yq update && apt install -yq flex && apt install -yq valgrind && apt install -yq libboost-all-dev
