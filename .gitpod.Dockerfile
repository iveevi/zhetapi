FROM gitpod/workspace-full

USER root

RUN apt-get -yq update && apt-get install -yq gcc-8 g++-8 && apt install -yq flex && apt install -yq valgrind && apt install -yq libboost-all-dev && apt-get install -yq asciidoctor && apt-get install -yq libcurl4-gnutls-dev
