FROM gitpod/workspace-full

USER root

RUN apt-get -yq update && apt install -yq flex
