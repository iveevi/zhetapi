FROM gitpod/workspace-full

RUN apt-get update \
    && apt-get install -y flex \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/* 
