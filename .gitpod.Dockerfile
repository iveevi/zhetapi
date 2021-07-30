FROM gitpod/workspace-full

USER root

RUN apt-get -yq update \
	&& apt-get install -yq gcc-8 g++-8 \
	&& apt-get install -yq clang-8 \
	&& apt-get install -yq valgrind \
	&& apt-get install -yq libboost-all-dev \
	&& apt-get install -yq asciidoctor \
	&& apt-get install -yq libcurl4-gnutls-dev \
	&& apt-get install -yq doxygen \
	&& apt-get install -yq texlive-latex-base \
	&& apt-get install -yq texlive-fonts-recommended \
	&& apt-get install -yq texlive-fonts-extra \
	&& apt-get install -yq texlive-latex-extra \
	&& apt-get install -yq graphviz \
	&& apt-get install -yq clang-tidy-8 \
	&& apt-get install -yq lcov \
	&& apt-get install -yq ninja-build \
	&& apt-get install -yq libsfml-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*
