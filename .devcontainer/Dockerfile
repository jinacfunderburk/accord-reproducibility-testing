FROM jupyter/scipy-notebook:2022-06-20

USER root

ARG COPY_CONTEXT=.
COPY ${COPY_CONTEXT}/library-scripts/common-debian.sh /tmp/library-scripts/
COPY --chown="${NB_UID}:${NB_GID}" ${COPY_CONTEXT}/library-scripts/get-token.sh /opt/conda/bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lmodern file && \
    groupadd jovyan && \
    usermod -g jovyan -a -G users jovyan && \
    bash /tmp/library-scripts/common-debian.sh && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts

RUN curl https://cli.github.com/packages/githubcli-archive-keyring.gpg | apt-key add - && \
    echo "deb [arch=amd64] https://cli.github.com/packages stable main" > \
        /etc/apt/sources.list.d/github.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh

USER ${NB_UID}

RUN pip install pytest pybind11

RUN conda install xeus-cling -c conda-forge