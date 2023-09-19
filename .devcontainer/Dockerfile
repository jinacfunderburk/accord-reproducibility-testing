FROM docker.io/jupyter/datascience-notebook:2022-06-20

USER root

### System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        lmodern file curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /tmp/library-scripts

USER ${NB_USER}

RUN pip install nbgitpuller && \
    jupyter serverextension enable --py nbgitpuller --sys-prefix 

### Prints Jupyter server token when terminal is opened
RUN echo "echo \"Jupyter server token: \$(jupyter server list 2>&1 | grep -oP '(?<=token=)[[:alnum:]]*')\"" > ${HOME}/.get-jupyter-url.sh && \
    echo "sh \${HOME}/.get-jupyter-url.sh" >> ${HOME}/.bashrc

RUN pip install pytest pybind11

# Install skggm from develop branch (the main branch doesn't install properly)
RUN pip install git+https://github.com/skggm/skggm.git@develop

# Install the archived SPACE package
RUN wget https://cran.r-project.org/src/contrib/Archive/space/space_0.1-1.tar.gz
RUN R CMD INSTALL space_0.1-1.tar.gz