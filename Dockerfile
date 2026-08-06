# This Dockerfile can be used to build a Docker image suitable for tango projects.

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /stage

COPY . .
RUN pip install --no-cache-dir .[all]

WORKDIR /workspace

RUN rm -rf /stage/

ENTRYPOINT ["tango"]
