.PHONY: develop teardown env clean notebook-remote notebook-local

SHELL := /bin/bash

# Find the conda version in the path
ENV_NAME := "2023_06_26_SRT_deconvolution_MS"
ENV_FILE := environment.yaml
MAMBA_BIN := $(shell which mamba || echo .mamba/bin/mamba)
MAMBA_DIR := $(shell readlink -f `dirname $(MAMBA_BIN)`/..)
WITH_ENV := . $(MAMBA_DIR)/bin/activate $(ENV_NAME)

# Set up host info
PORT := $(shell shuf -i 5000-6000 -n 1)
HOST_IP = $(shell ip route get 1.2.3.4 | awk '{print $$7}')

# Data management
S3_DATA_DIR := s3://gh-bi-lunar/analysis/
S3_RESOURCES := s3://gh-bi-lunar/shared/deconvolution/
LOCAL_DATA_DIR := /analysis/gh-msun/ojects/2023_06_26_SRT_deconvolution_MS/

notebook-remote: # develop
	$(WITH_ENV) && jupyter lab --no-browser --port=$(PORT) --ip=$(HOST_IP)

notebook-local: develop
	$(WITH_ENV) && jupyter lab

get-conda:
ifeq ("", "$(wildcard $(MAMBA_BIN))")
	wget -O tmp_mamba.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
	bash tmp_mamba.sh -b -p .mamba/
	rm tmp_mamba.sh
endif

env-exist: get-conda
	$(MAMBA_BIN) env list | grep $(ENV_NAME) -q || $(MAKE) env

env: environment.yaml
	$(MAMBA_BIN) env create -f $(ENV_FILE)

# develop: env-exist
# 	$(WITH_ENV) && jupyter contrib nbextension install --user && jupyter nbextensions_configurator enable --user

s3-up:
	aws s3 sync $(LOCAL_DATA_DIR) $(S3_DATA_DIR)

s3-down:
	aws s3 sync $(S3_DATA_DIR) $(LOCAL_DATA_DIR)

s3-down-resources:
	aws s3 sync $(S3_RESOURCES) $(LOCAL_DATA_DIR)
clean:
	find . |  grep -E "(__pycache__|\.pyc$\)" | xargs rm -rf
	rm -rf *.egg-info/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf .ipynb_checkpoints/

teardown:
	$(MAMBA_BIN) env list | grep $(ENV_NAME) -q  && $(MAMBA_BIN) env remove -n $(ENV_NAME) || echo "No env '$(ENV_NAME)' in conda path"
