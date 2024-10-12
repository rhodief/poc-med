#!/bin/bash
echo "Iniciando Script: Senha #-> gpu-jupyter <-#"
echo "Ver https://hub.docker.com/r/cschranz/gpu-jupyter"
docker run --gpus all -it -p 8848:8888 -v $(pwd)/src:/home/jovyan/work --device /dev/snd:/dev/snd --name jupyter-med -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --shm-size 10G  --user root cschranz/gpu-jupyter:v1.3_cuda-10.2_ubuntu-18.04_python-only
echo "Terminado"