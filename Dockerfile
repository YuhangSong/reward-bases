# docker build -t dryuhangsong/reward-bases:latest -f Dockerfile .
# docker run --rm -it dryuhangsong/reward-bases:latest
# docker push dryuhangsong/reward-bases:latest

FROM rayproject/ray-ml:2.4.0-py38-gpu

USER root

RUN conda install python=3.8 -y

RUN pip install --upgrade pip

RUN pip install ray[all] torch torchvision torchaudio seaborn tqdm visdom tabulate statsmodels h5py

RUN sudo apt-get update && sudo apt-get install julia -y

RUN sudo apt update && sudo apt upgrade git -y

RUN julia --eval 'using Pkg; Pkg.add(["MAT", "Plots", "GLM", "DataFrames", "Distributions", "StatsBase", "HypothesisTests", "StatsPlots", "Statistics", "NPZ", "JLD2"])'

RUN pip install -U seaborn
