FROM ubuntu:22.04

# Instale dependências básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxcb1 \
    libx11-6 \
    libxi6 \
    libxext6 \
    libxrender1 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

# Baixe e instale o FreeSurfer 7.4.1
RUN wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
    && tar -xzf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz -C /usr/local \
    && rm freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz

# Configure o ambiente
ENV FREESURFER_HOME=/usr/local/freesurfer
ENV PATH=$FREESURFER_HOME/bin:$PATH

# Copie a licença
COPY license.txt $FREESURFER_HOME/license.txt

# Configure o X11
ENV DISPLAY=:99