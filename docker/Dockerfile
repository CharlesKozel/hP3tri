# Stage 1: Build React frontend
FROM node:22-alpine AS web-build
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Build Kotlin application
FROM eclipse-temurin:21-jdk AS kotlin-build
WORKDIR /build
COPY gradlew gradlew.bat build.gradle.kts settings.gradle.kts ./
COPY gradle/ gradle/
COPY kotlin/ kotlin/
RUN ./gradlew :kotlin:installDist --no-daemon

# Stage 3: Runtime
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 via deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

# JDK 21 via Eclipse Adoptium (Jep needs JDK headers to compile)
RUN wget -qO- https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(. /etc/os-release && echo $VERSION_CODENAME) main" \
        > /etc/apt/sources.list.d/adoptium.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        temurin-21-jdk \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

# Python dependencies (jep compiles native extension against python3.12-dev + JDK headers)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /app

# Copy built Kotlin distribution
COPY --from=kotlin-build /build/kotlin/build/install/hp3tri /app/kotlin-dist

# Copy built React frontend
COPY --from=web-build /build/web/dist /app/web/dist

# Copy Python source
COPY python/ /app/python/

# Copy entrypoint
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Data volume for checkpoints, genomes, replays
RUN mkdir -p /app/data
VOLUME /app/data

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
