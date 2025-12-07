FROM ubuntu:latest
COPY . /app
WORKDIR /app
RUN make
