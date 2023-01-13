FROM registry-svc:25000/library/ubuntu_py3.8.1_torch8.1_cuda11:v1.0.0



# time zone set
#WORKDIR /usr/share
#ADD ./zoneinfo ./zoneinfo
#RUN  ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
#RUN echo "Asia/Shanghai" > /etc/timezone

# 创建目录
RUN mkdir /ntt
RUN mkdir /ntt/alphamind
RUN mkdir /ntt/tensorboard
RUN mkdir /ntt/datasets

# 复制文件
WORKDIR /opt
#ADD ./config ./config
ADD ./log ./log
ADD ./utils ./utils
ADD .env .
ADD flasktest.py .
#ADD flasktest.txt .
ADD release.sh .
ADD server.py .
ADD server.sh .
ADD train.py .
ADD train.sh .



