FROM ubuntu
RUN apt-get update
RUN apt-get install -y \
    git \
    cmake \
    gcc g++ \
    libpoco-dev libeigen3-dev



# LIBFRANKA
WORKDIR /

RUN git clone --recursive --branch 0.9.2 https://github.com/frankaemika/libfranka
RUN cd libfranka
RUN mkdir build
RUN cd build
WORKDIR /libfranka/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF ..
RUN cmake --build .
RUN cpack -G DEB
RUN dpkg -i libfranka*.deb


WORKDIR /

RUN apt-get install -y \
    python3.11 python3-pip pybind11-dev \
    catch2
RUN rm -f /usr/bin/python3
RUN ln -s /usr/bin/python3.11 /usr/bin/python3
RUN pip install pybind11

#panda-python 0.6.2 for libfranka 0.9.2 and python 3.11
RUN apt-get install wget unzip
RUN wget https://github.com/JeanElsner/panda-py/releases/download/v0.6.2/panda_py_0.6.2_libfranka_0.9.2.zip
RUN unzip panda_py_0.6.2_libfranka_0.9.2.zip
RUN pip install panda_python-0.6.2+libfranka.0.9.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Other python dependencies
RUN pip install numpy 
RUN pip install pyserial

