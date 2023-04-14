## Install pyrealsense2

- Build
```
git clone https://github.com/IntelRealSense/librealsense.git
cd ./librealsense
mkdir build
cd build
cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true
make -j4
sudo make install
```

- Build on Jetson Nano
```
git clone https://github.com/IntelRealSense/librealsense.git
cd ./librealsense
mkdir build
cd build
cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=... -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j4
sudo make install
```
Source: *https://github.com/IntelRealSense/librealsense/issues/6964*

- Update your PYTHONPATH environment variable to add the path to the pyrealsense library
```
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc
```