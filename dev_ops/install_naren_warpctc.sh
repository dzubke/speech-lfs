# this script installs Sean Naren's warpctc loss function found here:
## https://github.com/SeanNaren/warp-ctc

cd libs
git clone https://github.com/SeanNaren/warp-ctc.git warp-ctc-naren

# build the warp-ctc library
cd warp-ctc-naren
mkdir build; cd build
cmake ..
make

# install the pytorch bindings
cd ../pytorch_binding
python setup.py install

# if an import error for `_warp_ctc.cpython-36m-x86_64-linux-gnu.so` is encoutered,
## run the commented code below
# pip install warpctc_pytorch
# cp ~/miniconda3/envs/pyt_16/lib/python3.6/site-packages/warpctc_pytorch-0.1-py3.6-linux-x86_64.egg/warpctc_pytorch/_warp_ctc.cpython-36m-x86_64-linux-gnu.so ~/awni_speech/speech-lfs/libs/warp-ctc-naren/pytorch_binding/warpctc_pytorch

# update the `LD_LIBRARY_PATH` variable
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc-naren/build
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/libs/warp-ctc-naren/build' >> ~/awni_speech/speech-lfs/setup.sh


