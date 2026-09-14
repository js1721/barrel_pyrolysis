style=$1
export OPENBLAS_NUM_THREADS=1

if [ "$style" == "debug" ]; then
    echo "makeing clean debug"
    rm -rf build
    cmake -B build -DCMAKE_BUILD_TYPE='Debug'
    cmake --build build
elif [ "$style" == "release" ]; then
    echo "makeing clean release"
    rm -rf build
    cmake -B build -DCMAKE_BUILD_TYPE='Release'
    cmake --build build
else
    echo "makeing clean debug"
    rm -rf build
    cmake -B build goit-DCMAKE_BUILD_TYPE='Debug'
    cmake --build build
fi