#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${1:-"$SOURCE_DIR/build-cuda"}
CUDA_ARCH=${CUDA_ARCH:-sm_86}
JOBS=${JOBS:-2}
if [[ -z ${CUDA_HOME:-} ]]; then
    NVCC_PATH=$(command -v nvcc)
    CUDA_HOME=$(cd "$(dirname "$NVCC_PATH")/.." && pwd)
    export CUDA_HOME
fi

mkdir -p "$BUILD_DIR"
for path in "$SOURCE_DIR"/*; do
    name=${path##*/}
    case "$name" in
        *.c|*.h|*.cu|*.inc|Makefile)
            ln -sfn "$path" "$BUILD_DIR/$name"
            ;;
    esac
done
ln -sfn "$SOURCE_DIR/tests" "$BUILD_DIR/tests"

printf 'source=%s\nbuild=%s\ncuda_home=%s\ncuda_arch=%s\njobs=%s\n' \
    "$SOURCE_DIR" "$BUILD_DIR" "$CUDA_HOME" "$CUDA_ARCH" "$JOBS"

timeout "${BUILD_TIMEOUT:-1200}" \
    make -C "$BUILD_DIR" -j"$JOBS" cuda CUDA_ARCH="$CUDA_ARCH"

for binary in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
    test -x "$BUILD_DIR/$binary"
done

"$BUILD_DIR/ds4-server" --help >/dev/null
ldd "$BUILD_DIR/ds4-server" | grep -E 'libcudart|libcublas'
