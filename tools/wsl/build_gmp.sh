#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

GMP_VERSION=${GMP_VERSION:-6.3.0}
GMP_TARBALL="gmp-${GMP_VERSION}.tar.xz"
GMP_URL=${GMP_URL:-"https://gmplib.org/download/gmp/${GMP_TARBALL}"}

DOWNLOAD_DIR=${DOWNLOAD_DIR:-"${REPO_ROOT}/build/downloads"}
SOURCE_ROOT=${SOURCE_ROOT:-"${REPO_ROOT}/build/src"}
SOURCE_DIR="${SOURCE_ROOT}/gmp-${GMP_VERSION}"
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/gmp-build"}
PREFIX_DIR=${PREFIX_DIR:-"${REPO_ROOT}/build/deps/gmp-install"}
CC_BIN=${CC_BIN:-gcc-14}
CXX_BIN=${CXX_BIN:-g++-14}

mkdir -p "${DOWNLOAD_DIR}" "${SOURCE_ROOT}" "${BUILD_DIR}" "${PREFIX_DIR}"

download() {
    local url=$1
    local out=$2
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 "${url}" -o "${out}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${out}" "${url}"
    else
        echo "error: need curl or wget to download GMP" >&2
        exit 1
    fi
}

if [[ ! -f "${DOWNLOAD_DIR}/${GMP_TARBALL}" ]]; then
    download "${GMP_URL}" "${DOWNLOAD_DIR}/${GMP_TARBALL}"
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
    tar -xf "${DOWNLOAD_DIR}/${GMP_TARBALL}" -C "${SOURCE_ROOT}"
fi

cd "${BUILD_DIR}"

if [[ ! -f Makefile ]]; then
    CC="${CC_BIN}" CXX="${CXX_BIN}" "${SOURCE_DIR}/configure" \
        --prefix="${PREFIX_DIR}" \
        --enable-cxx \
        --disable-shared \
        CFLAGS="-O3 -march=native" \
        CXXFLAGS="-O3 -march=native"
fi

make -j"$(nproc)"
make install

echo "GMP ${GMP_VERSION} installed under ${PREFIX_DIR} using ${CC_BIN}/${CXX_BIN}"
