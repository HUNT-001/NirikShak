#!/usr/bin/env bash
# Cross-compile ExecuTorch's llama_main for Android arm64 (XNNPACK + KleidiAI) and push it.
# env: ET_ROOT (executorch checkout), ANDROID_NDK. Follows examples/models/llama/README.md of the
# pinned ExecuTorch tag; if your tag uses CMake presets, use those instead (same targets).
set -euo pipefail
: "${ET_ROOT:?}"; : "${ANDROID_NDK:?}"
cd "$ET_ROOT"
COMMON=(-DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-23 -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=cmake-out-android -DPYTHON_EXECUTABLE=python)
cmake "${COMMON[@]}" \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON -Bcmake-out-android .
cmake --build cmake-out-android -j"$(nproc)" --target install --config Release
cmake "${COMMON[@]}" -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
  -Bcmake-out-android/examples/models/llama examples/models/llama
cmake --build cmake-out-android/examples/models/llama -j"$(nproc)" --config Release
BIN=cmake-out-android/examples/models/llama/llama_main
DEV=${DEVICE_DIR:-/data/local/tmp/nirikshak}
adb shell mkdir -p "$DEV" && adb push "$BIN" "$DEV/" && adb shell chmod +x "$DEV/llama_main"
echo "pushed llama_main to $DEV"
