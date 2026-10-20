cmake_minimum_required(VERSION 3.16)

if(LINUX) 
	set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "" FORCE)
	set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "" FORCE)
endif()

option(BUILD_TESTING "Build the test suite" OFF)

if(WIN32)
	if(NOT DEFINED CUDAToolkit_ROOT AND WIN32)
	    # Check common installation path on Windows
	    set(CUDA_13_3_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")
	    if(EXISTS "${CUDA_13_3_PATH}")
		set(CUDAToolkit_ROOT "${CUDA_13_3_PATH}" CACHE PATH "Path to CUDA 13.3 Toolkit")
		message(STATUS "Auto-detected CUDA 13.3 at: ${CUDAToolkit_ROOT}")
	    endif()
	endif()
elseif(LINUX)
	set(CUDA_13_3_PATH "/usr/local/cuda-13.3")
	set(CMAKE_CUDA_COMPILER "${CUDA_13_3_PATH}/bin/nvcc")
endif()

if(BUILD_TESTING)
    set(VCPKG_MANIFEST_FEATURES "tests" CACHE STRING "" FORCE)
else()
    set(VCPKG_MANIFEST_FEATURES "" CACHE STRING "" FORCE)
endif()

#VCPGK

if(WIN32)
	set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
elseif(LINUX)
	set(CMAKE_TOOLCHAIN_FILE "~/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

set(VCPKG_USE_SYSTEM_BINARIES ON CACHE BOOL "" FORCE)
set(VCPKG_FORCE_SYSTEM_BINARIES ON CACHE BOOL "" FORCE)

project(TransformerC VERSION 0.1 LANGUAGES CXX)

if(WIN32)
	set(Torch_DIR "D:/Files/MakemoreC/ThirdParty/libtorch-win-shared-with-deps-2.13.0+cu132/libtorch/share/cmake/Torch")
elseif(LINUX)
	set(Torch_DIR "~/Files/TransformerC/ThirdParty/libtorch-shared-with-deps-2.13.0+cu132/libtorch/share/cmake/Torch")
endif()


find_package(Torch CONFIG REQUIRED)
find_package(CUDAToolkit 13.3 REQUIRED)

add_subdirectory(TransformerC)
add_subdirectory(Tokenizer)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(TransformerC)
endif()


