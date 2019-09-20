cmake_minimum_required(VERSION 2.7)
project(TensorStack)

set(TARGET "SHARED" CACHE STRING "STATIC or SHARED" FORCE)
add_definitions(-DBUILDING_TENSORSTACK)

# global root dir
set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DTS_SOLUTION_DIR="${SOLUTION_DIR}")
# add_definitions("-Wall -g") # add gcc option in FLAGS_GCC, not here!

add_definitions(-DTS_DISABLE_PARALLEL)
# dir for common cmake files
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${SOLUTION_DIR}/cmake)

#------------------------------------------------------------------------------#
# ============================= Options start ================================ #
#------------------------------------------------------------------------------#
# Install path
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build
    CACHE STRING "set install prefix" FORCE)

# Option for platform
set(PLATFORM "auto" CACHE STRING "auto, x86 or x64")
# Build type, configuration
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug or Release")

# If build test cases
option(TS_BUILD_TEST "[Optional] Build test" OFF)
# If build tool cases
option(TS_BUILD_TOOLS "[Optional] Build tools" OFF)
# If build with OpenCV, must set on if TS_BUILD_TEST=ON
option(TS_USE_OPENCV "[Optional] Use OpenCV" OFF)
# If build with debug API, must set on if TS_BUILD_TEST=ON
option(TS_USE_DEBUG_API "[Optional] Use Debug API" OFF)

# if use 
option(TS_USE_CUDA "[Optional] Use CUDA" OFF)
option(TS_USE_CUBLAS "[Optional] Use CUBLAS" OFF)
set(TS_CUDA_ARCH "" CACHE STRING "List arch, like: 60,61,62. Empty for all arch.")

# Options of optimizations
option(TS_USE_CBLAS "[Optional] Use CBLAS" OFF) # [discarded]
option(TS_USE_OPENMP "[Optional] Use OpenMP" ON)
option(TS_USE_SIMD "[Optional] Use SIMD" ON)
option(TS_USE_AVX "[Optional] Use AVX" OFF)
option(TS_USE_SSE "[Optional] Use SSE" OFF)
option(TS_USE_FMA "[Optional] Use FMA" OFF)
option(TS_USE_NEON "[Optional] Use NEON" OFF)
option(TS_USE_FAST_MATH "[Optional] Use fast math (-ffast-math)" OFF)
option(TS_USE_WINOGRAD "[Optional] Use Winograd" OFF)
# Special for armv7
option(TS_ON_ARMV7 "[Optional] Target is arm-v7a" OFF)

# Not change for now version
option(TS_USE_PROFILER "[Optional] Use Profiler" ON)
option(TS_USE_HOOK "[Optional] Use Hook" ON)

# Hidden options (do not uncomment, it already defined later)
# None

#------------------------------------------------------------------------------#
# ============================= Options end ================================== #
#------------------------------------------------------------------------------#

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CONFIGURATION "Release")
else ()
    set(CONFIGURATION ${CMAKE_BUILD_TYPE})
endif ()

message(STATUS "== Target: ${TARGET}")
message(STATUS "== Platform: ${PLATFORM}")
message(STATUS "== Configuration: ${CONFIGURATION}")

# set common compiler flags
include(LOCAL_FLAGS)
include(LOCAL_ENV)
include(tools)

if (TS_USE_PROFILER)
    add_definitions(-DTS_USE_PROFILER)
endif()
if (TS_USE_HOOK)
    add_definitions(-DTS_USE_HOOK)
endif()

if (TS_USE_DEBUG_API)
    message(STATUS "[Important] Use debug API: [ON]")
    add_definitions(-DTS_USE_DEBUG_API)
else (TS_USE_DEBUG_API)
    message(STATUS "[Important] Use debug API: [OFF]")
endif()

if (TS_USE_SIMD)
	message(STATUS "[Optional] Use SIMD: [ON]")
	add_definitions(-DTS_USE_SIMD)
    if (TS_USE_AVX)
		message(STATUS "[Optional] Use AVX: [ON]")
		add_definitions(-DTS_USE_AVX)
	elseif (TS_USE_SSE)
		message(STATUS "[Optional] Use SSE: [ON]")
		add_definitions(-DTS_USE_SSE)
	elseif (TS_USE_NEON)
		message(STATUS "[Optional] Use NEON: [ON]")
		add_definitions(-DTS_USE_NEON)
	endif()
    if (TS_USE_FMA)
        message(STATUS "[Optional] Use FMA: [ON]")
        add_definitions(-DTS_USE_FMA)
    else (TS_USE_FMA)
        message(STATUS "[Optional] Use FMA: [OFF]")
    endif()
endif()

# set winograd support
if (TS_USE_WINOGRAD)
    add_definitions(-DTS_USE_WINOGRAD)
    message(STATUS "[Optional] Use Winograd: [ON]")
else (TS_USE_WINOGRAD)
    message(STATUS "[Optional] Use Winograd: [OFF]")
endif()

# if target is target
if (TS_ON_ARMV7)
    add_definitions(-DTS_ON_ARMV7)
endif()

install(CODE "MESSAGE(\"Installing into ${CMAKE_INSTALL_PREFIX}\")")

set(LOCAL_OUTPUT_DIR ${PROJECT_SOURCE_DIR})

if (LOCAL_OUTPUT_DIR)
    set(EXECUTABLE_OUTPUT_PATH ${LOCAL_OUTPUT_DIR}/${ENV_RUNTIME_DIR})
    set(LIBRARY_OUTPUT_PATH ${LOCAL_OUTPUT_DIR}/${ENV_LIBRARY_DIR})
endif ()

include_directories(${SOLUTION_DIR}/include)
include_directories(${SOLUTION_DIR}/src)

set(SOURCE_DIR ${SOLUTION_DIR})

set(TENSORSTACK_MODULES
        api
        backend
        board
        compiler
        core
        encryption
        engine
        frontend
        global
        memory
        module
        runtime
        utils
        )
set(TENSORSTACK_ACHIVES common cpu) # in kernels and ops

#add_definitions(-DTS_USE_SSE)
# 4 types of achive
# blas
# Eigen
# Ordinary
# Ordinary with chipset result

set(third_libraries)

if (TS_USE_OPENCV)
    add_definitions(-DTS_USE_OPENCV)
    message(STATUS "[Optional] Use OpenCV: [ON]")

    find_package(OpenCV REQUIRED)
    message(STATUS "    OpenCV library status:")
    message(STATUS "        version: ${OpenCV_VERSION}")
    # message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "        include path: ${OpenCV_INCLUDE_DIRS}")

    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND third_libraries ${OpenCV_LIBS})
else ()
    message(STATUS "[Optional] Use OpenCV: [OFF]")
endif ()

if (TS_USE_OPENMP)
    add_definitions(-DTS_USE_OPENMP)
    message(STATUS "[Optional] Use OpenMP: [ON]")

    include(use_openmp)
    list(APPEND third_libraries ${OPENMP_LIBRARY})
else ()
    message(STATUS "[Optional] Use OpenMP: [OFF]")
endif ()

if (TS_USE_CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "[Optional] Use CUDA: [ON]; Found ${CUDA_VERSION}")
    list(APPEND TENSORSTACK_ACHIVES gpu)
    add_definitions(-DTS_USE_CUDA)

    include(use_cuda)

    set(TS_USE_CUBLAS ON)
else()
    message(STATUS "[Optional] Use CUDA: [OFF]")
endif()

if (WIN32)
    # list(APPEND third_libraries pthread)
elseif (APPLE)
    # list(APPEND third_libraries pthread)
    set(CMAKE_MACOSX_RPATH 1)
elseif (UNIX)
    list(APPEND third_libraries pthread)
    list(APPEND third_libraries m)
else ()
endif ()

if(TS_USE_CUBLAS)
	message(STATUS "[Optional] Use CUBLAS: [ON]")
	add_definitions(-DTS_USE_CUBLAS)
    list(APPEND third_libraries ${CUDA_CUBLAS_LIBRARIES})
else()
	message(STATUS "[Optional] Use CUBLAS: [OFF]")
endif()

if (TS_USE_CBLAS)
    message(STATUS "[Optional] Use CBLAS: [ON]")
    list(APPEND TENSORSTACK_ACHIVES cblas)
    add_definitions(-DTS_USE_CBLAS)

    include(use_cblas)
else()
    message(STATUS "[Optional] Use CBLAS: [OFF]")
endif()

set(INCLUDE_FILES)
set(SRC_FILES)
set(SRC_INCLUDE_FILES)

foreach (module ${TENSORSTACK_MODULES})
    FILE(GLOB_RECURSE INCLUDE_${module}_FILES ${SOURCE_DIR}/include/${module}/*.h)
    LIST(APPEND INCLUDE_FILES ${INCLUDE_${module}_FILES})

    FILE(GLOB_RECURSE SRC_${module}_FILES ${SOURCE_DIR}/src/${module}/*.cpp)
    LIST(APPEND SRC_FILES ${SRC_${module}_FILES})

    FILE(GLOB_RECURSE SRC_INCLUDE_${module}_FILES ${SOURCE_DIR}/src/${module}/*.h)
    LIST(APPEND SRC_INCLUDE_FILES ${SRC_INCLUDE_${module}_FILES})
endforeach ()

foreach (achive ${TENSORSTACK_ACHIVES})
    FILE(GLOB_RECURSE INCLUDE_ARCHIVE_${achive}_FILES
            ${SOURCE_DIR}/include/kernels/${achive}/*.h
            ${SOURCE_DIR}/include/ops/${achive}/*.h
            )
    LIST(APPEND INCLUDE_FILES ${INCLUDE_ARCHIVE_${achive}_FILES})

    FILE(GLOB_RECURSE SRC_ARCHIVE_${achive}_FILES
            ${SOURCE_DIR}/src/kernels/${achive}/*.cpp
            ${SOURCE_DIR}/src/ops/${achive}/*.cpp
            )
    LIST(APPEND SRC_FILES ${SRC_ARCHIVE_${achive}_FILES})

    if (TS_USE_CUDA)
        FILE(GLOB_RECURSE SRC_ARCHIVE_${achive}_CU_FILES
                ${SOURCE_DIR}/src/kernels/${achive}/*.cu
                ${SOURCE_DIR}/src/ops/${achive}/*.cu
                )
        LIST(APPEND SRC_FILES ${SRC_ARCHIVE_${achive}_CU_FILES})
    endif()

    FILE(GLOB_RECURSE SRC_INCLUDE_ARCHIVE_${achive}_FILES
            ${SOURCE_DIR}/src/kernels/${achive}/*.h
            ${SOURCE_DIR}/src/ops/${achive}/*.h
            )
    LIST(APPEND SRC_INCLUDE_FILES ${SRC_INCLUDE_ARCHIVE_${achive}_FILES})
endforeach ()

ts_add_library(${PROJECT_NAME}_LIB ${TARGET} ${INCLUDE_FILES} ${SRC_INCLUDE_FILES} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}_LIB ${third_libraries})

set_target_properties(${PROJECT_NAME}_LIB PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})

if (WIN32)
    set_target_properties(${PROJECT_NAME}_LIB PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/${ENV_RUNTIME_DIR})
endif()

if (IOS)
    set(TS_BUILD_FRAMEWORK ON)
else (IOS)
    set(TS_BUILD_FRAMEWORK OFF)
endif (IOS)

if (TS_BUILD_FRAMEWORK)
    set_target_properties(${PROJECT_NAME}_LIB PROPERTIES
            FRAMEWORK TRUE
            FRAMEWORK_VERSION A
            MACOSX_FRAMEWORK_IDENTIFIER com.seetatech.sdk.TensorStack
            # MACOSX_FRAMEWORK_INFO_PLIST Info.plist
            # "current version" in semantic format in Mach-O binary file
            VERSION 0.3.3
            # "compatibility version" in semantic format in Mach-O binary file
            SOVERSION 0.3.3
            # PUBLIC_HEADER "${FRAMEWORK_PUBLIC_HEADER_FILES}"
            # RESOURCE      "${RESOURCE_FILES1}"
            XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "iPhone Developer"
            )
    install(DIRECTORY ${SOLUTION_DIR}/include/api
            DESTINATION "framework/TensorStack.framework/Headers"
            )
else ()
    install(DIRECTORY ${SOLUTION_DIR}/include/api
            DESTINATION ${ENV_HEADER_DIR}
            )
endif (TS_BUILD_FRAMEWORK)

install(TARGETS ${PROJECT_NAME}_LIB
        RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
        LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
        ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
        FRAMEWORK DESTINATION framework
        )

install(FILES ${SOLUTION_DIR}/cmake/FindTensorStack.cmake
        DESTINATION cmake)

install(FILES ${SOLUTION_DIR}/cmake/FindTensorStack.cmake
        DESTINATION cmake
        RENAME TensorStackConfig.cmake)

# project files for clion
FILE(GLOB_RECURSE TEST_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
FILE(GLOB_RECURSE TOOL_FILES ${PROJECT_SOURCE_DIR}/tools/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

if (TS_BUILD_TEST)
message(STATUS "[Important] Build test: [ON]")
foreach (path ${TEST_FILES})
    string(REGEX MATCH "[^/]*.[(c)|(cc)|(cpp)]$" file_ext ${path})
    string(REGEX MATCH "^[^.]*" file ${file_ext})
    add_executable(test_${file} ${path})
    target_link_libraries(test_${file} ${PROJECT_NAME}_LIB)
    target_link_libraries(test_${file} ${third_libraries})
    # set_target_properties(test_${file} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/lib)
endforeach ()
else (TS_BUILD_TEST)
    message(STATUS "[Important] Build test: [OFF]")
endif(TS_BUILD_TEST)

if (TS_BUILD_TOOLS)
message(STATUS "[Important] Build tools: [ON]")
foreach (path ${TOOL_FILES})
    string(REGEX MATCH "[^/]*.[(c)|(cc)|(cpp)]$" file_ext ${path})
    string(REGEX MATCH "^[^.]*" file ${file_ext})
    add_executable(tool_${file} ${path})
    set_target_properties(tool_${file} PROPERTIES OUTPUT_NAME ${file})
    target_link_libraries(tool_${file} ${PROJECT_NAME}_LIB)
    target_link_libraries(tool_${file} ${third_libraries})
    install(TARGETS tool_${file}
            RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
            )
endforeach ()
else (TS_BUILD_TOOLS)
    message(STATUS "[Important] Build tools: [OFF]")
endif (TS_BUILD_TOOLS)

