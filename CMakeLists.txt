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

# option for platform
set(PLATFORM "auto" CACHE STRING "auto, x86 or x64")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug or Release")
set(CONFIGURATION ${CMAKE_BUILD_TYPE})

option(TS_USE_OPENMP "[Optional] Use OpenMP" ON)
option(TS_USE_OPENCV "[Optional] Use OpenCV" OFF)

option(TS_USE_CUDA "[Optional] Use CUDA" OFF)
option(TS_USE_CBLAS "[Optional] Use CBLAS" OFF)
option(TS_USE_CUBLAS "[Optional] Use CUBLAS" OFF)

option(TS_USE_PROFILER "[Optional] Use Profiler" ON)
if (TS_USE_PROFILER)
    add_definitions(-DTS_USE_PROFILER)
endif()
option(TS_USE_HOOK "[Optional] Use Hook" ON)
if (TS_USE_HOOK)
    add_definitions(-DTS_USE_HOOK)
endif()

option(TS_USE_DEBUG_API "[Optional] Use CBLAS" OFF)
if (TS_USE_DEBUG_API)
    add_definitions(-DTS_USE_DEBUG_API)
endif()

# set common compiler flags
include(LOCAL_FLAGS)
include(LOCAL_ENV)
include(tools)

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build CACHE STRING "set install prefix" FORCE)
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

add_definitions(-DTS_USE_SSE)
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
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

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
    message(STATUS "[Optional] Use CUDA: [ON]")

    list(APPEND TENSORSTACK_ACHIVES gpu)
    add_definitions(-DTS_USE_CUDA)
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

    set(CBLAS_FOUND OFF)

    if (WIN32)
        find_library(OpenBLAS REQUIRED)
        if (OpenBLAS_FOUND)
            set(CBLAS_FOUND ON)
            include_directories(${OpenBLAS_INCLUDE_DIR})
            list(APPEND third_libraries ${OpenBLAS_LIBRARY})
        endif()
    elseif (APPLE)
        message(STATUS "Using Max OS X: Accelerate")
        list(APPEND third_libraries -lcblas)
        set(CBLAS_FOUND ON)
    elseif (UNIX)
        find_library(OpenBLAS REQUIRED)
        if (OpenBLAS_FOUND)
            set(CBLAS_FOUND ON)
            include_directories(${OpenBLAS_INCLUDE_DIR})
            list(APPEND third_libraries ${OpenBLAS_LIBRARY})
        endif()
    else ()
        find_library(OpenBLAS REQUIRED)
        if (OpenBLAS_FOUND)
            set(CBLAS_FOUND ON)
            include_directories(${OpenBLAS_INCLUDE_DIR})
            list(APPEND third_libraries ${OpenBLAS_LIBRARY})
        endif()
    endif ()
    if (NOT CBLAS_FOUND)
        list(APPEND third_libraries -lopenblas)
    endif()
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

ts_add_library(${PROJECT_NAME}_LIB ${TARGET} ${INCLUDE_FILES} ${INCLUDE_SRC_FILES} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}_LIB ${third_libraries})

set_target_properties(${PROJECT_NAME}_LIB PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})

if (WIN32)
    set_target_properties(${PROJECT_NAME}_LIB PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/${ENV_RUNTIME_DIR})
endif()


install(TARGETS ${PROJECT_NAME}_LIB
        RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
        LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
        ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
        )

# TODO: update headers install script
install(DIRECTORY ${SOLUTION_DIR}/include/
        DESTINATION ${ENV_HEADER_DIR}
        )

# project files for clion
FILE(GLOB_RECURSE TEST_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
FILE(GLOB_RECURSE TOOL_FILES ${PROJECT_SOURCE_DIR}/tools/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

foreach (path ${TEST_FILES})
    string(REGEX MATCH "[^/]*.[(c)|(cc)|(cpp)]$" file_ext ${path})
    string(REGEX MATCH "^[^.]*" file ${file_ext})
    add_executable(test_${file} ${path})
    target_link_libraries(test_${file} ${PROJECT_NAME}_LIB)
    target_link_libraries(test_${file} ${third_libraries})
    # set_target_properties(test_${file} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/lib)
endforeach ()


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
