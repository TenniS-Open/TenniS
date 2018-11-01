cmake_minimum_required(VERSION 2.7)
project(TensorStack)

# global root dir
set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# dir for common cmake files
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)
list(APPEND CMAKE_PREFIX_PATH ${SOLUTION_DIR}/cmake)

# option for platform
set(PLATFORM "auto" CACHE STRING "auto, x86 or x64")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Debug or Release")
set(CONFIGURATION ${CMAKE_BUILD_TYPE})

option(TS_USE_CUDA "[Optional] Use CUDA" OFF)

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

set(TENSORSTACK_MODULES compiler core engine global module runtime utils)
set(TENSORSTACK_ACHIVES common cpu) # in kernels and ops

add_definitions(-DTS_USE_SSE)
# 4 types of achive
# blas
# Eigen
# Ordinary
# Ordinary with chipset result

if (TS_USE_CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "[Optional] Use CUDA: [ON]")

    list(APPEND TENSORSTACK_ACHIVES gpu)
    add_definitions(-DTS_USE_CUDA)
else()
    message(STATUS "[Optional] Use CUDA: [OFF]")
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

ts_add_library(${PROJECT_NAME}_STATIC STATIC ${INCLUDE_FILES} ${INCLUDE_SRC_FILES} ${SRC_FILES})
ts_add_library(${PROJECT_NAME}_SHARED SHARED ${INCLUDE_FILES} ${INCLUDE_SRC_FILES} ${SRC_FILES})

set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})
set_target_properties(${PROJECT_NAME}_STATIC PROPERTIES OUTPUT_NAME ${PROJECT_NAME}_static${ENV_SUFFIX})

set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/${ENV_RUNTIME_DIR})

install(TARGETS ${PROJECT_NAME}_SHARED ${PROJECT_NAME}_STATIC
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
    target_link_libraries(test_${file} ${PROJECT_NAME}_SHARED -lopenblas)
    # set_target_properties(test_${file} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LOCAL_OUTPUT_DIR}/lib)
endforeach ()

foreach (path ${TOOL_FILES})
    string(REGEX MATCH "[^/]*.[(c)|(cc)|(cpp)]$" file_ext ${path})
    string(REGEX MATCH "^[^.]*" file ${file_ext})
    add_executable(tool_${file} ${path})
    set_target_properties(tool_${file} PROPERTIES OUTPUT_NAME ${file})
    target_link_libraries(tool_${file} ${PROJECT_NAME}_SHARED)
    install(TARGETS tool_${file}
            RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
            )
endforeach ()