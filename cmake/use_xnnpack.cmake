cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(${CMAKE_VERSION} VERSION_GREATER 3.11)
    include(FetchContent)
    FetchContent_Declare(
        xnnpack
        GIT_REPOSITORY https://lvsen@gitlab.seetatech.com/lvsen/XNNPACK.git
        GIT_TAG tennis
    )

    if(${CMAKE_VERSION} VERSION_LESS 3.14)
        macro(FetchContent_MakeAvailable NAME)
            FetchContent_GetProperties(${NAME})
            if(NOT ${NAME}_POPULATED)
                FetchContent_Populate(${NAME})
                add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
            endif()
        endmacro()
    endif()

    FetchContent_MakeAvailable(xnnpack)

    set_target_properties(XNNPACK PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xnnpack")
    set_target_properties(cpuinfo PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cpuinfo")
    set_target_properties(pthreadpool PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool")
    set_target_properties(clog PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/clog")
#    list(APPEND third_libraries XNNPACK)
    target_link_libraries(${PROJECT_NAME}_LIB XNNPACK)
else()
    include(ExternalProject)

    set(XNNPACK_ROOT        ${CMAKE_BINARY_DIR}/xnnpack)
    set(XNNPACK_GIT_URL     https://lvsen@gitlab.seetatech.com/lvsen/XNNPACK.git)
    set(XNNPACK_GIT_TAG     tennis)
    set(XNNPACK_CONFIGURE   mkdir ${XNNPACK_ROOT}/src/xnnpack/build && cd ${XNNPACK_ROOT}/src/xnnpack/build && ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} ..)
    set(XNNPACK_MAKE        cd ${XNNPACK_ROOT}/src/xnnpack/build && ${CMAKE_COMMAND} --build . -- -j4)

    ExternalProject_Add(xnnpack
            PREFIX              ${XNNPACK_ROOT}
            GIT_REPOSITORY      ${XNNPACK_GIT_URL}
            GIT_TAG             ${XNNPACK_GIT_TAG}
            CONFIGURE_COMMAND   ${XNNPACK_CONFIGURE}
            BUILD_COMMAND       ${XNNPACK_MAKE}
            INSTALL_COMMAND     ""
            TEST_COMMAND        ""
            )
    add_dependencies(${PROJECT_NAME}_LIB xnnpack)

    set(XNNPACK_LIB ${XNNPACK_ROOT}/src/xnnpack/build/libXNNPACK.a)
    set(CPUINFO_LIB ${XNNPACK_ROOT}/src/xnnpack/build/cpuinfo/libcpuinfo.a)
    set(PTHREADPOOL_LIB ${XNNPACK_ROOT}/src/xnnpack/build/pthreadpool/libpthreadpool.a)
    set(CLOG_LIB ${XNNPACK_ROOT}/src/xnnpack/build/clog/libclog.a)
    target_link_libraries(${PROJECT_NAME}_LIB ${XNNPACK_LIB} ${CPUINFO_LIB} ${PTHREADPOOL_LIB} ${CLOG_LIB})
endif()
