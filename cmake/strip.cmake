# provide strip library method
if ("${CMAKE_STRIP_FLAGS}" STREQUAL "")
    set(CMAKE_STRIP_FLAGS)
endif()

if (ANDROID)
    # default use strip to strip almost symbol, if only strip debug, use cmake -DCMAKE_STRIP_FLAGS=--strip-debug.
    set(CMAKE_STRIP "${ANDROID_TOOLCHAIN_PREFIX}strip${ANDROID_TOOLCHAIN_SUFFIX}")
    # list(APPEND CMAKE_STRIP_FLAGS "--strip-debug")
endif()

# message(STATUS "CMAKE_STRIP_FLAGS: ${CMAKE_STRIP_FLAGS}")

function(STRIP_LIBRARY LIBRARY_NAME)  
    if ("${CONFIGURATION}" STREQUAL "Debug")
        return()
    endif()
    if ("${CMAKE_STRIP}" STREQUAL "")
        message(WARNING "Strip command not provide")
        return()
    endif()
    get_target_property(LIBRARY_NAME_OUTPUT ${LIBRARY_NAME} OUTPUT_NAME)
    add_custom_command(TARGET ${LIBRARY_NAME} POST_BUILD
             COMMAND ${CMAKE_STRIP} ${CMAKE_STRIP_FLAGS} "$<TARGET_FILE:${LIBRARY_NAME}>"
             COMMENT "Strip debug symbols on library ${LIBRARY_NAME_OUTPUT}.") 
endfunction()

