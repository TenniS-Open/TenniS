# base cmake tools

# ts_add_library(<name> [STATIC | SHARED | MODULE] source1 [source2 ...])
function(ts_add_library target_name target_type)
    # get all src_files
    set(src_files)
    set(INDEX 2)
    while(INDEX LESS ${ARGC})
        list(APPEND src_files ${ARGV${INDEX}})
        math(EXPR INDEX "${INDEX} + 1")
    endwhile()

    # add library target
    if (TS_USE_CUDA)
        cuda_add_library(${target_name} ${target_type} ${src_files})
    else()
        add_library(${target_name} ${target_type} ${src_files})
    endif()
endfunction()

# ts_add_executable(<name> source1 [source2 ...])
function(ts_add_executable target_name)
    # get all src_files
    set(src_files)
    set(INDEX 1)
    while(INDEX LESS ${ARGC})
        list(APPEND src_files ${ARGV${INDEX}})
        math(EXPR INDEX "${INDEX} + 1")
    endwhile()

    # add library target
    if (TS_USE_CUDA)
        cuda_add_executable(${target_name} ${src_files})
    else()
        add_executable(${target_name} ${src_files})
    endif()
endfunction()