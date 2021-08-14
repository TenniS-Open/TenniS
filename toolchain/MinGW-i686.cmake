# Set target OS is Windows
SET(CMAKE_SYSTEM_NAME Windows)

# Set C/C++ compiler
# Use posix version to support thread with libwinpthreads.
SET(CMAKE_C_COMPILER  i686-w64-mingw32-gcc-posix)
SET(CMAKE_CXX_COMPILER i686-w64-mingw32-g++-posix)
SET(CMAKE_RC_COMPILER i686-w64-mingw32-windres)

# Set Root
SET(CMAKE_FIND_ROOT_PATH  /usr/i686-w64-mingw32 )

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
