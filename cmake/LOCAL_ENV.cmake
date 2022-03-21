# set flags

set(ENV_RUNTIME_DIR "bin")
set(ENV_LIBRARY_DIR "lib")
set(ENV_ARCHIVE_DIR "lib")
set(ENV_HEADER_DIR "include")
set(ENV_SUFFIX "")

if (MSVC)
	if ("${PLATFORM}" STREQUAL "")
	elseif("${PLATFORM}" STREQUAL "auto")
	else ()
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR}/${PLATFORM})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}/${PLATFORM})
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}/${PLATFORM})
	endif ()
elseif (MINGW)
	if ("${PLATFORM}" STREQUAL "")
	elseif("${PLATFORM}" STREQUAL "auto")
	else ()
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR}/${PLATFORM})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}/${PLATFORM})
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}/${PLATFORM})
	endif ()
elseif (ANDROID)
	if ("${PLATFORM}" STREQUAL "")
	elseif("${PLATFORM}" STREQUAL "auto")
	else ()
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR}/${PLATFORM})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}/${PLATFORM})
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}/${PLATFORM})
	endif ()
elseif (IOS)
	if ("${PLATFORM}" STREQUAL "")
	elseif("${PLATFORM}" STREQUAL "auto")
	else ()
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR}/${PLATFORM})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}/${PLATFORM})
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}/${PLATFORM})
	endif ()
else()
	if ("${PLATFORM}" STREQUAL "")
	elseif("${PLATFORM}" STREQUAL "auto")
	elseif ("${PLATFORM}" STREQUAL "x86")
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}32)
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}32)
	elseif ("${PLATFORM}" STREQUAL "x64")
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}64)
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}64)
	else()
		set(ENV_RUNTIME_DIR ${ENV_RUNTIME_DIR}/${PLATFORM})
		set(ENV_LIBRARY_DIR ${ENV_LIBRARY_DIR}/${PLATFORM})
		set(ENV_ARCHIVE_DIR ${ENV_ARCHIVE_DIR}/${PLATFORM})
	endif()
endif()

if ("${CONFIGURATION}" STREQUAL "Debug")
	set(ENV_SUFFIX "d")
endif ()

set(ENV_INCLUDE_DIR ${ENV_HEADER_DIR})

