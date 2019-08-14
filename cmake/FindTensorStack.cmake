
# find TensorStack
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_VERSION
#<NAME>_DEFINITIONS

#variables:
#<NAME>_NAME
#<NAME>_INCLUDE_DIR
#<NAME>_LIBRARIE

set(TensorStack_NAME "TensorStack" CACHE STRING "The TensorStack library name")
set(TensorStack_VERSION "" CACHE STRING "The TensorStack library version")
set(TensorStack_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../")
#message(STATUS "SEETANET_HOME: " $ENV{SEETANET_HOME})
message(STATUS "TensorStack default module path: ${TensorStack_MODULE_PATH}" )
message(STATUS "TensorStack_ROOT_DIR: ${TensorStack_ROOT_DIR}")
	
if(BUILD_ANDROID)
	# if(TensorStack_ROOT_DIR STREQUAL "")
		# message(SEND_ERROR "Set the path to TensorStack root folder in the system variable TensorStack_ROOT_DIR ")
	# endif()
	set(TensorStack_INCLUDE_DIR "${TensorStack_MODULE_PATH}include")
	file(GLOB_RECURSE INCLUDE_FILE
		${TensorStack_INCLUDE_DIR}/TensorStackForward.h)
	if("${INCLUDE_FILE}" STREQUAL "")
		set(TensorStack_INCLUDE_DIR "${TensorStack_ROOT_DIR}/include")
	endif()
	message(STATUS "TensorStack include dir : ${TensorStack_INCLUDE_DIR}")
	file(GLOB TensorStack_LIBRARY_DEBUG
		${TensorStack_MODULE_PATH}${ENV_LIBRARY_DIR}/*${TensorStack_NAME}*.so)
	if("${TensorStack_LIBRARY_DEBUG}" STREQUAL "")
		file(GLOB TensorStack_LIBRARY_DEBUG
		${TensorStack_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${TensorStack_NAME}*.so)
	endif()
	file(GLOB TensorStack_LIBRARY_RELEASE
		${TensorStack_MODULE_PATH}${ENV_LIBRARY_DIR}/*${TensorStack_NAME}*.so)
	if("${TensorStack_LIBRARY_RELEASE}" STREQUAL "")
		file(GLOB TensorStack_LIBRARY_RELEASE
		${TensorStack_ROOT_DIR}/${ENV_LIBRARY_DIR}/*${TensorStack_NAME}*.so)
	endif()
else()
	find_path(TensorStack_INCLUDE_DIR
	  NAMES
		tensorstack.h
	  PATHS
		${TensorStack_ROOT_DIR}
		${TensorStack_MODULE_PATH}
		ENV TensorStack_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_HEADER_DIR})
	
	if("${TensorStack_INCLUDE_DIR}" STREQUAL "TensorStack_INCLUDE_DIR-NOTFOUND")
		set(TensorStack_INCLUDE_DIR "${TensorStack_MODULE_PATH}include")
	endif()
		
	find_library(TensorStack_LIBRARY_DEBUG
	  NAMES 
		${TensorStack_NAME}${TensorStack_VERSION}
	  PATHS
		${TensorStack_ROOT_DIR}
		${TensorStack_MODULE_PATH}
		ENV TensorStack_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})
		
	find_library(TensorStack_LIBRARY_RELEASE
	  NAMES 
		${TensorStack_NAME}${TensorStack_VERSION}
	  PATHS
		${TensorStack_ROOT_DIR}
		${TensorStack_MODULE_PATH}
		ENV TensorStack_ROOT_DIR
		usr
		usr/local
	  PATH_SUFFIXES
		${ENV_LIBRARY_DIR})	
endif()

if ("${CONFIGURATION}" STREQUAL "Debug")
	set(TensorStack_LIBRARY ${TensorStack_LIBRARY_DEBUG})
else()
	set(TensorStack_LIBRARY ${TensorStack_LIBRARY_RELEASE})
endif()

find_package(PackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(${TensorStack_NAME}
	FOUND_VAR
		TensorStack_FOUND
	REQUIRED_VARS
		TensorStack_INCLUDE_DIR
		TensorStack_LIBRARY
	FAIL_MESSAGE
		"Could not find TensorStack!try to set the path to TensorStack root folder in the system variable TensorStack_ROOT_DIR"
)

if(TensorStack_FOUND)
	set(TensorStack_LIBRARIES ${TensorStack_LIBRARY})
	set(TensorStack_INCLUDE_DIRS ${TensorStack_INCLUDE_DIR})
endif()
message(STATUS "TensorStack_FOUND: " ${TensorStack_FOUND})


foreach (inc ${TensorStack_INCLUDE_DIRS})
    message(STATUS "TensorStack include: " ${inc})
endforeach ()
foreach (lib ${TensorStack_LIBRARIES})
    message(STATUS "TensorStack library: " ${lib})
endforeach ()

