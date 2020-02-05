message(AUTHOR_WARNING "FindTensorStack.cmake was decrepted, please use FindTenniS.cmake instead.")

find_package(TenniS)

set(TensorStack_LIBRARY ${TenniS_LIBRARY})
set(TensorStack_INCLUDE_DIR ${TenniS_INCLUDE_DIR})

set(TensorStack_LIBRARIES ${TenniS_LIBRARIES})
set(TensorStack_INCLUDE_DIRS ${TenniS_INCLUDE_DIRS})
