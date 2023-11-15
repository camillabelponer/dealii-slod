CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET( TARGET_SRC dummy.cc )

INCLUDE_DIRECTORIES(include)

PROJECT(lod)

ADD_LIBRARY(lod  ${TARGET_SRC})

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug\nPlease call 'make' to build the debug project"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release\nPlease call 'make' to build the release project"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_SETUP_TARGET(lod)

IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

enable_testing()
ADD_SUBDIRECTORY(tests)
