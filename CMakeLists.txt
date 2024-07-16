CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

FIND_PACKAGE(deal.II 9.2
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

file(GLOB TARGET_SRC source/LODscalar.cc )

INCLUDE_DIRECTORIES(include)

PROJECT(lod)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_LIBRARY(lod SHARED ${TARGET_SRC})

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

# ADD_EXECUTABLE(ElasticityLOD app/main_Elasticity.cc)
# TARGET_LINK_LIBRARIES(ElasticityLOD lod)      
# DEAL_II_SETUP_TARGET(ElasticityLOD DEBUG)     
# # DEAL_II_SETUP_TARGET(ElasticLOD RELEASE)
ADD_EXECUTABLE(DiffusionLOD app/main_Diffusion.cc)
TARGET_LINK_LIBRARIES(DiffusionLOD lod)      
DEAL_II_SETUP_TARGET(DiffusionLOD DEBUG)  
# FOREACH(_main ${_main_files})
#       # Setup one executable per dimension/per build type
#       get_filename_component(_main_we ${_main} NAME_WE)
#       STRING(REPLACE "app_" "" _main_we ${_main_we})
#       SET(_exe "${_main_we}")
#       MESSAGE("-- Configuring executable ${_exe}")
#       ADD_EXECUTABLE(${_exe} ${_main})
#       TARGET_LINK_LIBRARIES(${_exe} lod)      
#       DEAL_II_SETUP_TARGET(${_exe} DEBUG)
# ENDFOREACH()


IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  SET(DEAL_II_HAVE_TESTS_DIRECTORY TRUE)
ENDIF()

add_custom_target(indent ./scripts/indent WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
ADD_SUBDIRECTORY(tests)
