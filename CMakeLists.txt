CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

SET(PROJECT_NAME weighted_hqp)
SET(PROJECT_DESCRIPTION "weighted_hqp")
SET(PROJECT_URL "http://github.com/ggory15/weighted_hqp")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)
SET(CMAKE_VERBOSE_MAKEFILE true)
SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++17 -O3)

#INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)
INCLUDE(cmake/ide.cmake)
INCLUDE(cmake/apple.cmake)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
#  CATKIN_DEPENDS 
 DEPENDS system_lib
)
include_directories(include)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# COMPUTE_PROJECT_ARGS(PROJECT_ARGS LANGUAGES CXX)
# PROJECT(${PROJECT_NAME} ${PROJECT_ARGS})
# SET(ADD_CXX_PATCH TRUE)
ADD_DEFINITIONS("-fpermissive")

#ADD_PROJECT_DEPENDENCY(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.2.0")
find_package(kimm_qpoases REQUIRED)
include_directories(${kimm_qpoases_INCLUDE_DIRS})
ADD_DEFINITIONS(-DEIGEN_RUNTIME_NO_MALLOC)
find_package(Boost REQUIRED COMPONENTS system unit_test_framework)


# SET(${EIGEN_INCLUDE_DIR} 
#   3rdparty/Eigen3
# )

SET(${PROJECT_NAME}_HEADERS
  include/weighted_hqp/Random.hpp
  include/weighted_hqp/InitSet.hpp
  include/weighted_hqp/HCod.hpp
  include/weighted_hqp/cod.hpp
  include/weighted_hqp/givens.hpp
  include/weighted_hqp/ehqp_primal.hpp
  include/weighted_hqp/eHQP_solver.hpp
  include/weighted_hqp/iHQP_solver.hpp
  include/weighted_hqp/step_length.hpp
  )

SET(${PROJECT_NAME}_SOURCES
  src/Random.cpp
  src/InitSet.cpp
  src/HCod.cpp
  src/cod.cpp
  src/givens.cpp
  src/ehqp_primal.cpp
  src/eHQP_solver.cpp
  src/iHQP_solver.cpp
  src/step_length.cpp
  )

include_directories(3rdparty/Eigen3)


ADD_HEADER_GROUP(${PROJECT_NAME}_HEADERS)
ADD_SOURCE_GROUP(${PROJECT_NAME}_SOURCES)

ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})

TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} SYSTEM PUBLIC ${kimm_qpoases_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})

#SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
#INSTALL(TARGETS ${PROJECT_NAME} EXPORT ${TARGETS_EXPORT_NAME} DESTINATION lib)


add_executable(ihqp_test unitTesting/ihqp_test.cpp)
add_executable(ehqp_test unitTesting/ehqp_test.cpp)
add_executable(ihqp_vs_qpoases unitTesting/ihqp_vs_qpoases.cpp)
target_link_libraries(ihqp_test ${PROJECT_NAME} ${GTEST_LIBRARIES} ${catkin_LIBRARIES} ${kimm_qpoases_LIBRARIES} pthread)
target_link_libraries(ehqp_test ${PROJECT_NAME} ${GTEST_LIBRARIES} ${catkin_LIBRARIES} ${kimm_qpoases_LIBRARIES} pthread)
target_link_libraries(ihqp_vs_qpoases ${PROJECT_NAME} ${GTEST_LIBRARIES} ${catkin_LIBRARIES} ${kimm_qpoases_LIBRARIES} pthread)

