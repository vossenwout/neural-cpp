cmake_minimum_required(VERSION 3.16)
project(neural_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parents project's compiler settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Main application sources (excluding main.cpp)
file(GLOB_RECURSE SOURCES "src/**/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

# Create a library with core functionality
add_library(neural_core ${SOURCES})

# Main executable
add_executable(cpp_neural src/main.cpp)
target_link_libraries(cpp_neural PRIVATE neural_core)

# Tests
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests PRIVATE neural_core gtest gtest_main)

# Register tests
include(GoogleTest)
gtest_discover_tests(run_tests)