# capd_check: independent CAPD check of the M = 1 certificate (proof/A_certify.md section 10).
# CAPD is not vendored. Build it first (build.sh does both) and point CAPD_CONFIG at its capd-config:
#   cmake -S . -B build -DCAPD_CONFIG=/path/to/capd/build/bin/capd-config && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(capd_check CXX)

set(CAPD_CONFIG "" CACHE FILEPATH "capd-config of a CAPD build (bin/capd-config in the CAPD build directory)")
if(NOT CAPD_CONFIG)
  find_program(CAPD_CONFIG capd-config)
endif()
if(NOT CAPD_CONFIG)
  message(FATAL_ERROR "capd-config not found: pass -DCAPD_CONFIG=/path/to/capd/build/bin/capd-config")
endif()

execute_process(COMMAND ${CAPD_CONFIG} --cflags OUTPUT_VARIABLE CAPD_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CAPD_CONFIG} --libs OUTPUT_VARIABLE CAPD_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(CAPD_CFLAGS UNIX_COMMAND "${CAPD_CFLAGS}")
separate_arguments(CAPD_LIBS UNIX_COMMAND "${CAPD_LIBS}")

add_executable(capd_check capd_check.cpp)
target_compile_options(capd_check PRIVATE ${CAPD_CFLAGS})     # includes -frounding-math, needed for intervals
target_link_libraries(capd_check PRIVATE ${CAPD_LIBS})
