cmake_minimum_required ( VERSION 2.6 )

# GCC is also in Courseware...
#SET ( CMAKE_CXX_COMPILER "g++-5.2.0" )
#SET ( CMAKE_C_COMPILER "gcc-5.2.0" )


## ---------------------------------------- START PROJECT ---------------------------------------- ##
project ( solve_equations )

SET ( CMAKE_VERBOSE_MAKEFILE ON )

set ( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set ( CMAKE_CXX_FLAGS "-g -std=c++11" )
set ( DEBUG_MODE ON )

#add_definitions ( -DDEBUG_PRINT )


INCLUDE_DIRECTORIES ( src )

file ( GLOB MY_HEADERS src/*.h
                        )
file ( GLOB MY_SOURCES src/*.cpp
                        )

add_executable ( solve
    ${MY_HEADERS}
    ${MY_SOURCES}
)

target_link_libraries ( solve

)

