cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

project(mlcapture)

enable_language(C)

# Release or Debug
set(CMAKE_BUILD_TYPE Debug)

# Cross compile settings
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Doesn't work very well when you're cross compiling (manually set the libs below)
# find_package(OpenCV REQUIRED)
find_package(nanomsg REQUIRED)

file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Vim autocomplete
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json")
		execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
				${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
				${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
		)
endif()

if(NOT UNIX)
		message(FATAL_ERROR "None-linux distributions aren't supported!")
endif()

add_executable(mlcapture ${SOURCE_FILES})

# Manually set the include and lib vars
set(G_B "/usr/arm-linux-gnueabihf")
set(G_INCLUDE_DIRS "${G_B}/include/")
set(OpenCV_INCLUDE_DIRS ${G_INCLUDE_DIRS})
set(CV_B "opencv_")
set(CV_V "") #".so.3.3.1")
set(OpenCV_LIBS "${CV_B}core${CV_V}" "${CV_B}highgui${CV_V}" "${CV_B}imgcodecs${CV_V}" "${CV_B}flann${CV_V}" "${CV_B}features2d${CV_V}" 
	"${CV_B}imgproc${CV_V}" "${CV_B}objdetect${CV_V}" "${CV_B}shape${CV_V}" "${CV_B}stitching${CV_V}" "${CV_B}videoio${CV_V}" 
	"${CV_B}video${CV_V}" "${CV_B}photo${CV_V}")

set(CMAKE_CXX_FLAGS "-std=c++11 -mhard-float -O2 -fpermissive -L${G_B}/lib")
message(STATUS "Compile flags: ${CMAKE_CXX_FLAGS}")

target_include_directories(mlcapture PUBLIC ${OpenCV_INCLUDE_DIRS} ${NANOMSG_INCLUDE_DIR} "${PROJECT_SOURCE_DIR}/include/")
target_link_libraries(mlcapture LINK_PUBLIC ${OpenCV_LIBS} ${NANOMSG_LIBRARIES} v4l2 pthread)
