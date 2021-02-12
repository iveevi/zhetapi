# Include directories
include_directories(PUBLIC engine)
include_directories(PUBLIC glad)
include_directories(PUBLIC /usr/local/include)
include_directories(PUBLIC CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})