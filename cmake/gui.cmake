# Compile gui file
add_executable(gui experimental/gui.cpp ${EXT_SOURCE})
target_link_libraries(gui ${PNG_LIBRARY} ${OPENGL_LIBRARIES} glfw zhp-shared)