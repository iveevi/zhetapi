# Compile gui file
add_executable(gui source/exp/gui.cpp ${EXT_SOURCE})
target_link_libraries(gui ${PNG_LIBRARY} ${OPENGL_LIBRARIES} glfw zhp-shared)