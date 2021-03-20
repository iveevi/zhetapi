add_executable(imv samples/features/image_viewer.cpp)
target_link_libraries(imv ${PNG_LIBRARY} ${OPENGL_LIBRARIES} glfw zhp-shared)
