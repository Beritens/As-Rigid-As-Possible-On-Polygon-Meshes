cmake_minimum_required(VERSION 3.16)
project(deformation)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)


# Enable the target igl::glfw
igl_include(glfw)
# Other modules you could enable
#igl_include(embree)
igl_include(imgui)
#igl_include(opengl)
#igl_include(stb)
#igl_include(predicates)
#igl_include(xml)
#igl_include(copyleft cgal)
#igl_include(copyleft comiso)
#igl_include(copyleft core)
#igl_include(copyleft cork)
#igl_include(copyleft tetgen)
#igl_include(restricted matlab)
#igl_include(restricted mosek)
#igl_include(restricted triangle)

# Add your project files
include_directories("include/")
file(GLOB SRCFILES source/*.cpp)
message(STATUS "Source files: ${SRCFILES}")


add_library(srcfiles ${SRCFILES})
target_link_libraries(srcfiles igl::core igl::glfw )

add_executable(deformation "main.cpp")




# Link igl (and the glfw module) to your project
target_link_libraries(deformation srcfiles igl::core igl::glfw igl::imgui
  ## Other modules you could link to
  # igl::embree
  # igl::opengl
  # igl::stb
  # igl::predicates
  # igl::xml
  # igl_copyleft::cgal
  # igl_copyleft::comiso
  # igl_copyleft::core
  # igl_copyleft::cork
  # igl_copyleft::tetgen
  # igl_restricted::matlab
  # igl_restricted::mosek
  # igl_restricted::triangle
  )
