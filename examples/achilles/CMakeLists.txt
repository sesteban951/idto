include_directories(${PROJECT_SOURCE_DIR})

add_executable(achilles achilles.cc)
target_link_libraries(achilles
                      drake::drake
                      example_base
                      find_resource
                      gflags)

configure_file(achilles.yaml
               ${CMAKE_BINARY_DIR}/idto/examples/achilles/achilles.yaml COPYONLY)

add_test(NAME achilles_test COMMAND achilles --test)
