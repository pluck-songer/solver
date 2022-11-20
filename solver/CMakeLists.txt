add_library(gauss_newton gauss_newton.cpp)

install(TARGETS gauss_newton
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)