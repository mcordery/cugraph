# Parse version information from version header:
include("${CMAKE_CURRENT_LIST_DIR}/libhipcxx-header-search.cmake")

file(READ "${_libhipcxx_VERSION_INCLUDE_DIR}/hip/std/detail/__config"
  libhipcxx_VERSION_HEADER
)

string(REGEX MATCH
  "#define[ \t]+_LIBCUDACXX_CUDA_API_VERSION[ \t]+([0-9]+)" unused_var
  "${libhipcxx_VERSION_HEADER}"
)

set(libhipcxx_VERSION_FLAT ${CMAKE_MATCH_1})
math(EXPR libhipcxx_VERSION_MAJOR "${libhipcxx_VERSION_FLAT} / 1000000")
math(EXPR libhipcxx_VERSION_MINOR "(${libhipcxx_VERSION_FLAT} / 1000) % 1000")
math(EXPR libhipcxx_VERSION_PATCH "${libhipcxx_VERSION_FLAT} % 1000")
set(libhipcxx_VERSION_TWEAK 0)

set(libhipcxx_VERSION
  "${libhipcxx_VERSION_MAJOR}.${libhipcxx_VERSION_MINOR}.${libhipcxx_VERSION_PATCH}.${libhipcxx_VERSION_TWEAK}"
)

set(PACKAGE_VERSION ${libhipcxx_VERSION})
set(PACKAGE_VERSION_COMPATIBLE FALSE)
set(PACKAGE_VERSION_EXACT FALSE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)

if(PACKAGE_VERSION VERSION_GREATER_EQUAL PACKAGE_FIND_VERSION)
  if(PACKAGE_FIND_VERSION_MAJOR VERSION_EQUAL libhipcxx_VERSION_MAJOR)
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
  endif()

  if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
