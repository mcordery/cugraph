# Parse version information from version header:
unset(_libhipcxx_VERSION_INCLUDE_DIR CACHE) # Clear old result to force search
find_path(_libhipcxx_VERSION_INCLUDE_DIR hip/std/detail/__config NO_DEFAULT_PATH # Only search explicit paths below:
          PATHS "${CMAKE_CURRENT_LIST_DIR}/../../../include" # Install tree
)
set_property(CACHE _libhipcxx_VERSION_INCLUDE_DIR PROPERTY TYPE INTERNAL)
