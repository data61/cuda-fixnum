
set(nose_cmd "nosetests3")
set(testcases "cudapaillier.py")

find_program(Nose ${nose_cmd}
  HINTS ENV PATH
  DOC "Location of the Nose test framework")

if (Nose STREQUAL "Nose-NOTFOUND")
  message(WARNING "Couldn't find the Nose test framework (for Python 3); test suite will not be available.")
else()
  add_custom_target(check
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src" ${nose_cmd} ${testcases}
    DEPENDS paillier
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Run tests")
endif ()
