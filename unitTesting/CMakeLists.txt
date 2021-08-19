MACRO(ADD_TEST_CFLAGS target flag)
  SET_PROPERTY(TARGET ${target} APPEND_STRING PROPERTY COMPILE_FLAGS " ${flag}")
ENDMACRO(ADD_TEST_CFLAGS)

MACRO(ADD_TESTCASE NAME PKGS)
  ADD_UNIT_TEST(${NAME} ${NAME})

  SET_TARGET_PROPERTIES(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_DYN_LINK") 
  SET(MODULE_NAME "${NAME}Test")
  STRING(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
  ADD_TEST_CFLAGS(${NAME} "-DBOOST_TEST_MODULE=${MODULE_NAME}") 

  # FOREACH(PKG ${PKGS})
  #   PKG_CONFIG_USE_DEPENDENCY(${NAME} ${PKG})
  # ENDFOREACH(PKG)
  #TARGET_INCLUDE_DIRECTORIES(${NAME} ${PROJECT_NAME} ${kimm_qpoases_INCLUDE_DIRS})
  TARGET_LINK_LIBRARIES(${NAME} ${PROJECT_NAME})
  TARGET_LINK_LIBRARIES(${NAME} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${kimm_qpoases_LIBRARIES})

ENDMACRO(ADD_TESTCASE)

# --- RULES -------------------------------------------------------------------
# --- RULES -------------------------------------------------------------------
# --- RULES -------------------------------------------------------------------

ADD_TESTCASE(Random "kimm_qpoases")
ADD_TESTCASE(InitSet "kimm_qpoases")
ADD_TESTCASE(hcod "kimm_qpoases")
ADD_TESTCASE(cod "kimm_qpoases")
ADD_TESTCASE(ehqp_test "kimm_qpoases")
ADD_TESTCASE(ihqp_test "kimm_qpoases")