

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
IF (${PACKAGE_NAME}_ENABLE_Triutils)

   TRIBITS_ADD_EXECUTABLE_AND_TEST(
     Tpetra_QOR
     SOURCES test_qor.cpp 
     ARGS 
     COMM serial mpi
     )

ENDIF()
