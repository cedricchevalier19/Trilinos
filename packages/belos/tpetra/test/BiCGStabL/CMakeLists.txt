

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
IF (${PACKAGE_NAME}_ENABLE_Triutils)

   TRIBITS_ADD_EXECUTABLE_AND_TEST(
     Tpetra_BiCGStabL
     SOURCES test_bicgstabl.cpp 
     ARGS 
     COMM serial mpi
     )

ENDIF()
