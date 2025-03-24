add_custom_target(Gws)
set_target_properties(Gws PROPERTIES FOLDER "MLIR/Examples")

macro(add_gws name)
  add_dependencies(Gws ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_gws name)

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_gws(gwsc
  gwsc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp

  DEPENDS
  GwsOpsIncGen
  )
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(gwsc
  PRIVATE
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms)
