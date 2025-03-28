add_custom_target(Gws)
set_target_properties(Gws PROPERTIES FOLDER "MLIR/Examples")

macro(add_gws name)
  add_dependencies(Gws ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_gws name)

include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/GwsCombine.td)
mlir_tablegen(GwsCombine.inc -gen-rewriters)
add_public_tablegen_target(GwsCombineIncGen)

add_gws(gwsc
  gwsc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/ShapeInferencePass.cpp
  mlir/GwsCombine.cpp

  DEPENDS
  GwsOpsIncGen
  GwsShapeInferenceInterfaceIncGen
  GwsCombineIncGen
  )
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(gwsc
  PRIVATE
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRCallInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms)
