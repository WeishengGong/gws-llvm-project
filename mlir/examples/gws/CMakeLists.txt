add_custom_target(Gws)
set_target_properties(Gws PROPERTIES FOLDER "MLIR/Examples")

macro(add_gws name)
  add_dependencies(Gws ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_gws name)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_gws(gwsc
  gwsc.cpp
  parser/AST.cpp
  )
include_directories(include/)
target_link_libraries(gwsc
  PRIVATE
    MLIRSupport)
