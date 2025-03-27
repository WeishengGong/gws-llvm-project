//===- Dialect.h - Dialect definition for the Gws IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Gws language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_GWS_DIALECT_H
#define MLIR_GWS_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the gws
/// dialect.
#include "gws/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// gws operations.
#define GET_OP_CLASSES
#include "gws/Ops.h.inc"

#endif // MLIR_GWS_DIALECT_H
