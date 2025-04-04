//===- Passes.h - Gws Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Gws.
//
//===----------------------------------------------------------------------===//

#ifndef GWS_PASSES_H
#define GWS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace gws {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Gws IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

} // namespace gws
} // namespace mlir

#endif // GWS_PASSES_H
