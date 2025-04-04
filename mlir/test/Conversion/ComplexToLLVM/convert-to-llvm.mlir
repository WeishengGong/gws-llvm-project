// RUN: mlir-opt %s -convert-complex-to-llvm | FileCheck %s

// Same below, but using the `ConvertToLLVMPatternInterface` entry point
// and the generic `convert-to-llvm` pass.
// RUN: mlir-opt --convert-to-llvm="filter-dialects=complex" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @complex_create
// CHECK-SAME:    (%[[REAL0:.*]]: f32, %[[IMAG0:.*]]: f32)
// CHECK-NEXT:    %[[CPLX0:.*]] = llvm.mlir.poison : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[CPLX1:.*]] = llvm.insertvalue %[[REAL0]], %[[CPLX0]][0] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[CPLX2:.*]] = llvm.insertvalue %[[IMAG0]], %[[CPLX1]][1] : !llvm.struct<(f32, f32)>
func.func @complex_create(%real: f32, %imag: f32) -> complex<f32> {
  %cplx2 = complex.create %real, %imag : complex<f32>
  return %cplx2 : complex<f32>
}

// CHECK-LABEL: func @complex_constant
// CHECK-NEXT:    llvm.mlir.constant([1.000000e+00, 2.000000e+00]) : !llvm.struct<(f64, f64)>
func.func @complex_constant() -> complex<f64> {
  %cplx2 = complex.constant [1.000000e+00, 2.000000e+00] : complex<f64>
  return %cplx2 : complex<f64>
}

// CHECK-LABEL: func @complex_extract
// CHECK-SAME:    (%[[CPLX:.*]]: complex<f32>)
// CHECK-NEXT:    %[[CAST0:.*]] = builtin.unrealized_conversion_cast %[[CPLX]] : complex<f32> to !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[REAL:.*]] = llvm.extractvalue %[[CAST0]][0] : !llvm.struct<(f32, f32)>
// CHECK-NEXT:    %[[IMAG:.*]] = llvm.extractvalue %[[CAST0]][1] : !llvm.struct<(f32, f32)>
func.func @complex_extract(%cplx: complex<f32>) {
  %real1 = complex.re %cplx : complex<f32>
  %imag1 = complex.im %cplx : complex<f32>
  return
}

// CHECK-LABEL: func @complex_addition
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fadd %[[A_REAL]], %[[B_REAL]] : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fadd %[[A_IMAG]], %[[B_IMAG]] : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func.func @complex_addition() {
  %a_re = arith.constant 1.2 : f64
  %a_im = arith.constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = arith.constant 5.6 : f64
  %b_im = arith.constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.add %a, %b : complex<f64>
  return
}

// CHECK-LABEL: func @complex_substraction
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fsub %[[A_REAL]], %[[B_REAL]] : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fsub %[[A_IMAG]], %[[B_IMAG]] : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func.func @complex_substraction() {
  %a_re = arith.constant 1.2 : f64
  %a_im = arith.constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = arith.constant 5.6 : f64
  %b_im = arith.constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.sub %a, %b : complex<f64>
  return
}

// CHECK-LABEL: func @complex_div
// CHECK-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs : complex<f32>
  return %div : complex<f32>
}
// CHECK-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// CHECK-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// CHECK: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// CHECK-DAG: %[[RHS_RE_SQ:.*]] = llvm.fmul %[[RHS_RE]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[RHS_IM_SQ:.*]] = llvm.fmul %[[RHS_IM]], %[[RHS_IM]]  : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[RHS_RE_SQ]], %[[RHS_IM_SQ]]  : f32

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_IM]]  : f32
// CHECK: %[[REAL_TMP_2:.*]] = llvm.fadd %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]]  : f32
// CHECK: %[[IMAG_TMP_2:.*]] = llvm.fsub %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// CHECK: %[[REAL:.*]] = llvm.fdiv %[[REAL_TMP_2]], %[[SQ_NORM]]  : f32
// CHECK: %[[IMAG:.*]] = llvm.fdiv %[[IMAG_TMP_2]], %[[SQ_NORM]]  : f32
// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0] : ![[C_TY]]
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1] : ![[C_TY]]
//
// CHECK: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// CHECK: return %[[CASTED_RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_mul
// CHECK-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_mul(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs : complex<f32>
  return %mul : complex<f32>
}
// CHECK-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// CHECK-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]
// CHECK: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[RHS_RE]], %[[LHS_RE]]  : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[RHS_IM]], %[[LHS_IM]]  : f32
// CHECK: %[[REAL:.*]] = llvm.fsub %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]]  : f32
// CHECK: %[[IMAG:.*]] = llvm.fadd %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0]
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1]

// CHECK: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// CHECK: return %[[CASTED_RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[CASTED_ARG:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : complex<f32> to ![[C_TY:.*>]]
// CHECK: %[[REAL:.*]] = llvm.extractvalue %[[CASTED_ARG]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.extractvalue %[[CASTED_ARG]][1] : ![[C_TY]]
// CHECK-DAG: %[[REAL_SQ:.*]] = llvm.fmul %[[REAL]], %[[REAL]]  : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = llvm.fmul %[[IMAG]], %[[IMAG]]  : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[REAL_SQ]], %[[IMAG_SQ]]  : f32
// CHECK: %[[NORM:.*]] = llvm.intr.sqrt(%[[SQ_NORM]]) : (f32) -> f32
// CHECK: return %[[NORM]] : f32

// CHECK-LABEL: func @complex_addition_with_fmf
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fadd %[[A_REAL]], %[[B_REAL]] {fastmathFlags = #llvm.fastmath<reassoc>} : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fadd %[[A_IMAG]], %[[B_IMAG]] {fastmathFlags = #llvm.fastmath<reassoc>} : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func.func @complex_addition_with_fmf() {
  %a_re = arith.constant 1.2 : f64
  %a_im = arith.constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = arith.constant 5.6 : f64
  %b_im = arith.constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.add %a, %b fastmath<reassoc> : complex<f64>
  return
}

// CHECK-LABEL: func @complex_substraction_with_fmf
// CHECK-DAG:     %[[A_REAL:.*]] = llvm.extractvalue %[[A:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_REAL:.*]] = llvm.extractvalue %[[B:.*]][0] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[A_IMAG:.*]] = llvm.extractvalue %[[A]][1] : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[B_IMAG:.*]] = llvm.extractvalue %[[B]][1] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C0:.*]] = llvm.mlir.poison : !llvm.struct<(f64, f64)>
// CHECK-DAG:     %[[C_REAL:.*]] = llvm.fsub %[[A_REAL]], %[[B_REAL]] {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f64
// CHECK-DAG:     %[[C_IMAG:.*]] = llvm.fsub %[[A_IMAG]], %[[B_IMAG]] {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f64
// CHECK:         %[[C1:.*]] = llvm.insertvalue %[[C_REAL]], %[[C0]][0] : !llvm.struct<(f64, f64)>
// CHECK:         %[[C2:.*]] = llvm.insertvalue %[[C_IMAG]], %[[C1]][1] : !llvm.struct<(f64, f64)>
func.func @complex_substraction_with_fmf() {
  %a_re = arith.constant 1.2 : f64
  %a_im = arith.constant 3.4 : f64
  %a = complex.create %a_re, %a_im : complex<f64>
  %b_re = arith.constant 5.6 : f64
  %b_im = arith.constant 7.8 : f64
  %b = complex.create %b_re, %b_im : complex<f64>
  %c = complex.sub %a, %b fastmath<nnan,ninf> : complex<f64>
  return
}

// CHECK-LABEL: func @complex_div_with_fmf
// CHECK-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// CHECK-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// CHECK-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// CHECK: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// CHECK-DAG: %[[RHS_RE_SQ:.*]] = llvm.fmul %[[RHS_RE]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK-DAG: %[[RHS_IM_SQ:.*]] = llvm.fmul %[[RHS_IM]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[RHS_RE_SQ]], %[[RHS_IM_SQ]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK: %[[REAL_TMP_2:.*]] = llvm.fadd %[[REAL_TMP_0]], %[[REAL_TMP_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK: %[[IMAG_TMP_2:.*]] = llvm.fsub %[[IMAG_TMP_0]], %[[IMAG_TMP_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// CHECK: %[[REAL:.*]] = llvm.fdiv %[[REAL_TMP_2]], %[[SQ_NORM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK: %[[IMAG:.*]] = llvm.fdiv %[[IMAG_TMP_2]], %[[SQ_NORM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0] : ![[C_TY]]
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1] : ![[C_TY]]
//
// CHECK: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// CHECK: return %[[CASTED_RESULT]] : complex<f32>
func.func @complex_div_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs fastmath<nsz,arcp> : complex<f32>
  return %div : complex<f32>
}


// CHECK-LABEL: func @complex_mul_with_fmf
// CHECK-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// CHECK-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// CHECK-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]
// CHECK: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[RHS_RE]], %[[LHS_RE]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[RHS_IM]], %[[LHS_IM]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32
// CHECK: %[[REAL:.*]] = llvm.fsub %[[REAL_TMP_0]], %[[REAL_TMP_1]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32
// CHECK: %[[IMAG:.*]] = llvm.fadd %[[IMAG_TMP_0]], %[[IMAG_TMP_1]] {fastmathFlags = #llvm.fastmath<contract, afn>} : f32

// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0]
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1]

// CHECK: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// CHECK: return %[[CASTED_RESULT]] : complex<f32>
func.func @complex_mul_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs fastmath<contract,afn>  : complex<f32>
  return %mul : complex<f32>
}

// CHECK-LABEL: func @complex_abs_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
// CHECK: %[[CASTED_ARG:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : complex<f32> to ![[C_TY:.*>]]
// CHECK: %[[REAL:.*]] = llvm.extractvalue %[[CASTED_ARG]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.extractvalue %[[CASTED_ARG]][1] : ![[C_TY]]
// CHECK-DAG: %[[REAL_SQ:.*]] = llvm.fmul %[[REAL]], %[[REAL]] {fastmathFlags = #llvm.fastmath<contract>} : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = llvm.fmul %[[IMAG]], %[[IMAG]] {fastmathFlags = #llvm.fastmath<contract>} : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[REAL_SQ]], %[[IMAG_SQ]] {fastmathFlags = #llvm.fastmath<contract>} : f32
// CHECK: %[[NORM:.*]] = llvm.intr.sqrt(%[[SQ_NORM]]) : (f32) -> f32
// CHECK: return %[[NORM]] : f32
func.func @complex_abs_with_fmf(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg fastmath<contract> : complex<f32>
  return %abs : f32
}
