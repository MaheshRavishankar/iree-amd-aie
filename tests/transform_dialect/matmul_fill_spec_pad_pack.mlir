func.func @matmul_i32() {
  %c0_i32 = arith.constant 0: i32
  %c0 = arith.constant 0 : index
  %arg0_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1000x2000xi32>>
  %arg0 = flow.dispatch.tensor.load %arg0_binding, offsets = [0, 0], sizes = [1000, 2000], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1000x2000xi32>> -> tensor<1000x2000xi32>
  %arg1_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2000x500xi32>>
  %arg1 = flow.dispatch.tensor.load %arg1_binding, offsets = [0, 0], sizes = [2000, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2000x500xi32>> -> tensor<2000x500xi32>
  %arg2_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) flags(None) : !flow.dispatch.tensor<writeonly:tensor<1000x500xi32>>
  %empty = tensor.empty() : tensor<1000x500xi32>
  %0 = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1000x500xi32>) -> tensor<1000x500xi32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<1000x2000xi32>, tensor<2000x500xi32>)
      outs(%0 : tensor<1000x500xi32>) -> tensor<1000x500xi32>
  flow.dispatch.tensor.store %1, %arg2_binding, offsets = [0, 0], sizes = [1000, 500], strides = [1, 1] : tensor<1000x500xi32> -> !flow.dispatch.tensor<writeonly:tensor<1000x500xi32>>
  return
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.read_only}) {
    %ops = transform.structured.match ops{["linalg.fill", "linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %fill, %matmul = transform.split_handle %ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // First level tile to forall.
    %first_level_tiled_matmul, %outer_forall =
      transform.structured.tile_using_forall %matmul tile_sizes [64, 64]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill operation into the forall loop.
    %fused_fill, %1 = transform.structured.fuse_into_containing_op %fill into %outer_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pack the matmul.
    %first_level_tiled_transposed_l2_packed_matmul = transform.structured.pack %first_level_tiled_matmul packed_sizes = [64, 64, 256]
      : (!transform.any_op) -> (!transform.any_op)

    %lhs_l2_pack = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[0] : (!transform.any_op) -> (!transform.any_op)
 
    %rhs_transposed_l2_pack_op = transform.get_producer_of_operand %first_level_tiled_transposed_l2_packed_matmul[1] : (!transform.any_op) -> (!transform.any_op)
    %first_level_tiled_l2_packed_matmul, %rhs_l2_pack, %rhs_unpack =
      transform.structured.pack_transpose %rhs_transposed_l2_pack_op with_compute_op(%first_level_tiled_transposed_l2_packed_matmul)
      outer_perm = [0, 1] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Promote the fused fill to memory
    %result_l2 = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %result_l2_buffer, %result_t2_new = transform.structured.bufferize_to_allocation %result_l2
        {memory_space = 1, bufferize_destination_only, mempcy = "linalg.copy", emit_dealloc} : !transform.any_op

    // First level for loop.
    %first_level_tiled_reduction_matmul, %outer_for_loop =
      transform.structured.tile_using_for %first_level_tiled_l2_packed_matmul [0, 0, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"scf.for">)

    // Fuse the pack operations in.
    %fused_lhs_l2_pack, %7 = transform.structured.fuse_into_containing_op %lhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l2_pack, %8 = transform.structured.fuse_into_containing_op %rhs_l2_pack into %outer_for_loop : (!transform.any_op, !transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    // Promote the lhs to shared memory
    %lhs_l2_pack_buffer, %lhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Promote the rhs to shared memory
    %rhs_l2_pack_buffer, %rhs_l2_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l2_pack
       {memory_space = 1, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Second level tile to forall with tile_sizes.
    %second_level_tiled_matmul, %inner_forall =
      transform.structured.tile_using_forall %first_level_tiled_reduction_matmul tile_sizes [0, 0, 0, 32, 32]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.generic.
    %l1_packed = transform.structured.pack %second_level_tiled_matmul packed_sizes = [0, 0, 0, 4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    %l1_packed_lhs = transform.get_producer_of_operand %l1_packed[0]
      : (!transform.any_op) -> (!transform.any_op)
    %lhs_l1_packed_matmul, %lhs_l1_pack_op, %lhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_lhs with_compute_op(%l1_packed)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    %l1_packed_rhs = transform.get_producer_of_operand %lhs_l1_packed_matmul[1]
      : (!transform.any_op) -> (!transform.any_op)
    %operands_l1_packed_matmul, %rhs_l1_pack_op, %rhs_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_rhs with_compute_op(%lhs_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %l1_packed_output = transform.get_consumers_of_result %operands_l1_packed_matmul[0]
      : (!transform.any_op) -> (!transform.any_op)
    %l1_packed_matmul, %output_l1_pack_op, %output_l1_unpack_op =
      transform.structured.pack_transpose %l1_packed_output with_compute_op(%operands_l1_packed_matmul)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Promote the result to local memory
    %output_l1_pack_op_source_buffer, %output_l1_pack_op_new = transform.structured.bufferize_to_allocation %output_l1_pack_op
        {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Second level for loop.
    %second_level_tiled_reduction_matmul, %inner_for_loop =
      transform.structured.tile_using_for %l1_packed_matmul [0, 0, 0, 0, 0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse pack op with the consumer.
    %fused_lhs_l1_pack, %9 = transform.structured.fuse_into_containing_op %lhs_l1_pack_op into %inner_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_rhs_l1_pack, %10 = transform.structured.fuse_into_containing_op %rhs_l1_pack_op into %inner_for_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote the LHS to local memory.
    %lhs_l1_pack_buffer, %lhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_lhs_l1_pack
      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote the RHS to local memory.
    %rhs_l1_pack_buffer, %rhs_l1_pack_new = transform.structured.bufferize_to_allocation %fused_rhs_l1_pack
      {memory_space = 2, bufferize_destination_only, memcpy_op = "linalg.copy", emit_dealloc} : !transform.any_op

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Hoist static alloc out of the loops
    %func = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.iree.hoist_static_alloc %func : (!transform.any_op) -> ()

    // Peel the for loop
    %peeled_outer_loop, %remainder = transform.loop.peel %outer_for_loop {peel_front = true}
    : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Find the fill operation to fuse
    %fill_op = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    // Get the consumers of the fill. It has to be a scf.forall op.
    %peeled_loop = transform.get_consumers_of_result %fill_op[0] : (!transform.any_op) -> (!transform.op<"scf.forall">)

    // Fuse the fill within the loop.
    %peel_fused, %13 = transform.structured.fuse_into_containing_op %fill_op into %peeled_loop : (!transform.any_op, !transform.op<"scf.forall">) -> (!transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %14 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: func.func @matmul_i32()
//   CHECK-DAG:   %[[LHS_ALLOC:.+]] = memref.alloc() : memref<4x4x8x8xi32, 2>
//   CHECK-DAG:   %[[RHS_ALLOC:.+]] = memref.alloc() : memref<4x8x4x8xi32, 2>
//   CHECK-DAG:   %[[RESULT_ALLOC:.+]] = memref.alloc() : memref<4x8x4x8xi32, 2>
//   CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
//   CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
//   CHECK-DAG:   %[[RESULT_BINDING:.+]] = hal.interface.binding.subspan set(0) binding(2)
//   CHECK-DAG:   %[[RESULT:.+]] = flow.dispatch.tensor.load %[[RESULT_BINDING]]
//   CHECK-DAG:   %[[OUTER:.+]] = scf.forall
//  CHECK-SAME:       shared_outs(%[[OUTER_ITER_ARG:.+]] = %[[RESULT]])
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//   CHECK-DAG:     %[[RESULT_SLICE:.+]] = tensor.extract_slice %[[OUTER_ITER_ARG]]
//   CHECK-DAG:     %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xi32>
//       CHECK:     %[[PEELED_LHS_PAD_INPUT:.+]] = tensor.extract_slice %[[LHS_SLICE]]
//       CHECK:     %[[PEELED_LHS_PAD:.+]] = tensor.pad %[[PEELED_LHS_PAD_INPUT]]
//       CHECK:     %[[PEELED_RHS_PAD_INPUT:.+]] = tensor.extract_slice %[[RHS_SLICE]]
//       CHECK:     %[[PEELED_RHS_PAD:.+]] = tensor.pad %[[PEELED_RHS_PAD_INPUT]]
//       CHECK:     %[[PEELED_INNER_FORALL:.+]] = scf.forall
//  CHECK-SAME:         shared_outs(%[[PEELED_INNER_ITERARG:.+]] = %[[EMPTY]])
//   CHECK-DAG:       %[[PEELED_LHS_PAD_SLICE:.+]] = tensor.extract_slice %[[PEELED_LHS_PAD]]
//   CHECK-DAG:       %[[PEELED_RHS_PAD_SLICE:.+]] = tensor.extract_slice %[[PEELED_RHS_PAD]]
//   CHECK-DAG:       %[[PEELED_RESULT_ITERARG_SLICE:.+]] = tensor.extract_slice %[[PEELED_INNER_ITERARG]]
//       CHECK:       %[[PEELED_RESULT_BUFFER:.+]] = bufferization.to_tensor %[[RESULT_ALLOC]]
//       CHECK:       %[[PEELED_RESULT_FILL:.+]] = linalg.fill
//  CHECK-SAME:           outs(%[[PEELED_RESULT_BUFFER]] :
//       CHECK:       %[[PEELED_INNER_FOR:.+]] = scf.for
//  CHECK-SAME:           iter_args(%[[PEELED_INNER_FOR_ITER_ARG:.+]] = %[[PEELED_RESULT_FILL]])
//       CHECK:         %[[PEELED_LHS_INNER_SLICE:.+]] = tensor.extract_slice %[[PEELED_LHS_PAD_SLICE]]
//       CHECK:         %[[PEELED_LHS_BUFFER:.+]] = bufferization.to_tensor %[[LHS_ALLOC]]
//       CHECK:         %[[PEELED_LHS_PACK:.+]] = tensor.pack %[[PEELED_LHS_INNER_SLICE]]
//  CHECK-SAME:             into %[[PEELED_LHS_BUFFER]]
//       CHECK:         %[[PEELED_RHS_INNER_SLICE:.+]] = tensor.extract_slice %[[PEELED_RHS_PAD_SLICE]]
//       CHECK:         %[[PEELED_RHS_BUFFER:.+]] = bufferization.to_tensor %[[RHS_ALLOC]]
//       CHECK:         %[[PEELED_RHS_PACK:.+]] = tensor.pack %[[PEELED_RHS_INNER_SLICE]]
//  CHECK-SAME:             into %[[PEELED_RHS_BUFFER]]
//       CHECK:         %[[PEELED_GENERIC:.+]] = linalg.generic
//  CHECK-SAME:             ins(%[[PEELED_LHS_PACK]], %[[PEELED_RHS_PACK]] :
//  CHECK-SAME:             outs(%[[PEELED_RESULT_FILL]] :
//       CHECK:         scf.yield %[[PEELED_GENERIC]]
//       CHECK:       %[[PEELED_RESULT_UNPACK:.+]] = tensor.unpack %[[PEELED_INNER_FOR]]
//  CHECK-SAME:           into %[[PEELED_RESULT_ITERARG_SLICE]]
//       CHECK:       scf.forall.in_parallel
//  CHECK-NEXT:         tensor.parallel_insert_slice %[[PEELED_RESULT_UNPACK]] into %[[PEELED_INNER_ITEERARG]]
