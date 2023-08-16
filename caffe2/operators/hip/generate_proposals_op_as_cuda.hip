#include "caffe2/operators/generate_proposals_op.h"
#
namespace caffe2 {

template <>
bool GenerateProposalsOp<CUDAContext>::RunOnDevice() {
  return GenerateProposalsOp<HIPContext>::RunOnDevice();
}

REGISTER_HIP_OPERATOR(GenerateProposals, GenerateProposalsOp<CUDAContext>);
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_HIP(
    GenerateProposals,
    caffe2::GenerateProposalsOp<caffe2::CUDAContext>);
