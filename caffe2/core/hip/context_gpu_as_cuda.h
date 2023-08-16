#ifndef CAFFE2_CORE_CONTEXT_GPU_AS_CUDA_H_
#define CAFFE2_CORE_CONTEXT_GPU_AS_CUDA_H_

#include "caffe2/core/common.h"
#include "caffe2/core/hip/common_gpu.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_base.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/numa.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"

#include <c10/core/Device.h>

namespace caffe2 {

class CAFFE2_HIP_API CUDAContext final : public HIPContext {
 public:
  // The default cuda context constructor.
  explicit CUDAContext(DeviceIndex gpu_id = -1) : HIPContext(gpu_id) {}
  explicit CUDAContext(const DeviceOption& option) : HIPContext(option) {}
  explicit CUDAContext(Device device) : HIPContext(device) {}

  ~CUDAContext() override {}

  at::Device device() const override {
    return at::Device(CUDA, gpu_id_);
  }

  DeviceType device_type() const override {
    return CUDA;
  }

  static constexpr DeviceType GetDeviceType() {
    return CUDA;
  }
};

using TensorCUDA = Tensor;

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_GPU_AS_CUDA_H_
