#ifndef _FLEXFLOW_RESHAPE_H
#define _FLEXFLOW_RESHAPE_H

#include "flexflow/device.h"
#include "flexflow/fftype.h"
#include "flexflow/layer.h"
#include "flexflow/node.h"
#include "flexflow/op_meta.h"
#include "flexflow/operator.h"
#include "flexflow/ops/reshape_params.h"

namespace FlexFlow {

class ReshapeMeta : public OpMeta {
public:
  ReshapeMeta(FFHandler handler);
  DataType data_type;
};

class Reshape : public Op {
public:
  using Params = ReshapeParams;
  using Input = ParallelTensor;
  Reshape(FFModel &model,
          const ParallelTensor input,
          std::vector<int> const &shape,
          char const *name);
  Reshape(FFModel &model,
          Params const &params,
          const Input input,
          char const *name = nullptr);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);

  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void forward_kernel(const T *input_ptr,
                             T *output_ptr,
                             size_t num_elements,
                             ffStream_t stream);
  template <typename T>
  static void forward_kernel_wrapper(const T *input_ptr,
                                     T *output_ptr,
                                     size_t num_elements);
  template <typename T>
  static void backward_kernel(T *input_grad_ptr,
                              const T *output_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  template <typename T>
  static void backward_kernel_wrapper(T *input_grad_ptr,
                                      const T *output_grad_ptr,
                                      size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  void serialize(Legion::Serializer &s) const override;
  static PCG::Node deserialize(FFModel &ff,
                               Legion::Deserializer &d,
                               ParallelTensor inputs[],
                               int num_inputs);
  Op *materialize(FFModel &ff,
                  ParallelTensor inputs[],
                  int num_inputs) const override;
  Params get_params() const;

public:
  size_t shape_length;
  int shape_array[MAX_TENSOR_DIM];
};

}; // namespace FlexFlow

#endif
