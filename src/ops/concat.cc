/* Copyright 2017 Stanford, NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/ops/concat.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::PhysicalRegion;
using Legion::coord_t;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using PCG::Node;

Tensor FFModel::concat(int n,
                       const Tensor* tensors,
                       int axis,
                       const char *name)
{
  Layer* concat = new Layer(this, OP_CONCAT, name, n/*inputs*/,
                            0/*weights*/, 1/*outputs*/, tensors);
  int numdim = tensors[0]->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdim; i++)
    dims[i] = tensors[0]->dims[i];
  for (int i = 1; i < n; i++) {
    assert(tensors[i]->data_type == tensors[0]->data_type);
    assert(tensors[i]->num_dims == tensors[0]->num_dims);
    for (int j = 0; j < numdim; j++) {
      if (j != numdim - axis - 1) {
        assert(tensors[i]->dims[j] == tensors[0]->dims[j]);
      } else {
        dims[j] += tensors[i]->dims[j];
      }
    }
  }
  concat->outputs[0] = create_tensor_legion_ordering(
      numdim, dims, tensors[0]->data_type, concat, 0, true/*create_grad*/);
  concat->add_int_property("legion_axis", numdim-axis-1);
  layers.push_back(concat);
  return concat->outputs[0];
#ifdef DEADCODE
  assert(axis < 0);
  Concat *cat = new Concat(*this, n, tensors, -1-axis, name);
  layers.push_back(cat);
  return cat->outputs[0];
#endif
}

Op* Concat::create_operator_from_layer(
    FFModel& model,
    const Layer* layer,
    const std::vector<ParallelTensor>& inputs) {
  long long value;
  layer->get_int_property("legion_axis", value);
  int legion_axis = value;
  return new Concat(model, inputs.size(), inputs.data(), legion_axis, layer->name);
}


Concat::Concat(FFModel& model,
               int _n, const ParallelTensor* _tensors,
               int _legion_axis,
               const char* name)
: Op(model, OP_CONCAT, name, _n/*inputs*/, 0/*weights*/, 1/*outputs*/, _tensors),
  axis(_legion_axis)
{
  //TODO: swich to use the Legion dim ordering
  int num_dim = inputs[0]->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < num_dim; i++)
    dims[i] = inputs[0]->dims[num_dim-1-i];
  for (int i = 1; i < numInputs; i++) {
    assert(inputs[i]->data_type == inputs[0]->data_type);
    assert(inputs[i]->num_dims == inputs[0]->num_dims);
    for (int j = 0; j < num_dim; j++) {
      if (j != axis)
        assert(inputs[i]->dims[j] == inputs[0]->dims[j]);
      else {
        // Assert that the concat dim cannot be parallelized
        assert(inputs[i]->dims[j].parallel_idx == -1);
        assert(inputs[i]->dims[j].degree == 1);
        dims[num_dim-1-j].size += inputs[i]->dims[j].size;
      }
    }
  }
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor(num_dim, dims, inputs[0]->data_type, this);
}

void Concat::init_meta(ConcatMeta *m) const
{
  m->axis = this->axis;
}

void Concat::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(CONCAT_INIT_TASK_ID, parallel_is,
    TaskArgument(this, sizeof(Concat)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part_grad, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, inputs[i]->region_grad));
    launcher.add_field(i + numInputs + 1, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  Concat* cc = (Concat*) task->args;
  FFHandler handler = *((const FFHandler*) task->local_args);
  ConcatMeta* m = new ConcatMeta(handler);
  // Note that our internal axis index ordering is opposite to other frameworks
  cc->init_meta(m);
  m->profiling = cc->profiling;
  std::strcpy(m->op_name, cc->name);
  return m;
}

void Concat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(CONCAT_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Concat)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i]->region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/
void Concat::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  const ConcatMeta* m = *((ConcatMeta**) task->local_args);
  // Note that our internal axis index ordering is opposite to other frameworks
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  //assert(out_domain.get_dim() == cc->outputs[0].num_dims);
  Domain in_domain[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_domain[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  float *output = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  const float *inputs[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    inputs[i] = helperGetTensorPointerRO<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
  
  Concat::forward_kernel_wrapper(m, output, inputs, cc->numInputs, cc->axis, out_domain, in_domain);
}

void Concat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(CONCAT_BWD_TASK_ID, parallel_is,
    TaskArgument(this, sizeof(Concat)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(inputs[i]->part_grad, 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i]->region_grad));
    //LogicalRegion lr = inputs[i]->region_grad;
    //printf("concat[%d]: region(%d,%d,%d)\n", i+1, lr.get_index_space().get_id(), lr.get_field_space().get_id(), lr.get_tree_id());
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): output_grad
  regions[1..numInputs](I/O): input_grad
*/
void Concat::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  const ConcatMeta* m = *((ConcatMeta**) task->local_args);
  // Note that our internal axis index ordering is opposite to other frameworks
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  //assert(out_grad_domain.get_dim() == cc->outputs[0].num_dims);
  Domain in_grad_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    in_grad_domains[i] = runtime->get_index_space_domain(
        ctx, task->regions[i+1].region.get_index_space());
  const float *output_grad = helperGetTensorPointerRO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  float *input_grads[MAX_NUM_INPUTS];
  for (int i = 0; i < cc->numInputs; i++)
    input_grads[i] = helperGetTensorPointerRW<float>(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);

  Concat::backward_kernel_wrapper(m, output_grad, input_grads, cc->numInputs, cc->axis,
                                  out_grad_domain, in_grad_domains);
}

bool Concat::get_int_parameter(PMParameter para, int* value) const
{
  switch (para) {
    case PM_AXIS:
      *value = axis;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Concat::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  assert (numInputs <= MAX_NUM_INPUTS);
  ParallelTensorBase sub_inputs[MAX_NUM_INPUTS], sub_output;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, op_type)) {
    return false;
  }
  for (int i = 0; i < numInputs; i++) {
    if (!inputs[i]->get_input_sub_tensor(pc, sub_inputs[i], op_type)) {
      return false;
    }
  }

  ConcatMeta *m = sim->concat_meta;
  this->init_meta(m);

  sim->free_all();
  float *input_ptrs[MAX_NUM_INPUTS];
  float *input_grad_ptrs[MAX_NUM_INPUTS];
  bool out_of_memory = false;
  for (int i = 0; i < numInputs; i++) {
    input_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (input_ptrs[i] == NULL);
  }
  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  out_of_memory = out_of_memory || (output_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }

  Domain out_domain = sub_output.get_domain();
  Domain in_domains[MAX_NUM_INPUTS];
  for (int i = 0; i < numInputs; i++) {
    in_domains[i] = sub_inputs[i].get_domain();
  }

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m, output_ptr, input_ptrs, numInputs, axis, out_domain, in_domains);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    for (int i = 0; i < numInputs; i++) {
      input_grad_ptrs[i] = (float *)sim->allocate(sub_inputs[i].get_volume(), DT_FLOAT);
      out_of_memory = out_of_memory || (input_grad_ptrs[i] == NULL);
    }
    float *output_grad_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    out_of_memory = out_of_memory || (output_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel_wrapper(m, output_grad_ptr, input_grad_ptrs,
                              numInputs, axis, out_domain, in_domains);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Concat] name(%s) forward_time(%.4lf) backward_time(%.4lf)\n",
        name,
        cost_metrics.forward_time,
        cost_metrics.backward_time);
  } else {
    printf("[Measure Concat] name(%s) forward_time(%.4lf)\n",
        name, cost_metrics.forward_time);
  }

  return true;
}

Node FFModel::get_or_create_concat_node(int num_inputs,
                                        const ParallelTensor* inputs,
                                        int axis)
{
  size_t hash = std::hash<int>()(num_inputs);
  hash = hash * 31 + std::hash<int>()(axis);
  for (int i = 0; i < num_inputs; i++)
    hash = hash * 31 + inputs[i]->get_owner_independent_hash();
  const auto& it = cached_concat_ops.find(hash);
  Concat* concat = NULL;
  if (it != cached_concat_ops.end()) {
    concat = it->second;
  } else {
    concat = new Concat(*this, num_inputs, inputs, axis, NULL);
    cached_concat_ops[hash] = concat;
  }
  Node ret;
  ret.guid = node_global_guid ++;
  ret.ptr = concat;
  return ret;
}

size_t Concat::get_params_hash() const {
  size_t hash = std::hash<int>()(this->numInputs);
  hash_combine(hash, this->axis);
  for (int i = 0; i < this->numInputs; i++) {
    hash_combine(hash, inputs[0]->get_owner_independent_hash());
  }

  return hash;
}

}; // namespace FlexFlow