/* Copyright 2021 Stanford, Facebook
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

#include "flexflow/utils/cuda_helper.h"
#include "clip.h"

void DataLoader::load_text_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  TensorAccessorR<float, 3> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_batch_input(regions[1],
                                            task->regions[1],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            false /*readOutput*/);

  /// @warning why batch_size is the last dimension?
  int batch_size = acc_batch_input.rect.hi[3] - acc_batch_input.rect.lo[3] + 1;
  int embed_size = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  int seq_length = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;

  assert(acc_batch_input.rect.hi[0] == acc_full_input.rect.hi[0]);
  assert(acc_batch_input.rect.lo[0] == acc_full_input.rect.lo[0]);
  assert(acc_batch_input.rect.hi[1] == acc_full_input.rect.hi[1]);
  assert(acc_batch_input.rect.lo[1] == acc_full_input.rect.lo[1]);
  assert(acc_batch_input.rect.hi[2] == acc_full_input.rect.hi[2]);
  assert(acc_batch_input.rect.lo[2] == acc_full_input.rect.lo[2]);

  float *input_zc;
  checkCUDA(cudaHostAlloc(&input_zc,
                          sizeof(float) * acc_batch_input.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * embed_size * seq_length;
    for (int j = 0; j < embed_size * seq_length; j++)
      input_zc[i * embed_size * seq_length + j] =
          acc_full_input.ptr[base_offset + j];
  }
  checkCUDA(cudaMemcpy(acc_batch_input.ptr,
                       input_zc,
                       sizeof(float) * acc_batch_input.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(input_zc));
}

void DataLoader::load_visual_input(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;

  /// @warning why does # of dims have to be 5 instead of 4?
  TensorAccessorR<float, 5> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 5> acc_batch_input(regions[1],
                                            task->regions[1],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            false /*readOutput*/);
  coord_t batch_size =
      acc_batch_input.rect.hi[3] - acc_batch_input.rect.lo[3] + 1;
  coord_t channels =
      acc_batch_input.rect.hi[2] - acc_batch_input.rect.lo[2] + 1;
  coord_t height = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  coord_t width = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  // FIXME: currently assume continous indices
  assert(batch_size == meta->num_samples);
  for (int i = 1; i < batch_size; i++)
    assert(meta->idxs[i] == meta->idxs[0] + i);
  coord_t start_idx = meta->idxs[0];
  float const *input_zc =
      acc_full_input.ptr + start_idx * channels * height * width;
  copy_kernel<<<GET_BLOCKS(acc_batch_input.rect.volume()), CUDA_NUM_THREADS>>>(
      acc_batch_input.ptr, input_zc, acc_batch_input.rect.volume());
  checkCUDA(cudaDeviceSynchronize());
}


void DataLoader::load_label(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  SampleIdxs *meta = (SampleIdxs *)task->local_args;
  TensorAccessorR<float, 2> acc_full_label(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 3> acc_batch_label(regions[1],
                                            task->regions[1],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            false /*readOutput*/);
  int batch_size = acc_batch_label.rect.hi[1] - acc_batch_label.rect.lo[1] + 1;
  int num_label = acc_batch_label.rect.hi[0] - acc_batch_label.rect.lo[0] + 1;
//  assert(num_label == 1); // Kaggle dataset a has single label
  assert(acc_batch_label.rect.hi[0] == acc_full_label.rect.hi[0]);
  assert(acc_batch_label.rect.lo[0] == acc_full_label.rect.lo[0]);
  float *label_zc;
  checkCUDA(cudaHostAlloc(&label_zc,
                          sizeof(float) * acc_batch_label.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * num_label;
    for (int j = 0; j < num_label; j++)
      label_zc[i * num_label + j] = acc_full_label.ptr[base_offset + j];
    // printf("meta->idxs[%d]=%d label=%.2lf\n", i, meta->idxs[i], label_zc[i]);
  }
  checkCUDA(cudaMemcpy(acc_batch_label.ptr,
                       label_zc,
                       sizeof(float) * acc_batch_label.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(label_zc));
}

