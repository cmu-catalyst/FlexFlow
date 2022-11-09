/* Copyright 2021 Facebook, Stanford
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

#include "flexflow/model.h"
#define MAX_NUM_SAMPLES 65536

using namespace Legion;
using namespace FlexFlow;

struct UniTConfig {
  UniTConfig(void);

  // Text Transformer arguments
  // hidden_size = embed_dim (for multi-head attention) = transformer_width
  int tt_hidden_size, tt_num_heads,  tt_num_layers, sequence_length;

  // Vision Transformer arguments
  int vt_hidden_size, vt_num_heads, vt_num_layers;

  // Decoder Transformer arguments
  int dec_hidden_size, dec_num_heads, dec_num_layers;

  int in_channels, image_size,  kernel_size, padding;
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             UniTConfig const &tf,
             Tensor const &_text_input,
             Tensor const &_visual_input,
             Tensor const &_label,
             Tensor const &_output_tensor);

  void next_batch(FFModel &ff);
  void reset();
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  static void load_text_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);
  static void load_visual_input(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);
  static void load_label(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);

public:
  int num_samples, next_index;

private:
  Tensor full_text_input, full_visual_input, batch_text_input, batch_visual_input;
  Tensor full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};
