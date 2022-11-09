/* Copyright 2021 Facebook
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

#include "unit.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("UniT");

Tensor create_residual_attention_block(FFModel *model,
                                       Tensor const &input,
                                       int hidden_size,
                                       int num_heads,
                                       int kv_dim) {
  /// @warning we skip attention mask and LayerNorm for now
  /// cuz it is marginal compared to other ops in terms of compute cost

  /// LayerNorm

  /// Multi-head attention
  /// @warning It requires the input shape to be (N, L, D)
  /// where N: batch size, L: seq length, H: width of Transformer
  /// Note that it's different from PyTorch default (L, N, D)
  Tensor t = model->multihead_attention(
      input, input, input, hidden_size, num_heads, kv_dim, kv_dim);

  /// LayerNorm

  /// MLP: Linear, GELU (-->RELU), Linear
  t = model->dense(model->dense(t, hidden_size, AC_MODE_RELU, false /*bias*/),
               hidden_size, AC_MODE_NONE, false /*bias*/);

  return t;
}

/// Basically, it's Transformer model
Tensor create_transformer(FFModel *model,
                          Tensor const &input,
                          int hidden_size,
                          int num_heads,
                          int kv_dim,
                          int num_layers) {
  Tensor t = input;
  for (int i = 0; i < num_layers; i++)
    t = create_residual_attention_block(model, t, hidden_size, num_heads, kv_dim);

  return t;
}

Tensor create_text_encoder(FFModel *model,
                          Tensor const &input,
                          int hidden_size,
                          int num_heads,
                          int kv_dim,
                          int num_layers) {
  Tensor t = input;

//  std::cout << "tensor before text encoder: ";
//  for (int i=0; i<5; ++i) {
//    std::cout << t->dims[i] << ",";
//  }
//  std::cout << std::endl;

  /// Transformer
  t = create_transformer(model, t, hidden_size, num_heads, kv_dim, num_layers);

  /// Layernorm

  /// Text projection


  return t;
}
Tensor BottleneckBlock(FFModel *ff, Tensor input, int out_channels, int stride) {
  Tensor t = ff->conv2d(input, out_channels, 1, 1, 1, 1, 0, 0, AC_MODE_NONE);
  // t = ff.batch_norm(t);

  t = ff->conv2d(t, out_channels, 3, 3, stride, stride, 1, 1, AC_MODE_NONE);
  // t = ff.batch_norm(t);

  t = ff->conv2d(t, 4 * out_channels, 1, 1, 1, 1, 0, 0);
  // t = ff.batch_norm(t, false);

  if ((stride > 1) || (input->dims[2] != out_channels * 4)) {
//    printf("input->dims = %d out_channels*4 = %d\n", input->dims[2], out_channels * 4);
    input = ff->conv2d(input, 4 * out_channels, 1, 1, stride, stride, 0, 0, AC_MODE_NONE);
    // input = ff.batch_norm(input, false);
  }
  t = ff->add(input, t);

  return ff->relu(t, false);
}

Tensor create_resnet34(FFModel *ff,
                       Tensor const &input) {

  // shape = [*, width, grid, grid]
  Tensor t = ff->conv2d(input, 64, 7, 7, 2, 2, 3, 3);
  // t = ff.batch_norm(t);
  t = ff->pool2d(t, 3, 3, 2, 2, 1, 1);

  for (int i = 0; i < 3; i++)
    t = BottleneckBlock(ff, t, 64, 1);

  for (int i = 0; i < 4; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(ff, t, 128, stride);
  }

  for (int i = 0; i < 6; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(ff, t, 256, stride);
  }

  for (int i = 0; i < 3; i++) {
    int stride = (i == 0) ? 2 : 1;
    t = BottleneckBlock(ff, t, 512, stride);
  }

  return t;
}

/// Basically, it's VisionTransformer model
Tensor create_image_encoder(FFModel *model,
                            Tensor const &input,
                            int kernel_size,
                            int stride,
                            int padding,
                            int hidden_size,
                            int num_heads,
                            int kv_dim,
                            int num_layers,
                            int batch_size,
                            int image_size) {

  // shape = [*, width, grid, grid]
  // FIXME: use_bias better be false
  Tensor t = create_resnet34(model, input);

  std::cout << "tensor after resnet-32: ";
  for (int i=0; i<5; ++i) {
    std::cout << t->dims[i] << ",";
  }
  std::cout << std::endl;

  /// Self type-inference for reshape: shape = [*, grid ** 2, width]
  std::vector<int> shape{t->dims[3], t->dims[2], t->dims[0]*t->dims[1]};
  t = model->reshape(t, shape);
  std::vector<int> perm1{0, 2, 1};
  t = model->transpose(t, perm1);


  /// Projection to match hidden dimension of vision Transformer
  /// The (output tensor).dims should be HLN (= NLH)
  t = model->dense(t, hidden_size, AC_MODE_RELU, false /*bias*/);


  /// Transformer
  t = create_transformer(model,
                         t,
                         hidden_size,
                         num_heads,
                         kv_dim,
                         num_layers);

  /// LayerNorm

  return t;
}

UniTConfig::UniTConfig(void) {
  // Text Transformer arguments
  // We assume hidden_size = embed_dim for convenience
  // hidden_size (for multi-head attention) = transformer_width
  /// @warning FF runtime fails to run 768 (hidden size) and 12 (# of heads)
  /// CU: cuEventSynchronize(e) = 700 (CUDA_ERROR_ILLEGAL_ADDRESS): an illegal memory access was encountered
  tt_hidden_size = 768; // 768
  tt_num_heads = 12; // 12
  tt_num_layers = 12; // 12

  sequence_length = 512; // 128, 256, 512 vary depending on tasks

  // Vision Transformer arguments
  vt_hidden_size = 768; // 256, 768
  vt_num_heads = 8; // 8
  vt_num_layers = 6; // 6

  // Decoder Transformer arguments
  dec_hidden_size = 768; // 768
  dec_num_heads = 8; // 8
  dec_num_layers = 6; // 6

  // Vision Transformer conv arguments
  // Candidates: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14-336px
  // B and L means "Base" and "Large", /32 means input patch size = 32x32

  // out_channels = vt_hidden_size
  /// @warning FF runtime fails to run 336 (image size) and 14 (kernel size)
  /// CU: cuEventSynchronize(e) = 700 (CUDA_ERROR_ILLEGAL_ADDRESS): an illegal memory access was encountered
  in_channels = 3;
  image_size = 600; // random sampling btw 384 - 600
  // stride = kernel_size --> Image is kxk words
  kernel_size = 99999; // Doesn't matter
  padding = 0;
}

void FlexFlow::top_level_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  FFConfig ffConfig;
  UniTConfig tfConfig;
  FFModel ff(ffConfig);

  /// Text Input - GLUE benchmark (sequence length is default of BERT-base)
  Tensor text_input; // NLH
  {
    int const dims[] = {
        ffConfig.batchSize, tfConfig.sequence_length, tfConfig.tt_hidden_size};
    text_input = ff.create_tensor<3>(dims, DT_FLOAT);
  }

//  std::cout << "Text Input shape : ";
//  for (int i=0; i<5; ++i) {
//    std::cout << text_input->dims[i] << ",";
//  }
//  std::cout << std::endl;

  /// Image Input - COCO dataset for VQA (randomly sampled from 384 <= w,h <= 600)
  /// But we assume the largest image size for convenience
  Tensor visual_input; // NCHW
  {
    int const dims[] = {ffConfig.batchSize, tfConfig.in_channels,
                        tfConfig.image_size, tfConfig.image_size};
    visual_input = ff.create_tensor<4>(dims, DT_FLOAT);
  }

  /// Text encoder (Basically Transformer model)
  /// A series of ResidualAttentionBlock
  Tensor tt = text_input; // Encoded vector for text
  tt = create_text_encoder(&ff,
                          tt,
                          tfConfig.tt_hidden_size,
                          tfConfig.tt_num_heads,
                          tfConfig.tt_hidden_size / tfConfig.tt_num_heads,
                          tfConfig.tt_num_layers);
  /// Projection
  tt = ff.dense(tt, tfConfig.dec_hidden_size, AC_MODE_RELU, false /*bias*/);


  /// Image encoder
  Tensor vt = visual_input; // Encoded vector for image
  vt = create_image_encoder(&ff,
                            vt,
                            tfConfig.kernel_size,
                            tfConfig.kernel_size,
                            tfConfig.padding,
                            tfConfig.vt_hidden_size,
                            tfConfig.vt_num_heads,
                            tfConfig.vt_hidden_size / tfConfig.vt_num_heads,
                            tfConfig.vt_num_layers,
                            ffConfig.batchSize,
                            tfConfig.image_size);
  /// Projection
  vt = ff.dense(vt, tfConfig.dec_hidden_size, AC_MODE_RELU, false /*bias*/);

//  /// FIXME: Temporary operator to fix mismatch of tt and vt
//  std::vector<int> tt_shape{ffConfig.batchSize, vt->dims[0], 38};
//  tt = ff.reshape(tt, tt_shape);

  /// Cosine similarity (Matmul between image and text features)
  std::cout << "Final TE shape : ";
  for (int i=0; i<5; ++i) {
    std::cout << tt->dims[i] << ",";
  }
  std::cout << std::endl;

  std::cout << "Final VE shape : ";
  for (int i=0; i<5; ++i) {
    std::cout << vt->dims[i] << ",";
  }
  std::cout << std::endl;

  /// FIXME: Later, we need to think about how to consider matrix of
  /// (batch_size, batch_size) with our pipeline scheme instead of
  /// (micro_batch_size, micro_batch_size) * (batch_size/micro_batch_size).
  /// Currently, we are missing out (batch_size - micro_batch_size) * batch_size pairs
  Tensor text_and_image[2];
  text_and_image[0] = tt;
  text_and_image[1] = vt;

  Tensor ot = ff.concat(2, text_and_image, 1);

//  ot = ff.dense(ot, tfConfig.dec_hidden_size, AC_MODE_RELU, false /*bias*/);

  std::cout << "concatenated tensor : ";
  for (int i=0; i<5; ++i) {
    std::cout << ot->dims[i] << ",";
  }
  std::cout << std::endl;

  ot = create_transformer(&ff, ot,
                          tfConfig.dec_hidden_size,
                          tfConfig.dec_num_heads,
                          tfConfig.dec_hidden_size / tfConfig.dec_num_heads,
                          tfConfig.dec_num_layers);



  std::cout << "output tensor : ";
  for (int i=0; i<5; ++i) {
    std::cout << ot->dims[i] << ",";
  }
  std::cout << std::endl;

  /// Scaling logits

  Optimizer *optimizer = new SGDOptimizer(&ff, 0.01f);
  std::vector<MetricsType> metrics;
  // metrics.push_back(METRICS_ACCURACY);
//   metrics.push_back(METRICS_MEAN_SQUARED_ERROR);

  /// @warning: Code exits when we compile the model if we turn on op profiling
  ff.compile(optimizer, LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics);


//  std::cout << "Code reaches here after compilation" << std::endl;

  // Data Loader
  DataLoader loader(ff, tfConfig, text_input, visual_input, ff.label_tensor, ot);
  loader.next_batch(ff);
  loader.reset();
  ff.init_operators();

  // Start timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  log_app.print("Warmup finished...Start timer...");
  log_app.print("Num. epochs = %d", ffConfig.epochs);
  log_app.print("Num. iterations/epoch = %d",
                loader.num_samples / ffConfig.batchSize);
  printf("parameters.size() = %lu\n", ff.parameters.size());
  double ts_start = Realm::Clock::current_time_in_microseconds();
  for (int epoch = 0; epoch < ffConfig.epochs; epoch++) {
    loader.reset();
    ff.reset_metrics();
    int iterations = loader.num_samples / ffConfig.batchSize;
    for (int iter = 0; iter < iterations; iter++) {
      // Only load data once for random input
      if (iter == 0 && epoch == 0)
        loader.next_batch(ff);
      runtime->begin_trace(ctx, 111 /*trace_id*/);
      ff.forward();
      ff.zero_gradients();
      ff.backward();
      ff.update();
      runtime->end_trace(ctx, 111 /*trace_id*/);
    }
  }
  // End timer
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n",
         run_time,
         loader.num_samples * ffConfig.epochs / run_time);
}

DataLoader::DataLoader(FFModel &ff,
                       UniTConfig const &tf,
                       Tensor const &_text_input,
                       Tensor const &_visual_input,
                       Tensor const &_label,
                       Tensor const &_output_tensor) {
  /// Set up context & # of samples to process
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  log_app.print("Use random dataset...");
  num_samples =
      ff.config.batchSize * ff.config.workersPerNode * ff.config.numNodes;
  log_app.print("Number of random samples = %d\n", num_samples);

  /// Set up input and output
  {
    batch_text_input = _text_input;
    int const dims[] = {num_samples, tf.sequence_length, tf.tt_hidden_size};
    full_text_input = ff.create_tensor<3>(dims, DT_FLOAT);
  }
  {
    batch_visual_input = _visual_input;
    int const dims[] = {num_samples, tf.in_channels, tf.image_size, tf.image_size};
    full_visual_input = ff.create_tensor<4>(dims, DT_FLOAT);
  }
  {
    batch_label = _label;
//    std::cout << "output tensor dims[1] : " << _output_tensor->dims[1] << std::endl;
    int const dims[] = {num_samples, tf.tt_hidden_size, _output_tensor->dims[1]};
    full_label = ff.create_tensor<3>(dims, DT_FLOAT);
  }

  /// Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1, TaskArgument(NULL, 0));
  // regions[0]: full_text_input
  launcher.add_region_requirement(
      RegionRequirement(full_text_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_text_input->parallel_tensor->region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);

  // regions[1]: full_visual_input
  launcher.add_region_requirement(
      RegionRequirement(full_visual_input->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_visual_input->parallel_tensor->region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);

  // regions[2]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label->parallel_tensor->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_label->parallel_tensor->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);

  // Note that these instances are in ZCM, can only use
  // TensorAccessorW with readOutput flag
  AccessorWO<float, 3> const acc_text_input(regions[0], FID_DATA);
  AccessorWO<float, 3> const acc_visual_input(regions[1], FID_DATA);
  AccessorWO<float, 3> const acc_label(regions[2], FID_DATA);

  Rect<3> rect_text_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<3> rect_visual_input = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<3> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());

  assert(acc_text_input.accessor.is_dense_arbitrary(rect_text_input));
  assert(acc_visual_input.accessor.is_dense_arbitrary(rect_visual_input));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  float *text_input_ptr = acc_text_input.ptr(rect_text_input.lo);
  float *visual_input_ptr = acc_visual_input.ptr(rect_visual_input.lo);
  float *label_ptr = acc_label.ptr(rect_label.lo);
  // assert(rect_input == rect_label);

  for (size_t i = 0; i < rect_text_input.volume(); i++)
    text_input_ptr[i] = ((float)std::rand()) / RAND_MAX;
  for (size_t i = 0; i < rect_visual_input.volume(); i++)
    visual_input_ptr[i] = ((float)std::rand()) / RAND_MAX;
  for (size_t i = 0; i < rect_label.volume(); i++)
    label_ptr[i] = std::rand() % 2;
}

void DataLoader::next_batch(FFModel &ff) {
  return;
//  Context ctx = ff.config.lg_ctx;
//  Runtime *runtime = ff.config.lg_hlr;
//
//  // Load Text Input
//  {
//    Domain domain = runtime->get_index_space_domain(
//        ctx, batch_text_input->parallel_tensor->parallel_is);
//    ArgumentMap argmap;
//    int idx = next_index;
//    for (Domain::DomainPointIterator it(domain); it; it++) {
//      SampleIdxs meta;
//      assert(ff.config.batchSize % batch_input->parallel_tensor->dims[2].size ==
//             0);
//      meta.num_samples =
//          ff.config.batchSize / batch_input->parallel_tensor->dims[2].size;
//      for (int i = 0; i < meta.num_samples; i++)
//        meta.idxs[i] = idx++;
//      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
//    }
//    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
//                           batch_input->parallel_tensor->parallel_is,
//                           TaskArgument(NULL, 0),
//                           argmap,
//                           Predicate::TRUE_PRED,
//                           false /*must*/,
//                           0 /*mapper_id*/,
//                           batch_input->parallel_tensor->machine_view.hash());
//    // Full dataset in ZCM
//    launcher.add_region_requirement(
//        RegionRequirement(full_input->parallel_tensor->region,
//                          0 /*projection id*/,
//                          READ_ONLY,
//                          EXCLUSIVE,
//                          full_input->parallel_tensor->region,
//                          MAP_TO_ZC_MEMORY));
//    launcher.add_field(0, FID_DATA);
//    launcher.add_region_requirement(
//        RegionRequirement(batch_input->parallel_tensor->part,
//                          0 /*projection id*/,
//                          WRITE_ONLY,
//                          EXCLUSIVE,
//                          batch_input->parallel_tensor->region));
//    launcher.add_field(1, FID_DATA);
//    runtime->execute_index_space(ctx, launcher);
//  }
//  // Load Visual Input
//  {
//    Domain domain = runtime->get_index_space_domain(
//        ctx, batch_input->parallel_tensor->parallel_is);
//    ArgumentMap argmap;
//    int idx = next_index;
//    for (Domain::DomainPointIterator it(domain); it; it++) {
//      SampleIdxs meta;
//      assert(ff.config.batchSize % batch_input->parallel_tensor->dims[2].size ==
//             0);
//      meta.num_samples =
//          ff.config.batchSize / batch_input->parallel_tensor->dims[2].size;
//      for (int i = 0; i < meta.num_samples; i++)
//        meta.idxs[i] = idx++;
//      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
//    }
//    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
//                           batch_input->parallel_tensor->parallel_is,
//                           TaskArgument(NULL, 0),
//                           argmap,
//                           Predicate::TRUE_PRED,
//                           false /*must*/,
//                           0 /*mapper_id*/,
//                           batch_input->parallel_tensor->machine_view.hash());
//    // Full dataset in ZCM
//    launcher.add_region_requirement(
//        RegionRequirement(full_input->parallel_tensor->region,
//                          0 /*projection id*/,
//                          READ_ONLY,
//                          EXCLUSIVE,
//                          full_input->parallel_tensor->region,
//                          MAP_TO_ZC_MEMORY));
//    launcher.add_field(0, FID_DATA);
//    launcher.add_region_requirement(
//        RegionRequirement(batch_input->parallel_tensor->part,
//                          0 /*projection id*/,
//                          WRITE_ONLY,
//                          EXCLUSIVE,
//                          batch_input->parallel_tensor->region));
//    launcher.add_field(1, FID_DATA);
//    runtime->execute_index_space(ctx, launcher);
//  }
//  // Load Labels
//  {
//    Domain domain = runtime->get_index_space_domain(
//        ctx, batch_label->parallel_tensor->parallel_is);
//    ArgumentMap argmap;
//    int idx = next_index;
//    for (Domain::DomainPointIterator it(domain); it; it++) {
//      SampleIdxs meta;
//      assert(ff.config.batchSize % batch_label->parallel_tensor->dims[2].size ==
//             0);
//      meta.num_samples =
//          ff.config.batchSize / batch_label->parallel_tensor->dims[2].size;
//      for (int i = 0; i < meta.num_samples; i++)
//        meta.idxs[i] = idx++;
//      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
//    }
//    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
//                           batch_label->parallel_tensor->parallel_is,
//                           TaskArgument(NULL, 0),
//                           argmap,
//                           Predicate::TRUE_PRED,
//                           false /*must*/,
//                           0 /*mapper_id*/,
//                           batch_label->parallel_tensor->machine_view.hash());
//    // Full dataset in ZCM
//    launcher.add_region_requirement(
//        RegionRequirement(full_label->parallel_tensor->region,
//                          0 /*projection id*/,
//                          READ_ONLY,
//                          EXCLUSIVE,
//                          full_label->parallel_tensor->region,
//                          MAP_TO_ZC_MEMORY));
//    launcher.add_field(0, FID_DATA);
//    launcher.add_region_requirement(
//        RegionRequirement(batch_label->parallel_tensor->part,
//                          0 /*projection id*/,
//                          WRITE_ONLY,
//                          EXCLUSIVE,
//                          batch_label->parallel_tensor->region));
//    launcher.add_field(1, FID_DATA);
//    runtime->execute_index_space(ctx, launcher);
//  }
//  // progress next_index
//  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}

void FlexFlow::register_custom_tasks() {
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load Sparse Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Text Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_text_input>(
        registrar, "Load Sparse Inputs Task");
  }
  // Load Dense Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Visual Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_visual_input>(
        registrar, "Load Dense Inputs Task");
  }
  // Load Labels
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(registrar,
                                                              "Load Labels");
  }
}
