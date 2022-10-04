from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.torch.model import PyTorchModel

#from accuracy import ModelAccuracy
from PIL import Image
import numpy as np

def top_level_task():
    ffconfig = FFConfig()
    alexnetconfig = NetConfig()
    print(alexnetconfig.dataset_path)
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 64, 256]
    input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    output_tensors = PyTorchModel.file_to_ff("test_net.ff", ffmodel, [input])
    # t = ffmodel.softmax(output_tensors[0])
    t = output_tensors[0]

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE, metrics=[MetricsType.METRICS_ROOT_MEAN_SQUARED_ERROR])
    output = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    num_samples = 2048

    (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

    full_input_np = np.random.uniform(-1, 1, size=dims_input).astype("float32")
    full_output_np = np.random.uniform(-1, 1, size=dims_input).astype("float32")

    dataloader_input = ffmodel.create_data_loader(input, full_input_np)
    dataloader_label = ffmodel.create_data_loader(output, full_output_np)

    num_samples = dataloader_input.num_samples
    assert dataloader_input.num_samples == dataloader_label.num_samples

    ffmodel.init_layers()

    epochs = ffconfig.epochs

    ts_start = ffconfig.get_current_time()

    #ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)
    ffmodel.eval(x=dataloader_input, y=dataloader_label)

    ts_end = ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start);
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
    # perf_metrics = ffmodel.get_perf_metrics()
    # accuracy = perf_metrics.get_accuracy()
    # if accuracy < ModelAccuracy.CIFAR10_ALEXNET.value:
    #   assert 0, 'Check Accuracy'

if __name__ == "__main__":
    print("test net torch")
    top_level_task()
