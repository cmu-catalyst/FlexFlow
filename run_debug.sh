CUDA_VISIBLE_DEVICES=0,1 ./examples/cpp/SimpleNet/simplenet -ll:gpu 2 -ll:fsize 2048 -ll:zsize 12192 -e 10 -b 64 --import-strategy ./examples/cpp/SimpleNet/S_parallel_strategy_nonuniform.txt
