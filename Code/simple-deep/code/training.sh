CUDA_VISIBLE_DEVICES=3 $CAFFE_ROOT/build/tools/caffe train --solver ../models/solver.prototxt 2>&1 | tee ../models/model-train.log
