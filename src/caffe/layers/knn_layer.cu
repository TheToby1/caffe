// Knn search layer.
// Adapted from https://github.com/vincentfpgarcia/kNN-CUDA.git Vincent Garcia, Éric Debreuve, Michel Barlaud

#include <vector>

#include "caffe/layers/knn_layer.hpp"

namespace caffe {
    template <typename Dtype>
    void SinLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    
    CUDA_POST_KERNEL_CHECK;
    }

    INSTANTIATE_LAYER_GPU_FUNCS(KnnLayer);


}  // namespace caffe