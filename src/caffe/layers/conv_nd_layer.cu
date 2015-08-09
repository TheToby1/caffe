#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

template <typename Dtype>
void ConvolutionNDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data, n * this->bottom_dim_, weight,
          top_data, n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ConvolutionNDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  if (this->device_context_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->param_propagate_down_[0]) {
      caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
    }
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      caffe_gpu_set(this->blobs_[1]->count(), Dtype(0),
          this->blobs_[1]->mutable_gpu_diff());
    }
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    if (this->param_propagate_down_[0]) {
      greentea_gpu_set<Dtype>(this->device_context_->id(),
                       this->blobs_[0]->count(), Dtype(0),
                       (cl_mem)weight_diff, 0);
    }
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      greentea_gpu_set<Dtype>(this->device_context_->id(),
                       this->blobs_[1]->count(), Dtype(0),
          (cl_mem)(this->blobs_[1]->mutable_gpu_diff()), 0);
    }
#endif  // USE_GREENTEA
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, n * this->bottom_dim_,
              top_diff, n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff, n * this->top_dim_, weight,
              bottom_diff, n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionNDLayer);

}  // namespace caffe