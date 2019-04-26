#ifndef CAFFE_KNN_LAYER_HPP_
#define CAFFE_KNN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe
{

template <typename Dtype>
class KnnLayer : public NeuronLayer<Dtype>
{
  public:
    explicit KnnLayer(const LayerParameter &param)
        : NeuronLayer<Dtype>(param) {}

    virtual inline const char *type() const { return "Knn"; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);
    virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

    virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                              const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

    float compute_distance(const Blob<Dtype> *ref, const Blob<Dtype> *query, int ref_index, int query_index);
    void modified_insertion_sort(const Blob<Dtype> *dist, int *index);

    int axis_, channels_;
    int k_;
    int ref_size_, query_size_;
};

} // namespace caffe

#endif // CAFFE_KNN_LAYER_HPP_