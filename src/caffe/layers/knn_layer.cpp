// Knn search layer.
// Adapted from https://github.com/vincentfpgarcia/kNN-CUDA.git Vincent Garcia, Éric Debreuve, Michel Barlaud

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "caffe/layers/knn_layer.hpp"

namespace caffe {

template <typename Dtype>
void KnnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const KnnParameter& param = this->layer_param_.knn_param();

    ignore_self_ = param.ignore_self();
    k_ = param.k();
    axis_ = param.axis();
    channels_ = bottom[0]->shape(1);

    ref_size_ = bottom[0]->shape(axis_);
    query_size_ = bottom[1]->shape(axis_);
}

template <typename Dtype>
void KnnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    vector<int> top_shape;

    top_shape.push_back(bottom[1]->shape(0));
    top_shape.push_back(k_);
    top_shape.push_back(bottom[1]->shape(axis_));
    top_shape.push_back(1);

    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void KnnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* ref = bottom[0]->cpu_data();
    const Dtype* query = bottom[1]->cpu_data();
    Dtype* k_index = top[0]->mutable_cpu_data();

    int batch_size = bottom[0]->shape(0);

    for (int b = 0; b < batch_size; ++b) {
        // Process one query point at a time
        Dtype* dist = (Dtype*)malloc(ref_size_ * sizeof(Dtype));
        int* index = (int*)malloc(ref_size_ * sizeof(int));

        const Dtype* cur_ref = ref + b * channels_ * ref_size_;
        const Dtype* cur_query = query + b * channels_ * query_size_;
        for (int i = 0; i < query_size_; ++i) {
            // Compute all distances / indexes
            for (int j = 0; j < ref_size_; ++j) {
                dist[j] = compute_distance(cur_ref, cur_query, j, i);
                index[j] = j;
            }

            // Sort distances / indexes
            modified_insertion_sort(dist, index);

            // Copy k smallest distances and their associated index
            int start = ignore_self_ ? 1 : 0;
            for (int j = start; j < k_ + start; ++j) {
                k_index[(b * k_ + (j - start)) * ref_size_ + i] = index[j - start];
            }
        }
    }
}

/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * @param ref          refence points
 * @param query        query points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
template <typename Dtype>
float KnnLayer<Dtype>::compute_distance(const Dtype* ref,
    const Dtype* query,
    int ref_index,
    int query_index)
{
    Dtype sum = 0.f;
    for (int d = 0; d < channels_; ++d) {
        const Dtype diff = ref[d * ref_size_ + ref_index] - query[d * query_size_ + query_index];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

/**
 * Gathers at the beginning of the `dist` array the k smallest values and their
 * respective index (in the initial array) in the `index` array. After this call,
 * only the k-smallest distances are available. All other distances might be lost.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist    array containing the `length` distances
 * @param index   array containing the index of the k smallest distances
 * @param length  total number of distances
 */
template <typename Dtype>
void KnnLayer<Dtype>::modified_insertion_sort(Dtype* dist, int* index)
{

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i = 1; i < ref_size_; ++i) {

        // Store current distance and associated index
        Dtype curr_dist = dist[i];
        int curr_index = i;
        int extra = ignore_self_ ? 0 : 1;
        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k_ + (1 - extra) && curr_dist >= dist[k_ - extra]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right

        int j = std::min(i, k_ - extra);
        while (j > 0 && dist[j - 1] > curr_dist) {
            dist[j] = dist[j - 1];
            index[j] = index[j - 1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j] = curr_dist;
        index[j] = curr_index;
    }
}

#ifdef CPU_ONLY
STUB_GPU(KnnLayer);
#endif

INSTANTIATE_CLASS(KnnLayer);
REGISTER_LAYER_CLASS(Knn);

} // namespace caffe