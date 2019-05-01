// Knn search layer.
// Adapted from https://github.com/vincentfpgarcia/kNN-CUDA.git Vincent Garcia, Éric Debreuve, Michel Barlaud

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "caffe/layers/knn_layer.hpp"

namespace caffe {
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
void modified_insertion_sort(Dtype* dist, Dtype* index, Dtype* dist_out, const int ref_size, const int k, const int dist_index, const int top_index)
{
    // Initialise the first index
    index[top_index] = 0;
    dist_out[top_index] = dist[dist_index];

    // Go through all points
    for (int i = 1; i < ref_size; ++i) {
        // Store current distance and associated index
        Dtype curr_dist = dist[dist_index + i];
        int curr_index = i;
        // Skip the current value if its index is > k and if it's higher the k-th already sorted smallest value
        if (i >= k && curr_dist >= dist[dist_index + k - 1])
            continue;

        // Shift values (and indexes) higher that the current distance to the right

        int j = std::min(i, k - 1);
        while (j > 0 && dist[dist_index + j - 1] > curr_dist) {
            dist[dist_index + j] = dist[dist_index + j - 1];
            dist_out[top_index + j] = dist[dist_index + j - 1];
            index[top_index + j] = index[top_index + j - 1];
            --j;
        }

        // Write the current distance and index at their position
        dist[dist_index + j] = curr_dist;
        dist_out[top_index + j] = curr_dist;
        index[top_index + j] = curr_index;
    }
}

template <typename Dtype>
void KnnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const KnnParameter& param = this->layer_param_.knn_param();

    k_ = param.k();
    axis_ = param.axis();

    this->blobs_.resize(1);
}

template <typename Dtype>
void KnnLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    channels_ = bottom[0]->shape(1);
    ref_size_ = bottom[0]->shape(axis_);
    query_size_ = bottom[1]->shape(axis_);
    vector<int> top_shape;

    top_shape.push_back(bottom[1]->shape(0));
    top_shape.push_back(1);
    top_shape.push_back(query_size_);
    top_shape.push_back(k_);

    top[0]->Reshape(top_shape);
    top[1]->Reshape(top_shape);

    this->blobs_[0].reset(new Blob<Dtype>(bottom[1]->shape(0), 1, query_size_, ref_size_));
}

template <typename Dtype>
void KnnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* ref = bottom[0]->cpu_data();
    const Dtype* query = bottom[1]->cpu_data();
    Dtype* k_index = top[0]->mutable_cpu_data();
    Dtype* dist_out = top[1]->mutable_cpu_data();
    Dtype* dist_mtx = this->blobs_[0]->mutable_cpu_data();

    int batch_size = bottom[0]->shape(0);

    for (int b = 0; b < batch_size; ++b) {
        // Process one query point at a time
        for (int i = 0; i < query_size_; ++i) {
            const int dist_index = (b * query_size_ + i) * ref_size_;
            const int top_index = (b * query_size_ + i) * k_;
            // Compute all distances / indexes
            for (int j = 0; j < ref_size_; ++j) {
                Dtype sum = 0;
                for (int c = 0; c < channels_; ++c) {
                    sum += pow(ref[(b * channels_ + c) * ref_size_ + j] - query[(b * channels_ + c) * query_size_ + i], 2);
                }
                dist_mtx[dist_index + j] = sum;
            }

            // Sort distances / indexes
            modified_insertion_sort(dist_mtx, k_index,
                dist_out, ref_size_, k_, dist_index, top_index);
        }
    }
}

template <typename Dtype>
void KnnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    const Dtype* ref = bottom[0]->cpu_data();
    const Dtype* query = bottom[1]->cpu_data();
    const Dtype* k_index = top[0]->cpu_data();
    const Dtype* top_diff = top[1]->cpu_diff();

    int batch_size = bottom[0]->shape(0);

    for (int bot = 0; bot < bottom.size(); ++bot) {
        if (propagate_down[bot]) {
            Dtype* bottom_diff = bottom[bot]->mutable_cpu_diff();
            int sign = -2;
            if (bot == 1)
                sign = 2;

            memset(bottom_diff, 0, sizeof(Dtype) * bottom[bot]->count());
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < query_size_; ++i) {
                    for (int j = 0; j < k_; ++j) {
                        const int row_ind = (b * k_ + j) * query_size_ + i;
                        const int ref_row = k_index[row_ind];

                        for (int c = 0; c < channels_; ++c) {
                            const int part_ind = (b * channels_ + c);
                            const int ref_ind = part_ind * ref_size_ + ref_row;
                            const int query_ind = part_ind * query_size_ + i;

                            bottom_diff[bot == 0 ? ref_ind : query_ind] += sign * (query[query_ind] - ref[ref_ind])
                                * top_diff[row_ind];
                        }
                    }
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(KnnLayer);
#endif

INSTANTIATE_CLASS(KnnLayer);
REGISTER_LAYER_CLASS(Knn);

} // namespace caffe