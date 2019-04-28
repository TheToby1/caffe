// Knn search layer.
// Adapted from https://github.com/vincentfpgarcia/kNN-CUDA.git Vincent Garcia,
// Éric Debreuve, Michel Barlaud

#include <vector>

#include "caffe/layers/knn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the
 * top of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire
 * array would not be very efficient if k is relatively small. Instead, we
 * perform a simple insertion sort by eventually inserting a given distance in
 * the first k values.
 *
 * @param dist         distance matrix
 * @param index        index matrix
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
template <typename Dtype>
__global__ void modified_insertion_sort(int n, Dtype* dist, Dtype* index, int width, int k)
{
    // Row position
    CUDA_KERNEL_LOOP(yIndex, n)
    {
        // Pointer shift
        Dtype* p_dist = dist + yIndex * width;
        Dtype* p_index = index + yIndex * k;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i = 1; i < width; ++i) {
            // Store current distance and associated index
            Dtype curr_dist = p_dist[i];
            Dtype curr_index = i;

            // Skip the current value if its index is >= k and if it's higher the k-th
            // slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[k - 1])
                continue;

            // Shift values (and indexes) higher that the current distance to the
            // right
            int j = min(i, k - 1);
            while (j > 0 && p_dist[j - 1] > curr_dist) {
                p_dist[j] = p_dist[j - 1];
                p_index[j] = p_index[j - 1];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j] = curr_dist;
            p_index[j] = curr_index;
        }
    }
}

template <typename Dtype>
__global__ void compute_distances(const int n, const Dtype* ref,
    const Dtype* query, const int ref_dim,
    const int query_dim, const int inner_dim,
    Dtype* out)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int b = index / (query_dim * ref_dim);
        const int ref_index = (index % ref_dim) * inner_dim + (b * (ref_dim * inner_dim));
        const int query_index = ((index / ref_dim) % query_dim) * inner_dim + (b * (query_dim * inner_dim));
        for (int i = 0; i < inner_dim; ++i) {
            out[index] += ref[i * ref_dim + ref_index] - query[i * query_dim + query_index];
        }
        out[index] = sqrt(out[index] * out[index]);
    }

} // namespace caffe

template <typename Dtype>
void KnnLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* ref_data = bottom[0]->gpu_data();
    const Dtype* query_data = bottom[1]->gpu_data();
    Dtype* k_index = top[0]->mutable_gpu_data();
    Dtype* dist_mtx = this->blobs_[0]->mutable_gpu_data();
    // CHECK_EQ(this->blobs_[0]->offset(1), top[0]->offset(1)) << "Offsets of
    // memory in blobs must be the same";
    const int count = top[0]->count();

    compute_distances<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, ref_data, query_data, ref_size_, query_size_, channels_,
            dist_mtx);

    modified_insertion_sort<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(query_size_), CAFFE_CUDA_NUM_THREADS>>>(
            top[0]->shape(0) * top[0]->shape(2), dist_mtx, k_index, ref_size_, k_);

    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(KnnLayer);

} // namespace caffe