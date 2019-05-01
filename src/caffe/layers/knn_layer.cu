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
__global__ void modified_insertion_sort(int n, Dtype* dist, Dtype* index, Dtype* dist_out, int width, int k)
{
    // Row position
    CUDA_KERNEL_LOOP(yIndex, n)
    {
        // Pointer shift
        Dtype* p_dist = dist + yIndex * width;
        Dtype* p_index = index + yIndex * k;
        Dtype* p_d_out = dist_out + yIndex * k;

        // Initialise the first index
        p_index[0] = 0;
        p_d_out[0] = p_dist[0];
        // Go through all points
        for (int i = 1; i < width; ++i) {
            // Store current distance and associated index
            Dtype curr_dist = p_dist[i];
            Dtype curr_index = i;

            // Skip the current value if its index is >= k and if it's higher the k-th
            // already sorted smallest value
            if (i >= k && curr_dist >= p_dist[k - 1])
                continue;

            // Shift values (and indexes) higher that the current distance to the
            // right
            int j = min(i, k - 1);
            while (j > 0 && p_dist[j - 1] > curr_dist) {
                p_dist[j] = p_dist[j - 1];
                p_d_out[j] = p_dist[j - 1];
                p_index[j] = p_index[j - 1];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j] = curr_dist;
            p_d_out[j] = curr_dist;
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
        const int ref_index = index % ref_dim;
        const int query_index = (index / ref_dim) - (b * query_dim);
        out[index] = 0;
        for (int i = 0; i < inner_dim; ++i) {
            out[index] += pow(ref[(b * inner_dim + i) * ref_dim + ref_index] - query[(b * inner_dim + i) * query_dim + query_index], 2);
        }
    }

} // namespace caffe

template <typename Dtype>
__global__ void compute_diff(const int n, const Dtype* ref,
    const Dtype* query, const Dtype* k_index, const Dtype* top_diff,
    const int ref_dim, const int query_dim, const int inner_dim, const int k,
    const int sign, Dtype* out)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int b = index / (query_dim * k);
        const int ref_row = k_index[index];
        const int query_row = (index / k) - (b * query_dim);

        for (int i = 0; i < inner_dim; ++i) {
            const int ref_ind = (b * inner_dim + i) * ref_dim + ref_row;
            const int query_ind = (b * inner_dim + i) * query_dim + query_row;
            atomicAdd(out + (sign < 0 ? ref_ind : query_ind), (sign * (query[query_ind] - ref[ref_ind])) * top_diff[index]);
        }
    }

} // namespace caffe

template <typename Dtype>
void KnnLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* ref_data = bottom[0]->gpu_data();
    const Dtype* query_data = bottom[1]->gpu_data();
    Dtype* k_index = top[0]->mutable_gpu_data();
    Dtype* dist_out = top[1]->mutable_gpu_data();
    Dtype* dist_mtx = this->blobs_[0]->mutable_gpu_data();
    const int count = this->blobs_[0]->count();

    compute_distances<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, ref_data, query_data, ref_size_,
            query_size_, channels_, dist_mtx);

    modified_insertion_sort<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(top[0]->shape(0) * query_size_), CAFFE_CUDA_NUM_THREADS>>>(
            top[0]->shape(0) * query_size_, dist_mtx, k_index, dist_out, ref_size_, k_);

    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void KnnLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    const Dtype* ref_data = bottom[0]->gpu_data();
    const Dtype* query_data = bottom[1]->gpu_data();
    const Dtype* k_index = top[0]->gpu_data();
    const Dtype* top_diff = top[1]->gpu_diff();
    const int count = top[0]->count();

    for (int i = 0; i < bottom.size(); ++i) {
        if (propagate_down[i]) {
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            int sign = -2;
            if (i == 1)
                sign = 2;

            caffe_gpu_set(bottom[i]->count(), (Dtype)0, bottom_diff);
            compute_diff<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
                <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    count, ref_data, query_data, k_index, top_diff,
                    ref_size_, query_size_, channels_, k_, sign, bottom_diff);
        }
    }

    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(KnnLayer);

} // namespace caffe