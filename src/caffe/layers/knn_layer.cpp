// Knn search layer.
// Adapted from https://github.com/vincentfpgarcia/kNN-CUDA.git Vincent Garcia, Éric Debreuve, Michel Barlaud

#include <vector>

#include "caffe/layers/knn_layer.hpp"

namespace caffe
{
template <typename Dtype>
void SinLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top)
{
    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();

    // Process one query point at the time
    for (int i = 0; i < query_nb; ++i)
    {

        // Compute all distances / indexes
        for (int j = 0; j < ref_nb; ++j)
        {
            dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

        // Sort distances / indexes
        modified_insertion_sort(dist, index, ref_nb, k);

        // Copy k smallest distances and their associated index
        for (int j = 0; j < k; ++j)
        {
            knn_dist[j * query_nb + i] = dist[j];
            knn_index[j * query_nb + i] = index[j];
        }
    }
}

template <typename Dtype>
void SinLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                   const vector<bool> &propagate_down,
                                   const vector<Blob<Dtype> *> &bottom)
{
    if (propagate_down[0])
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        const Dtype *top_diff = top[0]->cpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        Dtype bottom_datum;
        for (int i = 0; i < count; ++i)
        {
            bottom_datum = bottom_data[i];
            bottom_diff[i] = top_diff[i] * cos(bottom_datum);
        }
    }
}

/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * @param ref          refence points
 * @param ref_nb       number of reference points
 * @param query        query points
 * @param query_nb     number of query points
 * @param dim          dimension of points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
float compute_distance(const float *ref,
                       int ref_nb,
                       const float *query,
                       int query_nb,
                       int dim,
                       int ref_index,
                       int query_index)
{
    float sum = 0.f;
    for (int d = 0; d < dim; ++d)
    {
        const float diff = ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
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
 * @param k       number of smallest distances to locate
 */
void modified_insertion_sort(float *dist, int *index, int length, int k)
{

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i = 1; i < length; ++i)
    {

        // Store current distance and associated index
        float curr_dist = dist[i];
        int curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist[k - 1])
        {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = std::min(i, k - 1);
        while (j > 0 && dist[j - 1] > curr_dist)
        {
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