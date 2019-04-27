#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/layers/knn_layer.hpp"

namespace caffe {

template <typename TypeParam>
class KnnLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

protected:
    KnnLayerTest()
        : blob_bottom_(new Blob<Dtype>(2, 3, 4, 1))
        , blob_bottom_2_(new Blob<Dtype>(2, 3, 4, 1))
        , blob_top_(new Blob<Dtype>())
        , blob_top_2_(new Blob<Dtype>())
    {
        Caffe::set_random_seed(1701);
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_bottom_vec_.push_back(blob_bottom_2_);
        blob_top_vec_.push_back(blob_top_);
        blob_top_vec_.push_back(blob_top_2_);
    }
    virtual ~KnnLayerTest()
    {
        delete blob_bottom_;
        delete blob_bottom_2_;
        delete blob_top_;
    }
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_bottom_2_;
    Blob<Dtype>* const blob_top_;
    Blob<Dtype>* const blob_top_2_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};
TYPED_TEST_CASE(KnnLayerTest, TestDtypesAndDevices);

TYPED_TEST(KnnLayerTest, TestSetup)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    int k = 3;
    KnnParameter* knn_param = layer_param.mutable_knn_param();
    knn_param->set_k(k);

    FillerParameter filler_param;
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);

    shared_ptr<Layer<Dtype>> layer(
        new KnnLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_top_->shape(0), 2);
    EXPECT_EQ(this->blob_top_->shape(1), k);
    EXPECT_EQ(this->blob_top_->shape(2), this->blob_bottom_2_->shape(2));
    EXPECT_EQ(this->blob_top_->shape(3), 1);
}

TYPED_TEST(KnnLayerTest, TestForward)
{
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    int k = 3;
    KnnParameter* knn_param = layer_param.mutable_knn_param();
    knn_param->set_k(k);
    knn_param->set_ignore_self(false);

    Dtype* ref = this->blob_bottom_->mutable_cpu_data();
    Dtype* query = this->blob_bottom_2_->mutable_cpu_data();

    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
        ref[i] = i;
        query[i] = i;
    }

    shared_ptr<Layer<Dtype>> layer(
        new KnnLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const int ans[] = { 0, 1, 2, 1, 0, 2, 2, 1, 3, 3, 2, 1, 0, 1, 2, 1, 0, 2, 2, 1, 3, 3, 2, 1 };
    const Dtype* top_idx = this->blob_top_->cpu_data();
    const Dtype* top_dist = this->blob_top_2_->cpu_data();
    for (int i = 0; i < this->blob_top_->count(); ++i) {
        EXPECT_EQ(static_cast<int>(top_idx[i]), -1);
        EXPECT_EQ(top_dist[i], -1.);
    }
}

} // namespace caffe