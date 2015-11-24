#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/data_transformer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class TwinImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TwinImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {
      blob_top_data_2_ = new Blob<Dtype>();
  }

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_data_2_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
        outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " <<
            EXAMPLES_SOURCE_DIR "images/cat.jpg " << i;
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg "
                << EXAMPLES_SOURCE_DIR "images/cat.jpg " << 0;
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg "
                << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " << 1;
    reshapefile.close();

    filename_imagenet_ = std::string("/home/abhinav/ann_project/code/combined_model/new_train.txt");
  }

  virtual ~TwinImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
    delete blob_top_data_2_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
    string filename_imagenet_;
    Blob<Dtype>* const blob_top_data_;
    Blob<Dtype>* blob_top_data_2_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TwinImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(TwinImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TwinImageDataParameter* image_data_param = param.mutable_twin_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  TwinImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);

  EXPECT_EQ(this->blob_top_data_2_->num(), 5);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 360);
  EXPECT_EQ(this->blob_top_data_2_->width(), 480);


  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(TwinImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TwinImageDataParameter* image_data_param = param.mutable_twin_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  TwinImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);

  EXPECT_EQ(this->blob_top_data_2_->num(), 5);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 256);
  EXPECT_EQ(this->blob_top_data_2_->width(), 256);


  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    }
  }
}

TYPED_TEST(TwinImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TwinImageDataParameter* image_data_param = param.mutable_twin_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  TwinImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_data_2_->num(), 1);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 360);
  EXPECT_EQ(this->blob_top_data_2_->width(), 480);
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
  EXPECT_EQ(this->blob_top_data_2_->num(), 1);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 323);
  EXPECT_EQ(this->blob_top_data_2_->width(), 481);

}

TYPED_TEST(TwinImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TwinImageDataParameter* image_data_param = param.mutable_twin_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  TwinImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_data_2_->num(), 5);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 360);
  EXPECT_EQ(this->blob_top_data_2_->width(), 480);

  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
}

TYPED_TEST(TwinImageDataLayerTest, TestImagenet) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  TwinImageDataParameter* image_data_param = param.mutable_twin_image_data_param();
  image_data_param->set_batch_size(256);
  image_data_param->set_source(this->filename_imagenet_.c_str());
  image_data_param->set_mean_file("/media/data/abhinav/ann_project_data/imagenet_resized_mean.binaryproto");
  image_data_param->set_mean_file_2("/media/data/abhinav/ann_project_data/imagenet_edge_resized_mean.binaryproto");
  image_data_param->set_is_train(true);
  image_data_param->set_is_color_2(false);
  image_data_param->set_shuffle(true);
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);


  TransformationParameter* transform_param = param.mutable_transform_param();

  transform_param->set_mirror(true);
  transform_param->set_crop_size(227);

  TwinImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 256);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 227);
  EXPECT_EQ(this->blob_top_data_->width(), 227);

  EXPECT_EQ(this->blob_top_data_2_->num(), 256);
  EXPECT_EQ(this->blob_top_data_2_->channels(), 1);
  EXPECT_EQ(this->blob_top_data_2_->height(), 227);
  EXPECT_EQ(this->blob_top_data_2_->width(), 227);


  EXPECT_EQ(this->blob_top_label_->num(), 256);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    //for (int i = 0; i < 256; ++i) {
    //EXPECT_EQ(i, this->blob_top_label_->cpu_data()[i]);
    //}
  }
}


}  // namespace caffe
#endif  // USE_OPENCV
