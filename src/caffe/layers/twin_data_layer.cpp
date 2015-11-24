#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_transformer.hpp"

namespace caffe {

    template <typename Dtype>
    TwinImageDataLayer<Dtype>::TwinImageDataLayer(const LayerParameter& param)
        : BasePrefetchingTwinDataLayer<Dtype>(param) {
        //TransformationParameter trans_param_(param);
        if (this->layer_param_.twin_image_data_param().has_mean_file()) {
            //CHECK_EQ(this->layer_param_.twin_image_data_param().mean_value_size(), 0) <<
            // "Cannot specify mean_file and mean_value at the same time";
           const string& mean_file = this->layer_param_.twin_image_data_param().mean_file();
           if (Caffe::root_solver()) {
               LOG(INFO) << "Loading mean file from: " << mean_file;
           }
           BlobProto blob_proto;
           ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
           data_mean_.FromProto(blob_proto);
        }
        //TransformationParameter trans_param_(param);
        if (this->layer_param_.twin_image_data_param().has_mean_file_2()) {
            //CHECK_EQ(this->layer_param_.twin_image_data_param().mean_value_size(), 0) <<
            //  "Cannot specify mean_file and mean_value at the same time";
            const string& mean_file = this->layer_param_.twin_image_data_param().mean_file_2();
            if (Caffe::root_solver()) {
                LOG(INFO) << "Loading mean file from: " << mean_file;
            }
            BlobProto blob_proto;
            ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
            data_mean_2_.FromProto(blob_proto);
        }
    }


    template <typename Dtype>
    TwinImageDataLayer<Dtype>::~TwinImageDataLayer<Dtype>() {
        this->StopInternalThread();
    }

    template <typename Dtype>
    void TwinImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
        const int new_height = this->layer_param_.twin_image_data_param().new_height();
        const int new_width  = this->layer_param_.twin_image_data_param().new_width();
        const bool is_color  = this->layer_param_.twin_image_data_param().is_color();
        const bool is_color2 = this->layer_param_.twin_image_data_param().is_color_2();
        string root_folder = this->layer_param_.twin_image_data_param().root_folder();

        CHECK((new_height == 0 && new_width == 0) ||
              (new_height > 0 && new_width > 0)) << "Current implementation requires "
            "new_height and new_width to be set at the same time.";
        // Read the file with filenames and labels
        const string& source = this->layer_param_.twin_image_data_param().source();

        LOG(INFO) << "Opening file " << source;

        this->InitRand();

        std::ifstream infile(source.c_str());

        string filename;
        string filename2;
        int label;
        while (infile >> filename >> filename2 >> label) {
            lines_.push_back(std::make_pair(std::make_pair(filename, filename2),
                                            label));
        }

        if (this->layer_param_.twin_image_data_param().shuffle()) {
            // randomly shuffle data
            LOG(INFO) << "Shuffling data";
            const unsigned int prefetch_rng_seed = caffe_rng_rand();
            prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
            ShuffleImages();
        }
        LOG(INFO) << "A total of " << lines_.size() << " images.";

        lines_id_ = 0;
        // Check if we would need to randomly skip a few data points
        if (this->layer_param_.twin_image_data_param().rand_skip()) {
            unsigned int skip = caffe_rng_rand() %
                this->layer_param_.twin_image_data_param().rand_skip();
            LOG(INFO) << "Skipping first " << skip << " data points.";
            CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
            lines_id_ = skip;
        }
        // Read an image, and use it to initialize the top blob.
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                          new_height, new_width, is_color);
        cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                           new_height, new_width, is_color2);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;
        CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_].first.second;
        // Use data_transformer to infer the expected blob shape from a cv_image.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
        vector<int> top_shape2 = this->data_transformer_->InferBlobShape(cv_img2);
        this->transformed_data_.Reshape(top_shape);
        this->transformed_data_2_.Reshape(top_shape2);
        // Reshape prefetch_data and top[0] according to the batch_size.
        const int batch_size = this->layer_param_.twin_image_data_param().batch_size();
        CHECK_GT(batch_size, 0) << "Positive batch size required";
        top_shape[0] = batch_size;
        top_shape2[0] = batch_size;
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(top_shape);
            this->prefetch_[i].data_2_.Reshape(top_shape2);
        }
        top[0]->Reshape(top_shape);
        top[1]->Reshape(top_shape2);
        LOG(INFO) << "output data size: " << top[0]->num() << ","
                  << top[0]->channels() << "," << top[0]->height() << ","
                  << top[0]->width();
        LOG(INFO) << "output data size: " << top[1]->num() << ","
                  << top[1]->channels() << "," << top[1]->height() << ","
                  << top[1]->width();
        // label
        vector<int> label_shape(1, batch_size);
        top[2]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }

    }
    template <typename Dtype>
    void TwinImageDataLayer<Dtype>::ShuffleImages() {
        caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

// This function is called on prefetch thread
    template <typename Dtype>
    void TwinImageDataLayer<Dtype>::load_batch(TwinBatch<Dtype>* batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(batch->data_.count());
        CHECK(this->transformed_data_.count());

        CHECK(batch->data_2_.count());
        CHECK(this->transformed_data_2_.count());
        TwinImageDataParameter twin_image_data_param = this->layer_param_.twin_image_data_param();
        const int batch_size = twin_image_data_param.batch_size();
        const int new_height = twin_image_data_param.new_height();
        const int new_width = twin_image_data_param.new_width();
        const bool is_color = twin_image_data_param.is_color();
        const bool is_color2 = twin_image_data_param.is_color_2();

        string root_folder = twin_image_data_param.root_folder();

        // Reshape according to the first image of each batch
        // on single input batches allows for inputs of varying dimension.
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                          new_height, new_width, is_color);
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;
        cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                          new_height, new_width, is_color2);

        CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_].first.second;
        // Use data_transformer to infer the expected blob shape from a cv_img.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
        vector<int> top_shape2 = this->data_transformer_->InferBlobShape(cv_img2);
        this->transformed_data_.Reshape(top_shape);

        this->transformed_data_2_.Reshape(top_shape2);
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        top_shape2[0] = batch_size;
        batch->data_.Reshape(top_shape);
        batch->data_2_.Reshape(top_shape2);
        Dtype* prefetch_data = batch->data_.mutable_cpu_data();
        Dtype* prefetch_data2 = batch->data_2_.mutable_cpu_data();
        Dtype* prefetch_label = batch->label_.mutable_cpu_data();

        // datum scales
        const int lines_size = lines_.size();
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            // get a blob
            timer.Start();
            CHECK_GT(lines_size, lines_id_);
            cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first.first,
                                              new_height, new_width, is_color);
            cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines_[lines_id_].first.second,
                                              new_height, new_width, is_color2);
            CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first.first;
            CHECK(cv_img2.data) << "Could not load " << lines_[lines_id_].first.second;
            read_time += timer.MicroSeconds();
            timer.Start();
            // Apply transformations (mirror, crop...) to the image
            int offset = batch->data_.offset(item_id);
            int offset2 = batch->data_2_.offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_data + offset);
            this->transformed_data_2_.set_cpu_data(prefetch_data2 + offset2);
            this->Transform(cv_img, &(this->transformed_data_),
                            cv_img2, &(this->transformed_data_2_));
            //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
            //this->data_transformer_->Transform(cv_img2, &(this->transformed_data_2_));


            trans_time += timer.MicroSeconds();

            prefetch_label[item_id] = lines_[lines_id_].second;
            // go to the next iter
            lines_id_++;
            if (lines_id_ >= lines_size) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.twin_image_data_param().shuffle()) {
                    ShuffleImages();
                }
            }
        }
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

    }

    // TODO: Move to BaseTwinDataPrefetcher
    template <typename Dtype>
    void TwinImageDataLayer<Dtype>::Transform(const cv::Mat& cv_img,
                                              Blob<Dtype>* transformed_blob,
                                              const cv::Mat& cv_img2,
                                              Blob<Dtype>* transformed_blob2) {

        const int img_channels = cv_img.channels();
        const int img2_channels = cv_img2.channels();
        const int img_height = cv_img.rows;
        const int img_width = cv_img.cols;

        const int img2_height = cv_img2.rows;
        const int img2_width = cv_img2.cols;

        // Check dimensions.
        const int channels = transformed_blob->channels();
        const int height = transformed_blob->height();
        const int width = transformed_blob->width();
        const int num = transformed_blob->num();

        CHECK_EQ(channels, img_channels);
        CHECK_LE(height, img_height);
        CHECK_LE(width, img_width);
        CHECK_GE(num, 1);

        CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

        // Check dimensions.
        const int channels2 = transformed_blob2->channels();
        const int height2 = transformed_blob2->height();
        const int width2 = transformed_blob2->width();
        const int num2 = transformed_blob2->num();

        CHECK_EQ(channels2, img2_channels);
        CHECK_LE(height2, img2_height);
        CHECK_LE(width2, img2_width);
        CHECK_GE(num2, 1);

        CHECK(cv_img2.depth() == CV_8U) << "Image data type must be unsigned byte";

        const Dtype scale = this->layer_param_.transform_param().scale();
        const bool do_mirror = this->layer_param_.transform_param().mirror() && Rand(2);


        // Crop parameters
        const int crop_size = this->layer_param_.transform_param().crop_size();

        const bool has_mean_file = this->layer_param_.twin_image_data_param().has_mean_file();
        const bool has_mean_file_2 = this->layer_param_.twin_image_data_param().has_mean_file_2();

        CHECK_GT(img_channels, 0);
        CHECK_GE(img_height, crop_size);
        CHECK_GE(img_width, crop_size);

        CHECK_GT(img2_channels, 0);
        CHECK_GE(img2_height, crop_size);
        CHECK_GE(img2_width, crop_size);

        Dtype* mean = NULL;
        if (has_mean_file) {
            CHECK_EQ(img_channels, data_mean_.channels());
            CHECK_EQ(img_height, data_mean_.height());
            CHECK_EQ(img_width, data_mean_.width());
            mean = data_mean_.mutable_cpu_data();
        }

        Dtype* mean2 = NULL;
        if (has_mean_file_2) {
            CHECK_EQ(img2_channels, data_mean_2_.channels());
            CHECK_EQ(img2_height, data_mean_2_.height());
            CHECK_EQ(img2_width, data_mean_2_.width());
            mean2 = data_mean_2_.mutable_cpu_data();
        }

        int h_off = 0;
        int w_off = 0;

        cv::Mat cv_cropped_img = cv_img;
        cv::Mat cv_cropped_img2 = cv_img2;

        if (crop_size) {
            CHECK_EQ(crop_size, height);
            CHECK_EQ(crop_size, width);
            // We only do random crop when we do training.
            if (this->layer_param_.twin_image_data_param().is_train()) {
                h_off = Rand(img_height - crop_size + 1);
                w_off = Rand(img_width - crop_size + 1);
            } else {
                h_off = (img_height - crop_size) / 2;
                w_off = (img_width - crop_size) / 2;
            }
            cv::Rect roi(w_off, h_off, crop_size, crop_size);
            cv_cropped_img = cv_img(roi);
            cv_cropped_img2 = cv_img2(roi);
        } else {
            CHECK_EQ(img_height, height);
            CHECK_EQ(img_width, width);
        }

        CHECK(cv_cropped_img.data);
        CHECK(cv_cropped_img2.data);

        Dtype* transformed_data = transformed_blob->mutable_cpu_data();
        Dtype* transformed_data2 = transformed_blob2->mutable_cpu_data();
        int top_index;

        for (int h = 0; h < height; ++h) {
            const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
            const uchar* ptr2 = cv_cropped_img2.ptr<uchar>(h);
            int img_index = 0;
            int img_index2 = 0;
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < img_channels; ++c) {
                    if (do_mirror) {
                        top_index = (c * height + h) * width + (width - 1 - w);
                    } else {
                        top_index = (c * height + h) * width + w;
                    }
                    // int top_index = (c * height + h) * width + w;
                    Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                    Dtype pixel2;
                    if (c < img2_channels) {
                        pixel2 = static_cast<Dtype>(ptr2[img_index2++]);
                    }
                    if (has_mean_file) {
                        int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                        transformed_data[top_index] =
                            (pixel - mean[mean_index]) * scale;
                    } else {
                        transformed_data[top_index] = pixel * scale;
                    }
                    if (has_mean_file_2 && c < img2_channels) {
                        int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                        transformed_data2[top_index] =
                            (pixel2 - mean2[mean_index]) * scale;
                    } else if (c < img2_channels) {
                        transformed_data2[top_index] = pixel2 * scale;
                    }
                }
            }
        }
    }

    template <typename Dtype>
    int TwinImageDataLayer<Dtype>::Rand(int n) {
        CHECK(rng_);
        CHECK_GT(n, 0);
        caffe::rng_t* rng =
            static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
    }
    template <typename Dtype>
    void TwinImageDataLayer<Dtype>::InitRand() {
        const bool needs_rand = this->layer_param_.transform_param().mirror() ||
            (this->layer_param_.twin_image_data_param().is_train()
             && this->layer_param_.transform_param().crop_size());
        if (needs_rand) {
            const unsigned int rng_seed = caffe_rng_rand();
            rng_.reset(new Caffe::RNG(rng_seed));
        } else {
            rng_.reset();
        }
    }

    INSTANTIATE_CLASS (TwinImageDataLayer);
    REGISTER_LAYER_CLASS (TwinImageData);
}  // namespace caffe
