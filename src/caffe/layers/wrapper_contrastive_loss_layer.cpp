#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void printMatrix(const Dtype* matrix, const int X, const int Y){
  for (int x =0; x < X; x++){
    for(int y=0; y < Y; y++){
      std::cout << matrix[(x*Y) + y] << ", ";
    }
    std::cout << std::endl;
  }
    std::cout << std::endl << std::endl;
}


template <typename Dtype>
void WrapperContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1); //Check to ensure a single label for each input
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);

  bottom_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  bottom_p_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  sims_.Reshape(bottom[0]->num(), 1, 1, 1);

  bottom_vec_.clear();
  top_vec_.clear();

  for (int i = 0; i < layers_.size(); i++){
    delete layers_[i];
  }

  layers_.clear();

  bottom_vec_.push_back(&bottom_);
  bottom_vec_.push_back(&bottom_p_);
  bottom_vec_.push_back(&sims_);

  top_.Reshape(1,1,1,1);
  top_vec_.push_back(&top_);

  for(int i = 1; i < bottom[0]->num(); i++){
    ContrastiveLossLayer<Dtype>* l = new ContrastiveLossLayer<Dtype>(this->layer_param_);
    l->SetUp(bottom_vec_, top_vec_);
    layers_.push_back(l);
  }
}



template <typename Dtype>
void WrapperContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype loss(0.0);
  
  const int N = bottom[0]->num();
  const int F = bottom[0]->channels();
  const int totalData = bottom[0]->count();

  const Dtype* labels = bottom[1]->cpu_data();
  caffe_copy(totalData, bottom[0]->cpu_data(), bottom_.mutable_cpu_data());

  /*
  std::cout << "data:" << std::endl;
  printMatrix(bottom[0]->cpu_data(), N, F);

  std::cout << "labels:" << std::endl;
  printMatrix(bottom[1]->cpu_data(), N, 1);


  std::cout << "bottom:" << std::endl;
  printMatrix(bottom_.cpu_data(), N, F);
  */

  for (int l = 0; l < layers_.size(); l++){
    int idx = (l+1)*F; //Always skip the first one, otherwise will simply copy
    int firstChunk = totalData - idx;
    
    caffe_copy(firstChunk, bottom_.cpu_data() + idx, bottom_p_.mutable_cpu_data());
    caffe_copy(idx, bottom_.cpu_data(), bottom_p_.mutable_cpu_data() + firstChunk);

    //std::cout << "bottom_p (" << l+1 << "):" << std::endl;
    //printMatrix(bottom_p_.cpu_data(), N, F);

    Dtype* sims = sims_.mutable_cpu_data();
    for (int i = 0; i < N; i++){
      sims[i] = (labels[i] == labels[(i + (l+1))%N]) ? Dtype(1.0) : Dtype(0.0);
    }
	
    //std::cout << "sims (" << l+1 << "):" << std::endl;
    //printMatrix(sims, N, 1);

    layers_[l]->Forward(bottom_vec_, top_vec_);
    loss += top_.cpu_data()[0];

    //std::cout << "loss (" << l+1 << "): " << top_.cpu_data()[0] << std::endl;
    //std::cout << std::endl;
  }

  //std::cout << "Final loss: " << loss << std::endl;
  //std::cout << std::endl;

  top[0]->mutable_cpu_data()[0] = loss / static_cast<Dtype>(N-1)  / Dtype(2.0);
}

template <typename Dtype>
void WrapperContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int N = bottom[0]->num();

    vector<bool> prop_down;
    prop_down.push_back(true);
    prop_down.push_back(false);
    prop_down.push_back(false);

    top_.mutable_cpu_diff()[0] = top[0]->cpu_diff()[0];
    const Dtype* labels = bottom[1]->cpu_data();

    caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());

    for (int l = 0; l < layers_.size(); l++){
      //caffe_set(bottom_.count(), Dtype(0.0), bottom_.mutable_cpu_diff());
      
      //need to set sims again because Contrastive Loss uses this in gradient.
      Dtype* sims = sims_.mutable_cpu_data();
      for (int i = 0; i < N; i++){
        sims[i] = (labels[i] == labels[(i + (l+1))%N]) ? Dtype(1.0) : Dtype(0.0);
      }

      layers_[l]->Backward(top_vec_, prop_down, bottom_vec_);

      caffe_axpy(bottom[0]->count(), Dtype(1.0/(N-1)), bottom_.cpu_diff(), bottom[0]->mutable_cpu_diff());
    }

    //std::cout << "diff:" << std::endl;
    //printMatrix(bottom[0]->cpu_diff(), N, bottom[0]->channels());
  }
}

#ifdef CPU_ONLY
STUB_GPU(WrapperContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(WrapperContrastiveLossLayer);
REGISTER_LAYER_CLASS(WrapperContrastiveLoss);

}  // namespace caffe
