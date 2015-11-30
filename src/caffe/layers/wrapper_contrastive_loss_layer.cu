#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


/*
template <typename Dtype>
void WrapperContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype loss(0.0);
  
  const int N = bottom[0]->num();
  const int F = bottom[0]->channels();
  const int totalData = bottom[0]->count();

  const Dtype* labels = bottom[1]->cpu_data();
  caffe_copy(totalData, bottom[0]->gpu_data(), bottom_.mutable_gpu_data());


  for (int l = 0; l < layers_.size(); l++){
    int idx = (l+1)*F; //Always skip the first one, otherwise will simply copy
    int firstChunk = totalData - idx;
    
    caffe_copy(firstChunk, bottom_.gpu_data() + idx, bottom_p_.mutable_gpu_data());
    caffe_copy(idx, bottom_.gpu_data(), bottom_p_.mutable_gpu_data() + firstChunk);

    Dtype* sims = sims_.mutable_cpu_data();
    for (int i = 0; i < N; i++){
      sims[i] = (labels[i] == labels[(i + (l+1))%N]) ? Dtype(1.0) : Dtype(0.0);
    }

    layers_[l]->Forward(bottom_vec_, top_vec_);
    loss += top_.cpu_data()[0];
  }

  top[0]->mutable_cpu_data()[0] = loss / static_cast<Dtype>(N-1)  / Dtype(2.0);
}


template <typename Dtype>
void WrapperContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const int N = bottom[0]->num();

    vector<bool> prop_down;
    prop_down.push_back(true);
    prop_down.push_back(false);
    prop_down.push_back(false);

    top_.mutable_cpu_diff()[0] = top[0]->cpu_diff()[0];
    const Dtype* labels = bottom[1]->cpu_data();

    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_gpu_diff());

    for (int l = 0; l < layers_.size(); l++){
      //caffe_set(bottom_.count(), Dtype(0.0), bottom_.mutable_cpu_diff());
      
      //need to set sims again because Contrastive Loss uses this in gradient.
      Dtype* sims = sims_.mutable_cpu_data();
      for (int i = 0; i < N; i++){
        sims[i] = (labels[i] == labels[(i + (l+1))%N]) ? Dtype(1.0) : Dtype(0.0);
      }

      layers_[l]->Backward(top_vec_, prop_down, bottom_vec_);

      caffe_gpu_axpy(bottom[0]->count(), Dtype(1.0/(N-1)), bottom_.gpu_diff(), bottom[0]->mutable_gpu_diff());
    }

    //std::cout << "diff:" << std::endl;
    //printMatrix(bottom[0]->cpu_diff(), N, bottom[0]->channels());
  }
}
*/
//INSTANTIATE_LAYER_GPU_FUNCS(WrapperContrastiveLossLayer);

}  // namespace caffe
