#include <vector>

#include "caffe/layers/crop_layer.hpp"

namespace caffe {
template<typename Dtype>
__global__ void crop_kernel_forward(int threads,int crop_h,int corp_w,const Dtype* in,
  int in_h,int in_w,Dtype* out,int out_h,int out_w)
{
  int area_in=in_h*in_w;
  int area_out=out_h*out_w;
  CUDA_KERNEL_LOOP(index,threads)
  {
    int c_=index/out_h;
    int h_=index%out_h;
    const Dtype* in_offset=in+c_*area_in+(h_+crop_h)*in_w+corp_w;
    Dtype* out_offset=out+c_*area_out+h_*out_w;
    for(int i=0;i<in_w;i++)
    {
      out_offset[i]=in_offset[i];
    }
  }
}
template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int n = top[0]->count()/top[0]->shape(3);
  // NOLINT_NEXT_LINE(whitespace/operators)
  crop_kernel_forward<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,this->crop_h_,
                this->crop_w_,bottom_data,bottom[0]->shape(2),bottom[0]->shape(3),
                top_data,top[0]->shape(2),top[0]->shape(3));
}

template<typename Dtype>
__global__ void crop_kernel_backward(int threads,int crop_h,int corp_w,const Dtype* in,
  int in_h,int in_w,Dtype* out,int out_h,int out_w)
{
  int area_in=in_h*in_w;
  int area_out=out_h*out_w;
  CUDA_KERNEL_LOOP(index,threads)
  {
    int c_=index/in_h;
    int h_=index%in_h;
    Dtype* out_offset=out+c_*area_out+(h_+crop_h)*in_w+corp_w;
    const Dtype* in_offset=in+c_*area_in+h_*in_w;
    for(int i=0;i<in_w;i++)
    {
      out_offset[i]=in_offset[i];
    }
  }
}
template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int n = top[0]->count()/top[0]->shape(3);

  if (propagate_down[0]) {
    caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    // NOLINT_NEXT_LINE(whitespace/operators)
    crop_kernel_backward<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,this->crop_h_,
                  this->crop_w_,top_diff,top[0]->shape(2),top[0]->shape(3),bottom_diff,
                  bottom[0]->shape(2),bottom[0]->shape(3));
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);

}  // namespace caffe
