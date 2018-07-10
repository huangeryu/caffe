#include "caffe/layers/rescale_layer.hpp"
template<typename Dtype>
__global__ void rescale_forward_gpu(int threads, int rf,const Dtype* in,int count, Dtype* out)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        if(in[index]>rf || in[index]<Dtype(0.00001))
        {
            out[index]=Dtype(-1);
        }
        else
        {
            out[index]=2*in[index]/rf-1;
        }
    }
}
namespace caffe
{
template<typename Dtype>
void RescaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data=bottom[0]->gpu_data();
    Dtype* top_data=top[0]->mutable_gpu_data();
    int count=bottom[0]->count();
    rescale_forward_gpu<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,this->rf_,
            bottom_data,count,top_data);
}

template<typename Dtype>
void RescaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    //do-nothing
}
INSTANTIATE_LAYER_GPU_FUNCS(RescaleLayer);
}