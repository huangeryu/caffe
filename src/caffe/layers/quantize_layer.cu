#include "caffe/layers/quantize_layer.hpp"
namespace caffe
{
template<typename Dtype>
__global__ void quantize_forward_gpu(int threads,const Dtype* rfs,int size,const Dtype* in ,int area,
            Dtype* out1,Dtype* out2)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        int batch_size=index/area;
        for(int i=0;i<size;i++)
        {
            if(in[index]<rfs[i])
            {
                out1[index]=Dtype(i);
                return;
            }
        }
        out1[index]=Dtype(0.);
    }

}
template<typename Dtype>
void QuantizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Dtype* qlabel_data=top[0]->mutable_gpu_data();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    int n=bottom[0]->count();
    Dtype* rfs;
    int size=this->rfs_.size()*sizeof(Dtype);
    cudaError_t state=cudaMalloc((void**)&rfs,size);
    if(state!=cudaSuccess){LOG(FATAL)<<"cudaMalloc faill";}
    state=cudaMemcpy(rfs,this->rfs_.data(),size,cudaMemcpyHostToDevice);
    if(state!=cudaSuccess){LOG(FATAL)<<"cudaMemcpy fail";}
    quantize_forward_gpu<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,rfs,this->rfs_.size(),
                bottom_data,bottom[0]->count(2),qlabel_data);
    cudaFree(rfs);
}
template<typename Dtype>
void QuantizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    //do-nothing
}
INSTANTIATE_LAYER_GPU_FUNCS(QuantizeLayer);
}