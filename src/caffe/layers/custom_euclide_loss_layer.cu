#include "caffe/layers/custom_euclide_loss_layer.hpp"
namespace caffe
{
template<typename Dtype>
__global__ void custom_euclide_forward_gpu(int threads,const Dtype* label,const Dtype* in,Dtype* dis,Dtype* temp)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        if(label[index]>Dtype(-1))
        {
            temp[index]=static_cast<Dtype>(1);
            dis[index]=in[index]-label[index];
        }
        else dis[index]=in[index];
    }
}
template<typename Dtype>
__global__ void muti_alpha(int threads,const int area,const Dtype* in,const Dtype* alpha,const Dtype* label,Dtype* out)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        int batch_size=index/area;
        if(label[index]<Dtype(-1.))
        {
            out[index]=alpha[batch_size]*in[index];
        }
        else
        {
            out[index]=(1-alpha[batch_size])*in[index];
        }
    }
} 
template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
   const Dtype* bottom_data=bottom[0]->gpu_data();
   const Dtype* bottom_label=bottom[1]->gpu_data();
   this->alpha.clear();
   this->alpha.resize(bottom[0]->shape(0));
   Dtype* top_data=top[0]->mutable_cpu_data();
   Dtype* dis_data=this->dis_.mutable_gpu_data();
   Dtype* temp=this->dis_.mutable_gpu_diff();
   int n=bottom[0]->count(1);
   caffe_gpu_set(n,Dtype(0.),dis_data);
   caffe_gpu_set(n,Dtype(0.),temp);
   custom_euclide_forward_gpu<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,bottom_label,bottom_data,
                    dis_data,temp);
    for(int i=0;i<bottom[0]->shape(0);i++)
    {
        caffe_gpu_asum(n,temp,&(this->alpha[i]));
        this->alpha[i]/=bottom[0]->count(1);
    }
    caffe_gpu_powx(n,dis_data,static_cast<Dtype>(2.0),temp);
    Dtype* bate;
    cudaError_t state=cudaMalloc((void**)&bate,sizeof(Dtype)*this->alpha.size());
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc error!";
    state=cudaMemcpy(bate,&(this->alpha[0]),this->alpha.size()*sizeof(Dtype),cudaMemcpyHostToDevice);
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMemcpy error!";
    muti_alpha<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(),
                    bottom[0]->count(1),temp,bate,bottom_label,temp);
    Dtype loss=static_cast<Dtype>(0.0);
    caffe_gpu_asum(n,temp,&loss);
    top_data[0]=this->loss_param*loss/bottom[0]->shape(0);
    cudaFree(bate);
}


template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[0])
    {
        const Dtype* data=this->dis_.gpu_data();
        Dtype* temp=this->dis_.mutable_gpu_diff();
        Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();
        int n=bottom[0]->count();
        Dtype* bate;
        cudaError_t state=cudaMalloc((void**)&bate,sizeof(Dtype)*this->alpha.size());
        if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc error!";
        state=cudaMemcpy(bate,&(this->alpha[0]),this->alpha.size()*sizeof(Dtype),cudaMemcpyHostToDevice);
        if(state!=cudaSuccess)LOG(FATAL)<<"cudaMemcpy error!";
        muti_alpha<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,bottom[1]->count(1),data,bate,bottom[1]->gpu_data(),temp);
        caffe_gpu_axpby(n,Dtype(2.0/bottom[0]->shape(0)),temp,Dtype(0.),bottom_diff);
        cudaFree(bate);
    }
}
INSTANTIATE_LAYER_GPU_FUNCS(CustomEuclideLossLayer);
}