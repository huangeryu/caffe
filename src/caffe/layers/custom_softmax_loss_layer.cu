#include<cfloat>
#include "caffe/layers/custom_softmax_loss_layer.hpp"

namespace caffe
{

template<typename Dtype>
__global__ void comput_alpha(int threads,int stage,const Dtype* in,Dtype* sums)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        if(in[index]>stage-1 || in[index]<Dtype(0.001))
        {
            sums[index]=0.;
        }
        else
        {
            sums[index]=1.;
        }
    }
}
template<typename Dtype>
__global__ void custom_softmax_forward_gpu(int threads,int area,int stage,
            const Dtype* bata,const Dtype* data,const Dtype* label,Dtype* out)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        int channel=index/area;
        int offset=index%area;
        Dtype alpha=bata[channel];
        int k=label[index]>=stage?0:label[index];
        if(k>0)alpha=1-alpha;
        int i=channel*area*stage+k*area+offset;
        out[index]=alpha*log(std::max(data[i],Dtype(FLT_MIN)));
    }
}
template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    int batch_size=bottom[0]->shape(0);
    const Dtype* qlabel=bottom[1]->gpu_data();
    this->bate_.clear();
    this->bate_.resize(batch_size);
    this->softmaxlayer_->Forward(this->bottom_,this->top_);
    const Dtype* label=bottom[1]->gpu_data();
    const Dtype* data=this->pro_.gpu_data();
    Dtype  area=bottom[1]->count(1);
    Dtype* temp;
    cudaError_t state=cudaMalloc((void**)&temp,bottom[1]->count()*sizeof(Dtype));
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc fail!";
    comput_alpha<<<CAFFE_GET_BLOCKS(bottom[1]->count()),CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->count(),this->stage_,label,temp);
    for(int i=0;i<batch_size;i++)
    {
        caffe_gpu_asum(Dtype(area),&temp[i*int(area)],&(this->bate_[i]));
        this->bate_[i]/=area;
    }
    Dtype* alpha;
    state =cudaMalloc((void**)&alpha,this->bate_.size()*sizeof(Dtype));
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc fail!";
    state=cudaMemcpy(alpha,&(this->bate_[0]),batch_size*sizeof(Dtype),cudaMemcpyHostToDevice);
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMemcpy fail!";
    custom_softmax_forward_gpu<<<CAFFE_GET_BLOCKS(bottom[1]->count()),CAFFE_CUDA_NUM_THREADS>>>(bottom[1]->count(),
                    area,this->stage_,alpha,data,label,temp);
    Dtype loss=Dtype(0.);
    caffe_gpu_asum(bottom[1]->count(),temp,&loss);
    top[0]->mutable_cpu_data()[0]=this->scale_*loss/batch_size;
    cudaFree(temp);
    cudaFree(alpha);
}

template<typename Dtype>
__global__ void custom_softmax_backward_gpu(int threads,int stage,Dtype scale,int area,
    const Dtype* qlabel,const Dtype* data,const Dtype* alpha,Dtype* diff )
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        int batch_size=index/(area*stage);
        int channel=index/area;
        int k=channel%stage;
        int offset=index%area;
        Dtype bata=(channel==0)?alpha[batch_size]:1-alpha[batch_size];
        if(int(qlabel[batch_size*area+offset])==k)
        {
            diff[index]=scale*bata*(1.0 - data[index]);
        }
        else
        {
            diff[index]=-scale*bata*data[index];
        }
    }
}
template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom)
{
    int n=bottom[0]->count();
    int area=bottom[0]->count(2);
    const Dtype* qlabel=bottom[1]->gpu_data();
    const Dtype* data=this->pro_.gpu_data();
    Dtype* diff=bottom[0]->mutable_gpu_diff();
    Dtype* alpha;
    int size=this->bate_.size()*sizeof(Dtype);
    cudaError_t state=cudaMalloc((void**)&alpha,size);
    if(state!=cudaSuccess){LOG(FATAL)<<"malloc memeory on GPU fail";}
    state=cudaMemcpy(alpha,this->bate_.data(),size,cudaMemcpyHostToDevice);
    if(state!=cudaSuccess){LOG(FATAL)<<"copy memeory on GPU fail";}
    Dtype scale=-this->scale_/bottom[1]->shape(0);
    custom_softmax_backward_gpu<<<CAFFE_GET_BLOCKS(n),CAFFE_CUDA_NUM_THREADS>>>(n,this->stage_,scale
                area,qlabel,data,alpha,diff);
    cudaFree(alpha);
}
INSTANTIATE_LAYER_GPU_FUNCS(CustomSoftMaxLossLayer);
}