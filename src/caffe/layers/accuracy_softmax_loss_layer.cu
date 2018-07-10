#include "caffe/layers/accuracy_softmax_loss_layer.hpp"
namespace caffe
{
template<typename Dtype>
__global__ void accuracy_softmax_loss_forward_gpu(int threads,int area,int stage,const Dtype* data, 
                    const Dtype* label,Dtype* accur)
{
    CUDA_KERNEL_LOOP(index,threads)
    {
        int batch_size=index/area;
        int offset=index%area;
        Dtype max=Dtype(0.);
        int k=0;
        int base=batch_size*area*stage;
        for(int i=0;i<stage;i++)
        {
            int pos=base+i*area+offset;
            if(max<data[pos])
            {
                max=data[pos];
                k=i;
            }
        }
        int s=label[index]>=stage?0:label[index];
        accur[index]=Dtype(0.);
        if(k==s)accur[index]++;
    }
}
template<typename Dtype>
void AccuracySoftMaxLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    const Dtype* data=bottom[0]->gpu_data();
    const Dtype* label=bottom[1]->gpu_data();
    int stage=bottom[0]->shape(1);
    int area=bottom[0]->count(2);
    int count=bottom[1]->count();
    Dtype* temp=NULL;
    cudaError_t state=cudaMalloc((void**)&temp,sizeof(Dtype)*count);
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc memory errror.";
    accuracy_softmax_loss_forward_gpu<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,
                area,stage,data,label,temp);
    Dtype loss=Dtype(0.);
    caffe_gpu_asum(count,temp,&loss);
    top[0]->mutable_cpu_data()[0]=loss/count;
    cudaFree(temp);
}
template<typename Dtype>
void AccuracySoftMaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagete_down,
                        const vector<Blob<Dtype>*>& bottom )
{
    //do-nothing
}
INSTANTIATE_LAYER_GPU_FUNCS(AccuracySoftMaxLossLayer);
}