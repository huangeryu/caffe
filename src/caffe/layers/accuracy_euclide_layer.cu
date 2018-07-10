#include "caffe/layers/accuracy_euclide_layer.hpp"
namespace caffe
{
template<typename Dtype>
__global__ void accuracy_euclide_forward_gpu(int threads,const Dtype* data,const Dtype* label,Dtype* out,Dtype* count)
{
    count[0]=Dtype(0.);
    CUDA_KERNEL_LOOP(index,threads)
    {
        out[index]=Dtype(0.);
        if(label[index]>Dtype(-1))
        {
            out[index]=(data[index]-label[index])*(data[index]-label[index]);
            count[0]++;
        }
    }
}
template<typename Dtype>
void AccuracyEuclideLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    const Dtype* data=bottom[0]->gpu_data();
    const Dtype* label=bottom[1]->gpu_data();
    int count=bottom[0]->count();
    Dtype* temp;
    cudaError_t state=cudaMalloc((void**)&temp,sizeof(Dtype)*count);
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMalloc memory fail";
    Dtype * sum;
    state=cudaMalloc((void**)&sum,sizeof(Dtype));
    if(state!=cudaSuccess)LOG(FATAL)<<"cudaMaloc memory fail";
    accuracy_euclide_forward_gpu<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(count,data,label,temp,sum);
    Dtype hsum=Dtype(0.);
    Dtype loss=Dtype(0.);
    caffe_gpu_asum(count,sum,&hsum);
    caffe_gpu_asum(count,temp,&loss);
    CHECK(hsum>=Dtype(1.));
    top[0]->mutable_cpu_data()[0]=sqrt(loss/hsum);
    cudaFree(sum);
    cudaFree(temp);
}
template<typename Dtype>
void AccuracyEuclideLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagete_down,
                        const vector<Blob<Dtype>*>& bottom )
{
    //do-nothing
}
INSTANTIATE_LAYER_GPU_FUNCS(AccuracyEuclideLayer);
}