#include<cmath>
#include "caffe/layers/accuracy_euclide_layer.hpp"
namespace caffe
{
template<typename Dtype>
void AccuracyEuclideLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    CHECK(bottom.size()==2);
    vector<int> top_shape(0);
    top[0]->Reshape(top_shape);
}
template<typename Dtype>
void AccuracyEuclideLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    int num=bottom[0]->count();
    const Dtype* label=bottom[1]->cpu_data();
    const Dtype* data=bottom[0]->cpu_data();
    int count=0;
    Dtype sigma=Dtype(0.0);
    for(int i=0;i<num;i++)
    {
        if(label[i]>Dtype(-1.))
        {
            sigma+=(data[i]-label[i])*(data[i]-label[i]);
            count++;
        }
    }
    sigma=sqrt(sigma/count);
    top[0]->mutable_cpu_data()[0]=sigma;
}

template<typename Dtype>
void AccuracyEuclideLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                    const vector<Blob<Dtype>*>& bottom)
{
    //do-nothing
}
INSTANTIATE_CLASS(AccuracyEuclideLayer);
REGISTER_LAYER_CLASS(AccuracyEuclide);
#ifdef CPU_ONLY
STUB_GPU(AccuracyEuclideLayer)
#endif
}