#include "caffe/layers/accuracy_softmax_loss_layer.hpp"
namespace caffe
{
template<typename Dtype>
void AccuracySoftMaxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    vector<int> top_shape(0);
    top[0]->Reshape(top_shape);
    CHECK(bottom[0]->shape(0)==bottom[1]->shape(0));
    CHECK(bottom[0]->count(2)==bottom[0]->count(2));
}
template<typename Dtype>
void AccuracySoftMaxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    Dtype precision=Dtype(0.);
    const Dtype* data=bottom[0]->cpu_data();
    const Dtype* label=bottom[1]->cpu_data();
    int batch_size=bottom[0]->shape(0);
    int stage=bottom[0]->shape(1);
    int area=bottom[0]->count(2);
    int positive=0;
    int count=0;
    for(int i=0;i<batch_size;i++)
    {
       for(int j=0;j<area;j++)
       {
           int k=0;
           Dtype max=Dtype(0.);
           for(int c=0;c<stage;c++)
           {
               int index=i*area*stage+c*area+j;
               if(max<data[index])
               {
                   k=c;
                   max=data[index];
               }
           }
           int s=label[i*area+j]>=stage?0:label[i*area+j];
           if(s==k)positive++;
           count++;
       }
    }
    precision=Dtype(positive)/count;
    top[0]->mutable_cpu_data()[0]=precision;
}
template<typename Dtype>
void AccuracySoftMaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom)
{
    //do-nothing;
}
INSTANTIATE_CLASS(AccuracySoftMaxLossLayer);
REGISTER_LAYER_CLASS(AccuracySoftMaxLoss);
#ifdef CPU_ONLY
STUB_GPU(AccuracySoftMaxLossLayer)
#endif
}