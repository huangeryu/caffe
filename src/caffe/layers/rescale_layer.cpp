#include<vector>

#include "caffe/layers/rescale_layer.hpp"

namespace caffe
{
template<typename Dtype>
void RescaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    CHECK(this->layer_param_.has_rescale_param())<<"RescaleLayer not define.";
    CHECK(this->layer_param_.rescale_param().has_rf())<<"RescaleLayer not define receptive field.";
    rf_=this->layer_param_.rescale_param().rf();
}
template<typename Dtype>
void RescaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    top[0]->ReshapeLike(*bottom[0]);
}
template<typename Dtype>
void RescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    CHECK(bottom[0]->count()==top[0]->count())<<"RescaleLayer:bottom[0] and top[0] should have same count.";
    const Dtype* bm=bottom[0]->cpu_data();
    Dtype* tp=top[0]->mutable_cpu_data();
    for(int i=0;i<bottom[0]->count();i++)
    {
        if(bm[i]>Dtype(rf_) || bm[i]<Dtype(0.001))tp[i]=Dtype(-2);
        else tp[i]=2*bm[i]/rf_-1;
    }
}
template<typename Dtype>
void RescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom)
{
    //do_nothing
}

INSTANTIATE_CLASS(RescaleLayer);
REGISTER_LAYER_CLASS(Rescale);
#ifdef CPU_ONLY
STUB_GPU(RescaleLayer)
#endif
}
