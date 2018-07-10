#include<cstring>
#include "caffe/layers/quantize_layer.hpp"

namespace caffe
{
template<typename Dtype>
void QuantizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    CHECK(this->layer_param_.has_quantize_param() && 
            this->layer_param_.quantize_param().rf_size())<<"quantizelayer not define quantize_param or rf";
    caffe::QuantizeParamter quantize=this->layer_param_.quantize_param();
    this->scale_=Dtype(1.0);
    if(quantize.has_scale() && quantize.scale()>(Dtype)0.0001)this->scale_=Dtype(quantize.scale());
    for(int i=0;i<quantize.rf_size();i++)
    {
        float temp=quantize.rf(i) * this->scale_;
        this->rfs_.push_back(int(temp));
    }
    std::sort(this->rfs_.begin(),this->rfs_.end());

}

template<typename Dtype>
void QuantizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    CHECK(bottom[0]->shape(1)==1)<<"raw-labe should have one demension";
    vector<int> shape=bottom[0]->shape();
    top[0]->ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void QuantizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    Dtype* qlabel=top[0]->mutable_cpu_data();
    const Dtype* raw_label=bottom[0]->cpu_data();
    caffe_set(top[0]->count(),Dtype(0.0),qlabel);
    int dim=bottom[0]->count(1);//note bottom[0]->shape(1)==1
    for(int i=0;i<bottom[0]->shape(0);i++)
    {
        for(int j=0;j<dim;j++)
        {
            for(int k=0;k<this->rfs_.size();k++)
            {
                if(raw_label[i*dim+j]<=this->rfs_[k])
                {
                    qlabel[i*dim+j]=Dtype(k);
                    break;
                }
            }
        }
    }
}

template<typename Dtype>
void QuantizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom)
{
    //do-nothing
}
INSTANTIATE_CLASS(QuantizeLayer);
REGISTER_LAYER_CLASS(Quantize);
#ifdef CPU_ONLY
STUB_GPU(QuantizeLayer)
#endif
}//namespace