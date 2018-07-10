#ifndef CAFFE_CUSTOM_SOFTMAX_LOSS_LAYER_HPP_
#define CAFFE_CUSTOM_SOFTMAX_LOSS_LAYER_HPP_
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layer.hpp"
namespace caffe
{
template<typename Dtype>
class CustomSoftMaxLossLayer:public LossLayer<Dtype>
{
public:
    explicit CustomSoftMaxLossLayer(LayerParameter param):LossLayer<Dtype>(param),scale_(Dtype(1.))
    {}
    virtual ~CustomSoftMaxLossLayer()
    {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type()const{return "CustomSoftMaxLoss";}
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumtopBlobs()const {return 1;}
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
private:
    boost::shared_ptr<Layer<Dtype>> softmaxlayer_;
    vector<Blob<Dtype>*> bottom_;
    vector<Blob<Dtype>*> top_;
    Blob<Dtype> pro_;
    int stage_;
    Dtype scale_;
    vector<Dtype> bate_;
};

}
#endif