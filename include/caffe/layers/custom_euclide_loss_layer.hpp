#ifndef CAFFE_CUSTOM_EUCLIDE_LOSS_LAYER_HPP_
#define CAFFE_CUSTOM_EUCLIDE_LOSS_LAYER_HPP_

#include "caffe/layers/loss_layer.hpp"
namespace caffe
{
template<typename Dtype>
class CustomEuclideLossLayer:public LossLayer<Dtype>
{
public:
    explicit CustomEuclideLossLayer(LayerParameter param):LossLayer<Dtype>(param),loss_param_(0)
    {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type()const{return "CustomEuclideLoss";}
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
    Dtype loss_param_;
    int stage_;
    Blob<Dtype> dis_;
    vector<Dtype> alpha;
    
};

}//namespce
#endif