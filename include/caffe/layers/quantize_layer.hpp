#ifndef CAFFE_QUANTIZE_LAYER_HPP_
#define CAFFE_QUANTIZE_LAYER_HPP_
#include "caffe/layer.hpp"
namespace caffe
{
template<typename Dtype>
class QuantizeLayer:public Layer<Dtype>
{
public:
    explicit QuantizeLayer(LayerParameter param):Layer<Dtype>(param)
    {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

    virtual inline const char* type()const{return "Quantize";}
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
private:
    vector<int> rfs_;
    Dtype scale_;
};

}
#endif