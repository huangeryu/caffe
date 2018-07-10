#ifndef CAFFE_ACCURACY_EUCLIDE_LAYER_HPP_
#define CAFFE_ACCURACY_EUCLIDE_LAYER_HPP_
#include "caffe/layer.hpp"
#include <vector>
namespace caffe
{
template <typename Dtype>
class AccuracyEuclideLayer:public Layer<Dtype>
{
public:
    explicit AccuracyEuclideLayer(LayerParameter param):Layer<Dtype>(param)
    {}
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    inline virtual const char* type()const{return "AccuracyEuclide";}
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom);
};

} // caffe

#endif