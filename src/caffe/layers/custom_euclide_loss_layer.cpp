#include "caffe/layers/custom_euclide_loss_layer.hpp"
#include "caffe/util/io.hpp"
namespace caffe
{
template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom,top);
    CHECK(this->layer_param_.has_customeuclideloss_param())<<"customeuclideloss not define.";
    CHECK(this->layer_param_.customeuclideloss_param().has_stage())<<"customeuclideloss stage not define.";
    this->stage_=this->layer_param_.customeuclideloss_param().stage();
    this->loss_param_=Dtype(1.0);
    if(this->layer_param_.customeuclideloss_param().has_loss_scale())
    {
        this->loss_param_=this->layer_param_.customeuclideloss_param().loss_scale();
    }
}

template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::Reshape(bottom,top);
    this->dis_.ReshapeLike(*bottom[0]);
    CHECK(bottom.size()==2);
    CHECK(bottom[0]->count()==bottom[1]->count());
}

template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    int batch_size=bottom[1]->shape(0);
    int dim=bottom[1]->count(1);
    Dtype loss=Dtype(0.0);
    this->alpha.clear();
    this->alpha.resize(batch_size);
    const Dtype* data=bottom[0]->cpu_data();
    const Dtype* reglabel=bottom[1]->cpu_data();
    Dtype* dis_data=this->dis_.mutable_cpu_data();
    caffe_set(this->dis_.count(),(Dtype)0.,dis_data);
    for(int i=0;i<batch_size;i++)
    {
        Dtype temp=0;
        Dtype positive=Dtype(0.);
        Dtype pos_loss=Dtype(0.);
        Dtype neg_loss=Dtype(0.);
        for(int j=0;j<dim;j++)
        {
            if(reglabel[i*dim+j]>Dtype(-1.0))
            {
                positive++;
                temp=data[i*dim+j]-reglabel[i*dim+j];
                dis_data[i*dim+j]=temp;
                pos_loss+=temp*temp;
            }
            else
            {
                temp=data[i*dim+j];
                dis_data[i*dim+j]=temp;
                neg_loss+=temp*temp;
            }
        }
        this->alpha[i]=positive/Dtype(dim);
        loss+=this->alpha[i]*neg_loss+(1-this->alpha[i])*pos_loss;
    }
    loss*=this->loss_param_;
    loss/=bottom[1]->shape(0);
    top[0]->mutable_cpu_data()[0]=loss;
}

template<typename Dtype>
void CustomEuclideLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[1])
    {
        LOG(FATAL) << this->type()<< " Layer cannot backpropagate to label inputs.";
    }
    if(propagate_down[0])
    {
        const Dtype* label=bottom[1]->cpu_data();
        Dtype* data=this->dis_.mutable_cpu_data();
        for(int i=0;i<bottom[1]->count();i++)
        {
            int bs=i/bottom[1]->count(1);
            if(label[i]>Dtype(-1.))data[i]*=(1-this->alpha[bs]);
            else
            data[i]*=this->alpha[bs];
        }
        caffe_cpu_axpby(bottom[0]->count(),2*this->loss_param_/Dtype(bottom[1]->shape(0)),
                        this->dis_.mutable_cpu_data(),Dtype(0.),bottom[0]->mutable_cpu_diff());
    }
}
INSTANTIATE_CLASS(CustomEuclideLossLayer);
REGISTER_LAYER_CLASS(CustomEuclideLoss);

#ifdef CPU_ONLY
STUB_GPU(CustomEuclideLossLayer)
#endif

}//namespce