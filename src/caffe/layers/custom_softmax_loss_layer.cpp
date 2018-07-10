#include<cfloat>
#include "caffe/layers/custom_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe
{
template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::LayerSetUp(bottom,top);
    LayerParameter softmax_param(this->layer_param_);
    softmax_param.set_type("Softmax");
    this->softmaxlayer_=LayerRegistry<Dtype>::CreateLayer(softmax_param);
    this->bottom_.clear();
    this->bottom_.push_back(bottom[0]);
    this->top_.clear();
    this->top_.push_back(&this->pro_);
    this->softmaxlayer_->SetUp(this->bottom_,this->top_);
    CHECK(this->layer_param_.has_customsoftmaxloss_param())<<"customsoftmaxloss_param not define.";
    CHECK(this->layer_param_.customsoftmaxloss_param().has_stage())<<"customsoftmaxloss_param stage not define.";
    this->stage_=this->layer_param_.customsoftmaxloss_param().stage();
    if(this->layer_param_.customsoftmaxloss_param().has_loss_scale())
    {
        this->scale_=Dtype(this->layer_param_.customsoftmaxloss_param().loss_scale());
    }
}

template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    LossLayer<Dtype>::Reshape(bottom,top);
    CHECK(bottom[0]->shape(1)==this->stage_);
    CHECK(bottom[1]->shape(1)==1);
    CHECK(bottom[0]->count(2)==bottom[1]->count(2));
}

template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
    Dtype loss=Dtype(0.0);
    int batch_size=bottom[0]->shape(0);
	this->bate_.clear();
	this->bate_.resize(2*batch_size);
    int area=bottom[1]->count(2);
    this->softmaxlayer_->Forward(this->bottom_,this->top_);
    const Dtype* label=bottom[1]->cpu_data();
    const Dtype* data=this->pro_.cpu_data();
    int dim=bottom[0]->count(1);
    for(int i=0;i<batch_size;i++)
    {
        Dtype positive_loss=Dtype(0.);
        Dtype negative_loss=Dtype(0.);
        Dtype positive=Dtype(0.);
        Dtype negative=Dtype(0.);
        for(int j=0;j<area;j++)
        {
            if(label[i*area+j]>this->stage_-1 || label[i*area+j]<Dtype(0.001))
            {
                negative++;
                if(isnan(data[i*dim+j]))LOG(FATAL)<<"data["<<i*dim+j<<"]="<<data[i*dim+j];
                negative_loss+=log(std::max(data[i*dim+j],Dtype(FLT_MIN)));
            }
            else
            {
                int k=int(label[i*area+j]);
                positive++;
                if(isnan(data[i*dim+k*area+j]))LOG(FATAL)<<"data["<<i*dim+k*area+j<<"]="<<data[i*dim+k*area+j];
                positive_loss+=log(std::max(data[i*dim+k*area+j],Dtype(FLT_MIN)));
            }
        }
        bate_[i*2]=negative/Dtype(area);
        bate_[i*2+1]=positive/Dtype(area);
        loss+=bate_[2*i]*positive_loss+bate_[i*2+1]*negative_loss;
    }
    top[0]->mutable_cpu_data()[0]=-loss*this->scale_/batch_size;
}

template<typename Dtype>
void CustomSoftMaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,const vector<bool>& propagate_down,
                                const vector<Blob<Dtype>*>& bottom)
{
    if(propagate_down[1])
    {
        LOG(FATAL)<<this->type()<<" Layer cannot backpropagate to label inputs.";
    }
    if(propagate_down[0])
    {
        int batch_size=bottom[0]->shape(0);
        int channal=bottom[0]->shape(1);
        int dim=bottom[0]->count(2);
        int batch_count=bottom[0]->count(1);
        Dtype* diff=bottom[0]->mutable_cpu_diff();
        const Dtype* qlabel=bottom[1]->cpu_data();
        const Dtype* p=this->pro_.cpu_data();
        caffe_set(bottom[0]->count(),Dtype(0.),diff);
        for(int i=0;i<batch_size;i++)
        {
            for(int c=0;c<channal;c++)
            {
                for(int j=0;j<dim;j++)
                {
                    int k=int(qlabel[i*dim+j]);
                    int alpha=(k==c)?1:0;
                    int k_=(c==0);
                    int index=i*batch_count+c*dim+j;
                    Dtype temp=this->bate_[i*2+k_]*alpha+this->bate_[i*2+k_]*(-p[index]);
                    if(isnan(temp))
                    {
                        for(int m=0;m<this->bate_.size();m++)
                        {
                            LOG(INFO)<<"bate["<<m<<"]="<<this->bate_[m];
                        }
                        LOG(INFO)<<"[c k alpha]="<<c<<" "<<k<<" "<<alpha;
                        LOG(INFO)<<"p["<<index<<"]="<<p[index];
                        LOG(FATAL)<<"temp is nan";
                    }
                    diff[index]=-temp*this->scale_/Dtype(batch_size);
                }
            }
        }
    }
}
INSTANTIATE_CLASS(CustomSoftMaxLossLayer);
REGISTER_LAYER_CLASS(CustomSoftMaxLoss);
#ifdef CPU_ONLY
STUB_GPU(CustomSoftMaxLossLayer)
#endif
}//namespce
