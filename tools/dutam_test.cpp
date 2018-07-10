#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/util/db.hpp"
#include "caffe/util/db_lmdb.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/caffe.hpp"
#include<iostream>
#include<memory>
using namespace caffe;
using namespace caffe::db;
using namespace std;
typedef unsigned char uchar;
class images
{
public:
    explicit images(const string& pairfile):current_(0)
    {
        ifstream infile(pairfile.c_str());
        string img,label;
        while(infile>>img>>label)
        {
            this->lines.push_back(make_pair(img,label));
        }
    }
    bool compairTo(const Datum& datum)
    {
        if(current_<this->lines.size())
        {
            auto line=this->lines[current_];
            cv::Mat img=cv::imread(line.first,cv::IMREAD_COLOR);
            cv::Mat label=cv::imread(line.second,-1);
            CHECK_EQ(img.channels(),datum.channels())<<line.first;
            CHECK_EQ(img.rows,datum.height());
            CHECK_EQ(img.cols,datum.width());
            CHECK_EQ(label.channels(),1);
            CHECK_EQ(img.rows,label.rows);
            CHECK_EQ(img.cols,label.cols);
            const string& datum_data=datum.data();
            const string& datum_label_data=datum.label_data();
            this->current_++;
            for(int i=0;i<img.rows;i++)
            {
                auto ptr_img=img.ptr<uchar>(i);
                auto ptr_label=label.ptr<uchar>(i);
                int index=0;
                for(int j=0;j<img.cols;j++)
                {
                    for(int k=0;k<img.channels();k++)
                    {
                        int datum_index=k*(img.rows*img.cols)+i*img.cols+j;
                        if(ptr_img[index++]!=static_cast<uchar>(datum_data[datum_index]))
                        {
                            std::cout<<line.first<<" convert to lmdb errror."<<std::endl;
                            return false;
                        }
                    }
                    int datum_label_index=i*img.cols+j;
                    if(ptr_label[j]!=static_cast<uchar>(datum_label_data[datum_label_index]))
                    {
                        std::cout<<line.second<<" converto lmdb error."<<endl;
                        return false;
                    }
                }
            }
            return true;
        }
        else
        {
            LOG(INFO)<<"\ncurrent out of bound.";
            return false; 
        }
    }

public:
    vector<pair<string,string>> lines;
private:
    int current_;
};
int main(int argc,char* argv[])
{
    Datum datum;
    std::shared_ptr<DB> db_=std::make_shared<LMDB>();
    db_->Open(argv[1],db::READ);
    std::shared_ptr<Cursor> cursor_;
    cursor_.reset(db_->NewCursor());
    images images(argv[2]);
    int size=images.lines.size();
    int correct=0;
    int error=0;
    while(cursor_->valid())
    {
        datum.ParseFromString(cursor_->value());
        if(images.compairTo(datum))
        {
            correct++;
        }
        else
        {
            error++;
        }
        std::cout<<"[corrent error]/total :["<<correct<<" "<<error<<"]/"<<size<<endl;
        cursor_->Next();
    }
}