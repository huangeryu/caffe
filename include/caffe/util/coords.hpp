#ifndef CAFFE_UTIL_COORDS_H_
#define CAFFE_UTIL_COORDS_H_

#include <algorithm>
#include <utility>
#include <vector>

namespace caffe {

#define DISPLY(fn,param) \
LOG(INFO)<<#fn<<"("<<param[0].first<<","<<param[0].second<<")("<<param[1].first<<","<<param[1].second<<")"
template <typename Dtype>
class DiagonalAffineMap {
 public:
  explicit DiagonalAffineMap(const vector<pair<Dtype, Dtype> > coefs)
    : coefs_(coefs) { }
  static DiagonalAffineMap identity(const int nd) {
    return DiagonalAffineMap(vector<pair<Dtype, Dtype> >(nd, make_pair(1, 0)));
  }

  inline DiagonalAffineMap compose(const DiagonalAffineMap& other) const {
    CHECK_EQ(coefs_.size(), other.coefs_.size())
        << "Attempt to compose DiagonalAffineMaps of different dimensions";
    DiagonalAffineMap<Dtype> out;
//    DISPLY(compose,coefs_);
//    DISPLY(compose,other.coefs_);
    transform(coefs_.begin(), coefs_.end(), other.coefs_.begin(),
        std::back_inserter(out.coefs_), &compose_coefs);
//    DISPLY(compose,out.coefs_);
    return out;
  }
  inline DiagonalAffineMap inv() const {
    DiagonalAffineMap<Dtype> out;
//    DISPLY(inv,coefs_);
    transform(coefs_.begin(), coefs_.end(), std::back_inserter(out.coefs_),
        &inv_coefs);
//    DISPLY(inv,out.coefs_);
    return out;
  }
  inline vector<pair<Dtype, Dtype> > coefs() { return coefs_; }

 private:
  DiagonalAffineMap() { }
  static inline pair<Dtype, Dtype> compose_coefs(pair<Dtype, Dtype> left,
      pair<Dtype, Dtype> right) 
  {
  //  LOG(INFO)<<"compose_coefs "<<"make_pair("<<left.first<<"*"<<right.first<<","<<left.first<<"*"<<right.second<<"+"<<left.second<<")";
    return make_pair(left.first * right.first,
                     left.first * right.second + left.second);
  }
  static inline pair<Dtype, Dtype> inv_coefs(pair<Dtype, Dtype> coefs) 
  {
  //  LOG(INFO)<<"inv_coefs "<<"make_pair("<<"1/"<<coefs.first<<",-"<<coefs.second<<"/"<<coefs.first<<")";
    return make_pair(1 / coefs.first, - coefs.second / coefs.first);
  }
  vector<pair<Dtype, Dtype> > coefs_;
};

template <typename Dtype>
DiagonalAffineMap<Dtype> FilterMap(const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w) {
  vector<pair<Dtype, Dtype> > coefs;
//  LOG(INFO)<<"FilterMap "<<"(kernel,stride,pad)=["<<kernel_h<<"*"<<kernel_w<<","<<stride_h<<"*"<<stride_w<<","<<pad_h<<"*"<<pad_w<<"];";
  coefs.push_back(make_pair(stride_h,
        static_cast<Dtype>(kernel_h - 1) / 2 - pad_h));
  coefs.push_back(make_pair(stride_w,
        static_cast<Dtype>(kernel_w - 1) / 2 - pad_w));
//  DISPLY(FilterMap,coefs);
  return DiagonalAffineMap<Dtype>(coefs);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_COORDS_H_
