#include "weighted_hqp/InitSet.hpp"
#include <assert.h>

using namespace std;

namespace hcod{
    Initset::Initset(const std::vector<Eigen::VectorXi> &btype)
    : btype_(btype)
    {
        size_ = btype.size();
        aset_init_.clear();
        bounds_.clear();
        for (long unsigned int i = 0; i< size_; i++){
            aset_init_.push_back(Eigen::VectorXi(0));
            aset_bound_.push_back(Eigen::VectorXi(0));
        }
        aset_.clear();
        bounds_.clear();
        this -> calc_bounds();
    }
    Initset::Initset(const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
        aset_.clear();
        bounds_.clear();
        this -> calc_bounds();
        assert (false && "This is not supported initial guess");
    }
    
    void Initset::calc_bounds(){
        for (long unsigned int i=0; i<size_; i++){
            std::vector<int> atwin;
            for (int j=0; j<btype_[i].size(); j++){
                if (btype_[i](j) == Constraints::Etwin) 
                    atwin.push_back(j); // check 
            }
            int l = atwin.size();
            Eigen::VectorXi atwin_vec(l), btwin_vec(l);

            for (int j=0; j<l; j++)
                atwin_vec(j) = atwin[j];
            btwin_vec.setOnes();

            aset_.push_back(atwin_vec);
            bounds_.push_back(btwin_vec);
        }
    }
    
    

}