#ifndef __solver_Initset_h__
#define __solver_Initset_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    enum Constraints {
        Enone,
        Etwin,
        Edouble,
        Einf,
        Esup
    };

    class Initset{
        public:
            Initset(){
               
            };
            Initset(const std::vector<Eigen::VectorXi> & btype);
            Initset(const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            ~Initset(){};
        
        private: 
            void calc_bounds();

        public:
            std::vector<Eigen::VectorXi> & getactiveset(){
                return aset_;
            };
            std::vector<Eigen::VectorXi> & getbounds(){
                return bounds_;
            };
            const std::vector<Eigen::VectorXi> & getactiveset() const {
                return aset_;
            };
            const std::vector<Eigen::VectorXi> & getbounds() const {
                return bounds_;
            };

            void set_btype(const std::vector<Eigen::VectorXi> & btype){     
                btype_= btype;
                size_ = btype.size();
                aset_init_.clear();
                aset_bound_.clear();
                
                for (long unsigned int i = 0; i< size_; i++){
                    aset_init_.push_back(Eigen::VectorXi(0));
                    aset_bound_.push_back(Eigen::VectorXi(0));
                }
                aset_.clear();
                bounds_.clear();
                this -> calc_bounds();

                
            }

        private:
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;
            std::vector<Eigen::VectorXi> aset_;
            std::vector<Eigen::VectorXi> bounds_;     

            long unsigned int size_;    
    };
}

#endif