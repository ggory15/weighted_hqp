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
            Initset(const std::vector<Eigen::VectorXi> & btype);
            Initset(const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            ~Initset(){};
        
        private: 
            void calc_bounds();

        public:
            std::vector<Eigen::VectorXi> getactiveset(){
                return aset_;
            };
            std::vector<Eigen::VectorXi> getbounds(){
                return bounds_;
            };

        private:
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;
            std::vector<Eigen::VectorXi> aset_;
            std::vector<Eigen::VectorXi> bounds_;     

            long unsigned int size_;    
    };
}

#endif