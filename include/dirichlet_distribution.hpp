#pragma once

#include <random>
#include <limits>
#include <memory>

namespace stdpp{

  template <class RealType = double>
  struct dirichlet_distribution
  {
    typedef std::vector<RealType> result_type;
    typedef std::vector<RealType> param_type;

    dirichlet_distribution(): alphas_(nullptr) {}
    dirichlet_distribution(const param_type& alphas)
      : alphas_(std::unique_ptr<param_type>(new param_type(alphas))) {} // c++11 not support make_unique

    template <class RandomEngine>
    inline result_type operator()(RandomEngine& gen) const
    {
      assert( alphas_ );
      return this->operator()(gen, *alphas_);
    }
    
    template <class RandomEngine>
    result_type operator()(RandomEngine& gen, const param_type& alphas) const;

    inline void reset() {}

    inline const param_type& param() const { return *alphas_; }

    inline void param(const param_type& param)
    { alphas_ = std::unique_ptr<param_type>(new param_type(param)); } // c++11 not support make_unique

    inline const param_type& alphas() const { return *alphas_; }


  private:
    std::unique_ptr<param_type> alphas_;
  }; // end of dirichlet_distribution


  // ---------------------------------------------------------

  template <class RealType>
  template <class RandomEngine>
  std::vector<RealType>
  dirichlet_distribution<RealType>::operator()
    (RandomEngine& gen, const param_type& alphas) const
  {
    result_type result(alphas.size());

    std::gamma_distribution<RealType> gamma;

    double y = 0;
    for(std::size_t i = 0, I = alphas.size(); i < I; ++i){
      assert( alphas[i] > 0 );      
      auto x = gamma(gen, typename std::gamma_distribution<RealType>::param_type(alphas[i], 1.0));
      result[i] = x;
      y += x;
    }

    for(auto& elem : result){
      elem /= y;
      if( elem <= 0){
	elem = std::numeric_limits<RealType>::epsilon();
      }else if(elem >= 1){
	elem = 1 - std::numeric_limits<RealType>::epsilon();
      }
      assert( 0 < elem && elem < 1);
    }
    
    return result;
  }

  

} // end of namespace stdpp


