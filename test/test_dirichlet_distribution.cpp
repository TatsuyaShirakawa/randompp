/* -*- coding:utf-8 -*- */
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <gtest/gtest.h>

#include "randompp.hpp"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(DirichletDoubleTest, PredMeanSigma_1){
  typedef double T;

  std::vector<T> alphas;
  alphas.push_back(0.2);
  alphas.push_back(0.3);
  alphas.push_back(0.1);
  alphas.push_back(0.15);

  const size_t N = 1000000;

  std::mt19937 gen(0);
  stdpp::dirichlet_distribution<T> dirichlet;
  std::vector<double> s1(alphas.size(),0), s2(alphas.size(),0);
  for(std::size_t i=0;i<N;++i){
    std::vector<T> x(dirichlet(gen, alphas));
    for(std::size_t k=0;k<alphas.size();++k){
      s1[k] += x[k];
      s2[k] += x[k]*x[k];
    }
  }
  for(std::size_t k=0;k<alphas.size();++k){
    s1[k] /= N;
    s2[k] /= N; s2[k] -= s1[k]*s1[k];
  }
  
  double init=0;  
  T A = std::accumulate(alphas.begin(), alphas.end(), init);  
  
  std::vector<T> means(alphas.size()), sigma2s(alphas.size());
  for(std::size_t k=0;k<alphas.size();++k){
    means[k] = alphas[k]/A;
    sigma2s[k] = alphas[k]*(A-alphas[k])/(A*A*(A+1));
  }
  T err_mean = 0;
  T err_sigma = 0;
  for(std::size_t k=0;k<alphas.size();++k){
    err_mean += std::abs(means[k]-s1[k]);
    err_sigma += std::abs(std::sqrt(sigma2s[k])-std::sqrt(s2[k]));
  }
  EXPECT_LT(err_mean, 1e-2);
  EXPECT_LT(err_sigma, 1e-2);   
}

TEST(DirichletDoubleTest, PredMeanSigma_2){
  typedef double T;

  std::vector<T> alphas;
  alphas.push_back(1.7);
  alphas.push_back(0.3);
  alphas.push_back(1.1);
  alphas.push_back(2.4);

  const size_t N = 1000000;

  std::mt19937 gen(0);
  stdpp::dirichlet_distribution<T> dirichlet;

  std::vector<double> s1(alphas.size(),0), s2(alphas.size(),0);
  for(std::size_t i=0;i<N;++i){
    std::vector<T> x(dirichlet(gen, alphas));
    for(std::size_t k=0;k<alphas.size();++k){
      s1[k] += x[k];
      s2[k] += x[k]*x[k];
    }
  }
  for(std::size_t k=0;k<alphas.size();++k){
    s1[k] /= N;
    s2[k] /= N; s2[k] -= s1[k]*s1[k];
  }
  
  double init=0;  
  T A = std::accumulate(alphas.begin(), alphas.end(), init);  
  
  std::vector<T> means(alphas.size()), sigma2s(alphas.size());
  for(std::size_t k=0;k<alphas.size();++k){
    means[k] = alphas[k]/A;
    sigma2s[k] = alphas[k]*(A-alphas[k])/(A*A*(A+1));
  }
  T err_mean = 0;
  T err_sigma = 0;
  for(std::size_t k=0;k<alphas.size();++k){
    err_mean += std::abs(means[k]-s1[k]);
    err_sigma += std::abs(std::sqrt(sigma2s[k])-std::sqrt(s2[k]));
  }
  EXPECT_LT(err_mean, 1e-2);
  EXPECT_LT(err_sigma, 1e-2);   
}


int main(int argc, char** argv)
{
  InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}

