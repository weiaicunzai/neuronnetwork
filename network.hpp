
/*
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

*/

#ifndef __NET__
#define __NET__

#include <boost/random.hpp>
//#include <boost/container/detail/singleton.hpp>
#include "mnist_loader.hpp"
#include "number_generator.hpp"
#include <iostream>
#include <algorithm>

typedef vector<multi_array<double, 2> > weight;
typedef vector<multi_array<double, 1> > biase;
typedef vector<img_label_pair> mini_batch_container;
typedef vector<mini_batch_container> mini_batches_container;
//typedef singleton_default<mt19937_64, normal_distribution<> > var_gen_t;
using namespace boost;

class net
{
public:
  net(vector<int> sizes);

  //not suppose be here
  inline double sigmoid(double z)
  {
    return 1.0 / (1.0 + (double)exp(-z));
  }

  /*Train the neural network using mini-batch stochastic
gradient descent. 
The "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired
outputs.  The other non-optional parameters are
self-explanatory.  If "test_data" is provided then the
network will be evaluated against the test data after each
epoch, and partial progress printed out.  This is useful for
tracking progress, but slows things down substantially.*/
  void stochastic_gradient_descent(boost::shared_ptr<training_data_container> training_data,
                                   const int &epochs,
                                   const int &mini_batch_size,
                                   const double &eta,
                                   boost::shared_ptr<test_data_container> &test_data);

  /*Update the network's weights and biases by applying
 gradient descent using backpropagation to a single mini batch.
 The "mini_batch" is a vector of tuples "(x, y)", and "eta"
 is the learning rate.*/

  void update_mini_batch(mini_batch_container &mini_batch, const double &eta);
  void back_prop(const array784d &img, const array10i &label);

private:
  int num_layer;
  vector<int> sizes;
  boost::shared_ptr<weight> weights;
  boost::shared_ptr<biase> biases;
  boost::shared_ptr<weight> init_weights(const vector<int> &sizes);
  boost::shared_ptr<biase> init_biases(const vector<int> &sizes);

  void generate_biases_of_single_layer(multi_array<double, 1> &single_b_block, const int &neuron_num);
  void generate_biases_of_all_layers(boost::shared_ptr<biase> initialized_biases, const vector<int> &sizes);
  void generate_weights_of_single_layer(multi_array<double, 2> &single_w_block, const long unsigned int *block_shape);
  void generate_weights_of_all_layers(boost::shared_ptr<weight> initialized_weights, const vector<int> &sizes);

  // see if the first img is empty (all pixel value equals to 0)
  bool is_empty(boost::shared_ptr<test_data_container> test_data);
  boost::shared_ptr<mini_batches_container> generate_mini_batches(const boost::shared_ptr<training_data_container> training_data, const int &mini_batch_size);
  boost::shared_ptr<mini_batch_container> generate_mini_batch(const boost::shared_ptr<training_data_container> training_data,
                                                       const int &batch_index,
                                                       const int &mini_batch_size);

  boost::shared_ptr<biase> init_nabla_b();
  boost::shared_ptr<weight> init_nabla_w();

  boost::shared_ptr<vector<int> > get_z(const multi_array<double, 2> &w,
                                const vector<double> &activation,
                                const multi_array<double, 1> &b);
};

#endif