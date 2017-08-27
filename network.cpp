#include "network.hpp"




net::net(vector<int> sizes)
{
   num_layer = sizes.size(); 
   this->biases = init_biases(sizes); 
   this->weights = init_weights(sizes);
   this->sizes = sizes;
}

boost::shared_ptr<biase> net::init_biases(const vector<int> &sizes)
{
    boost::shared_ptr<biase> initialized_biases(new biase);
    generate_biases_of_all_layers(initialized_biases, sizes);
    return initialized_biases;
}

void net::generate_biases_of_all_layers(boost::shared_ptr<biase> initialized_biases, const vector<int> &sizes)
{
    for (vector<int>::const_iterator layer_index = sizes.begin() + 1; layer_index != sizes.end(); layer_index++)
    {
        array<size_t, 2> single_biase_block_sequence = {*layer_index};
        multi_array<double, 1> single_biase(single_biase_block_sequence);
        generate_biases_of_single_layer(single_biase, *layer_index);
        initialized_biases->push_back(single_biase); 
    } 
}

void net::generate_biases_of_single_layer(multi_array<double, 1> &single_b_block, const int &neuron_num)
{
    for (int biases_index = 0; biases_index < neuron_num; biases_index++)
        single_b_block[biases_index] = num_gen::get_instance().randn();
}

//
// for a 3 neurons  layer l-1 connect to  a  4 neurons  layer l,
// we store the biases in a 3 * 4  2-dim array with 3 rows and
// 4 cols
//

void net::generate_weights_of_single_layer(multi_array<double, 2> &single_w_block, const long unsigned int *block_shape)
{
    const int row_num = block_shape[0];
    const int col_num = block_shape[1];
    for (int row_index = 0; row_index < row_num; row_index++)
        for (int col_index = 0; col_index < col_num; col_index++)
            single_w_block[row_index][col_index] = num_gen::get_instance().randn();
}

void net::generate_weights_of_all_layers(boost::shared_ptr<weight> initialized_weights, const vector<int> &sizes)
{
    for (vector<int>::const_iterator iter = sizes.begin(); iter != sizes.end() - 1; iter++)
    {
        //array<size_t, 2> single_weight_block_sequence = {*iter, *(iter + 1)};
        array<size_t, 2> single_weight_block_sequence = {*(iter + 1), *iter};
        multi_array<double, 2> single_weight_block(single_weight_block_sequence);
        generate_weights_of_single_layer(single_weight_block, single_weight_block.shape());
        initialized_weights->push_back(single_weight_block);
    }
}

boost::shared_ptr<weight> net::init_weights(const vector<int> &sizes)
{
    boost::shared_ptr<weight> initialized_weights(new weight);
    generate_weights_of_all_layers(initialized_weights, sizes);
    return initialized_weights;
}

//replacing array with vector!
// we need assign the dim at run time, not compile time!
// array is really some stupid shiiit!
void net::stochastic_gradient_descent(boost::shared_ptr<training_data_container> training_data,
                                      const int &epochs,
                                      const int &mini_batch_size,
                                      const double &eta,
                                      boost::shared_ptr<test_data_container> &test_data)
{

    int test_num = 0;
    if(!is_empty(test_data)) test_num = (*test_data).size();
    for (int i = 0; i < epochs; i++)
    {
        random_shuffle((*training_data).begin(), (*training_data).end());
        mini_batches_container mini_batches = *generate_mini_batches(training_data, mini_batch_size);
        for (mini_batches_container::iterator mini_batch = mini_batches.begin(); mini_batch != mini_batches.end(); mini_batch++)
            update_mini_batch(*mini_batch, eta);
    }
    
}

// mini_batch.size() = 10
void net::update_mini_batch(mini_batch_container &mini_batch, const double &eta)
{
    biase nabla_b = *init_nabla_b();
    weight nabla_w = *init_nabla_w();
    for (mini_batch_container::iterator img_label_pair_index = mini_batch.begin();
         img_label_pair_index != mini_batch.end();
         img_label_pair_index++)
    {
        back_prop((*img_label_pair_index).get<0>(), (*img_label_pair_index).get<1>());

    }
}

// convert array784d to vector
void net::back_prop(const array784d &img, const array10i &label)
{
    biase nabla_b = *init_nabla_b();
    weight nabla_w = *init_nabla_w();
    vector<double> activation(img.begin(), img.end());
    vector<vector<double> > activations(1, activation);
    assert(nabla_b.size() == nabla_w.size());
    const int layer_num = nabla_b.size();
    for (int layer_index = 0; layer_index < layer_num; layer_index++)
    {
        multi_array<double, 1> b = this->biases->at(layer_index);
        multi_array<double, 2> w = this->weights->at(layer_index);
         //w * a + b
        get_z(w, activation, b);
    }
    
}

boost::shared_ptr<vector<int> > net::get_z(const multi_array<double, 2> &w, const vector<double> &activation, const multi_array<double, 1> &b)
{
    cout << w.shape()[1] << "---------------" << endl;
    cout << activation.size() << "+++++++++++++++" << endl;
    assert(w.shape()[1] == activation.size());
    assert(w.shape()[0] == b.shape()[0])
    const int biases_num = b.shape()[0];
    const int weights_num = w.shape()[1];
    boost::shared_ptr<vector<int> > z;
    for (int biases_index = 0; biases_index < biases_num; biases_index ++)
    {
        for(int weight_index = 0; weight_index < weights_num; weight_index++)
        {
            z->push_back(w[biases_index][weight_index] * );
        }
    }



    boost::shared_ptr<vector<int> > a(new vector<int>);
    return a;
}

boost::shared_ptr<biase> net::init_nabla_b()
{
    boost::shared_ptr<biase> nabla_b = make_shared<biase>();
    //deep copy
    *nabla_b = *this->biases;
    for (biase::iterator single_biase_block = nabla_b->begin();
         single_biase_block != nabla_b->end();
         single_biase_block++)
    {
        int element_num = (*single_biase_block).num_elements();
        vector<double> index(element_num, 0.0);
        (*single_biase_block).assign<vector<double>::iterator>(index.begin(), index.end());
    }
    return nabla_b;
}

boost::shared_ptr<weight> net::init_nabla_w()
{
    boost::shared_ptr<weight> nabla_w = make_shared<weight>();
    //deep copy
    *nabla_w = *this->weights;
    for (weight::iterator single_weight_block = nabla_w->begin();
         single_weight_block != nabla_w->end();
         single_weight_block++)
    {
        int element_num = (*single_weight_block).num_elements();
        vector<double> index(element_num, 0.0);
        (*single_weight_block).assign<vector<double>::iterator>(index.begin(), index.end());
    }
    return nabla_w;
}

bool net::is_empty(boost::shared_ptr<test_data_container> test_data)
{
    for (int pixel_index = 0; pixel_index < ROWS_NUMBER * COLS_NUMBER; pixel_index++)
        if((*test_data)[0].get<0>()[pixel_index] > 0)
            return true;
    return false;
}

boost::shared_ptr<mini_batches_container> net::generate_mini_batches(const boost::shared_ptr<training_data_container> training_data, const int &mini_batch_size)
{
    int training_data_num = (*training_data).size();
    int mini_batch_num = training_data_num / mini_batch_size;
    boost::shared_ptr<mini_batches_container> mini_batches =
        make_shared<mini_batches_container>();
    for (int mini_batch_index = 0; mini_batch_index < mini_batch_num; mini_batch_index++)
    {
        mini_batch_container mini_batch = *generate_mini_batch(training_data, mini_batch_index, mini_batch_size);
        mini_batches->push_back(mini_batch);
    }
    return mini_batches;
}

boost::shared_ptr<mini_batch_container> net::generate_mini_batch(const boost::shared_ptr<training_data_container> training_data,
                                                          const int &batch_index,
                                                          const int &mini_batch_size)
{
    boost::shared_ptr<mini_batch_container> mini_batch = 
        make_shared<mini_batch_container>();
    for (int tuple_index = 0; tuple_index < mini_batch_size; tuple_index++)
        mini_batch->push_back(training_data->at(batch_index * mini_batch_size + tuple_index));
    return mini_batch;
}

boost::shared_ptr<tuple<boost::shared_ptr<int>, boost::shared_ptr<int> > > test()
{

    int a = 121;
    boost::shared_ptr<int> b = make_shared<int>(a);
    boost::shared_ptr<int> c = make_shared<int>(44);
    tuple<boost::shared_ptr<int> > t = make_tuple(b);
    boost::shared_ptr<tuple<boost::shared_ptr<int>, boost::shared_ptr<int> > > my_b = make_shared<tuple<boost::shared_ptr<int>, boost::shared_ptr<int> > >(
        b,
        c
    );
    return my_b;
}

int main()
{

// boost::shared_ptr<vector<int> > a11(new vector<int>);
// boost::shared_ptr<vector<int> > b(new vector<int>);
// b->push_back(3);
// b->push_back(4);
//
// a11 = b;
//
boost::shared_ptr<tuple<boost::shared_ptr<int>, boost::shared_ptr<int> > > my = test();
    cout << (*my->get<0>()) << endl;
   cout << (*my->get<1>()) << endl;
    vector<int> cc;
    cc.push_back(784);
    cc.push_back(33);
    cc.push_back(10);
    //net a(cc);
    multi_array<int, 1> sizes(extents[3]);
    sizes[0] = 784;
    sizes[1] = 33;
    sizes[2] = 10;
    mnist_loader loader("/media/baiyu/A2FEE355FEE31FF1/mnist-dataset");
    net a(cc);
    boost::shared_ptr<wrapped_data> final_data = loader.load_data_wrapper();
     

   // boost::shared_ptr<training_data_container> training_data(new training_data_container);
    //training_data = final_data->get<0>();
       // make_shared<training_data_container>(final_data->get<0>());
    //traing_data = 
    boost::shared_ptr<training_data_container> training_data = final_data->get<0>();
    boost::shared_ptr<test_data_container> test_data = final_data->get<2>();
    cout <<"ffffffffffffffffffff" << endl;
    cout << final_data->get<0>()->at(33333).get<0>()[333] << "fff" << endl;
    //cout << loader.load_data_wrapper()->get<2>()->at(10).get<0>()[111] << endl;;
    cout << training_data->size() << endl;;
    cout << test_data->size() << endl;
    //cout << training_data->size() << endl;
    //cout << final_data->get<0>()->size() << endl;
    //cout << training_data->size() << endl;
        //make_shared<training_data_container>(final_data->get<0>());
  //  training_data = &final_data->get<0>();
 //   boost::shared_ptr<test_data_container> test_data =
 //       make_shared<test_data_container>(loader.load_data_wrapper()->get<2>());
    a.stochastic_gradient_descent(training_data, 33, 10, 3.0, test_data);
    //a.stochastic_gradient_descent();
    boost::array<int, 4> myarray= {1, 2, 0, 9};
    cout << myarray.size() << endl;
    cout << myarray.max_size() << endl;
    mt19937_64 rng(time(0));
    cout << myarray[0] << myarray[1] << myarray[2] << myarray[3] << endl;
    random_shuffle(myarray.begin(), myarray.end());
    cout << myarray[0] << myarray[1] << myarray[2] << myarray[3] << endl;
    boost::random::uniform_int_distribution<> ui(0, 9);
    for (int i = 0; i < 10; i++)
    {
        cout << ui(rng) << endl;
    }
    boost::array<size_t, 2> arr = {4, 3};
    multi_array<int, 2> my_multi(arr);
    int ccc = 0;

    for (multi_array<int, 2>::iterator iter = my_multi.begin(); iter != my_multi.end(); iter++)
    {
       // cout << ccc++ << endl;
        //cout << typeid(*iter).name() << endl;
        cout << (*iter)[1] << endl;
    }
    boost::array<int, 12> m;
    m.assign(7);
    my_multi.assign<array<int, 12>::iterator>(m.begin(), m.end());
    for(int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
        cout << my_multi[i][j] << endl;
    }
    cout << my_multi.num_elements() << endl;
    
 //   boost::shared_ptr<array<int, 3> > p_a = make_shared<array<int, 3> >();
 //   p_a->at(0) = 10;
 //   p_a->at(1) = 11;
 //   p_a->at(2) = 12;

 //   cout << p_a->at(1) << endl;

 //   boost::shared_ptr<array<int, 3> >p_test = make_shared<array<int, 3> >();
 //   *p_test = *p_a;

 //   cout << p_test->at(1) << endl;

 //   p_test->at(1) = 11111;
 //   cout << p_a->at(1) << endl;
 //   cout << p_test->at(1) << endl;
    //my_multi.assign<multi_array<int, 2>::iterator>(my_multi.begin(), my_multi.end());

   // a.print_num();
    //cc.push_back(22);
    //cout << a.sigmoid(-0.8) << endl;
  //  mnist_loader loader("/media/baiyu/A2FEE355FEE31FF1/mnist-dataset");
   // cout << "aa11" << endl;
    
}