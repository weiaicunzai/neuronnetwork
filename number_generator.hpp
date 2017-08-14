/***


a singleton to generate random numbers

****/

#include <boost/random.hpp>
#include <ctime>

using namespace std;
using namespace boost;

class num_gen
{
  public:
    //static mt19937_64 rng;
    //static mt19937_64 rng;
    //static normal_distribution <> nd();
    // static var_gen_t gen(rng, nd);
    //static var_gen_t gen(mt19937_64(), normal_distribution<>());

    static num_gen &get_instance()
    {
        static num_gen instance;
        return instance;
    }
    double randn()
    {
        normal_distribution<> nd(0, 1);
        mt19937_64 engine(time(0));
        static variate_generator<mt19937_64, normal_distribution<> > myrandn(engine, nd);
        return myrandn();
    }

  private:
    num_gen(){};
    num_gen(const num_gen &);
    void operator=(const num_gen &);
};
