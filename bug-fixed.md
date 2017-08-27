#v0.1
##1
add  endian_convert.hpp  to convert little endian to big endian, replace
the  byteswap.h file in the first version
##2
delete "using namespace boost" and "using namespace std", add namespace 
referrence in front of the class we use, to avoid "reference to 'xxx' is
ambiguous" error.

## work to do next
replace  boost::array and boost::multi_array with boost::uBLA::Matrix

