/*
*
*
*  this file loads pixel data and label data from mnist
*
*
*
*/
#ifndef __PARSER__
#define __PARSER__

#include <boost/multi_array.hpp>
#include <boost/filesystem/fstream.hpp>
#include <byteswap.h>

using namespace boost;
using namespace std;
using namespace boost::filesystem;

namespace boostifstream = boost::filesystem;

// load image file only need to read the input stream  no logic behavior
class parser
{
public:
  // int &number    not int number
  inline static void read_int(int &number, boostifstream::ifstream &ifs)
  {
    ifs.read((char *)&number, sizeof(int));
  }

  inline static void read_char(unsigned char &number, boostifstream::ifstream &ifs)
  {
    ifs.read((char *)&number, sizeof(char));
  }

  inline static bool is_equal(const int &extracted_number, const int &correct_number)
  {
    return extracted_number == correct_number;
  }

  inline static void convert_to_big_endian(int &number)
  {
    number = bswap_32(number);
  }
};

#endif
