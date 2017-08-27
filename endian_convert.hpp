/***

endian_convert can detect the endian of the os

*****/
#include <iostream>
#include <cassert>

using namespace std;

class endian_convert
{
  public:
    static inline bool is_big_endian()
    {
        unsigned short i = 0x0001;
        char *c = (char *)&i;
        return *c == 0;
    }

    //
    //swap every single byte
    //

    static inline void byte_swap_int32( int &number)
    {
        assert(sizeof(number) == 4);
        number = (number >> 24) | ((number & 0x0000ff00) << 8) | ((number & 0x00ff0000) >> 8) | (number << 24);
    }
private:
    endian_convert(){};
    endian_convert(const endian_convert &);
    void operator = (const endian_convert &);
};

//int main()
//{
//   // cout << endian_convert::is_big_endian() << endl;
//   int i = 0x12345678;
//   char *c = (char*) &i;
//   cout << hex << (int)(c[3]) << endl;
//   cout << hex << (i & 0x0ff00000) << endl;
//   cout << endian_convert::is_big_endian() << endl;
//   endian_convert::byte_swap_int32(i);
//   cout << hex << i << endl;
//}