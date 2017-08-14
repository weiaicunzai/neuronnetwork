/*****





this file stores image file properties


**********/

class image_file
{
public:
  int magic_number;
  int cols;
  int rows;
  int image_number;
  unsigned char pixel;
  static image_file &get_instance()
  {
    static image_file instance;
    return instance;
  }
  // image_file( const image_file &) = delete;
private:
  image_file(){};
  image_file(const image_file &);
  void operator=(const image_file &);
};