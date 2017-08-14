
/*****





this file stores label file properties


**********/

class label_file
{
  public:
    int magic_number;
    int label_number;
    unsigned char item;
    static label_file &get_instance()
    {
        static label_file instance;
        return instance;
    }

  private:
    label_file(){};                     // the brackets are needed here
    label_file(const label_file &);     // dont implement
    void operator=(const label_file &); //　dont implement
};