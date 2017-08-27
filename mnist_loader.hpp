/*

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.

*/

#ifndef  __MNIST__LOADER__
#define  __MNIST__LOADER__

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <string>
#include "parser.hpp"
#include "label_file.hpp"
#include "image_file.hpp"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::tuples;

typedef multi_array<int, 1> array1i;
typedef boost::shared_ptr<array1i> p_array1i;
typedef boost::tuple<p_array1i, p_array1i, p_array1i, p_array1i> origin_data;
typedef boost::array<double, 784>  array784d;
typedef boost::array<int, 10>  array10i;
typedef boost::tuple<array784d, array10i>  img_label_pair;
typedef boost::array<img_label_pair, 50000> training_data_container;
typedef boost::array<img_label_pair, 10000> validation_data_container;
typedef boost::array<img_label_pair, 10000> test_data_container;
typedef boost::tuple<boost::shared_ptr<training_data_container>,
              boost::shared_ptr<validation_data_container>,
              boost::shared_ptr<test_data_container> >
    wrapped_data;

const int IMAGE_MAGICNUMBER = 2051;
const int LABEL_MAGICNUMBER = 2049;
const int COLS_NUMBER = 28;
const int ROWS_NUMBER = 28;
const int IMAGES_NUMBER = 60000;
const int ITEMS_NUMBER = 10000;
const int TRAINING_NUMBER = 50000;
const int VALIDATION_NUMBER = 10000;

//typedef array1i single_image;

class mnist_loader
{

  private:
    path root_path;
    path train_image_path;
    path test_image_path;
    path train_label_path;
    path test_label_path;
    boostifstream::ifstream train_images_file;
    boostifstream::ifstream test_images_file;
    boostifstream::ifstream train_labels_file;
    boostifstream::ifstream test_labels_file;

  public:
    mnist_loader(string root_path);

    //********** traing image file ***********//

    //
    //when you load a new file header, the old one's 
    //parameters will be overwritten
    //

    void load_training_image_file_header();
    
    //
    //you need to load headers beafore you call this function
    //

    void check_training_image_file_header_parameters();

    //
    // you need to load traing image file header first
    // before you call this function
    //
    
    void load_single_training_image(array1i &all_image_pixels, const int &image_index);

    //
    //you need to load test image file header firset
    //before you call this function
    //

    void load_all_training_images(array1i &all_image_pixels);

    //****** test image  file********//

    //
    // same way to use as functions described above
    //
    void load_test_image_file_header();

    void check_test_image_file_header_parameters();

    void load_single_test_image(array1i &all_image_pixels, const int &image_index);

    void load_all_test_images(array1i &all_image_pixels);   

    //******** training label file  *************//

    //
    // same way to use as functions described above
    //

    void load_training_label_file_header();

    void check_training_label_file_header_parameters();

    void load_all_training_labels(array1i &all_label_items);

    void load_single_training_label(array1i &all_label_items, const int &label_index);

    //******** training label file  *************//

    //
    // same way to use as functions described above
    //

    void load_test_label_file_header();

    void check_test_label_file_header_parameters();

    void load_all_test_labels(array1i &all_label_items);

    void load_single_test_label(array1i &all_label_items, const int &label_index);

    //
    //only returns pixel or item value
    //

    boost::shared_ptr<array1i> get_training_image_data();

    boost::shared_ptr<array1i> get_test_image_data();

    boost::shared_ptr<array1i> get_training_label_data();

    boost::shared_ptr<array1i> get_test_label_data();



    //  
    //convert array1i to array784d * 60000
    //

    boost::shared_ptr<array784d> extract_single_image_from_original_data(boost::shared_ptr<array1i> original_data, const int &image_index);


    //
    //convert array1i to array10i * 60000
    //

    boost::shared_ptr<array10i> extract_single_label_from_original_data(boost::shared_ptr<array1i> original_data, const int &label_index);

    //
    // wrapper
    //

    boost::shared_ptr<training_data_container> get_wrapped_training_data(boost::shared_ptr<array1i> pixel_of_training_images, boost::shared_ptr<array1i> item_of_training_labels);

    boost::shared_ptr<validation_data_container> get_wrapped_validation_data(boost::shared_ptr<array1i> pixel_of_training_images, boost::shared_ptr<array1i> item_of_training_labels);

    boost::shared_ptr<test_data_container> get_wrapped_test_data(boost::shared_ptr<array1i> pixel_of_test_images, boost::shared_ptr<array1i> item_of_test_labels);


/*  return the original mnist data which are data extract from 
**  4 files including training_image, traing_label, test_image, 
**  test_label as a tuple contains 4 elements which is a multi_array
**  object.
**  
**  training_image is a multi_array object contains 60000 entries,
**  each entry  contains 28*28 values, and each value is 
**  between 0 - 255, so a traing_image contains 28*28*60000 values,
**  all stored in a one dimension multi_array object
**  
**  test_image is similar, except  contains only 10000 entries
**  
**  training_label is a multi_array object contains 60000 entries also,
**  each entries contains a unsigned char value between 0 - 1
**  
**  test_label is simlilar, except contains only 10000 entries
*/

    boost::shared_ptr<origin_data> load_data();


/*
//  returns a shared_prt pointer which points to a tuple contains
//  three elements  training_data, validation_data, test_data.
//  tuple(training_data, validation_data, test_data)
//
//  training_data is a one-dimension array contains 50000 entries,
//  each entry is a tuple which contains two elements(x, y), 
//  x is a one-dim array(array<float, 786>) containing the input image
//  each pixel was a float number and normallized by dividing over 255.0
//  y is a one-dim array(array<int, 10>), every elemetns equal to one 
//  except the one which representing the item number, we set it to 1
//  ex.  we have a label 7 ,then the corresbonding array is :
//  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 
//
//
//  validation_data and test_data are basiclly the same, except they only
//  have 10000 entries
*/
    boost::shared_ptr<wrapped_data> load_data_wrapper();

    inline bool is_correct_magic_number_of_image(const int &magic_number)
    {
        return parser::is_equal(magic_number, IMAGE_MAGICNUMBER);
    }

    inline bool is_correct_magic_number_of_label(const int &magic_number)
    {
        return parser::is_equal(magic_number, LABEL_MAGICNUMBER);
    }

    inline bool is_correct_training_image_number(const int &image_number)
    {
        return parser::is_equal(image_number, IMAGES_NUMBER);
    }

    inline bool is_correct_test_image_number(const int &image_number)
    {
        return parser::is_equal(image_number, ITEMS_NUMBER);
    }

    inline bool is_correct_cols_number(const int &cols)
    {
        return parser::is_equal(cols, COLS_NUMBER);
    }

    inline bool is_correct_rows_number(const int &rows)
    {
        return parser::is_equal(rows, ROWS_NUMBER);
    }

    inline bool is_correct_training_label_number(const int &items)
    {
        return parser::is_equal(items, IMAGES_NUMBER);
    }

    inline bool is_correct_test_label_number(const int &items)
    {
        return parser::is_equal(items, ITEMS_NUMBER);
    }

    //
    // load data 
    //

    inline void load_magic_number_of_labels(int &magic_number, boostifstream::ifstream &label_file)
    {
        parser::read_int(magic_number, label_file);
    }

    inline void load_magic_number_of_images(int &magic_number, boostifstream::ifstream &image_file)
    {
        parser::read_int(magic_number, image_file);
    }

    inline void load_number_of_images(int &image_number, boostifstream::ifstream &image_file)
    {
        parser::read_int(image_number, image_file);
    }

    inline void load_number_of_rows(int &ros, boostifstream::ifstream &image_file)
    {
        parser::read_int(ros, image_file);
    }

    inline void load_number_of_cols(int &cols, boostifstream::ifstream &image_file)
    {
        parser::read_int(cols, image_file);
    }

    inline void load_pixel(unsigned char &pixel, boostifstream::ifstream &image_file)
    {
        parser::read_char(pixel, image_file);
    }

    inline void load_label(unsigned char &label, boostifstream::ifstream &label_file)
    {
        parser::read_char(label, label_file);
    }

    inline void load_number_of_items(int &items, boostifstream::ifstream &label_file)
    {
        parser::read_int(items, label_file);
    }
};
#endif