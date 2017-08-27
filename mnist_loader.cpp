#include "mnist_loader.hpp"



mnist_loader::mnist_loader(std::string root_path)
{
    // need to add exception!!
    this->root_path = root_path;
    train_image_path = this->root_path / "train-images.idx3-ubyte";
    test_image_path = this->root_path / "t10k-images.idx3-ubyte";
    train_label_path = this->root_path / "train-labels.idx1-ubyte";
    test_label_path = this->root_path / "t10k-labels.idx1-ubyte";
    train_images_file.open(train_image_path, ios::binary); //wrong path will still work without error!
    test_images_file.open(test_image_path, ios::binary);
    train_labels_file.open(train_label_path, ios::binary);
    test_labels_file.open(test_label_path, ios::binary);
}

boost::shared_ptr<origin_data> mnist_loader::load_data()
{
    boost::shared_ptr<array1i> pixel_of_training_images = get_training_image_data();
    boost::shared_ptr<array1i> pixel_of_test_images = get_test_image_data();
    boost::shared_ptr<array1i> item_of_training_labels = get_training_label_data();
    boost::shared_ptr<array1i> item_of_test_labels = get_test_label_data();
    boost::shared_ptr<origin_data> all_data =
        make_shared<origin_data>(
            pixel_of_training_images,
            pixel_of_test_images,
            item_of_training_labels,
            item_of_test_labels
        );
    return all_data;
}

boost::shared_ptr<wrapped_data> mnist_loader::load_data_wrapper()
{
    cout << "load_data_wrapper" << endl;
    boost::shared_ptr<origin_data> all_data = load_data();
    boost::shared_ptr<array1i> pixel_of_training_images = (*all_data).get<0>();
    boost::shared_ptr<array1i> pixel_of_test_images = (*all_data).get<1>();
    boost::shared_ptr<array1i> item_of_training_labels = (*all_data).get<2>();
    boost::shared_ptr<array1i> item_of_test_labels = (*all_data).get<3>();
    boost::shared_ptr<training_data_container> training_data = 
        get_wrapped_training_data(pixel_of_training_images, item_of_training_labels);
    boost::shared_ptr<validation_data_container> validation_data =
        get_wrapped_validation_data(pixel_of_training_images, item_of_training_labels);
    boost::shared_ptr<test_data_container> test_data =
        get_wrapped_test_data(pixel_of_test_images, item_of_test_labels);
    boost::shared_ptr<wrapped_data> final_data = 
        make_shared<wrapped_data>(
            training_data,
            validation_data,
            test_data
        );  
    return final_data;
}

boost::shared_ptr<array784d> mnist_loader::extract_single_image_from_original_data(boost::shared_ptr<array1i> original_data, const int & image_index)
{
    boost::shared_ptr<array784d> single_img(new array784d);
    for (int row_index = 0; row_index < ROWS_NUMBER; row_index++)
        for(int col_index = 0; col_index < COLS_NUMBER; col_index++)
        //normalize the original pixel values by diving them over 255.0(double type)
            (*single_img)[row_index + col_index * COLS_NUMBER] = 
                (double)(*original_data)[row_index + col_index * COLS_NUMBER + image_index * ROWS_NUMBER * COLS_NUMBER] / 255.0;
    
    return single_img;
}

boost::shared_ptr<array10i> mnist_loader::extract_single_label_from_original_data(boost::shared_ptr<array1i> original_data, const int &label_index)
{
    // return a vector contains 10 elements,
    //ex. [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] stands for 2
    boost::shared_ptr<array10i> single_label(new array10i);
    const int item = (*original_data)[label_index];
    (*single_label).assign(0);
    (*single_label).at(item) = 1;
    return single_label;
}

boost::shared_ptr<training_data_container> mnist_loader::get_wrapped_training_data(boost::shared_ptr<array1i> pixel_of_training_images, boost::shared_ptr<array1i> item_of_training_labels)
{
    boost::shared_ptr<training_data_container> training_data(new training_data_container);
    for (int pair_index = 0; pair_index < TRAINING_NUMBER; pair_index++)
    {
        // can be refactored
        (*training_data).at(pair_index).get<0>() = 
            *extract_single_image_from_original_data(pixel_of_training_images, pair_index);
        (*training_data).at(pair_index).get<1>() =
            *extract_single_label_from_original_data(item_of_training_labels, pair_index);
    }
    return training_data;
}

boost::shared_ptr<validation_data_container> mnist_loader::get_wrapped_validation_data(boost::shared_ptr<array1i> pixel_of_training_images, boost::shared_ptr<array1i> item_of_training_labels)
{
    boost::shared_ptr<validation_data_container> validation_data(new validation_data_container);
    for (int pair_index = TRAINING_NUMBER; pair_index < IMAGES_NUMBER; pair_index++)
    {
        (*validation_data).at(pair_index - TRAINING_NUMBER).get<0>() =
            *extract_single_image_from_original_data(pixel_of_training_images, pair_index);
        (*validation_data).at(pair_index - TRAINING_NUMBER).get<1>() =
            *extract_single_label_from_original_data(item_of_training_labels, pair_index);
    }
    return validation_data;
}

boost::shared_ptr<test_data_container> mnist_loader::get_wrapped_test_data(boost::shared_ptr<array1i> pixel_of_test_images, boost::shared_ptr<array1i> item_of_test_labels)
{
    boost::shared_ptr<test_data_container> test_data(new test_data_container);
    for (int pair_index = 0; pair_index < ITEMS_NUMBER; pair_index++)
    {
        (*test_data).at(pair_index).get<0>() =
            *extract_single_image_from_original_data(pixel_of_test_images, pair_index);
        (*test_data).at(pair_index).get<1>() =
            *extract_single_label_from_original_data(item_of_test_labels, pair_index);
    }
    return test_data;
}

boost::shared_ptr<array1i> mnist_loader::get_training_image_data()
{
    load_training_image_file_header();
    check_training_image_file_header_parameters();
    const int cols = image_file::get_instance().cols;
    const int rows = image_file::get_instance().rows;
    const int image_number = image_file::get_instance().image_number;
    const int pixel_amount = cols * rows * image_number;
    boost::shared_ptr<array1i> pixel_of_training_images =
        make_shared<array1i>(extents[pixel_amount]);
    load_all_training_images(*pixel_of_training_images);
    return pixel_of_training_images;
}

boost::shared_ptr<array1i> mnist_loader::get_test_image_data()
{
    load_test_image_file_header();
    check_test_image_file_header_parameters();
    const int cols = image_file::get_instance().cols;
    const int rows = image_file::get_instance().rows;
    const int image_number = image_file::get_instance().image_number;
    const int pixel_amount = cols * rows * image_number;
    boost::shared_ptr<array1i> pixel_of_test_images = 
        make_shared<array1i>(extents[pixel_amount]);
    load_all_test_images(*pixel_of_test_images);
    return pixel_of_test_images;
}

boost::shared_ptr<array1i> mnist_loader::get_training_label_data()
{
   load_training_label_file_header();
   check_training_label_file_header_parameters();
   const int label_number = label_file::get_instance().label_number;
   boost::shared_ptr<array1i> item_of_training_labels =
        make_shared<array1i>(extents[label_number]);
   load_all_training_labels(*item_of_training_labels);
   return item_of_training_labels;
}

boost::shared_ptr<array1i> mnist_loader::get_test_label_data()
{
    load_test_label_file_header();
    check_test_label_file_header_parameters();
    const int label_number = label_file::get_instance().label_number;
    boost::shared_ptr<array1i> item_of_test_labels =
        make_shared<array1i>(extents[label_number]);
    load_all_test_labels(*item_of_test_labels);
    return item_of_test_labels;
}

//
// training image file
//

void mnist_loader::load_training_image_file_header()
{
    load_magic_number_of_images(image_file::get_instance().magic_number, train_images_file);
    load_number_of_images(image_file::get_instance().image_number, train_images_file);
    load_number_of_cols(image_file::get_instance().cols, train_images_file);
    load_number_of_rows(image_file::get_instance().rows, train_images_file);
}

void mnist_loader::check_training_image_file_header_parameters()
{
    parser::convert_to_big_endian(image_file::get_instance().magic_number);
    parser::convert_to_big_endian(image_file::get_instance().image_number);
    parser::convert_to_big_endian(image_file::get_instance().cols);
    parser::convert_to_big_endian(image_file::get_instance().rows);
    assert(is_correct_magic_number_of_image(image_file::get_instance().magic_number) == true);
    assert(is_correct_training_image_number(image_file::get_instance().image_number) == true);
    assert(is_correct_rows_number(image_file::get_instance().rows) == true);
    assert(is_correct_cols_number(image_file::get_instance().cols) == true);
}

void mnist_loader::load_single_training_image(array1i &all_image_pixels, const int & image_index)
{
    int cols = image_file::get_instance().cols;
    int rows = image_file::get_instance().rows;
    int pixel_index = image_index * cols * rows; 
    for (int row_index = 0; row_index < rows; row_index++)
        for (int col_index = 0; col_index < cols; col_index++)
        {
            int offset = row_index * rows + col_index;
            load_pixel(image_file::get_instance().pixel, train_images_file);
            all_image_pixels[pixel_index + offset] = (int)image_file::get_instance().pixel;
        }
}

void mnist_loader::load_all_training_images(array1i &all_image_pixels)
{
    int total_number = image_file::get_instance().image_number;
    for (int image_index = 0; image_index < total_number; image_index++)
        load_single_training_image(all_image_pixels, image_index);
}

//
// test image file
//

void mnist_loader::load_test_image_file_header()
{
    load_magic_number_of_images(image_file::get_instance().magic_number, test_images_file);
    load_number_of_images(image_file::get_instance().image_number, test_images_file);
    load_number_of_cols(image_file::get_instance().cols, test_images_file);
    load_number_of_rows(image_file::get_instance().rows, test_images_file);
}

void mnist_loader::check_test_image_file_header_parameters()
{
    parser::convert_to_big_endian(image_file::get_instance().magic_number);
    parser::convert_to_big_endian(image_file::get_instance().image_number);
    parser::convert_to_big_endian(image_file::get_instance().cols);
    parser::convert_to_big_endian(image_file::get_instance().rows);
    assert(is_correct_magic_number_of_image(image_file::get_instance().magic_number) == true);
    assert(is_correct_test_image_number(image_file::get_instance().image_number) == true);
    assert(is_correct_rows_number(image_file::get_instance().rows) == true);
    assert(is_correct_cols_number(image_file::get_instance().cols) == true);
}

void mnist_loader::load_single_test_image(array1i &all_image_pixels, const int & image_index)
{
    int cols = image_file::get_instance().cols;
    int rows = image_file::get_instance().rows;
    int pixel_index = image_index * cols * rows; 
    for (int row_index = 0; row_index < rows; row_index++)
        for (int col_index = 0; col_index < cols; col_index++)
        {
            int offset = row_index * rows + col_index;
            load_pixel(image_file::get_instance().pixel, test_images_file);
            all_image_pixels[pixel_index + offset] = (int)image_file::get_instance().pixel;
        }
}

void mnist_loader::load_all_test_images(array1i &all_image_pixels)
{
    int total_number = image_file::get_instance().image_number;
    for (int image_index = 0; image_index < total_number; image_index++)
        load_single_test_image(all_image_pixels, image_index);
}

//
//  train label file
//

void mnist_loader::load_training_label_file_header()
{
    load_magic_number_of_labels(label_file::get_instance().magic_number, train_labels_file);
    load_number_of_items(label_file::get_instance().label_number, train_labels_file);
}

void mnist_loader::check_training_label_file_header_parameters()
{
    parser::convert_to_big_endian(label_file::get_instance().magic_number);
    parser::convert_to_big_endian(label_file::get_instance().label_number);
    assert(is_correct_magic_number_of_label(label_file::get_instance().magic_number) == true);
    assert(is_correct_training_label_number(label_file::get_instance().label_number)== true);
}

void mnist_loader::load_single_training_label(array1i &all_label_items, const int &label_index)
{
    load_label(label_file::get_instance().item, train_labels_file); 
    all_label_items[label_index] = (int)label_file::get_instance().item;
}

void mnist_loader::load_all_training_labels(array1i &all_label_items)
{
    int total_number = label_file::get_instance().label_number;
    for (int item_index = 0; item_index < total_number; item_index++)
        load_single_training_label(all_label_items, item_index);
}

//
// test label file
//

void mnist_loader::load_test_label_file_header()
{
    load_magic_number_of_labels(label_file::get_instance().magic_number, test_labels_file);
    load_number_of_items(label_file::get_instance().label_number, test_labels_file);
}

void mnist_loader::check_test_label_file_header_parameters()
{
    parser::convert_to_big_endian(label_file::get_instance().magic_number);
    parser::convert_to_big_endian(label_file::get_instance().label_number);
    assert(is_correct_magic_number_of_label(label_file::get_instance().magic_number) == true);
    assert(is_correct_test_label_number(label_file::get_instance().label_number) == true);
}

void mnist_loader::load_single_test_label(array1i &all_label_items, const int &label_index)
{
    load_label(label_file::get_instance().item, test_labels_file);
    all_label_items[label_index] = (int) label_file::get_instance().item;
}

void mnist_loader::load_all_test_labels(array1i &all_label_items)
{
    const int total_number = label_file::get_instance().label_number;
    for (int label_index = 0; label_index < total_number; label_index++)
        load_single_test_label(all_label_items, label_index);
}