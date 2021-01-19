#include <iostream>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "image.h"

extern "C" int showImage(char file_name[])
{

	cv::Mat img;
	img = cv::imread(file_name, cv::IMREAD_COLOR);
	if(img.empty()){
		std::cout << "No Image" << std::endl;
		return -1;
	
	}
	cv::namedWindow("yaho", cv::WINDOW_AUTOSIZE);
	cv::imshow("yaho", img);
	
	cv::waitKey(0);
    
    return 0;
}

extern "C" unsigned int *extractValue(char file_name[])
{
    cv::Mat img;

    img = cv::imread(file_name, cv::IMREAD_GRAYSCALE);

    if(img.empty()){
		std::cout << "No Image" << std::endl;
    }
	
    return (unsigned int *)img.data;
}