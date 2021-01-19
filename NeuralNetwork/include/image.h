#ifndef __IMAGE_H__
#define __IMAGE_H__
#ifdef __cplusplus

#include <iostream>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "image.h"

extern "C" {
    int showImage(char file_name[]);
    unsigned int *extractValue(char file_name[]);
    }



#endif
#endif