#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "image.h"

double dotProd(
    int len,          // 벡터 길이
    double *vec1,   // 내적 연산에 사용될 벡터 변수
    double *vec2 )  // 내적 연산에 사용될 또 다른 벡터 변산
{
    int k, m;
    double sum;

    sum=0.0;
    k=len/4;  //벡터를 4개 그룹으로 나눔
    m=len%4;  //벡터를 4개 그룹으로 나눈 후 남은 나머지 

    while(k--){
        sum+=*vec1 * *vec2;
        sum+=*(vec1+1) * *(vec2+1);
        sum+=*(vec1+2) * *(vec2+2);
        sum+=*(vec1+3) * *(vec2+3);

        vec1+=4; vec2+=4;
    }

    while(m--){
        sum+=*vec1++ * *vec2++;
    }

    return sum;
}

double *activation(
    double *input,
    double *output,     // 결과 벡터
    unsigned short activation_type
)
{
    switch (activation_type){
        case 0:         // Linear         
            output=output;
            break;
    }

    return output;
}

int *padding(
    int *input,
    int input_width,
    int input_height,
    int padding_type
)
{
    int *output;
    int idx;
    output = malloc(sizeof(int) * (input_height+2) * (input_width+2));
    
    for(int col=0; col<input_height; col++){
        for(int row=0; row<input_width; row++){
            if ((row==0) || (row==input_width) || (col==0) ||(col==input_height))
                output[idx]=0;
            else{
                idx=(col-1)*input_width + (row-1);
                output[col*input_width+row]=input[(col-1)*input_width+(row-1)];
                }
        }
    }
    return output;
}

// TODO: 필터 개수 기능 구현해야 함.
double *convolution(
    int *input,
    int input_width,
    int input_height,
    int stride,
    int filter_width,
    int filter_height,
    int n_filters
)
{
    double *output;
    double *filter;
    int padding_size=1;
    int output_height=(input_height+2*padding_size-filter_height)/stride +1;
    int output_width=(input_width+2*padding_size-filter_width)/stride +1;
    double *array;

    output=malloc(sizeof(double)*output_height*output_width);
    filter=calloc(filter_width*filter_height, sizeof(double));
    array=malloc(sizeof(double)*filter_width*filter_height);

    for(int i=0; i<output_height; i++){
        for(int j=0; j<output_width; j++){
            
            for(int k=0; k<filter_height; k++){
                for(int l=0; l<filter_width; l++){
                    int idx = k*filter_width + l;
                    int idx1 = i*output_width + j + idx;
                    array[idx]=input[idx1];
                }
            }
            int idx2=i*output_width+j;
            output[idx2]=dotProd(output_height*output_width,array,filter);
        }
    }
    return output;
}

double max(
    double *input,
    int len_input
)
{
    double max_value=input[0];
    for (int i=0; i<len_input; i++){
        if (max_value>=input[i])
            max_value=input[i];
        else
            max_value=input[i];
    }
    return max_value;
}

double sum_array
(
    double *input,
    int len_input
)
{
    double sum=0;
    for (int i=0; i<len_input;i++)
    {
        sum+=input[i];
    }
    return sum;
}

// TODO: stride 구현 안됨.
double *pooling(
    double *input,
    int input_height,
    int input_width,
    int pooling_type,
    int stride
)
{
    double *output;
    int pooling_size=2;
    int output_height=(input_height-pooling_size)/stride +1;
    int output_width=(input_width-pooling_size)/stride +1;

    double *array;
    output=malloc(sizeof(double)*output_height*output_width);
    array=malloc(sizeof(double)*pooling_size*pooling_size);
    for(int i=0; i<output_height; i++){
        for(int j=0; j<output_width; j++){
            for(int k=0; k<pooling_size; k++){
                for(int l=0; l<pooling_size; l++){
                    int idx=k+pooling_size+l;
                    int idx1=i*output_width+j+k+pooling_size+l;
                    array[idx]=input[idx1];
                }
            }
            int idx2=i*output_width+j;
            output[idx2]=max(array, pooling_size*pooling_size);
        }
    }
    return output;
}

double *connected(
    double *input,
    int len_input,
    double *weights,
    int n_hid_neurons,
    double *outputs
)
{
    for (int i=0; i<n_hid_neurons; i++){
        double sum=0;   
        for (int j=0; j<len_input; j++){
            int idx=i*len_input + j;
            sum+=input[j]*weights[idx];
        }
        outputs[i]=sum;
    }
    return outputs;
}

void exponential(
    double *input,
    int len_input,
    double *output
)
{
    for (int i=0; i<len_input; i++){
        output[i]=exp(input[i]);
    }
}

int *softmax(
    double *input,
    int len_input,
    double *output
)
{
    double *input_exp;
    input_exp=malloc(sizeof(double)*len_input);
    exponential(input, len_input, input_exp);
    sum_array(input_exp, len_input);
    for (int i=0; i<len_input; i++){

    }
}

int *getLoss()
{

}


int main()
{
    int status, idx;
    int height=64;
    int width=64;
    u_char *img_data;
    char file_name[] = "/home/sdb/Desktop/mnist/images/t_00000_c5.png";

    status = showImage(file_name);

    img_data = extractValue(file_name);

    for(int col=0; col<height; col++){
        for(int row=0; row<width; row++){
            idx = col*width + row;
            printf("  %d ", img_data[idx]);
        }
        printf("\n");
    }

    return 0;
}