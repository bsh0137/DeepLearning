#include <stdio.h>
#include <math.h>

double dotprod(
    int n,          // 벡터 길이
    double *vec1,   // 내적 연산에 사용될 벡터 변수
    double *vec2 )  // 내적 연산에 사용될 또 다른 벡터 변산
{
    int k, m;
    double sum;

    sum=0.0;
    k=n/4;  //벡터를 4개 그룹으로 나눔
    m=n%4;  //벡터를 4개 그룹으로 나눈 후 남은 나머지 

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

void activation(
    int ninputs, // 길이
    double *input,
    double *coefs,
    double *output,
    int outlin
)
{
    double sum;
    
    sum=dotprod(ninputs, input, coefs);
    sum+=coefs[ninputs];

    if(outlin)
       *output=sum; 
    else
        *output = 1.0 / (1.0+exp(-sum));
}


static void trial_thr (
    double *input,
    int n_layer, // 레이어 개수 (입력 레이어 제외)
    int n_model_inputs,
    double *outputs,
    int ntarg,
    int *nhid_all,

    double *weights_opt[],
    double *hid_act[],

    double *final_layer_weights,
    int classifier
)
{
    int i, ilayer;
    double sum;
    
    for(ilayer=0; ilayer<n_layer; ilayer++){

        if(ilayer==0 && n_layer==0){
            for(i=0; i<ntarg; i++)
                activation(n_model_inputs, input, weights_opt[ilayer], outputs, 1);
        }

        else if(ilayer==0){
            for (i=0;i<nhid_all[ilayer];i++)
                activation(n_model_inputs, input, weights_opt[ilayer]+i+(n_model_inputs+1), hid_act[ilayer]+i, 0);
        }

        else if (ilayer < n_layer-1) { // 출력 레이어가 아니면
            for(i=0; i<nhid_all[ilayer];i++)
                activation(n_model_inputs, hid_act[ilayer]+i)
        }

    }
}

// int main(){
//     int n=6;
//     double result;
//     double vec1[6]={1.3, 2, 3, 4, 5, 6};
//     double vec2[6]={6, 5, 4, 3, 2, 1.2};

//     printf("%d \n", sizeof(vec1)/sizeof(double));

//     result=dotprod(n, &vec1, &vec2);
//     printf("result: %f\n", result);

//     return 1;
// }