#include <stdio.h>
#include <math.h>

double dotProd(
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
    
    sum=dotProd(ninputs, input, coefs);
    sum+=coefs[ninputs];

    if(outlin)
       *output=sum; 
    else
        *output = 1.0 / (1.0+exp(-sum));
}


static void trialThr (
    double *input,
    int n_layer, // 레이어 개수 (입력 레이어 제외)
    int n_model_inputs,
    double *outputs,
    int n_targ,
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
            for(i=0; i<n_targ; i++)
                activation(n_model_inputs, input, weights_opt[ilayer], outputs, 1);
        }

        else if(ilayer==0){
            for (i=0;i<nhid_all[ilayer];i++)
                activation(n_model_inputs, input, weights_opt[ilayer]+i+(n_model_inputs+1), hid_act[ilayer]+i, 0);
        }

        else if (ilayer < n_layer-1) { // 출력 레이어가 아니면
            for(i=0; i<nhid_all[ilayer];i++)
                activation(nhid_all[ilayer-1], hid_act[ilayer-1], weights_opt[ilayer]+i, hid_act[ilayer]+i, 0);
        }

        else {
            for (i=0;i<n_targ; i++)
                activation(nhid_all[ilayer-1], hid_act[ilayer-1], final_layer_weights+i*(nhid_all[ilayer-1]+1), outputs+i, 1);
        }
    }

    if (classifier) {
        sum=0.0;
        for(i=0; i<n_targ; i++){
            if(outputs[i]<300.0)
                outputs[i]=exp(outputs[i]);
            else
                outputs[i]=exp(300.0);
            sum+=outputs[i];
        }
        for(i=0;i<n_targ;i++)
            outputs[i]/=sum;
    }
}


double batchGradient(
    int start_data_idx,                     // 입력 행렬의 첫 번째 데이터 인덱스
    int last_data_idx,                      // 지난 마지막 데이터의 인덱스
    double *input,                  // 입력 행렬; 각 데이터의 길이 == max_neurons
    double *targets,                // 목표 행렬; 각 데이터의 길이 == ntarg
    int n_layers,                   // 출력은 포함하고 입력은 제외한 레이어의 개수
    int n_all_weights,              // 마지막 레이어와 모든 bias 항을 포함한 총 가중치 개수

    int n_model_inputs,             // 모델 입력의 개수, 입력 행렬은 더 많은 열을 가질 수도 있음.

    double *outputs,                // 모델의 출력 벡터, 여기서는 작업 벡터로 사용됨.
    int n_targ,                     // 출력의 개수
    int *n_all_hid_neuron,          // n_all_hid_neuron[i]은 i번째 레이어에 존재하는 뉴런 개수
    double *weights_opt[],          // weights_opt[i]는 i번째 은닉 레이어의 가중치 벡터를 가리키는 포인터
    double *hid_act[],              // hid_act[i]는 i번째 레이어의 활성화 벡터를 가리키는 포인터

    int max_neurons,                // 입력 행렬의 열 개수, n_model_inputs보다 최대치가 크다.

    double *prior_delta,            // 현재 레이어에 대한 델타 변수를 가리키는 포인터
    double *this_delta,             // 다음 단계에 사용하기 위해 이전 레이어에서 미리 저장해놓은 델타 변수를 가리키는 포인터

    double **grad_ptr,              // grad_ptr[i]는 i번째 레이어의 기울기를 가리키는 포인터

    double *final_layer_weights,    // 마지막 레이어의 가중치를 가리키는 포인터

    double *grad,                   // 계산된 모든 기울기로, 하나의 긴 벡터를 가리키는 포인터

    int classifier                  // 0이 아니면 SoftMax 결과 출력, 0이면 선형 결과 출력
)
{
    int i, j, icase, ilayer, n_prev, n_this, n_next, imax;
    double diff, *dptr, error, *targ_ptr, *prevact, *gradptr, delta, *next_coefs, tmax;

    for (i=0;i<n_all_weights; i++)  // 합산을 위해 기울기를 0으로 초기화
        grad[i]=0.0;                // 이 변수로 모든 레이어가 줄지어 저장된다.
    error=0.0;                      // 이 변수로 전체 오차 값을 누적해간다.

    for (icase=start_data_idx; icase<last_data_idx; icase++){
        dptr=input + icase * max_neurons;   // 현재 데이터를 가리킴.
        trialThr(dptr, n_layers, n_model_inputs, outputs, n_targ, n_all_hid_neuron, weights_opt, hid_act, final_layer_weights, classifier);

        targ_ptr=targets+icase*n_targ;

        if(classifier) {    // SoftMax를 사용한 경우
            tmax=-1.e30;
            for (i=0; i<n_targ; i++){
                if (targ_ptr[i] > tmax){
                    imax=i;
                    tmax=targ_ptr[i];
                }
                this_delta[i]=targ_ptr[i] - outputs[i];  // 교차 엔트로피를 입력(logit)으로 미분해 음의 부호를 취한 식
            }
            error -= log(outputs[imax] + 1.e-30); // 음의 로그 확률을 최소화 함.
        }

        else {
            for(i=0; i<n_targ; i++){
                diff=outputs[i]-targ_ptr[i];
                error += diff*diff;
                this_delta[i]=-2.0*diff; // i번째 뉴런의 입력으로 제곱 오차를 미분해 음의 부호를 취함.
            }
        }

        if (n_layers==1) {                          // 은닉 레이어가 없는 경우
            n_prev = n_model_inputs;                // 출력 레이어에 전달되는 입력의 개수
            prevact = input + icase * max_neurons;  // 현재 데이터를 가리키는 포인터
        }
        else {
            n_prev = n_all_hid_neuron[n_layers-2];  // n_layers-2 인덱스가 곧 마지막 은닉 레이어.
            prevact = hid_act[n_layers-2];          // 출력 레이어로 전달되는 레이어의 포인터 변수
        }
        gradptr = grad_ptr[n_layers-1];             // 기울기 벡터에서 출력 기울기를 가리키는 포인터

        for (i=0; i<n_targ; i++){
            delta = this_delta[i];                  // 평가 기준을 logit으로 편미분해 음수를 취한다.
            for (j=0; j<n_prev; j++)
                *gradptr++ += delta * prevact[j];   // 모든 훈련 데이터에 대한 결과를 누적함.
            
            *gradptr++ +=delta;                     // bias 활성화는 항상 1이다.
        }

        n_next=n_targ;                              // 한 레이어 되돌아갈 준비를 한다.
        next_coefs=final_layer_weights;

        for (ilayer=n_layers-2; ilayer>=0; ilayer--){
            n_this=n_all_hid_neuron[ilayer];            //현재 은닉 레이어상에 존재하는 뉴런 개수
            gradptr=grad_ptr[ilayer];                   //현재 레이어의 기울기를 가리키는 포인터

            for(i=0; i<n_this; i++) {
                delta=0.0;
                for (j=0; j<n_next; j++)
                    delta*=this_delta[j] * next_coefs[j*(n_this+1)+i];
                delta *=hid_act[ilayer][i] * (1.0-hid_act[ilayer][i]); // 미분 연산
                prior_delta[i] = delta;      // 다음 레이어를 위해 저장
                if (ilayer==0){
                    prevact=input + icase * max_neurons;    // 현재 데이터를 가리키는 포인터

                    for (j=0; j<n_model_inputs; j++)
                        *gradptr++ +=delta * prevact[j];
                }
                else{
                    prevact=hid_act[ilayer-1];
                    for (j=0; j<n_all_hid_neuron[ilayer-1]; j++)
                        *gradptr++ +=delta * prevact[j];
                }
                *gradptr++ +=delta;     // bias 활성화는 언제나 1이다.
            }   // 현재 은닉 레이어상의 모든 뉴런을 대상으로 한다.
            for (i=0; i<n_this; i++)    // 현재 델타 값을 이전 델타 값으로 저장
                this_delta[i] = prior_delta[i];

            n_next=n_all_hid_neuron[ilayer];    // 다음 레이어를 위한 준비
            next_coefs = weights_opt[ilayer];
        }   // 모든 레이어를 대상으로 거꾸로 진행해 나간다.
    }       // 모든 데이터를 대상으로 순환 실행

    return error;
}


