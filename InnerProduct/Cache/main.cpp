#include <iostream>

using namespace std;
#define n 5000
int a[n];
int b[n][n];
int sum[n];

int main()
{
    for(int i=0;i<n;i++){
        a[i]=i;
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            b[i][j]=i+j;
        }
    }


    for(int i=0;i<n;i++)
        sum[i]=0;
    for(int j=0;j<n;j++){
        for(int i=0;i<n;i++)
        {
            sum[i]+=b[j][i]*a[j];
        }
    }


    return 0;
}
