#include <iostream>

using namespace std;
#define n 10000
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

    //Æ½·²Ëã·¨
    for(int i=0;i<n;i++){
        sum[i]=0;
        for(int j=0;j<n;j++){
            sum[i]+=b[j][i]*a[j];
        }
    }

    return 0;
}
