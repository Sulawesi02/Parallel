#include<iostream>
using namespace std;
#define MAXN 10000000
int a[MAXN];
int sum,sum1,sum2=0;

int main()
{
    int t=50;
    for(int i=0;i<MAXN;i++)
    {
        a[i]=i;
    }

    //ÓÅ»¯Ëã·¨
    while(t--){
        for(int i=0;i<MAXN;i+=2){
            sum1+=a[i];
            sum2+=a[i+1];
        }
    }
    sum=sum1+sum2;

    return 0;
}
