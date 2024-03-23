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

    //Æ½·²Ëã·¨
    while(t--){
        for(int i=0;i<MAXN;i++)
            sum+=a[i];
    }
    return 0;
}
