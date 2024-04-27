#include<iostream>
#include<fstream>
#include<stdlib.h>
#include<ctime>

#include<windows.h>

#include<nmmintrin.h>  // SSE 4,2
#include<immintrin.h>  // AVX


using std::cin;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

void reset(float** a, int n);
void c_reset(float** a, int n);
void show(float** a, int n);
void GE(float** a, int n);
void C_GE(float** a, int n);
void SSE_GE(float** a, int n);
void AVX_GE(float** a, int n);
void SSE_U_GE(float** a, int n);
void AVX_U_GE(float** a, int n);

int n = 1000;
int lim = 1;
ofstream out("output.txt");
//ifstream in("input.txt");


void GE(float** a, int n) {  // ��׼�ĸ�˹��ȥ�㷨, Gauss Elimination��д
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j ++)
		{
			a[k][j] = a[k][j] / a[k][k];
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void C_GE(float** a, int n) {  // ��˹��ȥ�㷨��Cache�Ż��汾
	//__m128 va;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
		{
			a[k][j] = a[k][j] / t1;
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
			{
				a[i][j] -= t2 * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void SSE_GE(float** a, int n) {  // ʹ��SSE 4.2����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m128 va, vt, vaik, vakj, vaij, vx;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		vt = _mm_set1_ps(a[k][k]);  // �����ĸ��ظ�ֵ��vt��
		t1 = a[k][k];
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			if (j % 4 != 0) {
				a[k][j] = a[k][j] / t1;
				j -= 3;
				continue;
			}
			va = _mm_load_ps(&a[k][j]);
			va = _mm_div_ps(va, vt);
			_mm_store_ps(&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // �ƺ�
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 4 < n; j += 4)
			{
				if (j % 4 != 0) {
					a[i][j] -= t2 * a[k][j];
					j -= 3;
					continue;
				}
				vakj = _mm_load_ps(&a[k][j]);
				vaij = _mm_load_ps(&a[i][j]);
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void AVX_GE(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨
	__m256 va, vt, vaik, vakj, vaij, vx;
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);  
		t1 = a[k][k];
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			if (j % 8 != 0) {
				a[k][j] = a[k][j] / t1;
				j -= 7;
				continue;
			}
			va = _mm256_load_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_store_ps(&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // �ƺ�
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				if (j % 8 != 0) {
					a[i][j] -= t2 * a[k][j];
					j -= 7;
					continue;
				}
				vakj = _mm256_load_ps(&a[k][j]);
				vaij = _mm256_load_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_store_ps(&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
		
	}
}

void SSE_U_GE(float** a, int n) {  // ʹ��SSE 4.2����SIMD�Ż��ĸ�˹��ȥ�㷨��δ����
	__m128 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm_set1_ps(a[k][k]);  // �����ĸ��ظ�ֵ��vt��
		int j = 0;
		for (j = k + 1; j + 4 < n; j += 4)
		{
			va = _mm_loadu_ps(&a[k][j]);
			va = _mm_div_ps(va, vt);
			_mm_storeu_ps(&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // �ƺ�
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm_set1_ps(a[i][k]);
			for (j = k + 1; j + 4 < n; j += 4)
			{
				vakj = _mm_loadu_ps(&a[k][j]);
				vaij = _mm_loadu_ps(&a[i][j]);
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_storeu_ps(&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

void AVX_U_GE(float** a, int n) {  // ʹ��AVXָ�����SIMD�Ż��ĸ�˹��ȥ�㷨��δ����
	__m256 va, vt, vaik, vakj, vaij, vx;
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		int j = 0;
		for (j = k + 1; j + 8 < n; j += 8)
		{
			va = _mm256_loadu_ps(&a[k][j]);
			va = _mm256_div_ps(va, vt);
			_mm256_storeu_ps(&a[k][j], va);
		}
		for (j; j < n; j++)
		{
			a[k][j] = a[k][j] / a[k][k];  // �ƺ�
		}
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
				//a[i][j] -= a[i][k] * a[k][j];
			}
			for (j; j < n; j++)
			{
				a[i][j] -= a[i][k] * a[k][j];
			}
			a[i][k] = 0;
		}
	}
}

float** generate(int n) {
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
	}
	reset(m, n);
	return m;
}

float** aligned_generate(int n) {
	float** m = (float**)_aligned_malloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)_aligned_malloc(32 * n * sizeof(float*), 32);
	}
	reset(m, n);
	return m;
}

float** proof_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

float** aligned_proof_generate(int n) {
	ifstream inp("input.txt");
	inp >> n;
	float** m = (float**)_aligned_malloc(32 * n * sizeof(float**), 32);
	for (int i = 0; i < n; i++)
	{
		m[i] = (float*)_aligned_malloc(32 * n * sizeof(float*), 32);
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

void reset(float** a, int n)  // ��������ȷ��
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			a[i][j] = rand() % 10;
			//out << a[i][j] << " ";
		}
		//out << endl;
	}
}

void c_reset(float** a, int n)  // �ο�ָ����������ı�֤�����ȷ�ĳ�ʼ������
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < i; j++)
			a[i][j] = 0;
		a[i][i] = 1.0;
		for (int j = i + 1; j < n; j++)
			a[i][j] = rand();
	}
	for (int k = 0; k < n; k++)
		for (int i = k + 1; i < n; i++)
			for (int j = 0; j < n; j++)
				a[i][j] += a[k][j];
}

void show(float** a, int n) {  // ���ڹ۲�������н��
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << a[i][j] << " ";
			//out << a[i][j] << " ";
		}
		cout << endl;
		//out << endl;
	}
		
}



int main() {
	srand(time(0));
	//in >> n;
	cin >> n;
	out << n << endl;
	cout << "�����ģΪ" << n << "���㷨�����д���Ϊ" << lim << "��ʹ�ù̶���ʼֵ" << endl;
	long long head1, head2, head3, head4, head5, head6, tail1, tail2, tail3, tail4, tail5, tail6, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//-----------------------------------------------------------------

	//float** m1 = generate(n);
	float** m1 = proof_generate(n);
	//show(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head1);
	GE(m1, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
	cout << endl << endl << endl;
	//show(m1, n);

	cout << "GE: " << (tail1 - head1) * 1000.0 / freq
		<< "ms" << endl;

	//-----------------------------------------------------------------

	//float** m2 = generate(n);
	float** m2 = proof_generate(n);
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	C_GE(m2, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "C_GE: " << (tail2 - head2) * 1000.0 / freq
		<< "ms" << endl;
	//show(m2, n);
	
	//-----------------------------------------------------------------
	
	//float** m3 = generate(n);
	//float** m3 = proof_generate(n);
	//float** m3 = aligned_generate(n);
	float** m3 = aligned_proof_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head3);
	SSE_GE(m3, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
	
	cout << "SSE_GE: " << (tail3 - head3) * 1000.0 / freq
		<< "ms" << endl;
	//show(m3, n);

	//-----------------------------------------------------------------

	//float** m4 = generate(n);
	//float** m4 = proof_generate(n);
	//float** m4 = aligned_generate(n);
	float** m4 = aligned_proof_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head4);
	AVX_GE(m4, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail4);

	cout << "AVX_GE: " << (tail4 - head4) * 1000.0 / freq
		<< "ms" << endl;
	//show(m4, n);

	//-----------------------------------------------------------------

	//float** m5 = generate(n);
	float** m5 = proof_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head5);
	SSE_U_GE(m5, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail5);

	cout << "SSE_U_GE: " << (tail5 - head5) * 1000.0 / freq
		<< "ms" << endl;
	//show(m5, n);

	//-----------------------------------------------------------------

	//float** m6 = generate(n);
	float** m6 = proof_generate(n);

	QueryPerformanceCounter((LARGE_INTEGER*)&head6);
	AVX_U_GE(m6, n);
	QueryPerformanceCounter((LARGE_INTEGER*)&tail6);

	cout << "AVX_U_GE: " << (tail6 - head6) * 1000.0 / freq
		<< "ms" << endl;
	//show(m6, n);

	//-----------------------------------------------------------------

	system("pause");

}