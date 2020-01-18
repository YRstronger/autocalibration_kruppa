#ifndef AUTOCALIBRA_H
#define AUTOCALIBRA_H

// common
#include"common.h"

namespace Simple_OrbSlam
{
	class AutoCalibra
	{
	public:
		enum ACmethods
		{
			SimpliedK = 1

		};

		AutoCalibra(ACmethods meth);

		void CalAC(double *vec_K, vector<Matrix3d> vFM);

		ACmethods method;
	};


	//����<��Kruppa����>�Ĵ��ۺ���
	class SimpliedKruppa {

	public:
		SimpliedKruppa(const Matrix3d &FM) :F_(FM) {}


	//ģ��TΪjet����(Ceres�������ݱ���)
	template<typename T> 
	    bool operator()(
			const T* const vec_K,   //�ڲ�չ�������������ࡢ�������Ť������
			T* residual) const {


			typedef Eigen::Matrix<T, 3, 3> Mat3;
			typedef Eigen::Matrix<T, 3, 1> Vec3;

			////��ʼ����˳��fx��fy��s��cx��cy
			Mat3 K;
			K(0, 1) = /*T(0.0)*/vec_K[2];
			K(0, 0) = vec_K[0];
			K(0, 2) = vec_K[3];
			K(1, 1) = vec_K[1];
			K(1, 2) = vec_K[4];
			K(2, 2) = T(1.0);
			K(1, 0) = T(0.0);
			K(2, 0) = T(0.0);
			K(2, 1) = T(0.0);

			//cout << K << endl;
			//ת��Eigen��������Ϊceres::jet
			Mat3 F;
			//cout << F_ << endl;
			for (int i = 0; i < 9; ++i)
			{
				F(i) = T(F_(i));
			}


			Mat3 w_inv = K * K.transpose();   //���Ƶ�����ڲμ�����Զ�������
			JacobiSVD<Mat3> svd(F.transpose(), ComputeFullU | ComputeFullV);//�Ի�������F������ֵ�ֽ�
			//cout << K << endl;
			//cout<<F<<endl;


			Mat3 V = svd.matrixU();
			Mat3 U = svd.matrixV();

			Vec3 D = svd.singularValues();

			////�����kruppa����
			T A;
			T B;
			T C;
			T t1 = T(D(0) * D(0) * V.transpose().row(0)*w_inv*V.col(0));
			T t2 = T(U.transpose().row(1)*w_inv*U.col(1));
			A = t1 / t2;
			t1 = T(D(0) * D(1) * V.transpose().row(0)*w_inv*V.col(1));
			t2 = T(-U.transpose().row(0)*w_inv*U.col(1));
			B = t1 / t2;
			t1 = T(D(1) * D(1) * V.transpose().row(1)*w_inv*V.col(1));
			t2 = T(U.transpose().row(0)*w_inv*U.col(0));
			C = t1 / t2;


			//cout << A(0) << endl;

			//��������ʽ�໥��ֵ��Ϊ�в�
			/*residual[0] = abs(A - B) + abs(B - C) + abs(A - C);*/
			residual[0] = (A - B);
			residual[1] = (B - C);
			residual[2] = (A - C);

			T teempp = (A - B);
			
			//T scaleFactor =A.a;
			//��Լ���������顶����ͼ���Ρ��е�����
			residual[3] = 6e-9*vec_K[2];
			residual[4] = 3e-9*(vec_K[0]-vec_K[1]);
			return true;

		}

		//
		//private:
		const Matrix3d F_;


	};

}


#endif