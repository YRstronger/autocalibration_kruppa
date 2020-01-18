#include"AutoCalibra.h"

namespace Simple_OrbSlam
{
	AutoCalibra::AutoCalibra(ACmethods meth) {
		method = meth;
	}

   void AutoCalibra::CalAC(double *vec_K, vector<Matrix3d> vFM)
	{
	   Problem problem;

	   if (method == AutoCalibra::SimpliedK)
	   {
		   LossFunction* loss_function = new ceres::HuberLoss(0.001);

		   Matrix3d tempFm;
		   for (int i = 0; i < vFM.size(); i++)
		   {

			   tempFm = vFM[i];


			   CostFunction* cost_function =
				   new AutoDiffCostFunction
				   <SimpliedKruppa, 5, 5>(
					   new SimpliedKruppa(tempFm));
			   problem.AddResidualBlock(cost_function,
				   loss_function,
				   vec_K);
		   }

		   problem.SetParameterUpperBound(vec_K, 0, vec_K[0] + 300.0);
		   problem.SetParameterLowerBound(vec_K, 0, vec_K[0] - 300.0);
		   problem.SetParameterUpperBound(vec_K, 1, vec_K[1] + 300.0);
		   problem.SetParameterLowerBound(vec_K, 1, vec_K[1] - 300.0);
		   problem.SetParameterUpperBound(vec_K, 3, vec_K[3] + 100.0);
		   problem.SetParameterLowerBound(vec_K, 3, vec_K[3] - 100.0);
		   problem.SetParameterUpperBound(vec_K, 4, vec_K[4] + 100.0);
		   problem.SetParameterLowerBound(vec_K, 4, vec_K[4] - 100.0);


		   Solver::Options options;
		   options.minimizer_progress_to_stdout = true;   //是否显示迭代结果的标志
		   //options.initial_trust_region_radius = 100;
		   options.gradient_tolerance = 1e-20;
		   options.function_tolerance = 1e-20;
		   options.parameter_tolerance = 1e-16;
		   options.max_num_iterations = 150;
		   options.initial_trust_region_radius = 1e+9;

		   Solver::Summary summary;
		   ceres::Solve(options, &problem, &summary);
		   //	std::cout << summary.BriefReport() << "\n";
		   std::cout << summary.BriefReport() << "\n";
		   cout << summary.final_cost << endl;
		   cout << vec_K[0] << endl << vec_K[1] << endl << vec_K[2] << endl << vec_K[3] << endl << vec_K[4] << endl;
	   }
	}
}