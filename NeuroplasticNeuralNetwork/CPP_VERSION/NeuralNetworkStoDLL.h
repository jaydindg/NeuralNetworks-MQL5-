

#pragma once

#ifdef NeuralNetworkStoDLL_EXPORTS
#define NeuralNetworkStoDLL_API __declspec(dllexport)
#else
#define NeuralNetworkStoDLL_API __declspec(dllimport)
#endif

#include <random>
#include <math.h>
#include <thread>
#include <future>
#include <utility>
#include "NeuralNetworkStoDLL.h"
#include <Eigen/Dense>
#include <ctime>
#include <iostream>

NeuralNetworkStoDLL_API class NeuralNetwork {

private:

	int m_maxiters;
	double m_beta_1;
	double m_beta_2;
	bool m_verbose;
	double m_LearningRate;
	int m_deep;
	int m_depth;
	double m_sto_a_min;
	double m_sto_b_max;
	Eigen::MatrixXd m_input;
	Eigen::MatrixXd m_pred_input;
	Eigen::MatrixXd m_z_2;
	Eigen::MatrixXd m_a_2;
	Eigen::MatrixXd m_z_3;
	Eigen::MatrixXd m_yHat;
	Eigen::MatrixXd z_3_prime;
	Eigen::MatrixXd z_2_prime;
	Eigen::MatrixXd delta2;
	Eigen::MatrixXd delta3;
	Eigen::MatrixXd dJdW1;
	Eigen::MatrixXd dJdW2;
	Eigen::MatrixXd y_cor;

	double cost = 0;
	int iters = 0;
	double m_alpha;
	int m_outDim;
	Eigen::MatrixXd Forward_Prop(Eigen::MatrixXd Input);
	double Cost(Eigen::MatrixXd Input, Eigen::MatrixXd y_cor);

	void ComputeDerivatives(Eigen::MatrixXd Input, Eigen::MatrixXd y_);

public:

	Eigen::MatrixXd W_1;
	Eigen::MatrixXd W_2;

	NeuralNetwork(int in_DimensionRow, int in_DimensionCol, int Number_of_Neurons, int out_Dimension, double alpha, double LearningRate, double beta_1, double beta_2, int max_iterations, bool Verbose, double sto_a_min, double sto_b_max);
	void Train(Eigen::MatrixXd Input, Eigen::MatrixXd correct_Val, double sto_a_min, double sto_b_max);

	Eigen::MatrixXd Prediction(Eigen::MatrixXd Input);
	double GetCost();
	double GetIters();
	void ResetWeights();
	bool WriteWeights();
	bool LoadWeights();

};

double Sigmoid(double x);
double Sigmoid_Prime(double x);

extern "C" NeuralNetworkStoDLL_API int ConstructNNS(int in_DimensionRow, int in_DimensionCol, int Number_of_Neurons, int out_Dimension, double alpha, double LearningRate, double beta_1, double beta_2, int max_iterations, bool Verbose, double sto_a_min, double sto_b_max);

extern "C" NeuralNetworkStoDLL_API void TrainNN(int NN_Number, double* Input, double* correct_val, int Inputrow, int Inputcol, int CorRow, int CorCol, double sto_a_min, double sto_b_max);

extern "C" NeuralNetworkStoDLL_API double PredictionNN(int NN_Number, double* Input, int Inputrow, int Inputcol);
void TrainNNp(int NN_Number, Eigen::MatrixXd EInput, Eigen::MatrixXd Ecorrect_val, double sto_a_min, double sto_b_max);
Eigen::MatrixXd XavierInitialization(int in_dim, int out_dim, double learnBeta);
extern "C" __declspec(dllexport) void CloseLogFile();
extern "C" __declspec(dllexport) void LogMessage(const char* message);
extern "C" __declspec(dllexport) void InitializeLogFile();
extern "C" NeuralNetworkStoDLL_API double GetCost(int NN_Number);
extern "C" NeuralNetworkStoDLL_API double GetIters(int NN_Number);
extern "C" NeuralNetworkStoDLL_API void ParalellTrain(double* Inputs, double* correct_Vals, int Inputrow, int Inputcol, int CorRow, int CorCol, double sto_a_min, double sto_b_max);
std::string EigenToString(const Eigen::MatrixXd& matrix);
