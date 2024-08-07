#include "pch.h"
#include <string>
#include <random>
#include <math.h>
#include <thread>
#include <future>
#include <utility>
#include "NeuralNetworkStoDLL.h"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/Map.h>
#include <vector>
#include <fstream>
#include <iostream>


std::ofstream log_file;


using namespace std;
double aa = 1;
double bb = 1.01;


bool NeuralNetwork::LoadWeights(void) {

	return true;

};

bool NeuralNetwork::WriteWeights(void) {

	return true;

};

void NeuralNetwork::ResetWeights(void) {



	W_1 = Eigen::MatrixXd::Random(m_depth, m_deep) * 2;
	W_2 = Eigen::MatrixXd::Random(m_depth, m_outDim) * 2;

};

void NeuralNetwork::ComputeDerivatives(Eigen::MatrixXd Input, Eigen::MatrixXd y_) {


	Eigen::MatrixXd X = Input;
	Eigen::MatrixXd Y = y_;

	m_yHat = Forward_Prop(X);

	Eigen::MatrixXd cost = -1 * (Y - m_yHat);

	Eigen::MatrixXd m_a1 = Eigen::MatrixXd::Random(m_z_3.rows(), m_z_3.cols()).cwiseAbs() * (m_sto_b_max - m_sto_a_min);

	z_3_prime = (m_z_3.cwiseProduct(m_a1)).unaryExpr(&Sigmoid_Prime);

	delta3 = cost.cwiseProduct(z_3_prime);
	dJdW2 = m_a_2.transpose() * delta3;

	Eigen::MatrixXd m_a2 = Eigen::MatrixXd::Random(m_a_2.rows(), m_a_2.cols()).cwiseAbs() * (m_sto_b_max - m_sto_a_min);

	z_2_prime = (m_a_2.cwiseProduct(m_a2)).unaryExpr(&Sigmoid_Prime);
	delta2 = (delta3 * W_2.transpose());

	delta2 = (delta2.cwiseProduct(z_2_prime));

};

NeuralNetwork::NeuralNetwork(int in_DimensionRow, int in_DimensionCol, int Number_of_Neurons, int out_Dimension, double alpha, double LearningRate, double beta_1, double beta_2, int max_iterations, bool Verbose, double sto_a_min, double sto_b_max) {


	m_depth = in_DimensionCol;
	m_deep = Number_of_Neurons;
	m_alpha = alpha;
	m_outDim = out_Dimension;
	m_LearningRate = LearningRate;
	m_beta_1 = beta_1;
	m_beta_2 = beta_2;
	m_verbose = Verbose;

	m_sto_a_min = sto_a_min;
	m_sto_b_max = sto_b_max;
	m_maxiters = max_iterations;
	aa = m_sto_a_min;
	bb = m_sto_b_max;

	W_1 = Eigen::MatrixXd::Random(m_depth, m_deep) * 2;
	W_2 = Eigen::MatrixXd::Random(m_deep, m_outDim) * 2;

	//W_1 = XavierInitialization(m_depth, m_deep, 1) * 2;
	//W_2 = XavierInitialization(m_deep, m_outDim, 1) * 2;

};

Eigen::MatrixXd NeuralNetwork::Prediction(Eigen::MatrixXd Input) {


	m_pred_input = Input;

	Eigen::MatrixXd pred_z_2 = m_pred_input * W_1;

	Eigen::MatrixXd pred_a_2 = pred_z_2.unaryExpr(&Sigmoid);

	Eigen::MatrixXd pred_z_3 = pred_a_2 * W_2;

	Eigen::MatrixXd pred_yHat = pred_z_3.unaryExpr(&Sigmoid);

	return pred_yHat;

}



double NeuralNetwork::GetCost() {

	double res = cost;

	return res;
}

double NeuralNetwork::GetIters() {

	int res = iters;

	return res;

}

void NeuralNetwork::Train(Eigen::MatrixXd Input, Eigen::MatrixXd correct_val, double sto_a_min, double sto_b_max) {
	LogMessage("Starting training...");

	bool Train_condition = true;
	y_cor = correct_val;
	int iterations = 0;

	m_yHat = Forward_Prop(Input);
	ComputeDerivatives(Input, y_cor);

	Eigen::MatrixXd mt_1(W_1.rows(), W_1.cols());
	mt_1.fill(0.0);

	Eigen::MatrixXd mt_2(W_2.rows(), W_2.cols());
	mt_2.fill(0.0);

	double J = 0;
	double betaM1 = 1 - m_beta_1;
	double learnBeta = m_LearningRate * m_beta_2;

	while (Train_condition && (iterations < m_maxiters)) {
		m_yHat = Forward_Prop(Input);
		ComputeDerivatives(Input, y_cor);
		J = Cost(Input, y_cor);

		LogMessage(("Iteration: " + to_string(iterations) + " Cost: " + to_string(J)).c_str());

		if (J < m_alpha) {
			Train_condition = false;
		}

		mt_1 = m_beta_1 * mt_1 + betaM1 * dJdW1;
		mt_2 = m_beta_1 * mt_2 + betaM1 * dJdW2;

		LogMessage(("Updating weights for iteration: " + to_string(iterations)).c_str());
		LogMessage(("W_1 before update: " + to_string(W_1.sum())).c_str());
		LogMessage(("W_2 before update: " + to_string(W_2.sum())).c_str());

		//W_1 = XavierInitialization(m_depth, m_deep, learnBeta);
		//W_2 = XavierInitialization(m_deep, m_outDim, learnBeta);
		W_1 = W_1 - m_LearningRate * (m_beta_2 * mt_1);
		W_2 = W_2 - m_LearningRate * (m_beta_2 * mt_2);

		LogMessage(("W_1 after update: " + to_string(W_1.sum())).c_str());
		LogMessage(("W_2 after update: " + to_string(W_2.sum())).c_str());

		iterations++;
	}

	if (m_verbose == true) {
		iters = iterations;
		cost = J;
	}

	LogMessage("Training completed.");
}

double NeuralNetwork::Cost(Eigen::MatrixXd Input, Eigen::MatrixXd y_) {

	Eigen::MatrixXd X = Input;
	Eigen::MatrixXd Y = y_;
	m_yHat = Forward_Prop(X);

	Eigen::MatrixXd temp = (Y - m_yHat);
	temp = temp.cwiseProduct(temp);
	double J = .5 * (temp.sum()) / double(temp.cols() * temp.rows());
	return J;


}

Eigen::MatrixXd NeuralNetwork::Forward_Prop(Eigen::MatrixXd Input) {
	m_input = Input;

	m_z_2 = m_input * W_1;
	Eigen::MatrixXd m_a1 = Eigen::MatrixXd::Random(m_z_2.rows(), m_z_2.cols()).cwiseAbs() * (m_sto_b_max - m_sto_a_min);
	m_a_2 = (m_z_2.cwiseProduct(m_a1)).unaryExpr(&Sigmoid);

	LogMessage(("m_z_2: " + to_string(m_z_2.sum())).c_str());
	LogMessage(("m_a_2: " + to_string(m_a_2.sum())).c_str());

	m_z_3 = m_a_2 * W_2;
	Eigen::MatrixXd m_a2 = Eigen::MatrixXd::Random(m_z_3.rows(), m_z_3.cols()).cwiseAbs() * (m_sto_b_max - m_sto_a_min);
	Eigen::MatrixXd yHat = (m_z_3.cwiseProduct(m_a2)).unaryExpr(&Sigmoid);

	LogMessage(("m_z_3: " + to_string(m_z_3.sum())).c_str());
	LogMessage(("yHat: " + to_string(yHat.sum())).c_str());

	return yHat;
}

vector<NeuralNetwork> myvector;

int NumOfNet = 0;

extern "C" int ConstructNNS(int in_DimensionRow, int in_DimensionCol, int Number_of_Neurons, int out_Dimension, double alpha, double LearningRate, double beta_1, double beta_2, int max_iterations, bool Verbose, double sto_a_min, double sto_b_max) {

	NeuralNetwork net = NeuralNetwork(in_DimensionRow, in_DimensionCol, Number_of_Neurons, out_Dimension, alpha, LearningRate, beta_1, beta_2, max_iterations, Verbose, sto_a_min, sto_b_max);
	myvector.push_back(net);
	NumOfNet = NumOfNet + 1;
	return NumOfNet;
}

void TrainNNp(int NN_Number, Eigen::MatrixXd EInput, Eigen::MatrixXd Ecorrect_val, double sto_a_min, double sto_b_max) {


	myvector.at(NN_Number - 1).Train(EInput, Ecorrect_val, sto_a_min, sto_b_max);

}

extern "C" void TrainNN(int NN_Number, double* Input, double* correct_val, int Inputrow, int Inputcol, int CorRow, int CorCol, double sto_a_min, double sto_b_max) {

	Eigen::MatrixXd EInput = Eigen::Map< Eigen::MatrixXd>(Input, Inputrow, Inputcol);
	Eigen::MatrixXd Ecorrect_val = Eigen::Map< Eigen::MatrixXd>(correct_val, CorRow, CorCol);

	myvector.at(NN_Number - 1).Train(EInput, Ecorrect_val, sto_a_min, sto_b_max);

}

extern "C" void ParalellTrain(double* Inputs, double* correct_Vals, int Inputrow, int Inputcol, int CorRow, int CorCol, double sto_a_min, double sto_b_max) {

	Eigen::MatrixXd EInput = Eigen::Map< Eigen::MatrixXd>(Inputs, Inputrow, Inputcol);
	Eigen::MatrixXd Ecorrect_val = Eigen::Map< Eigen::MatrixXd>(correct_Vals, CorRow, CorCol);


	std::vector<std::thread> threads;

	double Res = 0;
	for (int i = 0; i < myvector.size(); i++) {

		Eigen::MatrixXd Input = EInput.block((i - 1) * Inputrow, 0, Inputrow, Inputcol);
		Eigen::MatrixXd correct_Val = Ecorrect_val.block((i - 1) * CorRow, 0, CorRow, CorCol);

		threads.emplace_back(TrainNNp, i, Input, correct_Val, sto_a_min, sto_b_max);

	}

	for (std::thread& t : threads) {
		t.join();
	}

}

extern "C" double GetCost(int NN_Number) {

	return myvector.at(NN_Number - 1).GetCost();
}

extern "C" double GetIters(int NN_Number) {

	return myvector.at(NN_Number - 1).GetIters();
}

extern "C" double PredictionNN(int NN_Number, double* Input, int Inputrow, int Inputcol) {

	Eigen::MatrixXd EInput = Eigen::Map< Eigen::MatrixXd>(Input, Inputrow, Inputcol);

	Eigen::MatrixXd pred = myvector.at(NN_Number - 1).Prediction(EInput);

	return pred.sum() / double(pred.cols() * pred.rows());

}

double Sigmoid(double x) {
	return (1 / (1 + exp(-x)));
}

double Sigmoid_Prime(double x) {

	double sigmoid_x = Sigmoid(x);
	return sigmoid_x * (1 - sigmoid_x);

	//return exp(-x) / ((1 + exp(-x)) * (1 + exp(-x)));

}

Eigen::MatrixXd XavierInitialization(int in_dim, int out_dim, double learnBeta) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0, sqrt(2.0 / (in_dim + out_dim)));

	Eigen::MatrixXd mat(out_dim, in_dim);
	for (int i = 0; i < out_dim; ++i) {
		for (int j = 0; j < in_dim; ++j) {
			mat(i, j) = d(gen);
		}
	}
	return mat;
}




extern "C" __declspec(dllexport) void InitializeLogFile() {
	log_file.open("C:\\Users\\jaydi\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Libraries\\NeuralNetworkStoDLL\\x64\\Release\\neural_network_log.txt", std::ios::out | std::ios::app);
	if (!log_file) {
		std::cerr << "Failed to open log file." << std::endl;
	}
}

extern "C" __declspec(dllexport) void LogMessage(const char* message) {
	if (log_file.is_open()) {
		log_file << message << std::endl;
	}
}

extern "C" __declspec(dllexport) void CloseLogFile() {
	if (log_file.is_open()) {
		log_file.close();
	}
}


// Utility function to convert Eigen::MatrixXd to string for logging
string EigenToString(const Eigen::MatrixXd& matrix) {
	std::ostringstream oss;
	oss << matrix;
	return oss.str();
}


// Example usage within a function
void ExampleFunction() {
	LogMessage("This is a log message.");
	// Your existing code...
}
