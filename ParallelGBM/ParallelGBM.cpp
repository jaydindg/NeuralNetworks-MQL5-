#include "pch.h"

#include <random>
#include <math.h>
#include <thread>
#include <future>
#include <utility>
#include <limits.h>
#include "ParalellGBM.h"

double ResVal = 0;

void GeometricBrownianMotion(double sigma, double mu, double time, double curr_price, int samples, int steps_into_future) {

	std::default_random_engine generator;

	double pred = 0;
	double t_ = std::sqrt(time);
	std::normal_distribution<double> distribution(0, t_);

	for (int sample = 0; sample < samples; sample++) {


		double price = curr_price;

		for (int steps = 0; steps < steps_into_future; steps++) {

			int error;
			price = price * std::exp((mu - (sigma * sigma) / 2) * time + sigma * distribution(generator));

		}

		pred = pred + price;

	}

	ResVal = ResVal + pred / double(samples);


}

double Paralellgbm(double sigma, double mu, double time, double curr_price, int samples, int steps_into_future, int numOfCores) {

	std::vector<std::thread> threads;
	double Res = 0;

	for (int i = 0; i < numOfCores; ++i) {

		threads.emplace_back(GeometricBrownianMotion, sigma, mu, time, curr_price, samples, steps_into_future);

	}

	for (std::thread& t : threads) {
		t.join();
	}

	double Val = ResVal / double(numOfCores);
	ResVal = 0;
	return Val;


}
