#pragma once

#include <torch/torch.h>

class FeedForward: public torch::nn::Module
{
private:

	torch::nn::Sequential net;

public:
	FeedForward(int64_t n_embd) :
		net(register_module("net", torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(n_embd, 4 * n_embd)),
			torch::nn::ReLU(),
			torch::nn::Linear(torch::nn::LinearOptions(4 * n_embd, n_embd)))))
	{
	}
	
	torch::Tensor forward(torch::Tensor x, torch::Tensor targets = torch::Tensor())
	{
		torch::Tensor out = net->forward(x);
		return out;
	}
};
