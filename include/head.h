#pragma once

#include <vector>

#include <torch/torch.h>
#include <utils.h>

class Head : public torch::nn::Module
{
private:

	torch::nn::Linear key;
	torch::nn::Linear query;
	torch::nn::Linear value;
	torch::Tensor tril;
	int64_t head_size;

public:
	Head(int64_t head_size, int64_t block_size, int64_t n_embd) :
		key(register_module("key", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		query(register_module("query", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		value(register_module("value", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		tril(register_buffer("tril", torch::tril(torch::ones({ block_size, block_size })))),
		head_size(head_size)
	{
	}

	torch::Tensor operator()(torch::Tensor x)
	{
		return forward(x);
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor targets = torch::Tensor())
	{
		auto B = x.sizes()[0];
		auto T = x.sizes()[1];
		auto C = x.sizes()[2];
		
#ifdef _HEAD_DEBUG		
		std::cout << "head x: \n";
		printTensorShape(x);
#endif		

		auto k = key(x);

#ifdef _HEAD_DEBUG			
		std::cout << "head k: \n";
		printTensorShape(k);
#endif			
		
		auto q = query(x);

#ifdef _HEAD_DEBUG			
		std::cout << "head q: \n";
		printTensorShape(q);
#endif			

		auto wei = q.matmul(k.transpose(-2, -1)) * sqrt(head_size);

#ifdef _HEAD_DEBUG			
		std::cout << "head wei: \n";
		printTensorShape(wei);
#endif			
		
		wei = wei.masked_fill(tril == 0, -std::numeric_limits<float>::infinity());

#ifdef _HEAD_DEBUG			
		std::cout << "head wei2: \n";
		printTensorShape(wei);
#endif			
		
		wei = torch::nn::functional::softmax(wei, -1);

#ifdef _HEAD_DEBUG			
		std::cout << "head wei3: \n";
		printTensorShape(wei);
#endif			
		
		auto v = value(x);
		
#ifdef _HEAD_DEBUG			
		std::cout << "head v: \n";
		printTensorShape(v);
#endif			
		
		auto out = wei.matmul(v);

#ifdef _HEAD_DEBUG			
		std::cout << "head out: \n";
		printTensorShape(out);
#endif			

		return out;
	}
};

class MultiHeadAttention : public torch::nn::Module
{
private:
	torch::nn::ModuleList heads;
public:
	MultiHeadAttention(int64_t num_heads, int64_t head_size, int64_t block_size, int64_t n_embd)
	{
		for (int64_t ii = 0; ii < num_heads; ii++)
		{
			auto head = register_module("head_" + std::to_string(ii), std::make_shared<Head>(head_size, block_size, n_embd));
			heads->push_back(head);
		}
		
		register_module("heads", heads);
	}
	
	torch::Tensor forward(torch::Tensor x, torch::Tensor targets = torch::Tensor())
	{
		std::vector<torch::Tensor> outputs;
		
		outputs.reserve(heads->size());
		
		for (int ii = 0; ii < heads->size(); ii++)
		{
			auto head = heads[ii];
			
			auto head_out = heads[ii]->as<Head>()->forward(x);
			outputs.push_back(head_out);
		}
		
		auto out = torch::cat(outputs, 2);
		return out;
	}
};
