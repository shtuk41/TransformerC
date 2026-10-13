#pragma once

#include <vector>

#include <torch/torch.h>
#include <feedForward.h>
#include <utils.h>


class Head : public torch::nn::Module
{
private:

	torch::nn::Linear key;
	torch::nn::Linear query;
	torch::nn::Linear value;
	torch::Tensor tril;
	int64_t head_size;
	torch::nn::Dropout drop;

public:
	Head(int64_t head_size, int64_t block_size, int64_t n_embd, double dropout) :
		key(register_module("key", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		query(register_module("query", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		value(register_module("value", torch::nn::Linear(torch::nn::LinearOptions(n_embd, head_size).bias(false)))),
		tril(register_buffer("tril", torch::tril(torch::ones({ block_size, block_size })))),
		head_size(head_size),
		drop(register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout))))
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

		wei = drop->forward(wei);

#ifdef _HEAD_DEBUG			
		std::cout << "drop wei: \n";
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
	torch::nn::Linear proj;
	torch::nn::Dropout drop;
	
public:
	MultiHeadAttention(int64_t num_heads, int64_t head_size, int64_t block_size, int64_t n_embd, double dropout) :
		proj(register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(n_embd, n_embd)))),
		drop(register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(dropout))))
	{
		for (int64_t ii = 0; ii < num_heads; ii++)
		{
			auto head = register_module("head_" + std::to_string(ii), std::make_shared<Head>(head_size, block_size, n_embd, dropout));
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
		out = proj->forward(out);
		out = drop->forward(out);
		
		return out;
	}
};

class Block : public torch::nn::Module
{
private:
	
	int64_t head_size;
	std::shared_ptr<MultiHeadAttention> sa; //multi attention
	std::shared_ptr<FeedForward> ffwd; 
	torch::nn::LayerNorm ln1;
	torch::nn::LayerNorm ln2;
	
public:
	Block(int64_t n_heads, int64_t block_size, int64_t n_embd, double dropout) :
			head_size(n_embd / n_heads),
			sa(register_module("sa_head", std::make_shared<MultiHeadAttention>(n_heads, n_embd / n_heads, block_size, n_embd, dropout))),
			ffwd(register_module("ffwd", std::make_shared<FeedForward>(n_embd, dropout))),
			ln1(register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd})))),
			ln2(register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd}))))
	{
	}
	
	torch::Tensor forward(torch::Tensor x)
	{
		x = ln1->forward(x);
		x = x + sa->forward(x);
		x = ln2->forward(x);
		x = x + ffwd->forward(x);
		
		return x;
	}
};
