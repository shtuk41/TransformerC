
#pragma once

//#define __BI_DEBUG

#include <memory>
#include <torch/torch.h>

#include <feedForward.h>
#include <head.h>


#include <utils.h>

using namespace torch::indexing;


class BigramLanguageModel5 : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
	torch::nn::Embedding position_embedding_table;
	torch::nn::Linear lm_head;
	int64_t block_size;
	
	torch::nn::Sequential blocks;
	
public:
	BigramLanguageModel5(int64_t vocab_size, int64_t number_heads, int64_t head_size, int64_t block_size, int64_t n_embd, double dropout) :
		token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embd))),
		position_embedding_table(register_module("position_embedding_table", torch::nn::Embedding(block_size, n_embd))),
		lm_head(register_module("lm_head", torch::nn::Linear(n_embd, vocab_size))),
		block_size(block_size),
		blocks(register_module("blocks", torch::nn::Sequential(
				std::make_shared<Block>(number_heads, block_size, n_embd, dropout),
				std::make_shared<Block>(number_heads, block_size, n_embd, dropout),
				std::make_shared<Block>(number_heads, block_size, n_embd, dropout),
				torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_embd})))))
	{}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto B = idx.sizes()[0];
		auto T = idx.sizes()[1];
		
		auto tok_emb = token_embedding_table(idx); //(B,T,C)

#ifdef __BI_DEBUG		
		std::cout << "tok_emb: \n";
		printTensorShape(tok_emb);
#endif		
		
		auto pos_emb = position_embedding_table(torch::arange(0, T, torch::kCUDA)); //(T,C)

#ifdef __BI_DEBUG			
		std::cout << "pos_emb: \n";
		printTensorShape(pos_emb);
#endif		
		
		auto x = tok_emb + pos_emb.unsqueeze(0);//(B,T,C)
	
#ifdef __BI_DEBUG			
		std::cout << "x after sum \n";
		printTensorShape(x);
#endif		
		
		x = blocks->forward(x);

#ifdef __BI_DEBUG		
		std::cout << "bigram x: \n";
		printTensorShape(x);
#endif		

		auto logits = lm_head(x); //(B,T,vocab_size)
		
#ifdef	__BI_DEBUG		
		std::cout << "bigram logits: \n";
		printTensorShape(logits);
#endif		

		torch::Tensor loss = torch::Tensor();
		if (targets.defined())
		{
			int64_t B = logits.size(0);
			int64_t T = logits.size(1);
			int64_t C = logits.size(2);

			auto logits_flat = logits.view({ B * T, C });
			auto targets_flat = targets.view({ B * T });

			loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
		}
#ifdef __BI_DEBUG	
		
		std::cout << "exit bigram forward: \n";
		
		printTensorShape(loss);
#endif		


		return { logits, loss };
		
	}

	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			int64_t seq_len = idx.size(1);
			int64_t start = std::max(static_cast<int64_t>(0), seq_len - block_size);

			auto idx_cond = idx.slice(1, start, seq_len);
			auto [logits, loss] = forward(idx_cond);
			auto last_logits = logits.index({ Slice(), -1});
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			auto idx_next = torch::multinomial(probs, 1);

			idx = torch::cat({ idx, idx_next }, 1);
		}

		return idx;
	}
};

class BigramLanguageModel4 : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
	torch::nn::Embedding position_embedding_table;
	std::shared_ptr<MultiHeadAttention> sa_head; //multi attention
	torch::nn::Linear lm_head;
	int64_t block_size;
	std::shared_ptr<FeedForward> ffwd; 
public:
	BigramLanguageModel4(int64_t vocab_size, int64_t number_heads, int64_t head_size, int64_t block_size, int64_t n_embd, double dropout) :
		token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embd))),
		position_embedding_table(register_module("position_embedding_table", torch::nn::Embedding(block_size, n_embd))),
		sa_head(register_module("sa_head", std::make_shared<MultiHeadAttention>(number_heads, head_size, block_size, n_embd, dropout))),
		lm_head(register_module("lm_head", torch::nn::Linear(n_embd, vocab_size))),
		block_size(block_size),
		ffwd(register_module("ffwd", std::make_shared<FeedForward>(n_embd, dropout)))
	{}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto B = idx.sizes()[0];
		auto T = idx.sizes()[1];
		
		auto tok_emb = token_embedding_table(idx); //(B,T,C)

#ifdef __BI_DEBUG		
		std::cout << "tok_emb: \n";
		printTensorShape(tok_emb);
#endif		
		
		auto pos_emb = position_embedding_table(torch::arange(0, T, torch::kCUDA)); //(T,C)

#ifdef __BI_DEBUG			
		std::cout << "pos_emb: \n";
		printTensorShape(pos_emb);
#endif		
		
		auto x = tok_emb + pos_emb.unsqueeze(0);//(B,T,C)
	
#ifdef __BI_DEBUG			
		std::cout << "x after sum \n";
		printTensorShape(x);
#endif		
		
		x = sa_head->forward(x);

#ifdef __BI_DEBUG		
		std::cout << "bigram x: \n";
		printTensorShape(x);
#endif		

		x = ffwd->forward(x);
		
#ifdef __BI_DEBUG			
		std::cout << "x after ffwd \n";
		printTensorShape(x);
#endif			

		
		auto logits = lm_head(x); //(B,T,vocab_size)
		
#ifdef	__BI_DEBUG		
		std::cout << "bigram logits: \n";
		printTensorShape(logits);
#endif		

		torch::Tensor loss = torch::Tensor();
		if (targets.defined())
		{
			int64_t B = logits.size(0);
			int64_t T = logits.size(1);
			int64_t C = logits.size(2);

			auto logits_flat = logits.view({ B * T, C });
			auto targets_flat = targets.view({ B * T });

			loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
		}
#ifdef __BI_DEBUG	
		
		std::cout << "exit bigram forward: \n";
		
		printTensorShape(loss);
#endif		


		return { logits, loss };
		
		
	}

	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			int64_t seq_len = idx.size(1);
			int64_t start = std::max(static_cast<int64_t>(0), seq_len - block_size);

			auto idx_cond = idx.slice(1, start, seq_len);
			auto [logits, loss] = forward(idx_cond);
			auto last_logits = logits.index({ Slice(), -1});
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			auto idx_next = torch::multinomial(probs, 1);

			idx = torch::cat({ idx, idx_next }, 1);
		}

		return idx;
	}
};

class BigramLanguageModel3 : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
	torch::nn::Embedding position_embedding_table;
	std::shared_ptr<Head> sa_head; //self attention
	torch::nn::Linear lm_head;
	int64_t block_size;
public:
	BigramLanguageModel3(int64_t vocab_size, int64_t head_size, int64_t block_size, int64_t n_embd, double dropout) :
		token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embd))),
		position_embedding_table(register_module("position_embedding_table", torch::nn::Embedding(block_size, n_embd))),
		sa_head(register_module("sa_head", std::make_shared<Head>(head_size, block_size, n_embd, dropout))),
		lm_head(register_module("lm_head", torch::nn::Linear(n_embd, vocab_size))),
		block_size(block_size)
	{}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto B = idx.sizes()[0];
		auto T = idx.sizes()[1];
		
		auto tok_emb = token_embedding_table(idx); //(B,T,C)

#ifdef __BI_DEBUG		
		std::cout << "tok_emb: \n";
		printTensorShape(tok_emb);
#endif		
		
		auto pos_emb = position_embedding_table(torch::arange(0, T, torch::kCUDA)); //(T,C)

#ifdef __BI_DEBUG			
		std::cout << "pos_emb: \n";
		printTensorShape(pos_emb);
#endif		
		
		auto x = tok_emb + pos_emb.unsqueeze(0);//(B,T,C)
	
#ifdef __BI_DEBUG			
		std::cout << "x after sum \n";
		printTensorShape(x);
#endif		
		
		x = sa_head->forward(x);

#ifdef __BI_DEBUG		
		std::cout << "bigram x: \n";
		printTensorShape(x);
#endif		
		
		auto logits = lm_head(x); //(B,T,vocab_size)
		
#ifdef	__BI_DEBUG		
		std::cout << "bigram logits: \n";
		printTensorShape(logits);
#endif		

		torch::Tensor loss = torch::Tensor();
		if (targets.defined())
		{
			int64_t B = logits.size(0);
			int64_t T = logits.size(1);
			int64_t C = logits.size(2);

			auto logits_flat = logits.view({ B * T, C });
			auto targets_flat = targets.view({ B * T });

			loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
		}
#ifdef __BI_DEBUG	
		
		std::cout << "exit bigram forward: \n";
		
		printTensorShape(loss);
#endif		


		return { logits, loss };
		
		
	}

	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			int64_t seq_len = idx.size(1);
			int64_t start = std::max(static_cast<int64_t>(0), seq_len - block_size);

			auto idx_cond = idx.slice(1, start, seq_len);
			auto [logits, loss] = forward(idx_cond);
			auto last_logits = logits.index({ Slice(), -1});
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			auto idx_next = torch::multinomial(probs, 1);

			idx = torch::cat({ idx, idx_next }, 1);
		}

		return idx;
	}
};

class BigramLanguageModel2 : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
	torch::nn::Embedding position_embedding_table;
	torch::nn::Linear lm_head;
public:
	BigramLanguageModel2(int64_t vocab_size, int64_t n_embd, int64_t block_size) : 
				token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embd))),
				position_embedding_table(register_module("position_embedding_table", torch::nn::Embedding(block_size, n_embd))),
				lm_head(register_module("lm_head", torch::nn::Linear(n_embd, vocab_size)))

	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto B = idx.sizes()[0];
		auto T = idx.sizes()[1];
		
		auto tok_emb = token_embedding_table(idx); //(B,T,C)
		auto pos_emb = position_embedding_table(torch::arange(T, torch::kCUDA));
		auto x = pos_emb + pos_emb;//(B,T,C)
		auto logits = lm_head(x); //(B,T,vocab_size)

        torch::Tensor loss = torch::Tensor();
        if (targets.defined()) 
        {
            int64_t B = logits.size(0);
            int64_t T = logits.size(1);
            int64_t C = logits.size(2);

            auto logits_flat = logits.view({B * T, C});
            auto targets_flat = targets.view({B * T});

            loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
        }

        return {logits, loss};
	}
	
	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			auto [logits, loss] =  forward(idx);
			
			auto last_logits = logits.index({Slice(), -1, Slice()}); 
			
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			
			auto idx_next = torch::multinomial(probs, 1);
			
			idx = torch::cat({idx, idx_next}, 1);
		}
		
		return idx;
	}
};


class BigramLanguageModel : public torch::nn::Module
{
private:
	torch::nn::Embedding token_embedding_table;
public:
	BigramLanguageModel(int64_t vocab_size) : token_embedding_table(register_module("token_embedding_table", torch::nn::Embedding(vocab_size, vocab_size)))
	{
	}

	std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor idx, torch::Tensor targets = torch::Tensor())
	{
		auto logits = token_embedding_table(idx);

        torch::Tensor loss = torch::Tensor();
        if (targets.defined()) 
        {
            int64_t B = logits.size(0);
            int64_t T = logits.size(1);
            int64_t C = logits.size(2);

            auto logits_flat = logits.view({B * T, C});
            auto targets_flat = targets.view({B * T});

            loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
        }

        return {logits, loss};
	}
	
	torch::Tensor generate(torch::Tensor idx, int max_new_tokens)
	{
		for (int ii = 0; ii < max_new_tokens; ii++)
		{
			auto [logits, loss] =  forward(idx);
			
			auto last_logits = logits.index({Slice(), -1, Slice()}); 
			
			auto probs = torch::nn::functional::softmax(last_logits, -1);
			
			auto idx_next = torch::multinomial(probs, 1);
			
			idx = torch::cat({idx, idx_next}, 1);
		}
		
		return idx;
	}
};
