
#include <limits>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include <torch/torch.h>

#include <bigramLanguageModel.h>
#include <head.h>
#include <utils.h>

struct configuration
{
	int head_size = 32;
	int batch_size = 32;
	int block_size = 8;
	int n_embd = 32;
	int max_new_tokens = 500;	
	double train_percent = 0.9; 
	int train_steps = 5000;
	float learning_rate = 1e-3;
	int train_loss_print_every_num_iter  = 1000;
	int estimate_loss_iterations = 200;
	
} configuration_params;

void dev1();
void dev2();
void dev3(configuration &params);
void dev4();
void dev5(configuration& params);
void dev6(configuration& params);

int main()
{
	torch::xpu::manual_seed(1337);
	
	//dev1();
	//dev2();
	//dev3(configuration_params);
	//dev4();
	//dev5(configuration_params);
	dev6(configuration_params);
	
	std::cout << "The end\n";
	return 0;
}

void dev6(configuration& params)
{
	std::cout << "dev6\n\n";

	torch::Device device(torch::kCUDA);

	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size());
		std::set<char> inputSet;

		for (const auto it : inputBuffer)
			inputSet.insert(it);

		encode encodeO(inputSet);

		torch::Tensor data = torch::tensor(encodeO(inputBuffer), torch::kLong);

		size_t trainSize = data.size(0) * params.train_percent;
		size_t valSize = data.size(0) - trainSize;

		std::cout << std::format("Train data size: {}\n", trainSize);
		std::cout << std::format("Validation data size: {}\n", valSize);
		torch::Tensor train_data = data.slice(0, 0, trainSize);
		torch::Tensor val_data = data.slice(0, trainSize, trainSize + valSize);

		std::cout << "Check point 1\n";

		auto get_batch = [&](std::string_view split) -> std::pair<torch::Tensor, torch::Tensor>
			{
				torch::Tensor current_data;

				if (split == "train")
				{
					current_data = train_data;
				}
				else
				{
					current_data = val_data;
				}

				// Generate random starting indices for the batch
				auto ix = torch::randint(0, current_data.size(0) - params.block_size, { params.batch_size }, torch::kLong);
				auto ix_accessor = ix.accessor<int64_t, 1>();

				std::vector<torch::Tensor> x_list;
				std::vector<torch::Tensor> y_list;
				x_list.reserve(params.batch_size);
				y_list.reserve(params.batch_size);

				for (int64_t i = 0; i < params.batch_size; ++i)
				{
					int64_t idx = ix_accessor[i];
					x_list.push_back(current_data.slice(0, idx, idx + params.block_size));
					y_list.push_back(current_data.slice(0, idx + 1, idx + params.block_size + 1));
				}

				auto x = torch::stack(x_list).to(device);
				auto y = torch::stack(y_list).to(device);

				return { x, y };
			};

		std::cout << "Check point 2\n";

		BigramLanguageModel3 m(65, params.head_size, params.block_size, params.n_embd);

		std::cout << "Check point 3\n";
		m.to(device);

		std::cout << "Check point 4\n";

		auto optimizer = torch::optim::Adam(m.parameters(), params.learning_rate);

		std::cout << "Check point 5\n";

		auto estimate_loss = [&](int iterations) -> std::unordered_map<std::string, float>
			{
				std::unordered_map<std::string, float> out;
				m.eval();  //switch to evaluation model;

				{
					torch::NoGradGuard no_grad;

					for (const auto& split : std::array{ "train", "val" })
					{
						float total_loss = 0.0f;

						for (int k = 0; k < iterations; k++)
						{
							auto [xb, yb] = get_batch(split);
							//std::cout << "Check point 7\n";
							auto [logits, loss] = m.forward(xb, yb);
							//std::cout << "Check point 8\n";
							total_loss += loss.item<float>();
						}
						out[split] = total_loss / iterations;
					}
				}

				m.train();  //switch back to train mode

				return out;
			};

		std::cout << "Check point 6\n";
		for (int step = 0; step < params.train_steps; step++)
		{
			if (step % params.train_loss_print_every_num_iter == 0)
			{
				auto losses = estimate_loss(params.estimate_loss_iterations);
				std::cout << "step:  " << step << ", " << "losses train: " << losses["train"] << " val: " << losses["val"] << std::endl;
			}

			auto [xb, yb] = get_batch("train");
			auto [logits, loss] = m.forward(xb, yb);

			optimizer.zero_grad(true);

			loss.backward();
			optimizer.step();
		}
		
		std::cout << "Check point 7\n";

		auto idx = torch::zeros({ 1,8 }, torch::kLong).to(device);

		idx = m.generate(idx, params.max_new_tokens);

		auto cpu_tensor = idx.to(torch::kCPU).contiguous();
		int64_t* ptr = cpu_tensor.data_ptr<int64_t>();
		std::vector<int64_t> vec(ptr, ptr + cpu_tensor.numel());

		decode<int64_t> decodeO(inputSet);

		auto output = decodeO(vec);

		for (char c : output)
		{
			std::cout << c;
		}

		std::cout << std::endl;
	}
	catch (std::exception& ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

void dev5(configuration& params)
{
	std::cout << "dev5\n\n";
	
	int64_t B = 4;
    int64_t T = 8;
    int64_t C = 32;
    
    auto x = torch::randn({B, T, C});
    
    int64_t head_size = 16;
    auto key_options = torch::nn::LinearOptions(C, head_size).bias(false);
    torch::nn::Linear key(key_options);
    auto query_options = torch::nn::LinearOptions(C, head_size).bias(false);
    torch::nn::Linear query(query_options);
	auto value_options = torch::nn::LinearOptions(C, head_size).bias(false);
	torch::nn::Linear value(value_options);
    auto k = key(x);
    auto q = query(x);
    
    auto wei = q.matmul(k.transpose(-2,-1)) * sqrt(head_size);
   	
	auto tril = torch::ones({T,T});
	tril = torch::tril(tril);
	
	//std::cout << "tril:\n";
	//std::cout << tril << std::endl;
	
	wei = wei.masked_fill(tril == 0, -std::numeric_limits<float>::infinity());
	wei = torch::nn::functional::softmax(wei, -1);

	auto v = value(x);
	auto out = wei.matmul(v);
	
	std::cout << "wei:\n";
	std::cout << wei << std::endl;

	std::cout << "out:\n";
	std::cout << out << std::endl;
	
	printTensorShape(out);
}

void dev4()
{
	std::cout << "dev4\n\n";
	
	int64_t B = 4;
    int64_t T = 8;
    int64_t C = 2;
    
    auto x = torch::randn({B, T, C});
    
    auto xbow = torch::zeros({B,T,C});
	
	for (int64_t b = 0; b < B; b++)
	{
		for (int64_t t = 0; t < T; t++)
		{
			auto x_b = x[b];
			auto xprev = x_b.slice(0, 0, t + 1);
			auto mean_val = torch::mean(xprev, 0);
			xbow.index_put_({b, t}, mean_val);
		}
	}
	
	std::cout << "XBOW\n";
	std::cout << xbow << std::endl;
	
	auto tril = torch::ones({T,T});
	tril = torch::tril(tril);
	auto wei = tril.masked_fill(tril == 0, -std::numeric_limits<float>::infinity());
	wei = torch::nn::functional::softmax(wei, -1);
	auto xbow2 = wei.matmul(x);
	
	std::cout << "XBOW2\n";
	std::cout << xbow2 << std::endl;
}

void dev3(configuration &params)
{
	std::cout << "dev3\n\n";
	
	torch::Device device(torch::kCUDA);
	
	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
		
		encode encodeO(inputSet);
		
		torch::Tensor data = torch::tensor(encodeO(inputBuffer), torch::kLong);
		
		size_t trainSize = data.size(0) * params.train_percent;
		size_t valSize = data.size(0) - trainSize;
		
		std::cout << std::format("Train data size: {}\n", trainSize);
		std::cout << std::format("Validation data size: {}\n", valSize);
		torch::Tensor train_data = data.slice(0, 0, trainSize);
		torch::Tensor val_data = data.slice(0,trainSize,trainSize + valSize);

		auto get_batch = [&](std::string_view split) -> std::pair<torch::Tensor, torch::Tensor>
		{
			torch::Tensor current_data;
			
			if (split == "train")
			{
				current_data = train_data;
			}
			else
			{
				current_data = val_data;
			}
			
			// Generate random starting indices for the batch
			auto ix = torch::randint(0, current_data.size(0) - params.block_size, {params.batch_size}, torch::kLong);
			auto ix_accessor = ix.accessor<int64_t, 1>();
			
			std::vector<torch::Tensor> x_list;
			std::vector<torch::Tensor> y_list;
			x_list.reserve(params.batch_size);
			y_list.reserve(params.batch_size);
			
			for (int64_t i = 0; i < params.batch_size; ++i)
			{
				int64_t idx = ix_accessor[i];
				x_list.push_back(current_data.slice(0, idx, idx + params.block_size));
				y_list.push_back(current_data.slice(0, idx + 1, idx + params.block_size + 1));
			}
			
			auto x = torch::stack(x_list).to(device);
			auto y = torch::stack(y_list).to(device);
			
			return {x, y};
		};
    
		BigramLanguageModel m(65);
		m.to(device);
	
		auto optimizer = torch::optim::Adam(m.parameters(), params.learning_rate);
		
		
		auto estimate_loss = [&](int iterations) -> std::unordered_map<std::string, float>
		{
			std::unordered_map<std::string, float> out;
			m.eval();  //switch to evaluation model;
			
			{
				torch::NoGradGuard no_grad;
				
				for (const auto& split: std::array{"train", "val"})
				{
					float total_loss = 0.0f;
					
					for (int k = 0; k < iterations; k++)
					{
						auto [xb, yb] = get_batch(split);
						auto [logits, loss] = m.forward(xb, yb);
						total_loss += loss.item<float>();
					}
					out[split] = total_loss / iterations;
				}
			}
			
			m.train();  //switch back to train mode
			
			return out;
		};
		
		
		for (int step = 0; step < params.train_steps; step++)
		{
			if (step % params.train_loss_print_every_num_iter == 0)
			{
				auto losses = estimate_loss(params.estimate_loss_iterations); 
				std::cout << "step:  " << step << ", " << "losses train: " << losses["train"] << " val: " << losses["val"] << std::endl;
			}
			
			auto [xb, yb] = get_batch("train");
			auto [logits, loss] =  m.forward(xb, yb);
			
			optimizer.zero_grad(true);
			
			loss.backward();
			optimizer.step();
		}
		
		auto idx = torch::zeros({1,1}, torch::kLong).to(device);
		
		idx = m.generate(idx, params.max_new_tokens);
		
		auto cpu_tensor = idx.to(torch::kCPU).contiguous();
		int64_t* ptr = cpu_tensor.data_ptr<int64_t>();
		std::vector<int64_t> vec(ptr, ptr + cpu_tensor.numel());
		
		decode<int64_t> decodeO(inputSet);
		
		auto output = decodeO(vec);
		
		for (char c : output)
		{
			std::cout << c;
		}
		
		std::cout << std::endl;
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

void dev2()
{
	std::cout << "dev2\n\n";
	
	try
	{
		int max_new_tokens = 100;	
		auto idx = torch::zeros({1,1},torch::kLong);
		
		auto m = BigramLanguageModel(65);
		idx = m.generate(idx, max_new_tokens);
		
		auto cpu_tensor = idx.to(torch::kCPU).contiguous();
		int64_t* ptr = cpu_tensor.data_ptr<int64_t>();
		std::vector<int64_t> vec(ptr, ptr + cpu_tensor.numel());
		
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
		
		decode<int64_t> decodeO(inputSet);
		
		auto output = decodeO(vec);
		
		for (char c : output)
		{
			std::cout << c;
		}
		
		std::cout << std::endl;
		
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

void dev1()
{
	std::cout << "dev1\n\n";
	
	int batch_size = 4;
	int block_size = 8;
	
	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::cout << std::format("Number of characters: {}\n", inputBuffer.size()); 
		std::set<char> inputSet;
		
		for (const auto it : inputBuffer)
			inputSet.insert(it);
			
		encode encodeO(inputSet);
		decode decodeO(inputSet);
	
		torch::Tensor data = torch::tensor(encodeO(inputBuffer), torch::kLong);
		
		printTensorShape(data);
		std::cout << "Data type: " << data.dtype() << std::endl;
		
		size_t trainSize = data.size(0) * 0.9;
		size_t valSize = data.size(0) - trainSize;
		
		std::cout << std::format("Train data size: {}\n", trainSize);
		std::cout << std::format("Validation data size: {}\n", valSize);
		torch::Tensor train_data = data.slice(0,0,trainSize);
		torch::Tensor val_data = data.slice(0,trainSize,trainSize + valSize);

		auto get_batch = [&](std::string_view split) -> std::pair<torch::Tensor, torch::Tensor>
		{
			torch::Tensor current_data;
			
			if (split == "train")
			{
				current_data = train_data;
			}
			else
			{
				current_data = val_data;
			}
			
			// Generate random starting indices for the batch
			auto ix = torch::randint(0, current_data.size(0) - block_size, {batch_size}, torch::kLong);
			auto ix_accessor = ix.accessor<int64_t, 1>();
			
			std::vector<torch::Tensor> x_list;
			std::vector<torch::Tensor> y_list;
			x_list.reserve(batch_size);
			y_list.reserve(batch_size);
			
			for (int64_t i = 0; i < batch_size; ++i)
			{
				int64_t idx = ix_accessor[i];
				x_list.push_back(current_data.slice(0, idx, idx + block_size));
				y_list.push_back(current_data.slice(0, idx + 1, idx + block_size + 1));
			}
			
			auto x = torch::stack(x_list);
			auto y = torch::stack(y_list);
			
			return {x, y};
		};
    
		// Use square brackets for C++ structured binding
		auto [xb, yb] = get_batch("train");
		
		std::cout << "xb:\n" << xb << std::endl;
		std::cout << "yb:\n" << yb << std::endl;
		
		auto m = BigramLanguageModel(65);
		auto [logits, loss] =  m.forward(xb, yb);
		
		std::cout << "Logits sizes: \n";
		printTensorShape(logits);
		
		//std::cout << "logits:\n" << logits << std::endl;
		std::cout << "loss:\n" << loss << std::endl;
		
		auto idx = xb;
		
		idx = m.generate(idx, 1);
		std::cout << "New ids: \n";
		std::cout << idx << std::endl;
		
	}
	catch (std::exception &ex)
	{
		std::cout << std::format("Exception: {}", ex.what());
	}
}

