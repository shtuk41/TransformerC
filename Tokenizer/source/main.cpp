
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

#include <utf8.h>

#include <tokenizer.h>


std::vector<char> readInputData();

int main()
{
#ifdef _WIN32
	UINT old_cp = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
#endif

	try
	{
		std::vector<char> inputBuffer = readInputData();
		std::vector<uint32_t> code_points;
		utf8::utf8to32(inputBuffer.begin(), inputBuffer.end(), std::back_inserter(code_points));
		std::cout << "Total utf points: " << code_points.size() << std::endl;

		BPE_Tokenizer tokenizer(code_points);

		std::cout << std::format("maximum token number: {}\n", tokenizer.GetMaxTokenNumber());

		uint32_t max_token_num = 0;

		do
		{
			tokenizer.TrainStep();

			max_token_num = tokenizer.GetMaxTokenNumber();
			//std::cout << std::format("maximum token number: {}\n", max_token_num);
		} while (max_token_num < 500);

		std::cout << std::format("maximum token number: {}\n", tokenizer.GetMaxTokenNumber());
		std::cout << std::format("vector size: {}\n", tokenizer.GetVectorSize());
	}
	catch (std::exception& e)
	{
		std::cout << "something failed: " << e.what() << std::endl;
	}

#ifdef _WIN32
	// 3. Set it back to normal right before the program terminates
	SetConsoleOutputCP(old_cp);
	std::cout << "\n[Restored Mode Test Check]: \xA3" << std::endl;
#endif

	//std::string text = "Привет, мир! ✓ ★";
    //size_t count = utf8::distance(text.begin(), text.end());
    //std::cout << text << "\n";
    //std::cout << "Character Count: " << count << "\n";

    return 0;
}

std::vector<char> readInputData()
{
	std::ifstream inputFile("input.txt", std::ios::binary);

	if (!inputFile.is_open())
	{
		std::cout << "Unable to open input.txt" << std::endl;
		throw std::runtime_error("unable to open input.txt");
	}

	inputFile.seekg(0, std::ios::end);
	std::streamsize size = inputFile.tellg();

	std::cout << std::format("input.txt size: {}\n", size);

	inputFile.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);

	if (inputFile.read(buffer.data(), size))
	{
		return buffer;
	}

	throw std::runtime_error("Failed to read file input.txt");
}