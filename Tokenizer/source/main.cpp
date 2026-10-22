
#include <stdint.h>
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
std::vector<uint32_t> readInputData32();
std::vector<uint32_t> readInputData32NoCast();


int main()
{
#ifdef _WIN32
	UINT old_cp = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
#endif

	try
	{
		//std::vector<char> inputBuffer = readInputData();
		//std::vector<uint32_t> code_points;
		//utf8::utf8to32(inputBuffer.begin(), inputBuffer.end(), std::back_inserter(code_points));

		std::vector<uint32_t> code_points = readInputData32();
		std::cout << "Total utf points: " << code_points.size() << std::endl;

		BPE_Tokenizer tokenizer(code_points);

		std::cout << std::format("maximum token number: {}\n", tokenizer.GetMaxTokenNumber());

		uint32_t max_token_num = 0;

		do
		{
			tokenizer.TrainStep();

			max_token_num = tokenizer.GetMaxTokenNumber();
			
			if (max_token_num %50 == 0)
				std::cout << std::format("maximum token number: {}\n", max_token_num);
		} while (max_token_num < 600);

		std::cout << std::format("maximum token number: {}\n", tokenizer.GetMaxTokenNumber());
		std::cout << std::format("vector size: {}\n", tokenizer.GetVectorSize());

		auto vocabluary = tokenizer.GetVocabluary();

		for (auto& p : vocabluary)
		{
			std::cout << std::format("{} : {}\n", p.first, p.second);
		}
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

std::vector<uint32_t> readInputData32()
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

	// Use a std::string buffer so iterators work natively with utf8cpp
	std::string buffer(size, '\0');
	if (!inputFile.read(buffer.data(), size))
	{
		throw std::runtime_error("Failed to read file input.txt");
	}

	std::vector<uint32_t> code_points;
	try
	{
		// Safely decode raw UTF-8 bytes into 32-bit Unicode code points
		utf8::utf8to32(buffer.begin(), buffer.end(), std::back_inserter(code_points));
	}
	catch (const std::exception& e)
	{
		std::cout << std::format("UTF-8 decoding error: {}\n", e.what());
		throw;
	}

	std::cout << std::format("Successfully decoded {} code points.\n", code_points.size());
	return code_points;
}

std::vector<uint32_t> readInputData32NoCast()
{
	std::ifstream inputFile("input.txt", std::ios::binary);
	if (!inputFile.is_open())
	{
		throw std::runtime_error("Unable to open input.txt");
	}

	inputFile.seekg(0, std::ios::end);
	std::streamsize size = inputFile.tellg();
	inputFile.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!inputFile.read(buffer.data(), size))
	{
		throw std::runtime_error("Failed to read file input.txt");
	}

	std::vector<uint32_t> code_points;
	code_points.reserve(size);

	// Direct byte-to-code-point mapping for Windows-1252 / Latin-1 files
	for (char c : buffer)
	{
		code_points.push_back(static_cast<uint32_t>(static_cast<unsigned char>(c)));
	}

	std::cout << std::format("Loaded {} code points from Windows-1252 file.\n", code_points.size());
	return code_points;
}
