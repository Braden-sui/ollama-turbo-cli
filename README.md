# Ollama Turbo CLI

A production-ready CLI application for interacting with OpenAI's gpt-oss:120b model through Ollama Turbo cloud service, featuring advanced tool calling capabilities and streaming responses.

## Features

- 🚀 **Ollama Turbo Integration**: Connect to gpt-oss:120b on datacenter-grade hardware
- 🔧 **Advanced Tool Calling**: Weather, calculator, file operations, system info
- 📡 **Streaming Responses**: Real-time response streaming with tool execution
- 💬 **Interactive Mode**: Continuous conversation with history management
- 🛡️ **Error Handling**: Robust retry logic and graceful error recovery
- ⚙️ **Configurable**: Environment variables and command-line options

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd ollama-turbo-cli
pip install -r requirements.txt
```

### 2. Get API Key

1. Visit [ollama.com/turbo](https://ollama.com/turbo)
2. Sign up for Ollama Turbo ($20/month)
3. Get your API key from [ollama.com/settings/keys](https://ollama.com/settings/keys)

### 3. Setup Environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Run

```bash
# Interactive mode
python -m src.cli --api-key YOUR_API_KEY

# Single message
python -m src.cli --api-key YOUR_API_KEY --message "What's 15 * 8?"

# With streaming
python -m src.cli --message "Weather in London" --stream
```

## Available Tools

### 🌤️ Weather Service
Get current weather for major cities worldwide.
```
"What's the weather in Tokyo?"
```

### 🧮 Mathematical Calculator
Perform complex mathematical calculations with support for functions.
```
"Calculate sin(pi/2) + sqrt(16)"
```

### 📁 File Operations
List files and directories with filtering options.
```
"List all Python files in the current directory"
```

### 💻 System Information
Get comprehensive system information including CPU, memory, and disk usage.
```
"Show me system information"
```

## Usage Examples

### Interactive Mode
```bash
python -m src.cli --api-key sk-your-key-here

👤 You: What's the weather in London and calculate 25 * 4?

🔧 Processing tool calls...
   1. Executing get_current_weather(city=London, unit=celsius)
      ✅ Result: Weather in London: Partly cloudy, 15°C, Humidity: 65%, Wind: 12 mph
   2. Executing calculate_math(expression=25 * 4)
      ✅ Result: Result: 25 * 4 = 100

🤖 Final response: Based on the results, London currently has partly cloudy weather at 15°C, and 25 multiplied by 4 equals 100.
```

### Single Message Mode
```bash
python -m src.cli --api-key sk-your-key --message "List files in /home/user/documents"
```

### Streaming Mode
```bash
python -m src.cli --message "Explain quantum computing" --stream
```

## Configuration

### Environment Variables
- `OLLAMA_API_KEY`: Your Ollama API key
- `OLLAMA_MODEL`: Model name (default: gpt-oss:120b)
- `OLLAMA_HOST`: API endpoint (default: https://ollama.com)
- `MAX_CONVERSATION_HISTORY`: Maximum messages to keep (default: 50)
- `STREAM_BY_DEFAULT`: Enable streaming by default (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)

### Command Line Options
```
--api-key        Ollama API key
--model          Model name (default: gpt-oss:120b)
--message        Single message mode
--stream         Enable streaming
--no-tools       Disable tool calling
--log-level      Set logging level
--version        Show version
--help           Show help
```

## Interactive Commands

While in interactive mode:
- `quit` or `exit`: Exit the application
- `clear`: Clear conversation history
- `history`: Show conversation history
- `Ctrl+C`: Exit gracefully

## Error Handling

The application includes robust error handling:
- **Network Issues**: Automatic retry with exponential backoff
- **API Errors**: Clear error messages and recovery
- **Tool Failures**: Graceful degradation with error reporting
- **Invalid Input**: Input validation and helpful suggestions

## Requirements

- Python 3.8+
- Ollama Turbo subscription ($20/month)
- Internet connection for API calls
- 4GB+ RAM recommended

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review error logs with `--log-level DEBUG`
3. Verify API key and subscription status
4. Open an issue on GitHub

## Troubleshooting

### Common Issues

#### API Key Errors
- Ensure your API key is correctly set in `.env` or via `--api-key`
- Verify the key is active at [ollama.com/settings/keys](https://ollama.com/settings/keys)

#### Connection Issues
- Check your internet connection
- Verify Ollama Turbo service status
- Try with `--log-level DEBUG` for detailed error messages

#### Tool Execution Failures
- Some tools require specific permissions (e.g., file operations)
- System info tool requires `psutil` package
- Weather data is available for major cities only

#### Performance
- Streaming mode provides better perceived performance
- Tool calls may add latency for complex operations
- Consider adjusting `MAX_CONVERSATION_HISTORY` for memory usage

## Development

### Project Structure
```
ollama-turbo-cli/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Main CLI application
│   ├── client.py           # Ollama Turbo client wrapper
│   ├── tools.py            # Tool definitions and implementations
│   └── utils.py            # Utility functions
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup configuration
├── README.md              # This file
├── .env.example           # Environment variables template
└── LICENSE                # MIT license
```

### Running Tests
```bash
# Test weather tool
python -m src.cli --message "What's the weather in Paris?"

# Test calculator
python -m src.cli --message "Calculate sqrt(144) + sin(pi/2)"

# Test file operations
python -m src.cli --message "List Python files in the current directory"

# Test system info
python -m src.cli --message "Show system information"

# Test multiple tools
python -m src.cli --message "Weather in London and calculate 15 * 8"
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Version History

### v1.0.0 (Current)
- Initial release with gpt-oss:120b support
- Four built-in tools: weather, calculator, files, system
- Streaming and non-streaming modes
- Interactive and single-message modes
- Comprehensive error handling and retry logic
