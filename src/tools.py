"""
Tool implementations and schemas for Ollama Turbo CLI.
"""

import os
import re
import sys
import math
import platform
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs, unquote, urljoin

try:
    import requests
except ImportError:
    requests = None

try:
    import psutil
except ImportError:
    psutil = None


def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather using real data from wttr.in with no static fallback.

    This function requires network access and the 'requests' library. It will return
    a clear error message if the lookup fails for any reason.
    """
    try:
        city = (city or "").strip()
        if not city:
            return "Error: city must be provided"

        # Validate unit
        unit_l = (unit or "celsius").lower()
        if unit_l not in ("celsius", "fahrenheit"):
            return "Error: unit must be 'celsius' or 'fahrenheit'"

        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use the weather tool."

        # Use wttr.in JSON API (no API key required)
        url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
        headers = {"User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)"}
        try:
            response = requests.get(url, timeout=8, headers=headers)
        except requests.RequestException as e:
            return f"Error fetching weather for '{city}': network error: {e}"

        if response.status_code != 200:
            return f"Error fetching weather for '{city}': HTTP {response.status_code}"

        try:
            data = response.json()
        except json.JSONDecodeError:
            return f"Error fetching weather for '{city}': invalid JSON response"

        try:
            current = data["current_condition"][0]
            condition = current["weatherDesc"][0]["value"]
            humidity = current["humidity"]
            wind_mph = current["windspeedMiles"]
            wind_dir = current.get("winddir16Point") or current.get("winddirDegree") or "N/A"
            if unit_l == "fahrenheit":
                temp = current["temp_F"]
                feels_like = current["FeelsLikeF"]
                unit_symbol = "°F"
            else:
                temp = current["temp_C"]
                feels_like = current["FeelsLikeC"]
                unit_symbol = "°C"
        except (KeyError, IndexError, TypeError):
            return f"Error: unexpected API response while fetching weather for '{city}'"

        return (
            f"Weather in {city.title()}: {condition}, {temp}{unit_symbol} "
            f"(feels like {feels_like}{unit_symbol}), "
            f"Humidity: {humidity}%, Wind: {wind_mph} mph {wind_dir}"
        )
    except Exception as e:
        return f"Error getting weather for '{city}': {str(e)}"


def calculate_math(expression: str) -> str:
    """Safe mathematical expression evaluator with comprehensive operations."""
    try:
        # Remove whitespace
        expression_orig = expression
        expression = expression.strip()
        
        # Define allowed characters for initial check
        basic_pattern = r'^[0-9+\-*/^().\s]+$'
        
        # Process the expression - handle special functions and constants
        expression_clean = expression.lower()
        
        # Replace math constants
        expression_clean = expression_clean.replace('pi', str(math.pi))
        expression_clean = expression_clean.replace('e', str(math.e))
        
        # Handle math functions with proper Python syntax
        expression_clean = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expression_clean)
        expression_clean = re.sub(r'sin\(([^)]+)\)', r'math.sin(\1)', expression_clean)
        expression_clean = re.sub(r'cos\(([^)]+)\)', r'math.cos(\1)', expression_clean)
        expression_clean = re.sub(r'tan\(([^)]+)\)', r'math.tan(\1)', expression_clean)
        expression_clean = re.sub(r'log\(([^)]+)\)', r'math.log(\1)', expression_clean)
        expression_clean = re.sub(r'log10\(([^)]+)\)', r'math.log10(\1)', expression_clean)
        expression_clean = re.sub(r'exp\(([^)]+)\)', r'math.exp(\1)', expression_clean)
        expression_clean = re.sub(r'abs\(([^)]+)\)', r'abs(\1)', expression_clean)
        expression_clean = re.sub(r'pow\(([^,]+),([^)]+)\)', r'pow(\1,\2)', expression_clean)
        
        # Replace ^ with ** for Python power operator
        expression_clean = expression_clean.replace('^', '**')
        
        # Check if only allowed characters remain after removing math. prefix
        test_expr = expression_clean.replace('math.', '').replace('abs', '').replace('pow', '')
        if not re.match(basic_pattern, test_expr):
            return f"Error: Invalid characters in expression '{expression_orig}'. Allowed: numbers, +, -, *, /, ^, (), ., sin, cos, tan, sqrt, log, exp, pi, e, abs, pow"
        
        # Evaluate the expression safely using eval with restricted namespace
        safe_dict = {
            "math": math,
            "abs": abs,
            "pow": pow,
            "__builtins__": {}
        }
        
        result = eval(expression_clean, safe_dict)
        
        # Format result nicely
        if isinstance(result, float):
            if abs(result - round(result)) < 1e-10:
                result = int(round(result))
            else:
                result = round(result, 8)
        
        return f"Result: {expression_orig} = {result}"
        
    except ZeroDivisionError:
        return f"Error: Division by zero in expression '{expression}'"
    except ValueError as e:
        return f"Error: Invalid mathematical operation in '{expression}': {str(e)}"
    except SyntaxError:
        return f"Error: Invalid syntax in expression '{expression}'"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def list_files(directory: str = ".", extension: Optional[str] = None) -> str:
    """List files in directory with optional extension filter."""
    try:
        # Resolve directory path
        directory = os.path.abspath(directory)
        
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist"
        
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory"
        
        # Get all items in directory
        try:
            items = os.listdir(directory)
        except PermissionError:
            return f"Error: Permission denied accessing '{directory}'"
        
        files = []
        dirs = []
        
        for item in items:
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    # Check extension filter
                    if extension is None or item.lower().endswith(extension.lower()):
                        # Get file size and modification time
                        size = os.path.getsize(item_path)
                        mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
                        
                        # Format size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024*1024:
                            size_str = f"{size/1024:.1f} KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f} MB"
                        
                        files.append(f"{item} ({size_str}, {mtime.strftime('%Y-%m-%d %H:%M')})")
                elif os.path.isdir(item_path):
                    if extension is None:  # Only show directories if no extension filter
                        dirs.append(f"{item}/")
            except (OSError, PermissionError):
                continue  # Skip inaccessible items
        
        # Sort files and directories
        files.sort()
        dirs.sort()
        
        result_items = dirs + files
        
        if not result_items:
            filter_msg = f" with extension '{extension}'" if extension else ""
            return f"No files found in '{directory}'{filter_msg}"
        
        # Limit output to prevent overwhelming response
        total_items = len(result_items)
        display_items = result_items[:15]
        
        result = f"Contents of '{directory}':\n"
        result += "\n".join(f"  {item}" for item in display_items)
        
        if total_items > 15:
            result += f"\n  ... and {total_items - 15} more items"
        
        return result
        
    except Exception as e:
        return f"Error listing files in '{directory}': {str(e)}"


def get_system_info() -> str:
    """Get comprehensive system information."""
    try:
        # Basic system info
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor() or "Unknown",
            "Python Version": platform.python_version(),
            "Python Executable": sys.executable,
            "Current Directory": os.getcwd(),
            "User": os.getenv('USER') or os.getenv('USERNAME') or "Unknown",
            "Hostname": platform.node(),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Memory information
        if psutil:
            try:
                memory = psutil.virtual_memory()
                system_info["Total RAM"] = f"{memory.total / (1024**3):.1f} GB"
                system_info["Available RAM"] = f"{memory.available / (1024**3):.1f} GB"
                system_info["RAM Usage"] = f"{memory.percent}%"
            except:
                system_info["Memory"] = "Unable to retrieve"
            
            # Disk information
            try:
                disk = psutil.disk_usage('/')
                system_info["Total Disk"] = f"{disk.total / (1024**3):.1f} GB"
                system_info["Free Disk"] = f"{disk.free / (1024**3):.1f} GB"
                system_info["Disk Usage"] = f"{(disk.used / disk.total) * 100:.1f}%"
            except:
                system_info["Disk"] = "Unable to retrieve"
            
            # CPU information
            try:
                system_info["CPU Cores"] = psutil.cpu_count(logical=False)
                system_info["CPU Threads"] = psutil.cpu_count(logical=True)
                system_info["CPU Usage"] = f"{psutil.cpu_percent(interval=1)}%"
            except:
                system_info["CPU"] = "Unable to retrieve"
        else:
            system_info["System Stats"] = "psutil not installed - install for detailed system info"
        
        # Format output
        result = "System Information:\n"
        for key, value in system_info.items():
            result += f"  {key}: {value}\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error retrieving system information: {str(e)}"


def web_fetch(url: str,
              method: str = "GET",
              params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None,
              timeout: float = 10,
              max_bytes: int = 8192,
              allow_redirects: bool = True,
              as_json: bool = False) -> str:
    """Fetch live web content from an HTTP/HTTPS URL.

    - Supports GET/HEAD with optional params and headers
    - Returns status, final URL, content type, and a truncated body snippet
    - If as_json is True, attempts to parse and return JSON
    """
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use web_fetch."

        url = (url or "").strip()
        if not url:
            return "Error: url must be provided"
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: url must start with http:// or https://"

        method_u = (method or "GET").upper()
        if method_u not in ("GET", "HEAD"):
            return "Error: method must be 'GET' or 'HEAD'"

        # Validate numeric bounds
        try:
            timeout = float(timeout)
            if timeout < 1 or timeout > 60:
                return "Error: timeout must be between 1 and 60 seconds"
        except Exception:
            return "Error: timeout must be a number"

        try:
            max_bytes = int(max_bytes)
            if max_bytes < 256 or max_bytes > 1048576:
                return "Error: max_bytes must be between 256 and 1048576"
        except Exception:
            return "Error: max_bytes must be an integer"

        # Merge headers with a sensible User-Agent
        hdrs = {
            "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)"
        }
        if isinstance(headers, dict):
            # Keep only str->str
            for k, v in headers.items():
                try:
                    hdrs[str(k)] = str(v)
                except Exception:
                    continue

        try:
            resp = requests.request(
                method=method_u,
                url=url,
                params=params if isinstance(params, dict) else None,
                headers=hdrs,
                timeout=timeout,
                allow_redirects=bool(allow_redirects)
            )
        except requests.RequestException as e:
            return f"Error fetching URL '{url}': network error: {e}"

        status = resp.status_code
        final_url = resp.url
        ctype = resp.headers.get("Content-Type", "")

        # HEAD has no body
        if method_u == "HEAD":
            return (
                f"HTTP {status}\n"
                f"Final URL: {final_url}\n"
                f"Content-Type: {ctype or 'unknown'}\n"
                f"Note: HEAD request has no body"
            )

        if as_json:
            try:
                data = resp.json()
                # Truncate JSON string representation if large
                body_str = json.dumps(data, ensure_ascii=False)[:max_bytes]
                if len(json.dumps(data, ensure_ascii=False)) > max_bytes:
                    body_str += f"... [truncated]"
                return (
                    f"HTTP {status}\n"
                    f"Final URL: {final_url}\n"
                    f"Content-Type: {ctype or 'unknown'}\n"
                    f"--- JSON Body (first {max_bytes} chars) ---\n"
                    f"{body_str}"
                )
            except ValueError:
                return (
                    f"HTTP {status}\n"
                    f"Final URL: {final_url}\n"
                    f"Content-Type: {ctype or 'unknown'}\n"
                    f"Error: Response is not valid JSON"
                )

        # Text mode
        body = resp.text or ""
        truncated = False
        if len(body) > max_bytes:
            body = body[:max_bytes] + "... [truncated]"
            truncated = True

        return (
            f"HTTP {status}\n"
            f"Final URL: {final_url}\n"
            f"Content-Type: {ctype or 'unknown'}\n"
            f"--- Body (first {max_bytes} chars) ---\n"
            f"{body}"
        )
    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"

def duckduckgo_search(query: str, max_results: int = 3) -> str:
    """Search using DuckDuckGo Instant Answer API (no API key required).

    Returns top results with title, URL, and snippet when available.
    """
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use duckduckgo_search."

        query = (query or "").strip()
        if not query:
            return "Error: query must be provided"

        try:
            max_results = int(max_results)
            if max_results < 1:
                max_results = 1
            if max_results > 5:
                max_results = 5
        except Exception:
            max_results = 3

        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "t": "ollama-turbo-cli"
        }
        headers = {
            "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)",
            "Accept": "application/json"
        }

        # Try API with small retries for transient statuses (e.g., 202, 429, 5xx)
        resp = None
        for attempt in range(5):
            try:
                resp = requests.get("https://api.duckduckgo.com/", params=params, headers=headers, timeout=8)
            except requests.RequestException as e:
                if attempt == 4:
                    return f"Error performing DuckDuckGo search: network error: {e}"
                time.sleep(0.5 * (2 ** attempt))
                continue
            if resp.status_code == 200:
                break
            if attempt < 4 and resp.status_code in (202, 429, 403, 500, 502, 503, 504):
                time.sleep(0.5 * (2 ** attempt))
                continue
            else:
                break

        if resp is not None and resp.status_code == 200:
            try:
                data = resp.json()
            except json.JSONDecodeError:
                return "Error performing DuckDuckGo search: invalid JSON response"
        else:
            # Fallback to HTML (Lite) endpoint and extract links
            html_headers = {
                "User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)",
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://duckduckgo.com/",
                "DNT": "1",
            }
            fallback_urls = [
                "https://duckduckgo.com/lite/",
                "https://html.duckduckgo.com/html/"
            ]
            collected = []
            for base in fallback_urls:
                try:
                    r = requests.get(base, params={"q": query}, headers=html_headers, timeout=8)
                except requests.RequestException:
                    continue
                if r.status_code != 200:
                    continue
                html = r.text or ""
                # Extract anchors; handle DDG redirect links (/l/?uddg=...)
                links = re.findall(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', html, flags=re.I)
                for href, text in links:
                    try:
                        absolute = href if href.lower().startswith("http") else urljoin(base, href)
                        parsed = urlparse(absolute)
                        netloc = parsed.netloc or ""
                    except Exception:
                        continue

                    resolved_url = None
                    if netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                        # Decode external URL from uddg param
                        try:
                            qs = parse_qs(parsed.query)
                            uddg = qs.get("uddg", [None])[0]
                            if uddg:
                                resolved_url = unquote(uddg)
                        except Exception:
                            pass
                    elif netloc.endswith("duckduckgo.com") or netloc.endswith("duck.com"):
                        # Skip other DDG internal links
                        continue
                    else:
                        resolved_url = absolute if absolute.lower().startswith("http") else None

                    if not resolved_url:
                        continue
                    if resolved_url in [c["url"] for c in collected]:
                        continue

                    title_text = re.sub(r"<[^>]+>", "", text)[:120].strip() or "(no title)"
                    snippet = re.sub(r"<[^>]+>", "", text)[:180].strip()
                    collected.append({"title": title_text, "url": resolved_url, "snippet": snippet})
                    if len(collected) >= max_results:
                        break
                if collected:
                    break

            if collected:
                out_lines = [f"DuckDuckGo (HTML fallback): Top {len(collected)} results for '{query}':"]
                for i, r in enumerate(collected[:max_results], 1):
                    title = re.sub(r"<[^>]+>", "", r.get("title") or "(no title)")
                    url = r.get("url") or ""
                    snippet = r.get("snippet") or ""
                    out_lines.append(f"{i}. {title} - {url}")
                    if snippet:
                        out_lines.append(f"   {snippet}")
                return "\n".join(out_lines)
            # If still nothing useful, report the last status
            code = resp.status_code if (resp is not None and hasattr(resp, 'status_code')) else 'unknown'
            return f"Error performing DuckDuckGo search: HTTP {code} and no fallback results"

        results = []

        # Prefer Instant Answer (Abstract/Answer)
        abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
        abstract_url = (data.get("AbstractURL") or "").strip()
        if abstract and abstract_url:
            results.append({"title": data.get("Heading") or "Instant Answer", "url": abstract_url, "snippet": abstract})

        # Flatten RelatedTopics
        def _flatten_topics(items):
            out = []
            for it in items or []:
                if isinstance(it, dict) and "FirstURL" in it:
                    out.append({
                        "title": (it.get("Text") or "").split(" - ")[0][:120],
                        "url": it.get("FirstURL"),
                        "snippet": it.get("Text") or ""
                    })
                elif isinstance(it, dict) and "Topics" in it:
                    out.extend(_flatten_topics(it.get("Topics") or []))
            return out

        results.extend(_flatten_topics(data.get("RelatedTopics")))

        # Deduplicate by URL and limit
        seen = set()
        unique = []
        for r in results:
            url = r.get("url")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)
            if len(unique) >= max_results:
                break

        if not unique:
            return f"DuckDuckGo: No results for '{query}'"

        out_lines = [f"DuckDuckGo: Top {len(unique)} results for '{query}':"]
        for i, r in enumerate(unique, 1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or ""
            snippet = r.get("snippet") or ""
            out_lines.append(f"{i}. {title} - {url}")
            if snippet:
                out_lines.append(f"   {snippet}")
        return "\n".join(out_lines)
    except Exception as e:
        return f"Error performing DuckDuckGo search: {str(e)}"


def wikipedia_search(query: str, limit: int = 3) -> str:
    """Search Wikipedia (no API key required) and return top results.

    Uses the MediaWiki search API and returns title, URL, and snippet.
    """
    try:
        if not requests:
            return "Error: Python 'requests' library is not installed. Install it to use wikipedia_search."

        query = (query or "").strip()
        if not query:
            return "Error: query must be provided"

        try:
            limit = int(limit)
            if limit < 1:
                limit = 1
            if limit > 5:
                limit = 5
        except Exception:
            limit = 3

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": "1",
            "srlimit": str(limit)
        }
        headers = {"User-Agent": "ollama-turbo-cli/1.0 (+https://ollama.com)"}
        try:
            resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=headers, timeout=8)
        except requests.RequestException as e:
            return f"Error performing Wikipedia search: network error: {e}"

        if resp.status_code != 200:
            return f"Error performing Wikipedia search: HTTP {resp.status_code}"

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return "Error performing Wikipedia search: invalid JSON response"

        search_results = (data.get("query") or {}).get("search") or []
        if not search_results:
            return f"Wikipedia: No results for '{query}'"

        def _strip_html(s: str) -> str:
            try:
                return re.sub(r"<[^>]+>", "", s)
            except Exception:
                return s

        lines = [f"Wikipedia: Top {min(len(search_results), limit)} results for '{query}':"]
        for i, item in enumerate(search_results[:limit], 1):
            title = item.get("title") or "(no title)"
            pageid = item.get("pageid")
            url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
            snippet = _strip_html(item.get("snippet") or "").replace("\n", " ")
            lines.append(f"{i}. {title} - {url}")
            if snippet:
                lines.append(f"   {snippet}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error performing Wikipedia search: {str(e)}"

# Tool schemas for Ollama API
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather information for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for (e.g., London, Paris, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Temperature unit - celsius or fahrenheit",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Evaluate mathematical expressions including basic operations and functions like sin, cos, sqrt, log, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'sqrt(16)')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories in a specified directory with optional extension filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list contents of",
                        "default": "."
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension filter (e.g., '.py', '.txt')"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get comprehensive system information including OS, CPU, memory, disk usage, and Python environment",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch live web content from a given URL (GET/HEAD). Returns status, headers, and a body snippet or JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to fetch"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "HEAD"],
                        "default": "GET"
                    },
                    "params": {
                        "type": "object",
                        "description": "Optional query parameters as key/value"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key/value"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds (1-60)",
                        "default": 10
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum number of body characters to return (256-1048576)",
                        "default": 8192
                    },
                    "allow_redirects": {
                        "type": "boolean",
                        "description": "Whether to follow redirects",
                        "default": True
                    },
                    "as_json": {
                        "type": "boolean",
                        "description": "If true, try to parse the response as JSON and return it",
                        "default": False
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "duckduckgo_search",
            "description": "Search the web using DuckDuckGo Instant Answer API (no API key). Returns top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results to return (1-5)", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia and return top results with title, URL, and snippet (no API key).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Number of results to return (1-5)", "default": 3}
                },
                "required": ["query"]
            }
        }
    }
]


# Tool function mapping
TOOL_FUNCTIONS = {
    "get_current_weather": get_current_weather,
    "calculate_math": calculate_math,
    "list_files": list_files,
    "get_system_info": get_system_info,
    "web_fetch": web_fetch,
    "duckduckgo_search": duckduckgo_search,
    "wikipedia_search": wikipedia_search
}
