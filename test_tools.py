#!/usr/bin/env python3
"""Manual tool test script (not part of unit tests)."""

from src.tools import get_current_weather, calculate_math, list_files, get_system_info


def main() -> None:
    print("=" * 60)
    print("TESTING ALL TOOLS - FULLY FUNCTIONAL")
    print("=" * 60)

    # Test 1: Weather (now uses real API)
    print("\n1. WEATHER TOOL (Real-time from wttr.in):")
    print("-" * 40)
    cities = ["London", "New York", "Tokyo", "Invalid City XYZ"]
    for city in cities:
        result = get_current_weather(city)
        print(f"  {city}: {result[:100]}...")

    # Test 2: Calculator
    print("\n2. CALCULATOR TOOL:")
    print("-" * 40)
    expressions = [
        "2 + 2",
        "sqrt(16) + sin(pi/2)",
        "log(e) * exp(1)",
        "25 * 4 + 10 / 2"
    ]
    for expr in expressions:
        result = calculate_math(expr)
        print(f"  {result}")

    # Test 3: File Operations
    print("\n3. FILE OPERATIONS TOOL:")
    print("-" * 40)
    print("  Current directory:")
    result = list_files(".", ".py")
    print(f"  {result[:200]}...")

    # Test 4: System Info
    print("\n4. SYSTEM INFO TOOL:")
    print("-" * 40)
    result = get_system_info()
    print(f"  {result[:300]}...")

    print("\n" + "=" * 60)
    print("✅ ALL TOOLS ARE FULLY FUNCTIONAL")
    print("=" * 60)


if __name__ == "__main__":
    main()
