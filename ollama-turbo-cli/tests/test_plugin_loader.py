import unittest

from src import client as client_module
from src import plugin_loader


class TestPluginLoader(unittest.TestCase):
    def setUp(self):
        # Ensure plugins reloaded fresh for each test
        plugin_loader.reload_plugins()

    def test_discovery_includes_builtin_tools(self):
        names = set(plugin_loader.TOOL_FUNCTIONS.keys())
        expected = {
            "get_current_weather",
            "calculate_math",
            "list_files",
            "get_system_info",
            "web_fetch",
            "duckduckgo_search",
            "wikipedia_search",
        }
        missing = expected - names
        self.assertFalse(missing, f"Missing built-in plugins: {missing}")

    def test_discovery_includes_third_party_tools(self):
        names = set(plugin_loader.TOOL_FUNCTIONS.keys())
        self.assertIn("hello", names)
        self.assertIn("echo", names)

    def test_execution_echo_and_hello(self):
        funcs = plugin_loader.TOOL_FUNCTIONS
        self.assertEqual(funcs["hello"](), "Hello, world! 👋")
        self.assertEqual(funcs["hello"](name="Braden"), "Hello, Braden! 👋")
        self.assertEqual(funcs["echo"](text="test"), "test")
        self.assertEqual(funcs["echo"](text="abc", uppercase=True), "ABC")
        self.assertEqual(funcs["echo"](text="abc", reverse=True), "cba")

    def test_schema_validation(self):
        funcs = plugin_loader.TOOL_FUNCTIONS
        with self.assertRaises(Exception):
            # Missing required 'expression' for calculate_math
            funcs["calculate_math"]()
        # Valid call
        out = funcs["calculate_math"](expression="2 + 2")
        self.assertIn("= 4", out)

    def test_client_imports_plugin_loader(self):
        # Ensure client uses plugin loader aggregates
        self.assertTrue(hasattr(client_module, "OllamaTurboClient"))
        # plugin schemas should be a list
        self.assertIsInstance(plugin_loader.TOOL_SCHEMAS, list)
        # plugin functions dict should be non-empty
        self.assertTrue(plugin_loader.TOOL_FUNCTIONS)


if __name__ == "__main__":
    unittest.main()
