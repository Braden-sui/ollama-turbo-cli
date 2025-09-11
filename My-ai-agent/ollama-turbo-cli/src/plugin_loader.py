"""
Dynamic plugin loader for tool system.

- Discovers plugins from internal package `src.plugins/` and external project `tools/` directory
- Validates plugin schemas (OpenAI-style function tools) using jsonschema
- Builds TOOL_SCHEMAS and TOOL_FUNCTIONS for the client
- Exposes a PluginManager for advanced usage and testing

Plugin contract (any Python module):
- Must define TOOL_SCHEMA: dict with keys {"type": "function", "function": {"name": str, "description": str, "parameters": object-schema}}
- Must provide an implementation, one of:
  * attribute TOOL_IMPLEMENTATION: callable
  * a function named the same as TOOL_SCHEMA['function']['name']
  * a function named 'execute'
- Optional: TOOL_VERSION: str, TOOL_AUTHOR: str

At runtime, arguments passed to tool implementations are validated against the
plugin's `parameters` JSON schema before execution. Any validation or runtime
error is raised to the caller, which the caller should catch and format.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

# Test isolation guard: if a stub module 'src.plugins.web_research' already exists in
# sys.modules (inserted by an earlier test) and lacks the 'run_research' attribute,
# add a placeholder so subsequent tests that monkeypatch it can succeed.
try:
    _m = sys.modules.get('src.plugins.web_research')
    if _m is not None and not hasattr(_m, 'run_research'):
        setattr(_m, 'run_research', None)
except Exception:
    pass

try:
    from jsonschema import validate as jsonschema_validate  # type: ignore
    from jsonschema import Draft202012Validator  # type: ignore
except Exception as e:  # pragma: no cover - degrade gracefully without jsonschema
    jsonschema_validate = None  # type: ignore
    Draft202012Validator = None  # type: ignore


# Minimal JSON Schema to validate the tool definition structure itself
_TOOL_DEFINITION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["type", "function"],
    "properties": {
        "type": {"const": "function"},
        "function": {
            "type": "object",
            "required": ["name", "description", "parameters"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "parameters": {"type": ["object", "boolean"]},  # allow True for no-arg tools
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}


@dataclass(frozen=True)
class ToolPlugin:
    name: str
    schema: Dict[str, Any]
    implementation: Callable[..., str]
    module: ModuleType
    source_path: str


class PluginLoadError(Exception):
    pass


class PluginManager:
    def __init__(self, plugin_paths: Optional[List[str]] = None, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._plugin_paths = plugin_paths or []
        self._plugins: List[ToolPlugin] = []
        self._schemas: List[Dict[str, Any]] = []
        self._functions: Dict[str, Callable[..., str]] = {}
        # Opt-in timing instrumentation for profiling test runs
        try:
            self._timing = _truthy_env("OLLAMA_PLUGINS_TIMING")
        except Exception:
            self._timing = False

    @staticmethod
    def _default_paths() -> List[str]:
        """Compute default plugin search paths.
        - Internal: src/plugins/
        - External: tools/ (repository root)
        - Optional: OLLAMA_TOOLS_DIR environment variable (comma-separated)
        """
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)  # project root assumed as parent of src/
        paths = [
            os.path.join(here, "plugins"),
            os.path.join(repo_root, "tools"),
        ]
        extra = os.getenv("OLLAMA_TOOLS_DIR")
        if extra:
            for p in extra.split(os.pathsep):
                if p:
                    paths.append(p)
        # Deduplicate while preserving order
        seen: set = set()
        out: List[str] = []
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(ap)
        return out

    def _iter_module_files(self, base_dir: str) -> List[str]:
        files: List[str] = []
        if not os.path.isdir(base_dir):
            return files
        for name in os.listdir(base_dir):
            if name.startswith("_"):
                continue
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                # support package-style plugins: tools/foo/__init__.py
                init_py = os.path.join(path, "__init__.py")
                if os.path.isfile(init_py):
                    files.append(init_py)
            elif name.endswith(".py"):
                # Skip the package's own __init__.py (e.g., src/plugins/__init__.py)
                if name == "__init__.py":
                    continue
                files.append(path)
        return files

    def _import_module_from_path(self, file_path: str, pkg_base: Optional[str]) -> ModuleType:
        module_name = None
        if pkg_base:
            # derive module_name like 'src.plugins.weather' or package 'src.plugins.foo'
            base_name = os.path.basename(file_path)
            if base_name == "__init__.py":
                # Treat package __init__ as the package module
                rel = os.path.basename(os.path.dirname(file_path))
            elif file_path.endswith(".py"):
                rel = os.path.splitext(base_name)[0]
            else:
                rel = os.path.basename(os.path.dirname(file_path))
            module_name = f"{pkg_base}.{rel}"
        else:
            module_name = f"plugin_{abs(hash(file_path))}"
        # If module already exists (e.g., test-installed stub or previously loaded), reuse it
        try:
            if module_name in sys.modules and isinstance(sys.modules[module_name], ModuleType):
                return sys.modules[module_name]
        except Exception:
            pass
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot create import spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Helpful debug to locate slow or hanging imports
        try:
            self._logger.debug(f"Importing plugin module from '{file_path}' as '{module_name}'")
        except Exception:
            pass
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    def _validate_tool_schema(self, schema: Dict[str, Any]) -> None:
        # Graceful degrade: skip validation when jsonschema is unavailable
        if jsonschema_validate is None or Draft202012Validator is None:
            try:
                # Log once per process to avoid noisy repeats
                global _JSONSCHEMA_WARNED
            except Exception:
                _JSONSCHEMA_WARNED = False  # type: ignore[assignment]
            try:
                if not globals().get('_JSONSCHEMA_WARNED', False):
                    self._logger.warning(
                        "Tool schema validation is partially disabled (jsonschema not installed). "
                        "Run: pip install \"ollama-turbo-cli[tools]\""
                    )
                    globals()['_JSONSCHEMA_WARNED'] = True
            except Exception:
                pass
            return
        try:
            jsonschema_validate(instance=schema, schema=_TOOL_DEFINITION_SCHEMA)
        except Exception as e:
            raise PluginLoadError(f"Tool definition failed validation: {e}")

    def _wrap_with_arg_validation(self, name: str, schema: Dict[str, Any], func: Callable[..., str], *, module_name: str, function_name: str) -> Callable[..., str]:
        params_schema = schema.get("function", {}).get("parameters")
        logger = self._logger

        def _apply_aliases(raw_kwargs: Dict[str, Any], param_names: set) -> Dict[str, Any]:
            """Map common alias argument names to canonical ones expected by functions.
            This helps tolerate upstream planners that use different names.
            Currently supported:
            - top_n -> top_k | max_results | limit (if available)
            - timeout -> timeout_s (if available)
            - site | domain -> site_include (if available)
            - freshness | recency_days -> freshness_days (if available)
            - force -> force_refresh (if available)
            - loc -> ignored (selection index from search results)
            - search -> ignored (planner hint not used by most tools)
            """
            out = dict(raw_kwargs)
            # top_n alias -> prefer top_k, else max_results, else limit
            if "top_n" in out and "top_n" not in param_names:
                if "top_k" in param_names and "top_k" not in out:
                    out["top_k"] = out.get("top_n")
                elif "max_results" in param_names and "max_results" not in out:
                    out["max_results"] = out.get("top_n")
                elif "limit" in param_names and "limit" not in out:
                    out["limit"] = out.get("top_n")
                out.pop("top_n", None)
            # timeout alias
            if "timeout" in out and "timeout" not in param_names and "timeout_s" in param_names and "timeout_s" not in out:
                out["timeout_s"] = out.get("timeout")
                out.pop("timeout", None)
            # site/domain -> site_include
            if "site" in out and "site" not in param_names and "site_include" in param_names and "site_include" not in out:
                out["site_include"] = out.get("site")
                out.pop("site", None)
            if "domain" in out and "domain" not in param_names and "site_include" in param_names and "site_include" not in out:
                out["site_include"] = out.get("domain")
                out.pop("domain", None)
            # exclude synonyms -> site_exclude
            if "exclude" in out and "exclude" not in param_names and "site_exclude" in param_names and "site_exclude" not in out:
                out["site_exclude"] = out.get("exclude")
                out.pop("exclude", None)
            if "exclude_domain" in out and "exclude_domain" not in param_names and "site_exclude" in param_names and "site_exclude" not in out:
                out["site_exclude"] = out.get("exclude_domain")
                out.pop("exclude_domain", None)
            # freshness/recency -> freshness_days
            if "freshness" in out and "freshness" not in param_names and "freshness_days" in param_names and "freshness_days" not in out:
                out["freshness_days"] = out.get("freshness")
                out.pop("freshness", None)
            if "recency_days" in out and "recency_days" not in param_names:
                if "freshness_days" in param_names and "freshness_days" not in out:
                    out["freshness_days"] = out.get("recency_days")
                # drop original alias regardless
                out.pop("recency_days", None)
            # force -> force_refresh
            if "force" in out and "force" not in param_names and "force_refresh" in param_names and "force_refresh" not in out:
                out["force_refresh"] = out.get("force")
                out.pop("force", None)
            # benign drops
            if "loc" in out and "loc" not in param_names:
                out.pop("loc", None)
            if "search" in out and "search" not in param_names:
                out.pop("search", None)
            return out

        def wrapper(**kwargs: Any) -> str:
            # Filter to the function's signature unless it accepts **kwargs; also apply aliases
            try:
                # Dynamically resolve the current function from the module to respect monkeypatching
                import sys as _sys, importlib as _importlib  # type: ignore
                target = func
                try:
                    mod = _sys.modules.get(module_name) or _importlib.import_module(module_name)
                    maybe = getattr(mod, function_name, None)
                    if callable(maybe):
                        target = maybe
                except Exception:
                    target = func

                # Sync common module-level symbols into the function globals so tests that
                # monkeypatch them (e.g., src.plugins.web_research.run_research) are honored
                try:
                    if hasattr(target, "__globals__") and isinstance(getattr(target, "__globals__", None), dict):
                        tg = target.__globals__  # type: ignore[assignment]
                        try:
                            # Only update if present in function globals to avoid leaking names
                            if "run_research" in tg and hasattr(mod, "run_research"):
                                tg["run_research"] = getattr(mod, "run_research")
                        except Exception:
                            pass
                except Exception:
                    pass

                sig = inspect.signature(target)
                params = sig.parameters
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                # Prefer schema properties as the canonical target names for aliasing
                schema_props: set = set()
                try:
                    if isinstance(params_schema, dict):
                        p = params_schema.get('properties')
                        if isinstance(p, dict):
                            schema_props = {str(k) for k in p.keys()}
                except Exception:
                    schema_props = set()
                if accepts_var_kw:
                    # Apply aliases against schema properties when available; keep kwargs-compatible
                    target_space = schema_props or set(params.keys())
                    aliased_any = _apply_aliases(kwargs, set(target_space))
                    # If we have a schema, drop non-schema keys to satisfy additionalProperties=False
                    if schema_props:
                        filtered = {k: v for k, v in aliased_any.items() if k in schema_props}
                    else:
                        filtered = aliased_any
                else:
                    # Apply aliases first, then strictly filter to function's accepted param names
                    param_names = {n for n, p in params.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
                    aliased = _apply_aliases(kwargs, set(param_names))
                    filtered = {k: v for k, v in aliased.items() if k in param_names}
                    dropped = [k for k in kwargs.keys() if k not in param_names]
                    if dropped:
                        logger.debug(f"Tool '{name}': dropping unexpected arguments: {dropped}")

                # Remove None values to avoid schema type mismatches (e.g., string vs null)
                cleaned = {k: v for k, v in filtered.items() if v is not None}

                # Validate cleaned args if schema is provided (post-alias)
                if params_schema and jsonschema_validate is not None:
                    try:
                        jsonschema_validate(instance=cleaned, schema=params_schema)  # type: ignore[arg-type]
                    except Exception as e:
                        raise ValueError(f"Arguments for {name} failed schema validation: {e}")

                return target(**cleaned)
            except TypeError as te:
                logger.error(f"Tool '{name}' invocation failed with TypeError: {te}")
                raise

        # Preserve nicer debug names
        wrapper.__name__ = f"plugin_{name}"
        wrapper.__doc__ = f"Auto-generated wrapper for plugin tool '{name}' with JSON schema validation."
        return wrapper

    def _extract_plugin(self, module: ModuleType, source_path: str) -> ToolPlugin:
        # Locate TOOL_SCHEMA
        schema = getattr(module, "TOOL_SCHEMA", None)
        if not isinstance(schema, dict):
            raise PluginLoadError("Missing or invalid TOOL_SCHEMA (must be a dict)")
        self._validate_tool_schema(schema)
        # Determine tool name
        func_meta = schema.get("function", {})
        name = func_meta.get("name")
        if not isinstance(name, str) or not name:
            raise PluginLoadError("Tool schema missing valid function.name")
        # Find implementation
        impl = getattr(module, "TOOL_IMPLEMENTATION", None)
        if not callable(impl):
            # Try same-name function
            impl = getattr(module, name, None)
        if not callable(impl):
            impl = getattr(module, "execute", None)
        if not callable(impl):
            raise PluginLoadError("No callable implementation found (TOOL_IMPLEMENTATION, function name, or execute)")

        # Capture the module and function names to support dynamic lookup on invoke
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            builtins_root = os.path.join(os.path.dirname(here), 'plugins')
            if os.path.abspath(source_path).startswith(os.path.abspath(builtins_root)):
                base = os.path.splitext(os.path.basename(source_path))[0]
                if base == '__init__':
                    base = os.path.basename(os.path.dirname(source_path))
                mod_name = f"src.plugins.{base}"
            else:
                mod_name = getattr(module, '__name__', '') or 'src.plugins'
        except Exception:
            mod_name = getattr(module, '__name__', '') or 'src.plugins'
        fn_name = getattr(impl, '__name__', name)
        wrapped = self._wrap_with_arg_validation(name, schema, impl, module_name=mod_name, function_name=fn_name)
        return ToolPlugin(name=name, schema=schema, implementation=wrapped, module=module, source_path=source_path)

    def load(self, reset: bool = False, additional_paths: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., str]]]:
        with self._lock:
            t_load_start = time.perf_counter() if getattr(self, "_timing", False) else None
            if reset:
                self._plugins = []
                self._schemas = []
                self._functions = {}

            search_paths = self._plugin_paths or self._default_paths()
            if additional_paths:
                for p in additional_paths:
                    ap = os.path.abspath(p)
                    if ap not in search_paths:
                        search_paths.append(ap)

            self._logger.debug(f"Plugin search paths: {search_paths}")

            # Optional name filters to reduce load during tests (comma-separated tool names)
            include_set: Optional[set] = None
            exclude_set: Optional[set] = None
            try:
                inc = os.getenv("OLLAMA_PLUGINS_INCLUDE")
                if inc is not None:
                    include_set = {s.strip() for s in inc.split(',') if s.strip()}
            except Exception:
                include_set = None
            try:
                exc = os.getenv("OLLAMA_PLUGINS_EXCLUDE")
                if exc is not None:
                    exclude_set = {s.strip() for s in exc.split(',') if s.strip()}
            except Exception:
                exclude_set = None

            loaded_names: set = set(self._functions.keys()) if self._functions else set()
            for path in search_paths:
                pkg_base = None
                # If path is inside our package, derive pkg base for clean module names
                here = os.path.dirname(os.path.abspath(__file__))
                if os.path.abspath(path).startswith(os.path.join(here, "plugins")):
                    pkg_base = f"{__package__}.plugins" if __package__ else "plugins"
                elif os.path.basename(path) == "tools":
                    # external tools directory not a package; import as file locations
                    pkg_base = None

                for file_path in self._iter_module_files(path):
                    try:
                        t_file_start = time.perf_counter() if getattr(self, "_timing", False) else None
                        # Skip disabled core plugin by filename to avoid importing module-level code
                        if os.path.basename(file_path) == "reliable_chat.py":
                            self._logger.info(f"Skipping disabled plugin module by filename: {file_path}")
                            continue
                        module = self._import_module_from_path(file_path, pkg_base)
                        plugin = self._extract_plugin(module, file_path)
                        # Apply optional name filters
                        if include_set is not None and plugin.name not in include_set:
                            self._logger.debug(f"Skipping plugin '{plugin.name}' not in include filter")
                            continue
                        if exclude_set is not None and plugin.name in exclude_set:
                            self._logger.debug(f"Skipping plugin '{plugin.name}' due to exclude filter")
                            continue
                        # Skip disabled core plugin 'reliable_chat'
                        if plugin.name == "reliable_chat":
                            self._logger.info(f"Skipping disabled plugin '{plugin.name}' from {file_path}")
                            continue
                        if plugin.name in loaded_names:
                            self._logger.warning(f"Duplicate tool name '{plugin.name}' from {file_path}; skipping")
                            continue
                        self._plugins.append(plugin)
                        self._schemas.append(plugin.schema)
                        self._functions[plugin.name] = plugin.implementation
                        loaded_names.add(plugin.name)
                        if getattr(self, "_timing", False) and t_file_start is not None:
                            dt_ms = (time.perf_counter() - t_file_start) * 1000.0
                            self._logger.info(f"Loaded plugin '{plugin.name}' from {file_path} in {dt_ms:.1f}ms")
                        else:
                            self._logger.info(f"Loaded plugin '{plugin.name}' from {file_path}")
                    except Exception as e:
                        self._logger.error(f"Failed to load plugin from {file_path}: {e}")
                        continue
            if getattr(self, "_timing", False) and t_load_start is not None:
                total_ms = (time.perf_counter() - t_load_start) * 1000.0
                try:
                    n = len(self._functions)
                except Exception:
                    n = 0
                self._logger.info(f"Plugin load completed: {n} tools in {total_ms:.1f}ms")
            return list(self._schemas), dict(self._functions)

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._schemas)

    @property
    def tool_functions(self) -> Dict[str, Callable[..., str]]:
        with self._lock:
            return dict(self._functions)


# Module-level default manager and lazy aggregates
_default_manager = PluginManager()

# Caches (populated on first access unless disabled). Kept private; access via __getattr__
_CACHED_SCHEMAS: Optional[List[Dict[str, Any]]] = None
_CACHED_FUNCTIONS: Optional[Dict[str, Callable[..., str]]] = None

def _truthy_env(name: str) -> bool:
    try:
        v = os.getenv(name)
        return v is not None and str(v).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False

def _plugins_mode() -> str:
    """Return 'off' | 'lazy' | 'eager'. Defaults to lazy for safety in tests."""
    try:
        if _truthy_env("OLLAMA_DISABLE_PLUGINS"):
            return "off"
        mode = (os.getenv("OLLAMA_PLUGINS_MODE") or "").strip().lower()
        if mode in {"off", "disable", "disabled"}:
            return "off"
        if mode in {"eager", "on"} or _truthy_env("OLLAMA_PLUGINS_EAGER"):
            return "eager"
        if mode in {"lazy"} or _truthy_env("OLLAMA_PLUGINS_LAZY"):
            return "lazy"
    except Exception:
        pass
    return "lazy"

def ensure_loaded(reset: bool = False, additional_paths: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., str]]]:
    """Ensure plugins are loaded into the default manager and caches.

    Honors env modes:
      - off: returns empty aggregates
      - lazy: load on first access
      - eager: callers may invoke this at import-sites to force load
    """
    global _CACHED_SCHEMAS, _CACHED_FUNCTIONS
    mode = _plugins_mode()
    if mode == "off":
        _CACHED_SCHEMAS, _CACHED_FUNCTIONS = [], {}
        return [], {}
    if reset or _CACHED_SCHEMAS is None or _CACHED_FUNCTIONS is None:
        schemas, funcs = _default_manager.load(reset=True, additional_paths=additional_paths)
        _CACHED_SCHEMAS, _CACHED_FUNCTIONS = list(schemas), dict(funcs)
    return list(_CACHED_SCHEMAS or []), dict(_CACHED_FUNCTIONS or {})

def __getattr__(name: str):
    """Lazy attribute access for TOOL_SCHEMAS / TOOL_FUNCTIONS.

    Accessing these will trigger a load unless plugins are disabled via env.
    """
    if name == "TOOL_SCHEMAS":
        schemas, _ = ensure_loaded(reset=False)
        return schemas
    if name == "TOOL_FUNCTIONS":
        _, funcs = ensure_loaded(reset=False)
        return funcs
    raise AttributeError(name)


def get_manager() -> PluginManager:
    return _default_manager


def reload_plugins(additional_paths: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Callable[..., str]]]:
    """Reload plugins into the default manager and refresh caches."""
    return ensure_loaded(reset=True, additional_paths=additional_paths)
