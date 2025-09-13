from __future__ import annotations

# Guard against package/module shadowing of src.web.rerank

def test_rerank_import_resolves_to_package():
    from src.web.rerank import chunk_text, rerank_chunks  # type: ignore

    # Functions should be defined from the package module (src.web.rerank.__init__)
    assert chunk_text.__module__.startswith("src.web.rerank"), chunk_text.__module__
    assert rerank_chunks.__module__.startswith("src.web.rerank"), rerank_chunks.__module__
