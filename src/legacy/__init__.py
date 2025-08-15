"""Legacy compatibility package."""

__all__ = ['OllamaTurboClient']


def __getattr__(name: str):  # pragma: no cover
    if name == 'OllamaTurboClient':
        from .client_facade import OllamaTurboClient as _C
        return _C
    raise AttributeError(name)
