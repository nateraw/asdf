from typing import List

def run(text: str, limit: int = 10) -> List[str]:
    return text.split()[:limit]

class FnWrap:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def wrapped_run(*args, **kwargs):
    return FnWrap(run)
