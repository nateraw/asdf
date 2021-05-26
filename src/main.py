from typing import List

def run(text: str, limit: int = 10) -> List[str]:
    return text.split()[:limit]
