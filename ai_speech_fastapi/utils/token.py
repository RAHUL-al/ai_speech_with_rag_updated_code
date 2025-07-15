from urllib.parse import unquote

def decode_token(raw_token: str) -> str:
    if not raw_token:
        return None
    decoded_once = unquote(raw_token)
    decoded_twice = unquote(decoded_once)
    if decoded_twice.startswith("Bearer "):
        return decoded_twice
    if decoded_once.startswith("Bearer "):
        return decoded_once
    return None
