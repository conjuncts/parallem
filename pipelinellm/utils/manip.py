import re


def maybe_snake_to_camel(snake_str: str, convert: bool = True) -> str:
    """Converts a snake_case string to CamelCase, if convert is True."""
    # https://github.com/googleapis/python-genai/blob/main/google/genai/_common.py#L250
    if not convert:
        return snake_str
    return re.sub(r"_([a-zA-Z])", lambda match: match.group(1).upper(), snake_str)


def to_snake_case(name: str) -> str:
    """Converts a string from camelCase or PascalCase to snake_case."""
    # https://github.com/googleapis/python-genai/blob/main/google/genai/_replay_api_client.py#L39
    if not isinstance(name, str):
        name = str(name)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


if __name__ == "__main__":
    print(to_snake_case("modelVersionWithManyHTTPParts"))
