def is_pydantic_model(obj):
    try:
        from pydantic import BaseModel

        return isinstance(obj, BaseModel)
    except ImportError:
        return False


def is_pydantic_v2_model_like(obj):
    """importing pydantic is strangely slow, so try to avoid it if we can"""
    required = [
        "model_fields",
        "model_construct",
        "model_dump",
        "model_json_schema",
        "model_dump_json",
        "model_validate",
        "model_rebuild",
    ]
    return all(hasattr(obj, attr) for attr in required)


if __name__ == "__main__":
    import pydantic

    class MyModel(pydantic.BaseModel):
        x: int
        y: str

    m = MyModel(x=1, y="hello")
    m.model_dump
    print(is_pydantic_model(m))  # True
