from typing import TYPE_CHECKING, Any, TypeGuard


if TYPE_CHECKING:
    import pydantic


def to_strict_json_schema(model: "pydantic.BaseModel") -> dict[str, Any]:
    """
    Bundled from openai=1.107.2:
    ```
    from openai.lib._pydantic import to_strict_json_schema
    ```

    Defers the real function, but if it fails, there is a fallback implementation.
    In that case, pydantic >= 2 is required. We also assume simply that the model is a BaseModel
    """
    try:
        from openai.lib._pydantic import to_strict_json_schema

        return to_strict_json_schema(model)
    except ImportError:
        pass

    # Fallback implementation for pydantic v2
    schema = model.model_json_schema()
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _is_dict(obj: object) -> TypeGuard[dict[object, object]]:
    return isinstance(obj, dict)


def _is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)


def _has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """
    Bundled from:
    ```
    from openai.lib._pydantic import to_strict_json_schema
    from openai.lib._pydantic import _ensure_strict_json_schema
    ```

    Original Description:
    > Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    """
    from openai import NOT_GIVEN

    try:
        from openai.lib._pydantic import _ensure_strict_json_schema

        return _ensure_strict_json_schema(json_schema, path=path, root=root)
    except ImportError:
        pass
    # if not is_dict(json_schema):
    # raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if _is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    definitions = json_schema.get("definitions")
    if _is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if _is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if _is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # unions
    any_of = json_schema.get("anyOf")
    if _is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if _is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", NOT_GIVEN) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and _has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = _resolve_ref(root=root, ref=ref)
        if not _is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied,
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid.
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def _resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert _is_dict(value), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved
