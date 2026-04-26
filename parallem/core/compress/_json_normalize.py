from collections.abc import Mapping, Sequence, Iterable
import json
import polars as pl

_POLARS_NORM_AVAILABLE = True
try:
    from polars.convert.normalize import _simple_json_normalize
except ImportError:
    _POLARS_NORM_AVAILABLE = False


def pl_json_normalize(
    data,
    *,
    separator: str = ".",
    max_level: int | None = None,
    schema=None,
    schema_overrides=None,
    strict: bool = True,
    infer_schema_length: int | None = 100,
    encoder=None,
):
    """Forked from pl.json_normalize, but where schema_overrides is exposed."""

    if not _POLARS_NORM_AVAILABLE:
        return pl.json_normalize(
            data,
            separator=separator,
            max_level=max_level,
            schema=schema,
            strict=strict,
            infer_schema_length=infer_schema_length,
            encoder=encoder,
        )
    if max_level is None:
        max_level = 1 << 32  # eg: u32
    max_level += 1

    if isinstance(data, Sequence) and len(data) == 0:
        return pl.DataFrame(schema=schema)
    elif isinstance(data, Mapping):
        data = [data]
    elif isinstance(data, Iterable) and not isinstance(data, str):  # type: ignore[redundant-expr]
        data = list(data)
    else:
        msg = "expected list or dict of objects"
        raise ValueError(msg)

    if encoder is None:
        encoder = json.dumps

    return pl.DataFrame(
        _simple_json_normalize(
            data,
            separator=separator,
            max_level=max_level,
            encoder=encoder,
        ),
        schema=schema,
        schema_overrides=schema_overrides,
        strict=strict,
        infer_schema_length=infer_schema_length,
    )
