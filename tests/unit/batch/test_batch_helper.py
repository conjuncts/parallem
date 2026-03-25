from parallem.utils._batch_helper import _split_batch_response


def test_split_batch_response_mixed_has_error_raw_output():
    content = '{"ok": 1}\n{"err": 1}'
    parsed_responses = [object()]
    parsed_errors = [object()]

    results = _split_batch_response(
        parsed_responses=parsed_responses,
        parsed_errors=parsed_errors,
        content=content,
        not_ok_i=[1],
    )

    assert len(results) == 2

    error_result = next(r for r in results if r.status == "error")
    assert error_result.raw_output is not None
    assert error_result.raw_output == '{"err": 1}'
