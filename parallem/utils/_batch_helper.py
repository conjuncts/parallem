from parallem.types import BatchResult


def _split_batch_response(
    *, parsed_responses, parsed_errors, content: str, not_ok_i: list[int]
) -> list[BatchResult]:
    """helper to split into a "ready" and an "error" BatchResult."""
    if not parsed_errors:
        # Perfect result
        return [
            BatchResult(
                status="ready",
                raw_output=content,
                parsed_responses=parsed_responses,
            )
        ]
    elif not parsed_responses:
        return [
            BatchResult(
                status="error",
                raw_output=content,
                parsed_responses=parsed_errors,
            )
        ]
    else:
        not_ok_set = set(not_ok_i)
        err_lines = [line for i, line in enumerate(content.strip().split("\n")) if i in not_ok_set]
        err_str = "\n".join(err_lines)

        return [
            BatchResult(
                status="ready",
                raw_output=content,
                parsed_responses=parsed_responses,
            ),
            BatchResult(
                status="error",
                raw_output=err_str,
                parsed_responses=parsed_errors,
            ),
        ]
