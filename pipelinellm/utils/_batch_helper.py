from pipelinellm.types import BatchResult


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
        # ok_str = ""
        # err_str = ""
        # for i, line in enumerate(content.strip().split("\n")):
        #     if i in not_ok_i:
        #         err_str += line + "\n"
        #     else:
        #         ok_str += line + "\n"

        return [
            BatchResult(
                status="ready",
                raw_output=content,
                parsed_responses=parsed_responses,
            ),
            BatchResult(
                status="error",
                raw_output=None,
                parsed_responses=parsed_errors,
            ),
        ]
