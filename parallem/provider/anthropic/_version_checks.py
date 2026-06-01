from parallem.core.exception import ProviderCompatibilityError
import importlib.metadata as importlib_metadata

_ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION = "0.77.0"


def _numeric_version_triplet(version_text: str) -> tuple[int, int, int]:
    numeric_parts = []
    current = ""

    for char in version_text:
        if char.isdigit():
            current += char
            continue
        if current:
            numeric_parts.append(int(current))
            current = ""
            if len(numeric_parts) >= 3:
                break

    if current and len(numeric_parts) < 3:
        numeric_parts.append(int(current))

    while len(numeric_parts) < 3:
        numeric_parts.append(0)

    return tuple(numeric_parts[:3])


def enforce_anthropic_min_version_for_structured_output() -> None:
    """Validate installed anthropic package supports output_config.format."""

    try:
        installed_version_text = importlib_metadata.version("anthropic")
    except importlib_metadata.PackageNotFoundError as exc:
        raise ImportError(
            "Structured output with Anthropic requires the 'anthropic' package to be installed."
        ) from exc

    installed_version = _numeric_version_triplet(installed_version_text)
    minimum_required = _numeric_version_triplet(_ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION)
    if installed_version < minimum_required:
        raise ProviderCompatibilityError(
            "Structured output with Anthropic requires anthropic>="
            f"{_ANTHROPIC_STRUCTURED_OUTPUT_MIN_VERSION}, found {installed_version_text}. "
            "Please upgrade the anthropic package."
        )
