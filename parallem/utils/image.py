import base64
from io import BytesIO
from typing import TypeGuard, Union
from PIL import Image

from parallem.types import ImageURLDocument



def is_image(obj) -> TypeGuard[Image.Image]:
    """Check if the object is a PIL Image."""
    return isinstance(obj, Image.Image)

def _get_image_type(obj: Image.Image):
    """Get preferred file type for this PIL Image"""
    out = obj.format
    if out is None:
        return "image/jpeg"
    return f"image/{out.lower()}"


def _image_to_b64(obj: Image.Image, image_type=None):
    """Convert a PIL Image to a base64-encoded string."""
    if image_type is None:
        image_type = obj.format
    buffered = BytesIO()
    obj.save(buffered, format=image_type)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_type_and_b64(obj: Image.Image, *, allowed=None):
    """Get the preferred file type and base64-encoded string for a PIL Image.

    :param allowed: A list of allowed image types. If the image's preferred type is not in this list,
        it will be converted to the first type in the list. If None, all types are allowed.
    """
    image_type = _get_image_type(obj)
    if image_type is None:
        # default to JPEG
        image_type = "image/jpeg"
    if allowed and image_type not in allowed:
        # If there is a whitelist, we have to convert
        convert_to = allowed[0]
        # obj = obj.convert("RGB")  # Convert to RGB if needed
        return convert_to, _image_to_b64(obj, convert_to.removeprefix("image/"))
    return image_type, _image_to_b64(obj, image_type.removeprefix("image/"))


def is_image_url(obj: ImageURLDocument) -> TypeGuard[ImageURLDocument]:
    return isinstance(obj, ImageURLDocument)


def to_image_url_str(obj: Union[Image.Image, str, "ImageURLDocument"], *, allowed_mimetypes=None) -> str:
    """Convert an object to an image_url string"""
    if isinstance(obj, ImageURLDocument):
        return obj.image_url
    elif isinstance(obj, str):
        return obj
    elif is_image(obj):
        image_type, b64_str = get_type_and_b64(obj, allowed=allowed_mimetypes)
        return f"data:{image_type};base64,{b64_str}"
