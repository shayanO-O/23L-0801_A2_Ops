import io
import pytest
from PIL import Image
from app.inference import validate_image_bytes, ValidationError, MAX_FILE_SIZE_BYTES


def test_validate_empty_bytes():
    # Verify Empty Input Triggers 400 Validation Error
    with pytest.raises(ValidationError) as exc_info:
        validate_image_bytes(b"", filename="empty.jpg")
    assert exc_info.value.status_code == 400
    assert "empty" in str(exc_info.value).lower()


def test_validate_oversized_payload():
    # Verify File Exceeding 5 MB Limit Triggers 413 Payload Too Large
    oversized_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(ValidationError) as exc_info:
        validate_image_bytes(oversized_bytes, filename="large.jpg")
    assert exc_info.value.status_code == 413
    assert "exceeds" in str(exc_info.value).lower()


def test_validate_unsupported_extension():
    # Verify Non-image Extension Triggers 400 Bad Request
    with pytest.raises(ValidationError) as exc_info:
        validate_image_bytes(b"some content", filename="malicious.exe")
    assert exc_info.value.status_code == 400
    assert "unsupported" in str(exc_info.value).lower()


def test_validate_corrupted_image():
    # Verify Unparseable Bytes Trigger 400 Validation Error
    corrupted_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01not_a_real_image"
    with pytest.raises(ValidationError) as exc_info:
        validate_image_bytes(corrupted_bytes, filename="corrupt.jpg")
    assert exc_info.value.status_code == 400
    assert "corrupted" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()


def test_validate_valid_image():
    # Verify Clean Image Bytes Pass Validation
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    valid_bytes = buf.getvalue()

    fmt, file_hash = validate_image_bytes(valid_bytes, filename="valid.jpg")
    assert fmt.lower() == "jpeg"
    assert len(file_hash) == 64
