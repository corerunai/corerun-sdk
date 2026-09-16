"""What the user is told when a request fails.

The Go API answers {"error": "<code>", "message": "<sentence>"} and the Python
data service answers FastAPI's {"detail": "<sentence>"}. The client read
`detail` then `error`, so every Go API failure arrived as a bare code --
`corerun models publish` on a model that does not exist said "Error: not_found"
and nothing else, while the handler had written "no model named tiny-gpt2".
"""

import pytest

from corerun.client import CoreRunClient
from corerun.exceptions import CoreRunError, NotFoundError, ValidationError


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.content = b"x"
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def handle(status_code, payload):
    return CoreRunClient._handle_response(CoreRunClient.__new__(CoreRunClient), FakeResponse(status_code, payload))


@pytest.mark.parametrize(
    "payload,expected",
    [
        # The Go API: the sentence, not the code.
        ({"error": "not_found", "message": "no model named tiny-gpt2"}, "no model named tiny-gpt2"),
        # The data service.
        ({"detail": "Storage account 'orgs3' has no bucket"}, "Storage account 'orgs3' has no bucket"),
        # A code alone still beats nothing.
        ({"error": "not_found"}, "not_found"),
        # An empty message must not win over the code.
        ({"error": "not_found", "message": ""}, "not_found"),
    ],
)
def test_the_readable_half_survives(payload, expected):
    with pytest.raises(NotFoundError) as caught:
        handle(404, payload)
    assert str(caught.value) == expected


def test_a_body_that_is_not_json_falls_back_to_the_text():
    with pytest.raises(ValidationError) as caught:
        handle(400, ValueError("not json"))
    assert "not json" in str(caught.value)


def test_an_unmapped_status_carries_the_message_too():
    with pytest.raises(CoreRunError) as caught:
        handle(409, {"error": "conflict", "message": "version 1 is PENDING"})
    assert "version 1 is PENDING" in str(caught.value)
