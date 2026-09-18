"""Which URLs already carry their own authorisation.

This mirrors the Go client's TestPresignedURLsAreRecognised. The two
implementations are deliberately the same shape, and a URL one of them sends a
credential to and the other does not is a difference that shows up as a 403 on
one code path only.
"""

from corerun.lfs import _presigned


def test_urls_that_carry_their_own_authorisation():
    for href in [
        "https://s3.example.com/b/o?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc",
        "https://s3.example.com/b/o?AWSAccessKeyId=k&Signature=xyz",
        "https://s.example.com/b/o?X-Goog-Signature=abc",
        "https://s.example.com/b/o?sv=2021&sig=abc",
        # CloudFront, which is what HuggingFace hands back for weights: signed
        # with a key pair, so it carries neither an access key id nor an
        # X-Amz-Signature.
        "https://us.aws.cdn.hf.co/repos/ab/cd/model.safetensors"
        "?Expires=1789700000&Policy=eyJTdGF0ZW1lbnQiOltdfQ__"
        "&Signature=Wm9v~abc__&Key-Pair-Id=K24J24Z295AELW",
    ]:
        assert _presigned(href), f"{href} was not recognised as presigned"


def test_urls_that_need_our_credential():
    # "signature" and "sig" are words an ordinary URL may contain, and
    # mistaking one for a presigned URL withholds the credential the request
    # actually needs.
    for href in [
        "https://dev.corerun.ai/api/v1/git/t/m.git/info/lfs/objects/oid",
        "https://dev.corerun.ai/api/v1/git/t/m.git/o?sig=notreally",
        "https://dev.corerun.ai/api/v1/git/t/m.git/o?signature=notreally",
    ]:
        assert not _presigned(href), f"{href} was mistaken for presigned"
