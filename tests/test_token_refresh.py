import sys, types, json
sys.path.insert(0, 'corerun-sdk/src')

# Exercise the client's refresh logic without importing the whole package.
src = open('corerun-sdk/src/corerun/client.py').read()
start = src.index('    def _exchange_refresh_token')
end = src.index('    def get(')
body = src[start:end]

class FakeResp:
    def __init__(self, code, payload): self.status_code, self._p = code, payload
    def json(self): return self._p

class Cfg:
    def __init__(self, rt): self.refresh_token, self.auth_token = rt, 'old'
    api_url, timeout, verify_ssl, auth_token_file = 'http://x/api/v1', 5, True, None
    saved = False
    def save(self): Cfg.saved = True

ns = {'httpx': types.SimpleNamespace(post=lambda *a, **k: FakeResp(200, {'access_token': 'cr-new-token'}))}
exec(body.replace('    def ', 'def ').replace('\n        ', '\n    '), ns)

class C:
    def __init__(self, cfg): self.config, self._client = cfg, None
    _exchange_refresh_token = ns['_exchange_refresh_token']
    _use_key = ns['_use_key']

c = C(Cfg('cr-refresh'))
assert c._exchange_refresh_token() is True, "should exchange"
assert c.config.auth_token == 'cr-new-token', c.config.auth_token
assert Cfg.saved, "new token should be written back"
print("1 ok: a good refresh token yields a new access token and is persisted")

ns['httpx'] = types.SimpleNamespace(post=lambda *a, **k: FakeResp(401, {}))
exec(body.replace('    def ', 'def ').replace('\n        ', '\n    '), ns)
C._exchange_refresh_token = ns['_exchange_refresh_token']
c2 = C(Cfg('cr-spent'))
assert c2._exchange_refresh_token() is False, "a refused refresh must not claim success"
print("2 ok: a spent refresh token reports failure rather than looping")

c3 = C(Cfg(None))
assert c3._exchange_refresh_token() is False
print("3 ok: no refresh token configured is not an error")

def boom(*a, **k): raise RuntimeError("network down")
ns['httpx'] = types.SimpleNamespace(post=boom)
exec(body.replace('    def ', 'def ').replace('\n        ', '\n    '), ns)
C._exchange_refresh_token = ns['_exchange_refresh_token']
c4 = C(Cfg('cr-refresh'))
assert c4._exchange_refresh_token() is False, "a network error must not escape"
print("4 ok: a network failure during refresh does not replace the real error")
