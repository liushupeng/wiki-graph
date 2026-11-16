# OAuth 2.0 PKCE

PKCE 由 code_verifier 和其派生的 code_challenge 组成。推荐的 S256 方法对 verifier 做 SHA-256 后再使用 base64url 编码。它降低授权码被截获后直接兑换令牌的风险。
