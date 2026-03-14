from __future__ import annotations

import gzip
import ipaddress
import json
import re
import socket
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser

from agent.tool import Tool, ToolException


_DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    )


def _host_resolves_to_private(host: str) -> bool:
    # If DNS lookup fails, treat as unsafe.
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return True

    for family, _, _, _, sockaddr in infos:
        if family == socket.AF_INET:
            ip = sockaddr[0]
        elif family == socket.AF_INET6:
            ip = sockaddr[0]
        else:
            continue
        if _is_private_ip(ip):
            return True
    return False


def _validate_url(url: str, allow_private: bool) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolException("web: only http/https URLs are allowed")
    if not parsed.netloc:
        raise ToolException("web: URL must include a host")

    host = parsed.hostname or ""
    if host.lower() == "localhost":
        raise ToolException("web: localhost is not allowed")
    if not allow_private and _host_resolves_to_private(host):
        raise ToolException("web: private/internal network targets are not allowed")
    return parsed


def _decode_body(headers: dict[str, str], body: bytes) -> str:
    enc = (headers.get("content-encoding") or "").lower()

    # Some servers send gzip-compressed bodies even when headers are missing.
    if ("gzip" in enc) or body.startswith(b"\x1f\x8b\x08"):
        try:
            body = gzip.decompress(body)
        except Exception:
            pass

    ctype = headers.get("content-type") or ""
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", ctype)
    charset = (m.group(1) if m else "utf-8").strip()
    try:
        return body.decode(charset, errors="replace")
    except Exception:
        return body.decode("utf-8", errors="replace")


def _http_get(
    url: str,
    timeout_s: int,
    max_bytes: int,
    user_agent: str,
    extra_headers: dict[str, str] | None = None,
) -> tuple[dict[str, str], bytes]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent or _DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip",
            **(extra_headers or {}),
        },
        method="GET",
    )

    # 直接创建一个跳过 SSL 验证的 context（适用于开发环境/代理环境）
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        raise ToolException(f"web: HTTP {e.code} {e.reason} - {url}")
    except urllib.error.URLError as e:
        raise ToolException(f"web: URL error ({e.reason}) - {url}")
    except Exception as e:
        raise ToolException(f"web: request failed ({e}) - {url}")

    if len(body) > max_bytes:
        raise ToolException("web: response too large (increase max_bytes if needed)")
    return headers, body


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "div", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        s = data.strip()
        if not s:
            return
        self._parts.append(s + " ")

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _extract_links(html: str, base_url: str, max_links: int) -> list[str]:
    hrefs = re.findall(r"""href\s*=\s*(['\"])(.*?)\1""", html, flags=re.IGNORECASE)
    out: list[str] = []
    seen: set[str] = set()
    for _, href in hrefs:
        if len(out) >= max_links:
            break
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        abs_url = urllib.parse.urljoin(base_url, href)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        out.append(abs_url)
    return out


@dataclass(frozen=True)
class _WebCfg:
    timeout_s: int
    max_bytes: int
    user_agent: str


class WebOpenTool(Tool):
    name = "web_open"
    description = "Fetch a web page and return readable text (basic HTML-to-text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to fetch."},
            "max_chars": {"type": "integer", "description": "Max characters to return.", "default": 20000},
            "include_links": {"type": "boolean", "description": "Append extracted links.", "default": False},
            "max_links": {"type": "integer", "description": "Max links to extract.", "default": 50},
            "allow_private": {"type": "boolean", "description": "Allow localhost/private targets (SSRF risk).", "default": False},
            "timeout_s": {"type": "integer", "description": "Request timeout seconds.", "default": 20},
            "max_bytes": {"type": "integer", "description": "Max bytes to download.", "default": 2000000},
            "user_agent": {"type": "string", "description": "Override User-Agent header."},
        },
        "required": ["url"],
    }

    def call(
        self,
        url: str,
        max_chars: int = 20000,
        include_links: bool = False,
        max_links: int = 50,
        allow_private: bool = False,
        timeout_s: int = 20,
        max_bytes: int = 2_000_000,
        user_agent: str | None = None,
        **kwargs,
    ) -> str:
        _validate_url(url, allow_private=allow_private)
        cfg = _WebCfg(timeout_s=timeout_s, max_bytes=max_bytes, user_agent=user_agent or _DEFAULT_UA)

        headers, body = _http_get(url, timeout_s=cfg.timeout_s, max_bytes=cfg.max_bytes, user_agent=cfg.user_agent)
        html = _decode_body(headers, body)
        text = _html_to_text(html)

        if include_links:
            links = _extract_links(html, base_url=url, max_links=max_links)
            text = text + "\n\nLinks:\n" + "\n".join(links)

        if len(text) > max_chars:
            return text[:max_chars] + "\n... (truncated)"
        return text


class WebLinksTool(Tool):
    name = "web_links"
    description = "Fetch a page and extract links (absolute URLs)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to fetch."},
            "max_links": {"type": "integer", "description": "Max links to return.", "default": 50},
            "allow_private": {"type": "boolean", "description": "Allow localhost/private targets (SSRF risk).", "default": False},
            "timeout_s": {"type": "integer", "description": "Request timeout seconds.", "default": 20},
            "max_bytes": {"type": "integer", "description": "Max bytes to download.", "default": 2000000},
            "user_agent": {"type": "string", "description": "Override User-Agent header."},
        },
        "required": ["url"],
    }

    def call(
        self,
        url: str,
        max_links: int = 50,
        allow_private: bool = False,
        timeout_s: int = 20,
        max_bytes: int = 2_000_000,
        user_agent: str | None = None,
        **kwargs,
    ) -> str:
        _validate_url(url, allow_private=allow_private)
        cfg = _WebCfg(timeout_s=timeout_s, max_bytes=max_bytes, user_agent=user_agent or _DEFAULT_UA)

        headers, body = _http_get(url, timeout_s=cfg.timeout_s, max_bytes=cfg.max_bytes, user_agent=cfg.user_agent)
        html = _decode_body(headers, body)
        links = _extract_links(html, base_url=url, max_links=max_links)
        return json.dumps({"url": url, "links": links}, ensure_ascii=False, indent=2)


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web using search engines to find relevant information. Returns search results with titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query keywords",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of search results to return. Default is 5, max is 10",
            },
        },
        "required": ["query"],
    }

    def call(
        self,
        query: str,
        num_results: int = 5,
        timeout_s: int = 20,
        max_bytes: int = 2_000_000,
        user_agent: str | None = None,
        **kwargs,
    ) -> str:
        if not query.strip():
            raise ToolException("web_search: query is empty")

        count = max(1, min(int(num_results), 10))

        base = "https://cn.bing.com/search"
        _validate_url(base, allow_private=False)
        cfg = _WebCfg(timeout_s=timeout_s, max_bytes=max_bytes, user_agent=user_agent or _DEFAULT_UA)

        params: dict[str, str] = {
            "q": query,
            "count": str(count),
        }
        url = base + "?" + urllib.parse.urlencode(params)

        headers, body = _http_get(
            url,
            timeout_s=cfg.timeout_s,
            max_bytes=cfg.max_bytes,
            user_agent=cfg.user_agent,
            extra_headers={
                "Accept-Encoding": "identity",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            },
        )
        html = _decode_body(headers, body)

        def _strip_tags(s: str) -> str:
            s = re.sub(r"<[^>]+>", " ", s)
            s = unescape(s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        results: list[dict[str, str]] = []
        for block in re.findall(
            r"""<li[^>]+class=\"b_algo\"[^>]*>.*?</li>""",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            if len(results) >= count:
                break

            m = re.search(
                r"""<h2[^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>""",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m:
                continue

            href = unescape(m.group(1)).strip()
            title = _strip_tags(m.group(2))

            snippet = ""
            sm = re.search(
                r"""<div[^>]+class=\"b_caption\"[^>]*>.*?<p[^>]*>(.*?)</p>""",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if sm:
                snippet = _strip_tags(sm.group(1))

            results.append({"title": title, "url": href, "snippet": snippet})

        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)