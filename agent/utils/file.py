import aiohttp
import asyncio
import base64
import os
import ssl
from pathlib import Path
from typing import Optional, Callable, AsyncGenerator, Union
from urllib.parse import urlparse


async def save_file(
    content: Union[bytes, str, AsyncGenerator[bytes, None]],
    save_path: str,
    filename: str,
    mode: str = "wb",
    overwrite: bool = True,
) -> str:
    """通用的文件保存方法

    Args:
        content: 文件内容，可以是 bytes、str 或异步生成器
        save_path: 保存目录
        filename: 文件名
        mode: 写入模式，默认 "wb"（二进制写入）
        overwrite: 是否覆盖已存在的文件，默认 False

    Returns:
        保存的文件路径

    Raises:
        ValueError: 文件已存在（当 overwrite=False 时）
    """
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    if file_path.exists() and not overwrite:
        raise ValueError(f"文件已存在: {file_path}")

    # 异步生成器模式（流式写入）
    if hasattr(content, '__aiter__'):
        with open(file_path, mode) as f:
            async for chunk in content:
                if isinstance(chunk, str):
                    f.write(chunk.encode())
                else:
                    f.write(chunk)
    else:
        # 直接写入
        with open(file_path, mode) as f:
            f.write(content)

    return str(file_path)


async def download_file(
    url: str,
    chunk_size: int = 8192,
    timeout: int = 300,
    headers: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ssl_verify: bool = True,
) -> tuple[str, AsyncGenerator[bytes, None], dict]:
    """下载网络文件（流式），不保存到本地

    Args:
        url: 文件的下载地址
        chunk_size: 下载块大小，默认 8192 字节
        timeout: 下载超时时间（秒），默认 300 秒
        headers: 自定义请求头
        progress_callback: 进度回调函数，参数为 (已下载字节数, 总字节数)
        ssl_verify: 是否验证 SSL 证书，默认 True

    Returns:
        (文件名, 内容异步生成器, 元信息字典)
        元信息包含: total_size, content_type 等

    Raises:
        ValueError: URL 无效
        aiohttp.ClientError: 网络请求失败
        asyncio.TimeoutError: 下载超时
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"无效的 URL: {url}")

    # 从 URL 提取文件名
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"

    # 设置请求头
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    if headers:
        default_headers.update(headers)

    timeout_config = aiohttp.ClientTimeout(total=timeout)
    # 配置 SSL
    ssl_config = None if ssl_verify else ssl.create_default_context()
    if not ssl_verify and ssl_config:
        ssl_config.check_hostname = False
        ssl_config.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(ssl=ssl_config)
    session = aiohttp.ClientSession(timeout=timeout_config, connector=connector)

    response = await session.get(url, headers=default_headers)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0))
    content_type = response.headers.get("Content-Type", "")
    meta = {
        "total_size": total_size,
        "content_type": content_type,
    }

    downloaded_size = 0

    async def content_generator():
        nonlocal downloaded_size
        try:
            async for chunk in response.content.iter_chunked(chunk_size):
                downloaded_size += len(chunk)
                if progress_callback:
                    progress_callback(downloaded_size, total_size)
                yield chunk
        finally:
            response.close()
            await session.close()

    return filename, content_generator(), meta


async def download_and_save(
    url: str,
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    chunk_size: int = 8192,
    timeout: int = 300,
    headers: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    overwrite: bool = False,
    ssl_verify: bool = True,
) -> str:
    """下载网络文件并保存到本地（便捷方法）

    Args:
        url: 文件的下载地址
        save_path: 保存目录，默认为当前工作目录
        filename: 保存的文件名，默认从 URL 中提取
        chunk_size: 下载块大小，默认 8192 字节
        timeout: 下载超时时间（秒），默认 300 秒
        headers: 自定义请求头
        progress_callback: 进度回调函数，参数为 (已下载字节数, 总字节数)
        overwrite: 是否覆盖已存在的文件，默认 False
        ssl_verify: 是否验证 SSL 证书，默认 True

    Returns:
        下载文件的本地路径
    """
    if save_path is None:
        save_path = os.getcwd()

    # 下载文件
    downloaded_filename, content_gen, meta = await download_file(
        url=url,
        chunk_size=chunk_size,
        timeout=timeout,
        headers=headers,
        progress_callback=progress_callback,
        ssl_verify=ssl_verify,
    )

    # 使用下载的文件名（如果未指定）
    if filename is None:
        filename = downloaded_filename

    # 保存文件
    return await save_file(
        content=content_gen,
        save_path=save_path,
        filename=filename,
        overwrite=overwrite,
    )


async def download_files(
    urls: list[str],
    save_path: Optional[str] = None,
    max_concurrent: int = 3,
    **kwargs,
) -> list[tuple[str, Optional[str], Optional[Exception]]]:
    """批量下载网络文件

    Args:
        urls: 文件下载地址列表
        save_path: 保存目录
        max_concurrent: 最大并发数，默认 3
        **kwargs: 传递给 download_and_save 的其他参数

    Returns:
        列表，每个元素为 (url, 本地路径或None, 异常或None)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def download_with_semaphore(url: str):
        async with semaphore:
            try:
                path = await download_and_save(url, save_path=save_path, **kwargs)
                return (url, path, None)
            except Exception as e:
                return (url, None, e)

    tasks = [download_with_semaphore(url) for url in urls]
    return await asyncio.gather(*tasks)


async def download_to_base64(
    url: str,
    chunk_size: int = 8192,
    timeout: int = 300,
    headers: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    encoding: str = "utf-8",
    ssl_verify: bool = True,
) -> tuple[str, str, dict]:
    """下载网络文件并转换为 base64 编码字符串

    Args:
        url: 文件的下载地址
        chunk_size: 下载块大小，默认 8192 字节
        timeout: 下载超时时间（秒），默认 300 秒
        headers: 自定义请求头
        progress_callback: 进度回调函数，参数为 (已下载字节数, 总字节数)
        encoding: base64 编码的字符串编码方式，默认 utf-8
        ssl_verify: 是否验证 SSL 证书，默认 True

    Returns:
        (文件名, base64编码字符串, 元信息字典)
        元信息包含: total_size, content_type 等

    Raises:
        ValueError: URL 无效
        aiohttp.ClientError: 网络请求失败
        asyncio.TimeoutError: 下载超时
    """
    filename, content_gen, meta = await download_file(
        url=url,
        chunk_size=chunk_size,
        timeout=timeout,
        headers=headers,
        progress_callback=progress_callback,
        ssl_verify=ssl_verify,
    )

    # 收集所有内容
    content = b""
    async for chunk in content_gen:
        content += chunk

    # 转换为 base64
    base64_str = base64.b64encode(content).decode(encoding)

    return filename, base64_str, meta


def file_to_base64(
    file_path: str,
    encoding: str = "utf-8",
) -> str:
    """将本地文件转换为 base64 编码字符串

    Args:
        file_path: 本地文件路径
        encoding: base64 编码的字符串编码方式，默认 utf-8

    Returns:
        base64 编码字符串

    Raises:
        FileNotFoundError: 文件不存在
    """
    with open(file_path, "rb") as f:
        content = f.read()
    return base64.b64encode(content).decode(encoding)


def detect_mime_type(base64_str: str) -> str:
    """根据 base64 编码内容检测 MIME 类型

    通过解析 base64 数据的文件头魔数来判断图片类型。

    Args:
        base64_str: base64 编码字符串（不含 data URI 前缀）

    Returns:
        MIME 类型字符串，如 "image/png"、"image/jpeg" 等
        如果无法识别则返回 "application/octet-stream"
    """
    # Base64 编码后的魔数前缀（每种类型的文件头特征）
    # 这些是常见图片格式的 base64 编码前缀
    mime_signatures = {
        # PNG: 文件头 89 50 4E 47 (base64: iVBORw0KGgo)
        "/9j/": "image/jpeg",  # JPEG: 文件头 FF D8 FF
        "iVBORw0KGgo": "image/png",  # PNG: 文件头 89 50 4E 47 0D 0A 1A 0A
        "R0lGOD": "image/gif",  # GIF: 文件头 47 49 46 38
        "UklGR": "image/webp",  # WebP: 文件头 52 49 46 46 + WEBP
        "Qk0": "image/bmp",  # BMP: 文件头 42 4D
        "PHN2Zw": "image/svg+xml",  # SVG: 文件头 <?xml 或 <svg
        "SUkq": "image/tiff",  # TIFF (little endian): 49 49 2A 00
        "TWljcm9zb2Z0IFdvcmQg": "application/msword",  # DOC
        "UEsDBBQABgAIAAAAIQ": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
        "JVBERi0xLj": "application/pdf",  # PDF: 文件头 %PDF
    }

    # 移除可能的空白字符
    base64_str = base64_str.strip()

    # 检查是否包含 data URI 前缀，如果有则提取 base64 部分
    if base64_str.startswith("data:"):
        # 格式: data:image/png;base64,xxxxx
        prefix_end = base64_str.find(",")
        if prefix_end != -1:
            # 直接从 data URI 中提取 MIME 类型
            mime_part = base64_str[5:prefix_end]
            if ";base64" in mime_part:
                mime_type = mime_part.split(";")[0]
                return mime_type
            base64_str = base64_str[prefix_end + 1:]

    # 根据文件头魔数判断类型
    for signature, mime_type in mime_signatures.items():
        if base64_str.startswith(signature):
            return mime_type

    # 默认类型
    return "application/octet-stream"


def base64_to_file(
    base64_str: str,
    save_path: str,
    filename: str,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> str:
    """将 base64 编码字符串保存为文件

    Args:
        base64_str: base64 编码字符串
        save_path: 保存目录
        filename: 文件名
        encoding: base64 编码的字符串编码方式，默认 utf-8
        overwrite: 是否覆盖已存在的文件，默认 True

    Returns:
        保存的文件路径
    """
    content = base64.b64decode(base64_str.encode(encoding))
    return asyncio.run(save_file(content, save_path, filename, overwrite=overwrite))
