#!/usr/bin/env python3
"""按关键词从 Pexels 批量下载多样化高清图 —— MemoSight 守门员训练数据采集工具。

守门员是二分类模型，需要多样化训练数据：
  - 正例（positive）：屏幕有用文字（PPT、投影、代码屏、文档）
  - 反例-噪声文字（negative_noise）：招牌、书脊、包装文字、锁屏……
  - 反例-无文字（negative_clean）：风景、人像、物体……

本脚本只负责"按关键词抓高清原图、按 类别/关键词 存盘"，不在这里降清。
降清 / 灰度 / resize 交给 scripts/extract_frames.py 处理。

关键词配置在 scripts/keywords.json，增删关键词改那个文件即可（不用动代码）。

数据图库：Pexels（免费，注册即得 key）。文档：https://www.pexels.com/api/documentation/
API key 从环境变量 PEXELS_API_KEY 读，或从项目根目录的 .env 文件读（.env 已被 .gitignore 忽略）。
.env 格式（每行一个 KEY=VALUE）：
    PEXELS_API_KEY=你的key

输出结构（默认根目录 data/raw）：
    data/raw/<类别>/<关键词slug>/<关键词slug>_<序号>_<pexels图片id>.jpg

文件名带 Pexels 图片 id：
  - **跨关键词/跨类别全局去重**：启动时递归扫描输出根目录建立全局 id 集，
    同一张 Pexels 图（同 id）在整个数据集里只下载一次。这从源头杜绝
    "同图被多关键词重复下载、散入不同子类、切分时跨 split 泄漏"
    （根因见 docs/data-leakage-audit.md）。
  - 带递增序号，天然防覆盖

依赖：requests（见 requirements.txt）。

示例：
    python3 scripts/download_images.py                    # 跑 keywords.json 里全部关键词
    python3 scripts/download_images.py --category positive  # 只跑正例
    python3 scripts/download_images.py --dry-run          # 只打印计划，不下载
    python3 scripts/download_images.py --config my.json --output-root data/raw2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit(
        "缺少依赖 requests。请先安装：pip install requests\n"
        "（依赖已列在 requirements.txt；本脚本不会替你自动安装。）"
    )

# Pexels 搜索接口。每页最多 80 张（来自官方文档，可能调整，以文档为准）。
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
PEXELS_MAX_PER_PAGE = 80

# 默认输出根目录（data/ 不进 git）。
DEFAULT_OUTPUT_ROOT = "data/raw"
# 默认关键词配置文件（与本脚本同目录）。
DEFAULT_CONFIG = Path(__file__).resolve().parent / "keywords.json"

# 限流：两次 API/下载请求之间的默认间隔（秒）。Pexels 免费额度 200 次/小时、20000 次/月，
# 加延时既是礼貌也避免触发限流。
DEFAULT_DELAY = 1.0
# 单次请求失败的默认重试次数与退避基数（秒）。
DEFAULT_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
# 请求超时（秒）。
REQUEST_TIMEOUT = 30

# 配置文件里以下划线开头的键是注释/说明，不当作类别处理。
_COMMENT_PREFIX = "_"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按关键词从 Pexels 批量下载多样化高清图，按 类别/关键词 存盘（守门员训练数据采集）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="关键词配置文件（JSON）。",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="下载输出根目录；其下按 类别/关键词 建子文件夹。",
    )
    parser.add_argument(
        "--category",
        action="append",
        metavar="NAME",
        help="只下载指定类别（可重复指定多个）。不指定则下载配置里全部类别。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="两次请求之间的间隔秒数（限流）。",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help="单次请求失败的重试次数。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要下载的计划，不真正请求 / 下载。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少进度输出。",
    )
    return parser.parse_args(argv)


def load_env_file(env_path: Path) -> None:
    """极简 .env 读取：把 KEY=VALUE 写进 os.environ（不覆盖已存在的环境变量）。

    只支持每行一个 KEY=VALUE，# 开头为注释，自动去掉值两侧引号。不引入额外依赖。
    """
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key(project_root: Path) -> str:
    """从环境变量或项目根目录 .env 取 PEXELS_API_KEY，取不到给清晰报错。"""
    load_env_file(project_root / ".env")
    key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not key:
        sys.exit(
            "未找到 Pexels API key。\n"
            "请到 https://www.pexels.com/api/ 免费申请一个，然后二选一设置：\n"
            "  1) 环境变量：  export PEXELS_API_KEY=你的key\n"
            f"  2) 项目根目录新建 .env 文件（已被 .gitignore 忽略），写入：\n"
            "       PEXELS_API_KEY=你的key\n"
            "拿到 key 前可以先用 --dry-run 看下载计划。"
        )
    return key


def slugify(query: str) -> str:
    """把搜索词转成文件夹/文件名安全的 slug：小写、非字母数字转下划线。"""
    s = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return s or "query"


def load_config(config_path: Path, only_categories: list[str] | None) -> list[dict]:
    """读配置，展开成 [{category, query, count}] 任务列表。下划线开头的键当注释跳过。"""
    if not config_path.is_file():
        sys.exit(f"配置文件不存在：{config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"配置文件不是合法 JSON：{config_path}\n{exc}")

    tasks: list[dict] = []
    for category, entries in data.items():
        if category.startswith(_COMMENT_PREFIX):
            continue
        if only_categories and category not in only_categories:
            continue
        if not isinstance(entries, list):
            sys.exit(f"类别 {category!r} 的值应是列表，实际是 {type(entries).__name__}。")
        for entry in entries:
            query = entry.get("query")
            count = entry.get("count")
            if not query or not isinstance(count, int) or count <= 0:
                sys.exit(f"类别 {category!r} 里有不合法的条目（需要 query + 正整数 count）：{entry}")
            tasks.append({"category": category, "query": query, "count": count})

    if only_categories:
        known = [c for c in data if not c.startswith(_COMMENT_PREFIX)]
        missing = [c for c in only_categories if c not in known]
        if missing:
            sys.exit(f"配置里没有这些类别：{missing}\n可用类别：{known}")
    if not tasks:
        sys.exit("没有可执行的下载任务（配置为空或 --category 过滤后为空）。")
    return tasks


def existing_photo_ids(dest_dir: Path) -> set[str]:
    """扫描目标文件夹里已下载的图，按文件名末尾的 _<id>.<ext> 提取 Pexels 图片 id。"""
    ids: set[str] = set()
    if not dest_dir.is_dir():
        return ids
    for p in dest_dir.iterdir():
        if not p.is_file():
            continue
        m = re.search(r"_(\d+)$", p.stem)
        if m:
            ids.add(m.group(1))
    return ids


def all_existing_photo_ids(output_root: Path) -> set[str]:
    """递归扫描整个输出根目录下所有已下载图片的 Pexels 图片 id（跨类别 / 跨关键词）。

    这是跨关键词全局去重的依据：同一张 Pexels 图常被多个关键词命中，
    若只在单文件夹内去重，同图就会以不同文件名散落到不同子类，进而在
    数据切分时造成跨 split 泄漏（见 docs/data-leakage-audit.md）。
    """
    ids: set[str] = set()
    if not output_root.is_dir():
        return ids
    for p in output_root.rglob("*"):
        if not p.is_file():
            continue
        m = re.search(r"_(\d+)$", p.stem)
        if m:
            ids.add(m.group(1))
    return ids


def request_with_retries(
    method: str,
    url: str,
    retries: int,
    quiet: bool,
    **kwargs,
) -> requests.Response | None:
    """带重试 + 指数退避的请求。返回 Response（即使 4xx/5xx 也返回，由调用方判断），
    全部重试用尽仍是网络异常则返回 None。"""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            # 429（限流）/ 5xx 值得重试；其余直接返回交给调用方处理。
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_BACKOFF_BASE ** (attempt - 1)
                if not quiet:
                    print(f"    HTTP {resp.status_code}，{wait:.0f}s 后重试（{attempt}/{retries}）...")
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as exc:
            wait = RETRY_BACKOFF_BASE ** (attempt - 1)
            if not quiet:
                print(f"    请求异常：{exc}；{wait:.0f}s 后重试（{attempt}/{retries}）...")
            time.sleep(wait)
    return None


def search_photos(
    api_key: str,
    query: str,
    want: int,
    delay: float,
    retries: int,
    quiet: bool,
) -> list[dict]:
    """调用 Pexels 搜索，翻页凑够 want 张，返回 [{id, url(original)}] 列表。"""
    headers = {"Authorization": api_key}
    photos: list[dict] = []
    page = 1
    while len(photos) < want:
        per_page = min(PEXELS_MAX_PER_PAGE, want - len(photos))
        params = {"query": query, "per_page": per_page, "page": page}
        resp = request_with_retries(
            "GET", PEXELS_SEARCH_URL, retries, quiet, headers=headers, params=params
        )
        if resp is None:
            print(f"    搜索 {query!r} 第 {page} 页失败（网络异常，重试用尽），跳过本词剩余。")
            break
        if resp.status_code == 401:
            sys.exit("Pexels 返回 401 未授权：API key 无效或已失效，请检查 PEXELS_API_KEY。")
        if resp.status_code != 200:
            print(f"    搜索 {query!r} 返回 HTTP {resp.status_code}，跳过本词剩余。")
            break

        batch = resp.json().get("photos", [])
        if not batch:
            if not quiet:
                print(f"    {query!r} 第 {page} 页无更多结果，实际可得 {len(photos)} 张。")
            break
        for ph in batch:
            src = ph.get("src", {}).get("original")
            if src:
                photos.append({"id": str(ph["id"]), "url": src})
        page += 1
        time.sleep(delay)
    return photos[:want]


def url_extension(url: str) -> str:
    """从 URL 路径推断图片扩展名，默认 .jpg。"""
    path = url.split("?", 1)[0]
    ext = Path(path).suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def download_photo(
    url: str,
    dest_path: Path,
    retries: int,
    quiet: bool,
) -> bool:
    """下载单张图到 dest_path。先写临时文件再改名，避免半截文件冒充已下载。"""
    resp = request_with_retries("GET", url, retries, quiet, stream=True)
    if resp is None or resp.status_code != 200:
        code = "网络异常" if resp is None else f"HTTP {resp.status_code}"
        if not quiet:
            print(f"    下载失败（{code}）：{url}")
        return False
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp_path.replace(dest_path)
        return True
    except OSError as exc:
        if not quiet:
            print(f"    写文件失败：{exc}")
        tmp_path.unlink(missing_ok=True)
        return False


def run_task(
    task: dict,
    api_key: str,
    output_root: Path,
    seen_ids: set[str],
    delay: float,
    retries: int,
    dry_run: bool,
    quiet: bool,
) -> tuple[int, int]:
    """执行一个 (category, query, count) 任务。返回 (新下载数, 跳过数)。

    seen_ids 是**全局**已下载图片 id 集合（跨类别/关键词），就地更新。
    同一 Pexels 图片 id 在整个数据集里只下载一次，从源头杜绝跨 split 泄漏。
    """
    category, query, want = task["category"], task["query"], task["count"]
    slug = slugify(query)
    dest_dir = output_root / category / slug

    already = existing_photo_ids(dest_dir)  # 本文件夹已有，用于续接序号
    print(f"[{category}/{slug}] 目标 {want} 张，本目录已有 {len(already)} 张，"
          f"全局已知 {len(seen_ids)} 个 id")

    if dry_run:
        print(f"    (dry-run) 将搜索 {query!r} 并最多下载 {want} 张到 {dest_dir}")
        return 0, 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    photos = search_photos(api_key, query, want, delay, retries, quiet)
    if not quiet:
        print(f"    搜索到 {len(photos)} 个候选")

    downloaded = 0
    skipped = 0
    seq = len(already)  # 续接已有序号
    for ph in photos:
        # 全局去重：该 id 在数据集任何角落出现过就跳过（跨关键词/跨类别）。
        if ph["id"] in seen_ids:
            skipped += 1
            continue
        seq += 1
        ext = url_extension(ph["url"])
        fname = f"{slug}_{seq:04d}_{ph['id']}{ext}"
        if download_photo(ph["url"], dest_dir / fname, retries, quiet):
            downloaded += 1
            seen_ids.add(ph["id"])  # 立即纳入全局集，后续关键词不再下同图
            if not quiet and downloaded % 20 == 0:
                print(f"    已下载 {downloaded} 张...")
        else:
            seq -= 1  # 下载失败，序号让回去
        time.sleep(delay)

    print(f"    完成：新下载 {downloaded} 张，跳过已存在/重复 {skipped} 张 → {dest_dir}")
    return downloaded, skipped


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root).expanduser()
    config_path = Path(args.config).expanduser()

    tasks = load_config(config_path, args.category)

    total_want = sum(t["count"] for t in tasks)
    print(f"配置：{config_path}")
    print(f"输出根目录：{output_root.resolve()}")
    print(f"任务数：{len(tasks)}，计划下载上限：{total_want} 张")
    if args.category:
        print(f"仅限类别：{args.category}")
    print("-" * 50)

    # dry-run 不需要 key；真下载才要。
    api_key = "" if args.dry_run else get_api_key(project_root)

    # 全局已下载 id 集合（跨类别/关键词）——跨关键词去重的依据，从源头杜绝泄漏。
    seen_ids = all_existing_photo_ids(output_root)
    print(f"全局已知图片 id：{len(seen_ids)} 个（跨关键词去重基线）")
    print("-" * 50)

    total_dl = 0
    total_skip = 0
    for task in tasks:
        dl, sk = run_task(
            task, api_key, output_root, seen_ids,
            args.delay, args.retries, args.dry_run, args.quiet
        )
        total_dl += dl
        total_skip += sk

    print("-" * 50)
    if args.dry_run:
        print("(dry-run) 仅打印计划，未下载任何文件。")
    else:
        print(f"全部完成：共新下载 {total_dl} 张，跳过已存在 {total_skip} 张 → {output_root.resolve()}")


if __name__ == "__main__":
    main()
