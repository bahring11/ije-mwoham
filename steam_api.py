import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("STEAM_API_KEY")


def resolve_steamid(user_input: str) -> str | None:
    """숫자 Steam ID(17자리)면 그대로, 커스텀 이름이면 숫자 ID로 변환"""
    user_input = user_input.strip().rstrip("/").split("/")[-1]  # 프로필 주소를 넣어도 동작
    if user_input.isdigit() and len(user_input) == 17:
        return user_input
    r = requests.get(
        "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/",
        params={"key": KEY, "vanityurl": user_input},
        timeout=10,
    ).json()
    return r["response"].get("steamid")


def get_owned_games(steamid: str) -> list[dict]:
    """보유 게임 목록을 플레이 시간 많은 순으로 반환"""
    r = requests.get(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/",
        params={"key": KEY, "steamid": steamid, "include_appinfo": 1},
        timeout=10,
    ).json()
    games = r["response"].get("games", [])
    return sorted(games, key=lambda g: g["playtime_forever"], reverse=True)


def get_prices(appids: list[int], cc: str = "kr") -> dict[int, dict]:
    """스팀 상점의 현재 가격·할인 정보 (한국 원화 기준)

    반환 예: {620: {"free": False, "now": "₩ 8,250", "orig": "₩ 16,500", "discount": 50}}
    가격 정보를 못 가져온 게임은 결과에 없음.
    """
    appids = [int(a) for a in dict.fromkeys(appids)]
    result = {}
    for i in range(0, len(appids), 50):  # 한 번에 너무 많이 보내지 않게 50개씩
        chunk = appids[i:i + 50]
        try:
            r = requests.get(
                "https://store.steampowered.com/api/appdetails",
                params={"appids": ",".join(map(str, chunk)), "cc": cc,
                        "filters": "price_overview"},
                timeout=8,
            ).json() or {}
        except Exception as e:
            print(f"      (가격 조회 실패: {e})")
            continue
        for appid in chunk:
            info = r.get(str(appid)) or {}
            if not info.get("success"):
                continue
            data = info.get("data")
            po = data.get("price_overview") if isinstance(data, dict) else None
            if po:
                result[appid] = {
                    "free": False,
                    "now": po.get("final_formatted", ""),
                    "orig": po.get("initial_formatted") or po.get("final_formatted", ""),
                    "discount": int(po.get("discount_percent") or 0),
                }
            else:  # 성공했는데 가격 정보가 없으면 무료 게임
                result[appid] = {"free": True, "now": "무료", "orig": "", "discount": 0}
    return result


# ---------------------------------------------------------------- 리뷰
REVIEW_TTL = 6 * 3600  # 한 번 가져온 리뷰는 6시간 동안 다시 요청하지 않음
_review_cache: dict[int, tuple[float, dict | None]] = {}

SCORE_KO = {
    "Overwhelmingly Positive": "압도적으로 긍정적", "Very Positive": "매우 긍정적",
    "Positive": "긍정적", "Mostly Positive": "대체로 긍정적", "Mixed": "복합적",
    "Mostly Negative": "대체로 부정적", "Negative": "부정적",
    "Very Negative": "매우 부정적", "Overwhelmingly Negative": "압도적으로 부정적",
}


def _clean_review(text: str, limit: int = 220) -> str:
    """BBCode([h1], [b] 등)와 줄바꿈을 정리하고 길면 자름"""
    text = re.sub(r"\[/?[^\]]{1,20}\]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _fetch_reviews(appid: int, language: str) -> dict | None:
    r = requests.get(
        f"https://store.steampowered.com/appreviews/{appid}",
        params={"json": 1, "language": language, "filter": "all",  # all = 도움이 된 순
                "num_per_page": 10, "purchase_type": "all", "review_type": "all"},
        timeout=6,
    ).json()
    return r if r.get("success") == 1 else None


def get_reviews(appid: int, n: int = 3) -> dict | None:
    """도움이 된 순 리뷰 n개 + 평가 요약. 한국어 리뷰가 부족하면 영어 리뷰로 보충

    반환 예: {"summary": "매우 긍정적", "total": 1234, "lang": "한국어",
             "items": [{"text": "...", "up": True, "hours": 52, "votes": 120}]}
    """
    appid = int(appid)
    cached = _review_cache.get(appid)
    if cached and time.time() - cached[0] < REVIEW_TTL:
        return cached[1]

    result = None
    try:
        for language, label in (("koreana", "한국어"), ("english", "영어")):
            data = _fetch_reviews(appid, language)
            if not data:
                continue
            items = []
            for rv in data.get("reviews", []):
                text = _clean_review(rv.get("review", ""))
                if len(text) < 15:  # 너무 짧은 리뷰("굿", "ㅇㅇ")는 건너뜀
                    continue
                items.append({
                    "text": text,
                    "up": bool(rv.get("voted_up")),
                    "hours": round((rv.get("author") or {}).get("playtime_forever", 0) / 60),
                    "votes": int(rv.get("votes_up") or 0),
                })
            items.sort(key=lambda x: x["votes"], reverse=True)
            if len(items) >= 2 or (items and language == "english"):
                qs = data.get("query_summary") or {}
                desc = qs.get("review_score_desc", "")
                result = {"summary": SCORE_KO.get(desc, desc), "total": int(qs.get("total_reviews") or 0),
                          "lang": label, "items": items[:n]}
                break
    except Exception as e:
        print(f"      (리뷰 조회 실패 {appid}: {e})")
        return None  # 실패는 캐시하지 않음 (다음에 다시 시도)

    _review_cache[appid] = (time.time(), result)
    return result


def get_reviews_many(appids: list[int], n: int = 3) -> dict[int, dict]:
    """여러 게임의 리뷰를 동시에(병렬로) 가져옴"""
    appids = list(dict.fromkeys(int(a) for a in appids))
    if not appids:
        return {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = pool.map(lambda a: get_reviews(a, n), appids)
    return {a: r for a, r in zip(appids, results) if r}


if __name__ == "__main__":
    if not KEY:
        print("STEAM_API_KEY를 못 찾았어요. .env 파일을 확인하세요.")
        raise SystemExit

    user = input("Steam ID 또는 프로필 주소: ")
    steamid = resolve_steamid(user)
    if not steamid:
        print("Steam ID를 찾지 못했어요. 프로필 주소를 확인하세요.")
        raise SystemExit

    games = get_owned_games(steamid)
    if not games:
        print("게임 목록이 비어 있어요. 프로필의 '게임 세부 정보'가 공개인지 확인하세요.")
        raise SystemExit

    print(f"\n보유 게임 {len(games)}개 / 많이 한 게임 TOP 10")
    for g in games[:10]:
        hours = g["playtime_forever"] / 60
        recent = g.get("playtime_2weeks", 0) / 60
        print(f"- {g['name']}: {hours:.1f}시간 (최근 2주 {recent:.1f}시간)")