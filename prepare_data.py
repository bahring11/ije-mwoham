import html
import json
import re
import pandas as pd
from datasets import load_dataset

TOP_N = 2000          # 남길 게임 수
MIN_REVIEWS = 1000    # 최소 리뷰 수
MIN_RATIO = 0.70      # 최소 긍정 비율
ABOUT_MAX = 1200      # 상세 설명 최대 글자 수

# 추천에 의미 있는 분류만 남김 (트레이딩 카드, 리모트 플레이 같은 건 제외)
KEEP_CATEGORIES = {
    "Single-player", "Multi-player", "PvP", "Online PvP", "LAN PvP",
    "Shared/Split Screen PvP", "Co-op", "Online Co-op", "LAN Co-op",
    "Shared/Split Screen Co-op", "Shared/Split Screen", "MMO",
    "Cross-Platform Multiplayer", "VR Support", "VR Only",
}


def clean_html(text):
    """상세 설명의 HTML 태그를 지우고 일정 길이로 자름"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:ABOUT_MAX]


def to_list(v):
    """태그/장르가 리스트·딕셔너리·문자열 어떤 형태로 와도 리스트로 변환"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, dict):
        v = {k: n for k, n in v.items() if n}          # 값이 있는 태그만
        return [k for k, _ in sorted(v.items(), key=lambda x: -x[1])]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if x]
    s = str(v).strip()
    if s.startswith(("[", "{")):
        try:
            return to_list(json.loads(s))
        except json.JSONDecodeError:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


print("데이터셋 다운로드 중... (처음 한 번만 오래 걸려요)")
ds = load_dataset("FronkonGames/steam-games-dataset", split="train")
df = ds.to_pandas()
print(f"전체 게임 수: {len(df):,}")

# 평가 좋은 인기 게임만 남기기
df["total_reviews"] = df["positive"] + df["negative"]
df = df[df["total_reviews"] >= MIN_REVIEWS]
df["positive_ratio"] = df["positive"] / df["total_reviews"]
df = df[df["positive_ratio"] >= MIN_RATIO]
df = df.sort_values("total_reviews", ascending=False).head(TOP_N)

# 필요한 정보만 정리
out = pd.DataFrame({
    "appid": df["appID"],
    "name": df["name"],
    "description": df["short_description"].fillna(""),
    "genres": df["genres"].apply(lambda v: ", ".join(to_list(v))),
    "about": df["detailed_description"].apply(clean_html),
    "categories": df["categories"].apply(
        lambda v: ", ".join(c for c in to_list(v) if c in KEEP_CATEGORIES)),
    "positive_ratio": df["positive_ratio"].round(3),
    "total_reviews": df["total_reviews"],
    "median_playtime_hours": (df["median_playtime_forever"] / 60).round(1),
    "price": df["price"],
})
out.to_csv("games.csv", index=False, encoding="utf-8-sig")

print(f"\n저장 완료: games.csv ({len(out)}개)")
for name in ["Stardew Valley", "Terraria", "Core Keeper"]:
    row = out[out["name"] == name]
    if not row.empty:
        r = row.iloc[0]
        print(f"\n[{r['name']}]\n 장르: {r['genres']}\n 분류: {r['categories']}\n 상세: {r['about'][:120]}...")