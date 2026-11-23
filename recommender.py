"""RAG 추천: 내 플레이 기록(Steam API) + 비슷한 게임 검색(ChromaDB) → Gemini가 추천 답변 생성"""
import json
import os
import re
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from steam_api import get_owned_games, resolve_steamid

load_dotenv()
# 60초 안에 답이 없으면 끊고, 몰래 여러 번 재시도하지 않게 설정
llm = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)),
)
col = chromadb.PersistentClient(path="chroma_db").get_collection("games")


def pick_models() -> list[str]:
    """쓸 수 있는 Flash 모델을 최신순으로 정리 (.env에 GEMINI_MODEL이 있으면 그걸 맨 앞에)"""
    skip = ("preview", "exp", "image", "tts", "live", "audio", "thinking")
    found = []
    for m in llm.models.list():
        name = m.name.replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" not in actions or "flash" not in name:
            continue
        if any(s in name for s in skip):
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        if ver:
            is_lite = "lite" in name
            found.append((is_lite, -float(ver.group(1)), name))  # 일반 Flash 먼저, 최신순
    models = [name for _, _, name in sorted(found)]
    if os.getenv("GEMINI_MODEL"):
        models.insert(0, os.getenv("GEMINI_MODEL"))
    if not models:
        raise SystemExit("쓸 수 있는 Gemini Flash 모델을 못 찾았어요. GEMINI_API_KEY를 확인하세요.")
    return models[:4]


MODELS = pick_models()
print(f"사용 모델 (앞에서부터 시도): {', '.join(MODELS)}")


def ask_llm(prompt: str, json_mode: bool = False) -> str:
    """모델이 바쁘면(503) 또는 한도 초과면(429) 다음 모델로 넘어가서 다시 시도"""
    config = types.GenerateContentConfig(
        temperature=0.2 if json_mode else 0.7,
        response_mime_type="application/json" if json_mode else None,
    )
    last_err = None
    for model in MODELS:
        try:
            start = time.time()
            resp = llm.models.generate_content(model=model, contents=prompt, config=config)
            print(f"    ({model} 응답 {time.time() - start:.1f}초)")
            return resp.text
        except errors.APIError as e:
            if e.code in (429, 500, 503, 404):
                print(f"    ({model} 사용 불가: {e.code}, 다음 모델로 시도)")
                last_err = e
                continue
            raise
    raise last_err


# ---------- 1. 내 플레이 기록 정리 (Steam API) ----------
def build_profile(steamid: str) -> dict:
    games = get_owned_games(steamid)
    return {
        "count": len(games),
        "top": [f"{g['name']} ({g['playtime_forever'] / 60:.0f}h)" for g in games[:10]],
        "recent": [f"{g['name']} ({g['playtime_2weeks'] / 60:.1f}h)"
                   for g in games if g.get("playtime_2weeks")],
        "owned": {g["appid"]: g for g in games},
    }


# ---------- 2. 질문 분석: 한국어 질문 → 영어 검색어 + 필터 ----------
def analyze(question: str, profile: dict) -> dict:
    prompt = f"""You turn a Korean game-recommendation request into a search plan.
User's most played games: {', '.join(profile['top'])}
Request: "{question}"

Return JSON with these keys:
- "search_query": 1-2 English sentences describing the kind of game wanted
  (genre, mood, mechanics). If the request is vague, base it on the user's most played games.
- "coop": true if they want to play together with friends, otherwise null
- "single": true if they want to play alone, otherwise null
- "owned_only": true only if they want games they already own, otherwise false"""
    return json.loads(ask_llm(prompt, json_mode=True))


# ---------- 3. 검색: 벡터 DB에서 후보 게임 찾기 ----------
def retrieve(plan: dict, profile: dict, n: int = 15) -> list[dict]:
    conds = [{k: True} for k in ("coop", "single") if plan.get(k)]
    if plan.get("owned_only"):
        conds.append({"appid": {"$in": [int(a) for a in profile["owned"]]}})
    where = {"$and": conds} if len(conds) > 1 else (conds[0] if conds else None)

    res = col.query(query_texts=[plan["search_query"]], n_results=n, where=where)
    cands = []
    for m, doc in zip(res["metadatas"][0], res["documents"][0]):
        mine = profile["owned"].get(m["appid"])
        cands.append({
            "name": m["name"],
            "genres": m["genres"],
            "modes": m["categories"],
            "positive": m["positive_ratio"],
            "median_hours": m["median_playtime_hours"],
            "owned": f"보유 중 ({mine['playtime_forever'] / 60:.0f}시간 플레이)" if mine else "미보유",
            "summary": doc[:400],
        })
    return cands


# ---------- 4. 생성: 기록 + 후보를 프롬프트에 넣어 추천 답변 ----------
def recommend(question: str, profile: dict) -> tuple[dict, list[dict], str]:
    print("  1/3 질문 분석 중...")
    plan = analyze(question, profile)
    print("  2/3 벡터 DB에서 후보 검색 중...")
    cands = retrieve(plan, profile)
    print("  3/3 추천 답변 작성 중...")
    context = "\n".join(
        f"[{i + 1}] {c['name']} | 장르: {c['genres']} | 방식: {c['modes']} | "
        f"긍정 {c['positive']:.0%} | 보통 총 플레이 {c['median_hours']}시간 | {c['owned']}\n"
        f"    {c['summary']}"
        for i, c in enumerate(cands)
    )
    prompt = f"""너는 스팀 게임 추천 도우미야. 아래 [사용자 정보]와 [후보 게임]만 근거로 답해.
후보 목록에 없는 게임은 절대 추천하지 마.

[사용자 정보]
- 보유 게임 {profile['count']}개
- 많이 한 게임: {', '.join(profile['top'])}
- 최근 2주 플레이: {', '.join(profile['recent']) or '없음'}

[질문]
{question}

[후보 게임]
{context}

[답변 규칙]
- 질문 조건에 가장 잘 맞는 게임 3개를 골라 추천해.
- 각 게임마다 왜 이 사용자에게 맞는지 플레이 기록과 연결해서 2~3문장으로 설명해.
- 이미 10시간 넘게 한 게임은 추천하지 마. 보유했지만 거의 안 한 게임이 조건에 맞으면 먼저 추천해.
- 마지막 줄에 "근거: 내 플레이 기록 · 스토어 게임 설명"이라고 적어.
- 한국어, 친근한 말투로."""
    return plan, cands, ask_llm(prompt)


if __name__ == "__main__":
    user = os.getenv("STEAM_ID") or input("Steam ID 또는 프로필 주소: ")
    steamid = resolve_steamid(user)
    profile = build_profile(steamid)
    print(f"보유 게임 {profile['count']}개 불러옴. 질문을 입력하세요 (빈 줄 입력 시 종료)\n")

    while True:
        q = input("질문> ").strip()
        if not q:
            break
        try:
            plan, cands, answer = recommend(q, profile)
        except errors.APIError as e:
            if e.code in (429, 503):
                print("\n[오류] 모든 Gemini 모델이 바쁘거나 한도에 걸렸어요. 1분 정도 뒤에 다시 질문하세요.\n")
            else:
                print(f"\n[오류] Gemini API 오류 {e.code}: {e.message}\n")
            continue
        except Exception as e:
            print(f"\n[오류] {type(e).__name__}: {e}\n")
            continue
        print(f"\n[검색 계획] {plan}")
        print(f"[후보] {', '.join(c['name'] for c in cands)}\n")
        print(answer, "\n")
