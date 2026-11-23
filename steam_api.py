import os
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