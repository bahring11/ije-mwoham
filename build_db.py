"""games.csv를 임베딩해서 ChromaDB(chroma_db 폴더)에 저장한다.
- 직접 실행하면: DB를 새로 만들고 검색 테스트까지 함
- agent.py에서는: DB가 없을 때 build()를 불러서 자동으로 만듦 (배포 서버용)
"""
import chromadb
import pandas as pd

BATCH = 200


def has(cats: str, *names: str) -> bool:
    items = {c.strip() for c in cats.split(",")}
    return any(n in items for n in names)


def build(client=None, csv_path: str = "games.csv"):
    client = client or chromadb.PersistentClient(path="chroma_db")
    df = pd.read_csv(csv_path).fillna("")
    print(f"게임 {len(df)}개 불러옴")

    ids, docs, metas = [], [], []
    for _, r in df.iterrows():
        cats = str(r["categories"])
        # 임베딩할 문서: 중요한 정보(이름, 장르, 플레이 방식, 짧은 설명)를 앞쪽에 둔다
        doc = (
            f"{r['name']}. Genres: {r['genres']}. Play modes: {cats}. "
            f"{r['description']} {r['about']}"
        )
        ids.append(str(r["appid"]))
        docs.append(doc)
        metas.append({
            "name": str(r["name"]),
            "appid": int(r["appid"]),
            "genres": str(r["genres"]),
            "categories": cats,
            "positive_ratio": float(r["positive_ratio"] or 0),
            "total_reviews": int(r["total_reviews"] or 0),
            "median_playtime_hours": float(r["median_playtime_hours"] or 0),
            "price": float(r["price"] or 0),
            # 필터용 값
            "single": has(cats, "Single-player"),
            "multi": has(cats, "Multi-player", "MMO"),
            "coop": has(cats, "Co-op", "Online Co-op", "LAN Co-op", "Shared/Split Screen Co-op"),
        })

    try:
        client.delete_collection("games")  # 다시 실행할 때 처음부터 새로 만들기
    except Exception:
        pass
    col = client.create_collection("games")

    print("임베딩 저장 중... (처음엔 임베딩 모델을 내려받느라 조금 걸려요)")
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i + BATCH], documents=docs[i:i + BATCH], metadatas=metas[i:i + BATCH])
        print(f"  {min(i + BATCH, len(ids))}/{len(ids)}")
    print(f"저장 완료: {col.count()}개")
    return col


if __name__ == "__main__":
    col = build()

    def search(query: str, where: dict | None = None, n: int = 5):
        res = col.query(query_texts=[query], n_results=n, where=where)
        print(f"\n검색: '{query}'" + (f"  필터: {where}" if where else ""))
        for m, d in zip(res["metadatas"][0], res["distances"][0]):
            print(f"  - {m['name']}  (거리 {d:.3f}, 긍정 {m['positive_ratio']:.0%})")

    search("cooperative survival crafting sandbox to play with friends", where={"coop": True})
    search("relaxing cozy farming life simulation")
    search("short roguelike action runs", where={"single": True})
