import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kalshi_mm.data.build import iter_games

for g in iter_games("data/raw", "nba", "2024-25", pregame_hours=6):
    s = g.stream
    nb = int((s.etype == "book").sum())
    nt = int((s.etype == "trade").sum())
    print(f"{g.ticker}  tip={g.tip_ts} ({g.tip_source})  close={g.meta['close_time']}")
    print(f"  pregame events: book={nb} trades={nt}")
    nn = s.dropna(subset=["bid", "ask"])
    if not nn.empty:
        spreads = (nn["ask"] - nn["bid"])
        print(f"  spread: median={spreads.median():.1f}c  mean={spreads.mean():.2f}c  "
              f"bid range [{nn['bid'].min():.0f},{nn['bid'].max():.0f}]")
