"""Stable cross-language rules, feature, action, and target contracts."""

from __future__ import annotations

RULES_SCHEMA_ID = "edgeconnect.star.rules.v1"
CONFORMANCE_SCHEMA_ID = "edgeconnect.star.conformance.v1"
ACTION_LAYOUT_SCHEMA_ID = "edgeconnect.star.action-layout.nodes-then-pass.v1"
EXTERNAL_FEATURE_SCHEMA_ID = "edgeconnect.star.model-features.external.v1"
RULES_VERSION = 1
RULES_HASH_ALGORITHM = "fnv1a64"
RULES_HASH_HEX = "cdb34fb02be82843"
RULES_HASH_WIRE = f"{RULES_HASH_ALGORITHM}:{RULES_HASH_HEX}"
RULES_HASH = 0xCDB34FB02BE82843

# Exact canonical bytes from testdata/star/conformance-v1.json. This gameplay
# contract is deliberately independent from FEATURE_CONTRACT below.
RULES_CANONICAL = (
    "double-star/rules-v1;"
    "rings=integer:3..12;"
    "node-count=5*r*(r+1)/2;"
    "node-order=x:1..r,s:0..4,y:0..x-1;"
    "node-id=5*x*(x-1)/2+s*x+y;"
    "sector-order=*:0,S:1,T:2,A:3,R:4:clockwise;"
    "sector-arithmetic=mod5;"
    "label=sector+ring-char(10->0)+decimal-y;"
    "peri=x==r;"
    "quark=x==r&&y==0;"
    "edges=node-order:cycle,radial,diagonal,corner-cross;"
    "then-ring1-k5-lexicographic;"
    "edge-dedupe=first-undirected-insertion;"
    "csr-neighbor-order=edge-insertion-order;"
    "cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));"
    "radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);"
    "diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);"
    "corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);"
    "bridge=K5((s,1,0),s=0..4);"
    "opening-placements=1;"
    "later-turn-placements=2;"
    "pie=false;"
    "actions=atomic-place-pass;"
    "action-wire=place(node)->node,pass->-1;"
    "legal-order=empty-node-id-ascending,pass-last;"
    "native-action-layout=node-u-at-u,pass-at-n;"
    "pass=forfeit-turn-remainder;"
    "placement-resets-pass-streak=0;"
    "terminal=full-or-two-consecutive-passes;"
    "full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,"
    "movesLeft-in-{0,1},midTurn=(movesLeft>0),passStreak=0,lastMove=final-node,"
    "currentTurnMoves=final-partial-turn;"
    "double-pass-terminal=first-pass-endTurn-to-movesLeft-2-and-clear-"
    "currentTurnMoves,second-pass-retain-actor,no-endTurn,passStreak=2,"
    "movesLeft=2,midTurn=false,lastMove=unchanged;"
    "pair-semantic=AB==BA-excluding-lastMove;"
    "stones=empty:-1,players:0,1;"
    "star=same-color-connected-group-with-at-least-two-directly-occupied-peries;"
    "territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-"
    "alive-color-set-is-exactly-one-player;"
    "score=peries+quark-peri+2*(opponent-stars-own-stars);"
    "tiebreak=quarks;"
    "terminal-value=toMove-perspective:win=1,draw=0,loss=-1;"
    "wdl-class=loss:0,draw:1,win:2;"
    "score-margin=toMove-total-opponent-total;"
    "terminal-legal-actions=empty;"
    "d5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;"
    "d5-coordinate=t=s*x+y(mod5*x);"
    "d5-rk=t+k*x(mod5*x);"
    "d5-fk=k*x-t(mod5*x);"
    "d5-action=map-place-node,pass-invariant"
)
RULES_CONTRACT = RULES_CANONICAL

FEATURE_SCHEMA_VERSION = 2
ACTION_LAYOUT_VERSION = 1
SCORE_MARGIN_MIN = -181
SCORE_MARGIN_MAX = 181
SOFT_POLICY_TEMPERATURE = 4.0


def fnv1a64(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


assert fnv1a64(RULES_CANONICAL) == RULES_HASH

FEATURE_CONTRACT = (
    "startrain/features/v2;"
    "semantic-key=rings,stones,to_move,moves_left,opening,pass_streak,terminal;"
    "perspective=current-player;"
    "node=empty,current,opponent,owner-current,owner-opponent,owner-unclaimed,"
    "alive-current,alive-opponent,peri,quark,ring-fraction,arm-distance,"
    "degree-fraction,bridge,legal;"
    "global=rings,occupancy,current-count,opponent-count,moves-left,opening,"
    "pass-streak,terminal,current-score,opponent-score,margin,current-peries,"
    "opponent-peries,current-quarks,opponent-quarks,current-stars,"
    "opponent-stars,contested-peries;"
    "edges=tangential,radial-diagonal,bridge;"
    "sample-actions=node[0:N],pass[N];"
    "batch-actions=node[0:maxN],pass[maxN];"
    "soft-policy=katago-temperature-4"
)
FEATURE_SCHEMA_HASH = fnv1a64(FEATURE_CONTRACT)

WDL_LOSS = 0
WDL_DRAW = 1
WDL_WIN = 2

# Missing labels use explicit availability masks. This value remains available
# as a real score-margin label because it lies inside the supported range.
MISSING_CLASS = -100

TARGET_POLICY = 1 << 0
TARGET_WDL = 1 << 1
TARGET_SCORE_MARGIN = 1 << 2
TARGET_OWNERSHIP = 1 << 3
TARGET_ALIVE = 1 << 4
TARGET_SOFT_POLICY = 1 << 5
ALL_TARGETS = (
    TARGET_POLICY
    | TARGET_WDL
    | TARGET_SCORE_MARGIN
    | TARGET_OWNERSHIP
    | TARGET_ALIVE
    | TARGET_SOFT_POLICY
)
