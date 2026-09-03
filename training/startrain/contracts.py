"""Stable cross-language rules, feature, action, and target contracts."""

from __future__ import annotations

RULES_SCHEMA_ID = "edgeconnect.star.rules.v3"
CONFORMANCE_SCHEMA_ID = "edgeconnect.star.conformance.v3"
ACTION_LAYOUT_SCHEMA_ID = "edgeconnect.star.action-layout.nodes-only.v1"
EXTERNAL_FEATURE_SCHEMA_ID = "edgeconnect.star.model-features.external.v3"
RULES_VERSION = 3
RULES_HASH_ALGORITHM = "fnv1a64"
RULES_HASH_HEX = "a5d932b0ef8354e8"
RULES_HASH_WIRE = f"{RULES_HASH_ALGORITHM}:{RULES_HASH_HEX}"
RULES_HASH = 0xA5D932B0EF8354E8

# The previous lineage's contract. Only the lineage-transfer tool and the
# cross-schema arena may accept artifacts carrying these identifiers.
LEGACY_RULES_SCHEMA_ID = "edgeconnect.star.rules.v2"
LEGACY_RULES_VERSION = 2
LEGACY_RULES_HASH_HEX = "2da3783519381453"
LEGACY_RULES_HASH_WIRE = f"{RULES_HASH_ALGORITHM}:{LEGACY_RULES_HASH_HEX}"
LEGACY_RULES_HASH = 0x2DA3783519381453

# Exact canonical bytes from ``src/lib/star/rules.ts``. This gameplay contract
# is deliberately independent from FEATURE_CONTRACT below.
RULES_CANONICAL = (
    "double-star/rules-v3;"
    "rings=even:{4,6,8,10};"
    "node-count=5*r*(r+1)/2;"
    "node-order=x:1..r,s:0..4,y:0..x-1;"
    "node-id=5*x*(x-1)/2+s*x+y;"
    "sector-order=*:0,S:1,T:2,A:3,R:4:clockwise;"
    "sector-arithmetic=mod5;"
    "label=sector+ring-char(10->0)+decimal-y;"
    "peri=x==r;"
    "quark=x==r&&y==0;"
    "edges=node-order:cycle,radial,diagonal,corner-cross;then-ring1-k5-lexicographic;"
    "edge-dedupe=first-undirected-insertion;"
    "csr-neighbor-order=edge-insertion-order;"
    "cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));"
    "radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);"
    "diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);"
    "corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);"
    "bridge=K5((s,1,0),s=0..4);"
    "modes={classic:turn-size-1,double:opening-1-then-2};"
    "handicap=k-consecutive-opening-placements-by-player0,k-in-1..9,k=1-is-standard;"
    "pie=optional:after-first-turn-player1-may-swap,recolor-opening-stones-to-player1,"
    "player0-moves-next-with-full-turn,swap-unavailable-after-any-placement;"
    "handicap-excludes-pie;"
    "variant-in-semantic-key=mode,handicap,pie;"
    "history-in-semantic-key=currentTurn,previousTurn,ownPreviousTurn,handicapStones;"
    "actions=atomic-place|swap;"
    "action-wire=place(node)->node,swap->node-count;"
    "legal-order=empty-node-id-ascending;"
    "native-action-layout=node-u-at-u;"
    "terminal=full;"
    "full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,"
    "movesLeft-below-turn-size,midTurn=(movesLeft>0),lastMove=final-node,"
    "currentTurnMoves=final-partial-turn;"
    "pair-semantic=AB==BA-excluding-lastMove;"
    "stones=empty:-1,players:0,1;"
    "star=same-color-connected-group-with-at-least-two-directly-occupied-peries;"
    "territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-"
    "alive-color-set-is-exactly-one-player;"
    "score=peries+quark-peri+2*(opponent-stars-own-stars);"
    "tiebreak=quarks;"
    "terminal-value=toMove-perspective:win=1,loss=-1,tie=invalid;"
    "outcome-class=loss:0,win:1;"
    "score-margin=toMove-total-opponent-total;"
    "terminal-legal-actions=empty;"
    "d5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;"
    "d5-coordinate=t=s*x+y(mod5*x);"
    "d5-rk=t+k*x(mod5*x);"
    "d5-fk=k*x-t(mod5*x);"
    "d5-action=map-place-node,swap-fixed"
)
RULES_CONTRACT = RULES_CANONICAL

FEATURE_SCHEMA_VERSION = 4
LEGACY_FEATURE_SCHEMA_VERSION = 3
ACTION_LAYOUT_VERSION = 1
SCORE_MARGIN_MIN = -151
SCORE_MARGIN_MAX = 151
SOFT_POLICY_TEMPERATURE = 4.0
MAX_HANDICAP = 9
# Playout-doubling advantage input range for the side to move.
MAX_PLAYOUT_DOUBLING_ADVANTAGE = 3

MODES = ("classic", "double")
MODE_INDEX = {"classic": 0, "double": 1}
MODE_TURN_SIZE = {"classic": 1, "double": 2}

# Mixture segments used for replay stratification and arena floors.
SEGMENT_STANDARD = "standard"
SEGMENT_CLASSIC = "classic"
SEGMENT_HANDICAP = "handicap"
SEGMENT_PIE = "pie"
SEGMENTS = (SEGMENT_STANDARD, SEGMENT_CLASSIC, SEGMENT_HANDICAP, SEGMENT_PIE)


def fnv1a64(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result = ((result ^ byte) * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


assert fnv1a64(RULES_CANONICAL) == RULES_HASH

FEATURE_CONTRACT = (
    "startrain/features/v4;"
    "semantic-key=rings,stones,to_move,moves_left,opening,terminal,mode,handicap,"
    "pie_pending,swap_available,current_turn,previous_turn,own_previous_turn,"
    "handicap_stones;"
    "context=history_known,pda;"
    "perspective=current-player;"
    "node=empty,current,opponent,owner-current,owner-opponent,owner-unclaimed,"
    "alive-current,alive-opponent,peri,quark,ring-fraction,arm-distance,"
    "degree-fraction,bridge,legal,placed-this-turn,own-previous-turn,"
    "opponent-previous-turn,handicap-stone;"
    "global=rings,occupancy,current-count,opponent-count,moves-left-of-turn,"
    "opening,terminal,current-score,opponent-score,margin,current-peries,"
    "opponent-peries,current-quarks,opponent-quarks,current-stars,"
    "opponent-stars,contested-peries,turn-size,handicap,handicap-phase,"
    "handicap-remaining,pie-pending,swap-available,history-known,pda;"
    "edges=tangential,radial-diagonal,bridge;"
    "relations=ring-difference,angular-offset-bucket,shortest-path-bucket,peri-pair;"
    "sample-actions=node[0:N];"
    "batch-actions=node[0:maxN];"
    "soft-policy=katago-temperature-4"
)
FEATURE_SCHEMA_HASH = fnv1a64(FEATURE_CONTRACT)

# The previous lineage's feature contract (schema v3), retained so the teacher
# checkpoint can be encoded during lineage transfer and cross-schema arenas.
LEGACY_FEATURE_CONTRACT = (
    "startrain/features/v3;"
    "semantic-key=rings,stones,to_move,moves_left,opening,terminal;"
    "perspective=current-player;"
    "node=empty,current,opponent,owner-current,owner-opponent,owner-unclaimed,"
    "alive-current,alive-opponent,peri,quark,ring-fraction,arm-distance,"
    "degree-fraction,bridge,legal;"
    "global=rings,occupancy,current-count,opponent-count,moves-left,opening,"
    "terminal,current-score,opponent-score,margin,current-peries,"
    "opponent-peries,current-quarks,opponent-quarks,current-stars,"
    "opponent-stars,contested-peries;"
    "edges=tangential,radial-diagonal,bridge;"
    "sample-actions=node[0:N];"
    "batch-actions=node[0:maxN];"
    "soft-policy=katago-temperature-4"
)
LEGACY_FEATURE_SCHEMA_HASH = fnv1a64(LEGACY_FEATURE_CONTRACT)
assert LEGACY_FEATURE_SCHEMA_HASH == 0x6B5B00F638E9C16B

OUTCOME_LOSS = 0
OUTCOME_WIN = 1

# Missing labels use explicit availability masks. This value remains available
# as a real score-margin label because it lies inside the supported range.
MISSING_CLASS = -100

TARGET_POLICY = 1 << 0
TARGET_OUTCOME = 1 << 1
TARGET_SCORE_MARGIN = 1 << 2
TARGET_OWNERSHIP = 1 << 3
TARGET_ALIVE = 1 << 4
TARGET_SOFT_POLICY = 1 << 5
# Teacher soft targets attached by the lineage-transfer importer.
TARGET_TEACHER = 1 << 6
ALL_TARGETS = (
    TARGET_POLICY
    | TARGET_OUTCOME
    | TARGET_SCORE_MARGIN
    | TARGET_OWNERSHIP
    | TARGET_ALIVE
    | TARGET_SOFT_POLICY
)
