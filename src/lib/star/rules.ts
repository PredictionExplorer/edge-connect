/**
 * Versioned Double *Star contract shared by TypeScript and parity ports.
 *
 * The hash covers STAR_RULES_CANONICAL, not implementation source. A
 * consumer can therefore reject fixtures or saved protocol data produced for
 * different semantics without depending on TypeScript formatting or builds.
 * Optional classic and pie-rule UI modes are outside this parity contract.
 */

export const STAR_RULES_VERSION = 1 as const;
export const STAR_RULES_HASH_ALGORITHM = 'fnv1a64' as const;
export const STAR_RULES_SCHEMA_ID = 'edgeconnect.star.rules.v1' as const;
export const STAR_CONFORMANCE_SCHEMA_ID =
  'edgeconnect.star.conformance.v1' as const;
/**
 * Model feature definitions live outside this TypeScript rules package.
 * Exported identifiers let training/inference artifacts pin their own schema
 * without folding model-only changes into the gameplay rules hash.
 */
export const STAR_FEATURE_SCHEMA_ID =
  'edgeconnect.star.model-features.external.v1' as const;
export const STAR_ACTION_LAYOUT_SCHEMA_ID =
  'edgeconnect.star.action-layout.nodes-then-pass.v1' as const;
export const STAR_PASS_ACTION_CODE = -1 as const;

export const STAR_RULES_CONTRACT = {
  schema: STAR_RULES_SCHEMA_ID,
  version: STAR_RULES_VERSION,
  variant: 'double-star',
  board: {
    minimumRings: 3,
    maximumRings: 12,
    nodeCount: '5*rings*(rings+1)/2',
    nodeOrder: 'ring-major, then sector-major, then position-major',
    ringStart: '5*x*(x-1)/2',
    nodeId: '5*x*(x-1)/2 + sector*x + position',
    sectorOrder: ['*', 'S', 'T', 'A', 'R'],
    sectorArithmetic: 'modulo 5',
    ringAddress: {
      sector: '0..4 clockwise',
      ring: '1..rings',
      position: '0..ring-1 clockwise from the sector arm',
    },
    perimeter: 'ring == rings',
    quark: 'ring == rings && position == 0',
    adjacency: [
      'cyclic successor on each ring',
      'radial edge to (sector,ring-1,position) when ring >= 2 and position <= ring-2',
      'diagonal edge to (sector,ring-1,position-1) when ring >= 2 and position >= 1',
      'corner-cross edge to (sector+1,ring-1,0) when ring >= 2 and position == ring-1',
      'complete graph K5 over the five ring-1 nodes',
    ],
    edgeOrder:
      'iterate nodes in node-id order; attempt cycle, radial, diagonal, corner-cross edges in that order; then ring-1 K5 pairs in lexicographic arm order; keep first undirected insertion',
    csrOrder: 'neighbors retain undirected edge insertion order',
    labels:
      'sector character + ring character (ring 10 is 0) + decimal position',
  },
  scoring: {
    emptyValue: -1,
    colors: [0, 1],
    star:
      'a same-color connected group is alive iff it directly occupies at least two perimeter nodes',
    territory:
      'after dead groups are removed, a maximal non-alive region belongs to a player iff every adjacent alive star has that color and at least one alive star is adjacent',
    peries: 'owned perimeter nodes, whether occupied by an alive star or territory',
    quarks: 'owned perimeter nodes with position 0',
    quarkPeri: 'one point when a player owns at least three quarks',
    award: '2 * (opponent alive-star count - own alive-star count)',
    total: 'peries + quarkPeri + award',
    leader: 'higher total, then higher quark count, otherwise tie',
    terminalValue:
      'from terminal toMove perspective: winner 1, loser -1, tie 0',
    wdlClass: 'loss 0, draw 1, win 2',
    scoreMargin: 'toMove total - opponent total',
  },
  game: {
    openingPlacements: 1,
    laterTurnPlacements: 2,
    pieRule: false,
    actionTypes: ['place', 'pass'],
    actionWireEncoding: {
      place: 'dense node id 0..n-1',
      pass: STAR_PASS_ACTION_CODE,
    },
    legalActionOrder: 'legal placements by ascending node id, then pass',
    nativeActionLayout: 'node u at slot u; pass at slot n',
    pass: 'forfeits the remainder of the current turn',
    termination: 'board full or two consecutive pass actions',
    fullBoardResidual:
      'the final placement decrements movesLeft and terminates before endTurn; actor and turnCount are retained; movesLeft is 0 or 1; midTurn is movesLeft > 0; passStreak is 0; lastMove is the final node; currentTurnMoves retains the final partial turn',
    doublePassResidual:
      'the first pass ends the turn with movesLeft 2 and clears currentTurnMoves; the second pass terminates without endTurn with passStreak 2, movesLeft 2, midTurn false, unchanged lastMove, and the second passer retained as toMove',
    terminalLegalActions: 'none',
    placement: 'only an empty in-range node is legal',
    replay: 'apply the ordered action log to a fresh initial state',
    pairEquivalence:
      'the two placements AB and BA in one complete nonterminal turn have the same semantic state; lastMove is presentation metadata',
  },
  symmetry: {
    group: 'D5',
    order: ['r0', 'r1', 'r2', 'r3', 'r4', 'f0', 'f1', 'f2', 'f3', 'f4'],
    ringCoordinate: 't = sector*ring + position modulo 5*ring',
    rotation: 'r(k): t -> t + k*ring for k in 0..4',
    reflection: 'f(k): t -> k*ring - t for k in 0..4',
    action: 'place transforms by the node map; pass is invariant',
  },
} as const;

/**
 * Compact ASCII wire contract. Rust and Python consumers mirror these exact
 * bytes so every runtime derives the same unsigned 64-bit fingerprint.
 */
export const STAR_RULES_CANONICAL = [
  'double-star/rules-v1;',
  'rings=integer:3..12;',
  'node-count=5*r*(r+1)/2;',
  'node-order=x:1..r,s:0..4,y:0..x-1;',
  'node-id=5*x*(x-1)/2+s*x+y;',
  'sector-order=*:0,S:1,T:2,A:3,R:4:clockwise;',
  'sector-arithmetic=mod5;',
  'label=sector+ring-char(10->0)+decimal-y;',
  'peri=x==r;',
  'quark=x==r&&y==0;',
  'edges=node-order:cycle,radial,diagonal,corner-cross;then-ring1-k5-lexicographic;',
  'edge-dedupe=first-undirected-insertion;',
  'csr-neighbor-order=edge-insertion-order;',
  'cycle=(s,x,y)-(y<x-1?(s,x,y+1):(s+1,x,0));',
  'radial=x>=2&&y<=x-2?(s,x,y)-(s,x-1,y);',
  'diagonal=x>=2&&y>=1?(s,x,y)-(s,x-1,y-1);',
  'corner-cross=x>=2&&y==x-1?(s,x,y)-(s+1,x-1,0);',
  'bridge=K5((s,1,0),s=0..4);',
  'opening-placements=1;',
  'later-turn-placements=2;',
  'pie=false;',
  'actions=atomic-place-pass;',
  'action-wire=place(node)->node,pass->-1;',
  'legal-order=empty-node-id-ascending,pass-last;',
  'native-action-layout=node-u-at-u,pass-at-n;',
  'pass=forfeit-turn-remainder;',
  'placement-resets-pass-streak=0;',
  'terminal=full-or-two-consecutive-passes;',
  'full-terminal=decrement-movesLeft,retain-actor-and-turnCount,no-endTurn,movesLeft-in-{0,1},midTurn=(movesLeft>0),passStreak=0,lastMove=final-node,currentTurnMoves=final-partial-turn;',
  'double-pass-terminal=first-pass-endTurn-to-movesLeft-2-and-clear-currentTurnMoves,second-pass-retain-actor,no-endTurn,passStreak=2,movesLeft=2,midTurn=false,lastMove=unchanged;',
  'pair-semantic=AB==BA-excluding-lastMove;',
  'stones=empty:-1,players:0,1;',
  'star=same-color-connected-group-with-at-least-two-directly-occupied-peries;',
  'territory=after-dead-removal,maximal-nonalive-component-owned-iff-adjacent-alive-color-set-is-exactly-one-player;',
  'score=peries+quark-peri+2*(opponent-stars-own-stars);',
  'tiebreak=quarks;',
  'terminal-value=toMove-perspective:win=1,draw=0,loss=-1;',
  'wdl-class=loss:0,draw:1,win:2;',
  'score-margin=toMove-total-opponent-total;',
  'terminal-legal-actions=empty;',
  'd5-order=r0,r1,r2,r3,r4,f0,f1,f2,f3,f4;',
  'd5-coordinate=t=s*x+y(mod5*x);',
  'd5-rk=t+k*x(mod5*x);',
  'd5-fk=k*x-t(mod5*x);',
  'd5-action=map-place-node,pass-invariant',
].join('');

/** Compute the unsigned 64-bit FNV-1a hash used by the parity contract. */
export function fnv1a64(value: string): string {
  let hash = BigInt('0xcbf29ce484222325');
  const prime = BigInt('0x100000001b3');
  const mask = BigInt('0xffffffffffffffff');
  for (const byte of new TextEncoder().encode(value)) {
    hash = ((hash ^ BigInt(byte)) * prime) & mask;
  }
  return hash.toString(16).padStart(16, '0');
}

/** Stable wire fingerprint for the complete gameplay contract above. */
export const STAR_RULES_HASH =
  'fnv1a64:cdb34fb02be82843' as const;
