/**
 * Deterministic cross-language conformance vectors for the *Star rules.
 *
 * The output intentionally uses only JSON primitives and arrays so a Rust
 * implementation can deserialize it without reproducing JavaScript typed
 * arrays or object-key conventions.
 */

import { getBoard, MAX_RINGS, MIN_RINGS, type Board } from './board';
import {
  applyAction,
  initialState,
  type GameAction,
  type GameConfig,
  type GameState,
} from './game';
import {
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_CONFORMANCE_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_PASS_ACTION_CODE,
  STAR_RULES_CANONICAL,
  STAR_RULES_CONTRACT,
  STAR_RULES_HASH,
  STAR_RULES_HASH_ALGORITHM,
  STAR_RULES_SCHEMA_ID,
  STAR_RULES_VERSION,
} from './rules';
import { EMPTY, scorePosition } from './scoring';
import {
  D5_SYMMETRIES,
  getD5Maps,
  inverseD5Symmetry,
} from './symmetry';

function adjacentNodes(board: Board, node: number): number[] {
  const adjacent: number[] = [];
  for (let e = board.adjOff[node]; e < board.adjOff[node + 1]; e++) {
    adjacent.push(board.adj[e]);
  }
  adjacent.sort((a, b) => a - b);
  return adjacent;
}

function boardVector(board: Board) {
  let maximumDegree = 0;
  for (let u = 0; u < board.n; u++) {
    maximumDegree = Math.max(
      maximumDegree,
      board.adjOff[u + 1] - board.adjOff[u],
    );
  }
  return {
    rings: board.rings,
    nodeCount: board.n,
    perimeterCount: board.periCount,
    edgeCount: board.adj.length / 2,
    maximumDegree,
    nodeOrdering: {
      order: ['ring', 'sector', 'position'],
      ringStart: '5*x*(x-1)/2',
      nodeId: '5*x*(x-1)/2 + sector*x + position',
    },
    sectorOf: Array.from(board.sectorOf),
    ringOf: Array.from(board.ringOf),
    positionOf: Array.from(board.posOf),
    perimeterMask: Array.from(board.isPeri),
    quarkMask: Array.from(board.isQuark),
    labels: [...board.labels],
    adjacencyOffsets: Array.from(board.adjOff),
    adjacency: Array.from(board.adj),
    bridge: [...board.bridge],
    nodes: Array.from({ length: board.n }, (_, id) => ({
      id,
      sector: board.sectorOf[id],
      ring: board.ringOf[id],
      position: board.posOf[id],
      label: board.labels[id],
      perimeter: board.isPeri[id] === 1,
      quark: board.isQuark[id] === 1,
      adjacent: adjacentNodes(board, id),
    })),
    symmetries: D5_SYMMETRIES.map((symmetry) => {
      const maps = getD5Maps(board, symmetry);
      return {
        id: symmetry.id,
        kind: symmetry.kind,
        turns: symmetry.turns,
        inverseId: inverseD5Symmetry(symmetry).id,
        map: Array.from(maps.forward),
        inverseMap: Array.from(maps.inverse),
      };
    }),
  };
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  return (mixed ^ (mixed >>> 16)) >>> 0;
}

function deterministicPosition(
  board: Board,
  seed: number,
  occupancyPermille: number,
): Int8Array {
  const stones = new Int8Array(board.n).fill(EMPTY);
  for (let u = 0; u < board.n; u++) {
    const occupiedRoll = mix32(seed ^ Math.imul(u + 1, 0x9e3779b1)) % 1000;
    if (occupiedRoll < occupancyPermille) {
      stones[u] = (mix32(seed ^ Math.imul(u + 1, 0x85ebca6b)) & 1) as 0 | 1;
    }
  }
  return stones;
}

function scoreExpectation(board: Board, stones: ArrayLike<number>) {
  const score = scorePosition(board, stones);
  return {
    players: score.players,
    nodeOwner: Array.from(score.nodeOwner),
    aliveStone: Array.from(score.aliveStone),
    contestedPeries: score.contestedPeries,
    leader: score.leader,
  };
}

function scoreVector(id: string, board: Board, stones: Int8Array) {
  return {
    id,
    rings: board.rings,
    stones: Array.from(stones),
    expected: scoreExpectation(board, stones),
  };
}

function gameStateVector(afterActions: number, state: GameState) {
  return {
    afterActions,
    opening: state.turnCount === 0,
    stones: Array.from(state.stones),
    stonesPlaced: state.stonesPlaced,
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    midTurn: state.midTurn,
    passStreak: state.passStreak,
    over: state.over,
    canSwap: state.canSwap,
    swapped: state.swapped,
    lastMove: state.lastMove,
    currentTurnMoves: [...state.currentTurnMoves],
    turnCount: state.turnCount,
  };
}

function actionCode(action: GameAction): number {
  switch (action.type) {
    case 'place':
      return action.node;
    case 'pass':
      return STAR_PASS_ACTION_CODE;
    case 'swap':
      throw new Error('swap is outside the Double *Star parity contract');
  }
}

function outcomeFor(leader: 0 | 1 | -1, player: 0 | 1): -1 | 0 | 1 {
  if (leader === -1) return 0;
  return leader === player ? 1 : -1;
}

function terminalVector(
  reason: 'board-full' | 'double-pass',
  state: GameState,
) {
  if (!state.over) throw new Error('terminal vector requires a terminal state');
  const score = scoreExpectation(state.board, state.stones);
  const valuesByPlayer = [
    outcomeFor(score.leader, 0),
    outcomeFor(score.leader, 1),
  ] as const;
  const scoreMarginsByPlayer = [
    score.players[0].total - score.players[1].total,
    score.players[1].total - score.players[0].total,
  ] as const;
  const perspectivePlayer = state.toMove;
  return {
    reason,
    winner: score.leader,
    score,
    valuesByPlayer,
    wdlClassByPlayer: [
      valuesByPlayer[0] + 1,
      valuesByPlayer[1] + 1,
    ] as const,
    scoreMarginsByPlayer,
    valuePerspective: {
      kind: 'toMove',
      player: perspectivePlayer,
      value: valuesByPlayer[perspectivePlayer],
      wdlClass: valuesByPlayer[perspectivePlayer] + 1,
      scoreMargin: scoreMarginsByPlayer[perspectivePlayer],
    },
  };
}

function gameTrace(
  id: string,
  config: GameConfig,
  actions: GameAction[],
  terminalReason: 'board-full' | 'double-pass',
) {
  let state = initialState(config);
  const states = [gameStateVector(0, state)];
  for (let i = 0; i < actions.length; i++) {
    state = applyAction(state, actions[i]);
    states.push(gameStateVector(i + 1, state));
  }
  return {
    id,
    config,
    actions,
    actionCodes: actions.map(actionCode),
    states,
    terminal: terminalVector(terminalReason, state),
  };
}

function doublePassGameVector() {
  const config: GameConfig = {
    rings: 6,
    mode: 'double',
    pieRule: false,
    playerNames: ['blue', 'red'],
  };
  const board = getBoard(config.rings);
  const actions: GameAction[] = [
    { type: 'place', node: board.idx(0, 6, 0) },
    { type: 'place', node: board.idx(1, 2, 0) },
    { type: 'place', node: board.idx(1, 2, 1) },
    { type: 'place', node: board.idx(0, 6, 1) },
    { type: 'pass' },
    { type: 'place', node: board.idx(2, 3, 0) },
    { type: 'place', node: board.idx(2, 3, 1) },
    { type: 'pass' },
    { type: 'pass' },
  ];
  return gameTrace(
    'double-midturn-two-pass-terminal',
    config,
    actions,
    'double-pass',
  );
}

function boardFullGameVector(rings: number, residualMovesLeft: 0 | 1) {
  const config: GameConfig = {
    rings,
    mode: 'double',
    pieRule: false,
    playerNames: ['blue', 'red'],
  };
  const board = getBoard(rings);
  const actions: GameAction[] = Array.from(
    { length: board.n },
    (_, node) => ({ type: 'place', node }) as const,
  );
  const trace = gameTrace(
    `rings-${rings}-board-full-residual-${residualMovesLeft}`,
    config,
    actions,
    'board-full',
  );
  const terminal = trace.states[trace.states.length - 1];
  if (terminal.movesLeft !== residualMovesLeft) {
    throw new Error(
      `rings ${rings} ended with movesLeft ${terminal.movesLeft}, expected ${residualMovesLeft}`,
    );
  }
  return trace;
}

function applyLog(config: GameConfig, actions: GameAction[]): GameState {
  let state = initialState(config);
  for (const action of actions) state = applyAction(state, action);
  return state;
}

function semanticStateVector(state: GameState) {
  return {
    rings: state.board.rings,
    stones: Array.from(state.stones),
    stonesPlaced: state.stonesPlaced,
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    opening: state.turnCount === 0,
    midTurn: state.midTurn,
    passStreak: state.passStreak,
    terminal: state.over,
    currentTurnMoves: [...state.currentTurnMoves],
    turnCount: state.turnCount,
  };
}

function pairEquivalenceVector() {
  const config: GameConfig = {
    rings: 4,
    mode: 'double',
    pieRule: false,
    playerNames: ['blue', 'red'],
  };
  const board = getBoard(config.rings);
  const opening: GameAction = { type: 'place', node: board.idx(0, 4, 0) };
  const a = board.idx(1, 3, 1);
  const b = board.idx(3, 3, 1);
  const abActions: GameAction[] = [
    opening,
    { type: 'place', node: a },
    { type: 'place', node: b },
  ];
  const baActions: GameAction[] = [
    opening,
    { type: 'place', node: b },
    { type: 'place', node: a },
  ];
  const afterAb = applyLog(config, abActions);
  const afterBa = applyLog(config, baActions);
  const semanticState = semanticStateVector(afterAb);
  if (
    JSON.stringify(semanticState) !== JSON.stringify(semanticStateVector(afterBa))
  ) {
    throw new Error('AB and BA must produce the same semantic state');
  }
  return {
    id: 'complete-turn-ab-ba',
    config,
    pair: { a, b },
    ab: {
      actions: abActions,
      actionCodes: abActions.map(actionCode),
      semanticState,
      lastMove: afterAb.lastMove,
    },
    ba: {
      actions: baActions,
      actionCodes: baActions.map(actionCode),
      semanticState: semanticStateVector(afterBa),
      lastMove: afterBa.lastMove,
    },
    equivalentFields: Object.keys(semanticState),
    excludedPresentationFields: ['lastMove'],
  };
}

function actionLayoutRow(rings: number, maximumNodes: number) {
  const nodeCount = getBoard(rings).n;
  return {
    rings,
    nodeCount,
    native: {
      actionCount: nodeCount + 1,
      placementSlots: [0, nodeCount - 1],
      passSlot: nodeCount,
    },
    padded: {
      actionCount: maximumNodes + 1,
      placementSlots: [0, nodeCount - 1],
      paddingSlots:
        nodeCount === maximumNodes ? [] : [nodeCount, maximumNodes - 1],
      passSlot: maximumNodes,
    },
    examples: [
      {
        action: { type: 'place', node: 0 },
        wireCode: 0,
        nativeIndex: 0,
        paddedIndex: 0,
      },
      {
        action: { type: 'place', node: nodeCount - 1 },
        wireCode: nodeCount - 1,
        nativeIndex: nodeCount - 1,
        paddedIndex: nodeCount - 1,
      },
      {
        action: { type: 'pass' },
        wireCode: STAR_PASS_ACTION_CODE,
        nativeIndex: nodeCount,
        paddedIndex: maximumNodes,
      },
    ],
  };
}

function mixedActionLayout(id: string, rings: number[]) {
  const maximumNodes = Math.max(...rings.map((size) => getBoard(size).n));
  return {
    id,
    rings,
    maximumNodes,
    batchActionCount: maximumNodes + 1,
    rows: rings.map((size) => actionLayoutRow(size, maximumNodes)),
  };
}

/** Build the complete deterministic JSON-compatible conformance document. */
export function createStarConformance() {
  const boards: ReturnType<typeof boardVector>[] = [];
  const scores: ReturnType<typeof scoreVector>[] = [];

  for (let rings = MIN_RINGS; rings <= MAX_RINGS; rings++) {
    const board = getBoard(rings);
    boards.push(boardVector(board));

    const empty = new Int8Array(board.n).fill(EMPTY);
    const singlePerimeter = empty.slice();
    singlePerimeter[board.idx(0, rings, 0)] = 0;
    const twoPerimeter = singlePerimeter.slice();
    twoPerimeter[board.idx(0, rings, 1)] = 0;
    scores.push(scoreVector(`rings-${rings}-empty`, board, empty));
    scores.push(
      scoreVector(`rings-${rings}-single-perimeter`, board, singlePerimeter),
    );
    scores.push(
      scoreVector(`rings-${rings}-two-perimeter-star`, board, twoPerimeter),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-sparse`,
        board,
        deterministicPosition(board, 0x51a7 ^ rings, 80),
      ),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-dense`,
        board,
        deterministicPosition(board, 0xd3e5e ^ rings, 720),
      ),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-full`,
        board,
        deterministicPosition(board, 0xf011 ^ rings, 1000),
      ),
    );
  }

  const twoPassGame = doublePassGameVector();
  const games = [
    twoPassGame,
    boardFullGameVector(3, 1),
    boardFullGameVector(5, 0),
  ];
  const terminal =
    twoPassGame.states[twoPassGame.states.length - 1];
  scores.push(
    scoreVector(
      'rings-6-pass-terminal',
      getBoard(6),
      Int8Array.from(terminal.stones),
    ),
  );

  return {
    schema: STAR_CONFORMANCE_SCHEMA_ID,
    schemas: {
      rules: STAR_RULES_SCHEMA_ID,
      conformance: STAR_CONFORMANCE_SCHEMA_ID,
      modelFeatures: STAR_FEATURE_SCHEMA_ID,
      actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    },
    rules: {
      version: STAR_RULES_VERSION,
      hashAlgorithm: STAR_RULES_HASH_ALGORITHM,
      hash: STAR_RULES_HASH,
      canonical: STAR_RULES_CANONICAL,
      contract: STAR_RULES_CONTRACT,
    },
    actionEncoding: {
      placementCode: 'dense node id',
      passCode: STAR_PASS_ACTION_CODE,
      legalOrder: 'ascending legal placement node ids, then pass',
      nativeLayout: 'node u at index u; pass at index nodeCount',
    },
    actionLayouts: {
      schema: STAR_ACTION_LAYOUT_SCHEMA_ID,
      modelFeatureSchema: STAR_FEATURE_SCHEMA_ID,
      mixedBatches: [
        mixedActionLayout('mini-medium', [3, 5]),
        mixedActionLayout('small-large-full', [4, 6, 12]),
      ],
    },
    boards,
    scores,
    games,
    pairEquivalences: [pairEquivalenceVector()],
  };
}

export type StarConformance = ReturnType<typeof createStarConformance>;

/** Stable pretty-printed representation used for checked-in test data. */
export function serializeStarConformance(
  conformance: StarConformance = createStarConformance(),
): string {
  return `${JSON.stringify(conformance, null, 2)}\n`;
}
