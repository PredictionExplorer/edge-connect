/** Player palette shared by the board and the panels. */
export const PLAYER_COLORS = [
  {
    name: 'Amber',
    base: '#ffb958',
    bright: '#ffd699',
    deep: '#8a5416',
    glow: 'rgba(255, 185, 88, 0.55)',
    soft: 'rgba(255, 185, 88, 0.16)',
  },
  {
    name: 'Azure',
    base: '#62c6ff',
    bright: '#b5e4ff',
    deep: '#175a86',
    glow: 'rgba(98, 198, 255, 0.55)',
    soft: 'rgba(98, 198, 255, 0.16)',
  },
] as const;

export const BOARD_PRESETS = [
  { rings: 4, label: 'Mini', nodes: 50, peries: 20 },
  { rings: 6, label: 'Small', nodes: 105, peries: 30 },
  { rings: 8, label: 'Medium', nodes: 180, peries: 40 },
  { rings: 10, label: 'Full', nodes: 275, peries: 50 },
] as const;
