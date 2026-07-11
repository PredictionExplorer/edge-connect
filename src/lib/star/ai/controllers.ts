import type { GameConfig } from '../game';

export const CONTROLLER_TYPES = ['human', 'server', 'local'] as const;

export type ControllerType = (typeof CONTROLLER_TYPES)[number];
export type PlayerControllers = [ControllerType, ControllerType];

export const HUMAN_CONTROLLERS: PlayerControllers = ['human', 'human'];

export function isControllerType(value: unknown): value is ControllerType {
  return typeof value === 'string' && CONTROLLER_TYPES.includes(value as ControllerType);
}

export function supportsAiControllers(
  config: Pick<GameConfig, 'mode' | 'pieRule'>,
): boolean {
  return config.mode === 'double' && config.pieRule === false;
}

/**
 * Persistence and setup both flow through this boundary. Invalid values, and
 * AI values attached to unsupported variants, become human controllers.
 */
export function normalizeControllers(
  config: Pick<GameConfig, 'mode' | 'pieRule'>,
  value: unknown,
): PlayerControllers {
  if (!supportsAiControllers(config) || !Array.isArray(value) || value.length !== 2) {
    return [...HUMAN_CONTROLLERS];
  }

  return [
    isControllerType(value[0]) ? value[0] : 'human',
    isControllerType(value[1]) ? value[1] : 'human',
  ];
}

export function controllerLabel(controller: ControllerType): string {
  switch (controller) {
    case 'human':
      return 'Human';
    case 'server':
      return 'Server AI';
    case 'local':
      return 'Local AI';
  }
}
