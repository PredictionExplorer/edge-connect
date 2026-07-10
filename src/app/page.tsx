'use client';

import { GameScreen } from '@/components/GameScreen';
import { SetupScreen } from '@/components/SetupScreen';
import { Starfield } from '@/components/Starfield';
import { useAppStore, useMounted } from '@/lib/store';

export default function Home() {
  const mounted = useMounted();
  const phase = useAppStore((s) => s.phase);

  return (
    <>
      <Starfield />
      {mounted && (phase === 'playing' ? <GameScreen /> : <SetupScreen />)}
    </>
  );
}
