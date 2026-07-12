'use client';

import { useLayoutEffect } from 'react';
import { GameScreen } from '@/components/GameScreen';
import { SetupScreen } from '@/components/SetupScreen';
import { Starfield } from '@/components/Starfield';
import { useAppStore, useMounted } from '@/lib/store';

function AppSkeleton() {
  return (
    <main
      aria-hidden
      className="app-skeleton screen-safe relative z-10 mx-auto flex w-full max-w-6xl flex-col"
    >
      <div className="mx-auto mt-5 h-12 w-40 rounded-2xl border border-gold/10 bg-gold/[0.04]" />
      <div className="mt-10 grid flex-1 items-center gap-8 md:grid-cols-[minmax(0,5fr)_minmax(22rem,4fr)]">
        <div className="mx-auto aspect-square w-full max-w-lg rounded-[2rem] border border-gold/10 bg-white/[0.02]" />
        <div className="panel-surface flex flex-col gap-4 rounded-3xl p-5">
          <div className="h-16 rounded-2xl bg-white/[0.035]" />
          <div className="h-28 rounded-2xl bg-white/[0.035]" />
          <div className="h-24 rounded-2xl bg-white/[0.035]" />
          <div className="mt-auto h-14 rounded-2xl bg-gold/10" />
        </div>
      </div>
    </main>
  );
}

export default function Home() {
  const mounted = useMounted();
  const phase = useAppStore((s) => s.phase);

  useLayoutEffect(() => {
    if (!mounted) return;
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [mounted, phase]);

  return (
    <>
      <Starfield />
      {mounted ? phase === 'playing' ? <GameScreen /> : <SetupScreen /> : <AppSkeleton />}
    </>
  );
}
