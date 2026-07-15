import { expect, type Page } from '@playwright/test';

export async function reachFourRingClinch(page: Page): Promise<number> {
  const clinchDialog = page.getByRole('dialog', {
    name: /cannot be caught/i,
  });
  for (let placed = 0; placed < 50; placed++) {
    if (await clinchDialog.isVisible()) return placed;
    await page
      .getByRole('button', { name: /empty .* may place here/i })
      .first()
      .click();
    if (await clinchDialog.isVisible()) return placed + 1;
  }
  throw new Error('expected the four-ring game to clinch before the board filled');
}

export async function fillFourRingGame(page: Page): Promise<void> {
  for (let placed = 0; placed < 50; placed++) {
    const clinchDialog = page.getByRole('dialog', {
      name: /cannot be caught/i,
    });
    if (await clinchDialog.isVisible()) {
      await clinchDialog
        .getByRole('button', { name: 'Continue playing' })
        .click();
    }
    await page
      .getByRole('button', { name: /empty .* may place here/i })
      .first()
      .click();
  }
  await expect(page.getByRole('dialog', { name: 'Game over' })).toBeVisible();
}
