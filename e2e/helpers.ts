import { expect, type Page } from '@playwright/test';

export async function fillFourRingGame(page: Page): Promise<void> {
  for (let placed = 0; placed < 50; placed++) {
    await page
      .getByRole('button', { name: /empty .* may place here/i })
      .first()
      .click();
  }
  await expect(page.getByRole('dialog', { name: 'Game over' })).toBeVisible();
}
