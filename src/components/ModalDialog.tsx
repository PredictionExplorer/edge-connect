'use client';

import {
  useEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent,
  type ReactNode,
  type RefObject,
} from 'react';

interface ModalDialogProps {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  labelledBy?: string;
  ariaLabel?: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
  className?: string;
}

function focusableElements(dialog: HTMLDialogElement): HTMLElement[] {
  return Array.from(
    dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute('hidden'));
}

export function ModalDialog({
  open,
  onClose,
  children,
  labelledBy,
  ariaLabel,
  initialFocusRef,
  closeOnBackdrop = true,
  closeOnEscape = true,
  className = '',
}: ModalDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    const dialog = dialogRef.current;
    if (!dialog) return;

    const opener = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const closeOnDocumentEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !closeOnEscape) return;
      event.preventDefault();
      event.stopPropagation();
      onCloseRef.current();
    };
    document.addEventListener('keydown', closeOnDocumentEscape, true);

    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }

    const focusTarget =
      initialFocusRef?.current ?? focusableElements(dialog)[0] ?? dialog;
    focusTarget.focus();

    return () => {
      document.removeEventListener('keydown', closeOnDocumentEscape, true);
      document.body.style.overflow = previousOverflow;
      if (dialog.open && typeof dialog.close === 'function') dialog.close();
      opener?.focus({ preventScroll: true });
    };
  }, [closeOnEscape, initialFocusRef, open]);

  if (!open) return null;

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDialogElement>) => {
    if (event.key !== 'Tab') return;

    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = focusableElements(dialog);
    if (focusable.length === 0) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const handleClick = (event: MouseEvent<HTMLDialogElement>) => {
    if (closeOnBackdrop && event.target === event.currentTarget) onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      aria-label={ariaLabel}
      aria-labelledby={labelledBy}
      aria-modal="true"
      onCancel={(event) => {
        event.preventDefault();
        if (closeOnEscape) onClose();
      }}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      className={`m-auto max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] max-w-none overflow-visible border-0 bg-transparent p-0 text-ink backdrop:bg-black/70 backdrop:backdrop-blur-md ${className}`}
    >
      {children}
    </dialog>
  );
}
