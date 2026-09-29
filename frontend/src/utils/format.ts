/**
 * Formatting utilities for Money Migration Atlas frontend.
 */

export function truncateAddress(address: string, startChars: number = 6, endChars: number = 4): string {
  if (!address) return '';
  if (address.length <= startChars + endChars) return address;
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
}

export function formatScore(score?: number | null, fallback: string = 'N/A'): string {
  if (score === null || score === undefined || isNaN(score)) return fallback;
  return Number(score).toFixed(2);
}

export function formatPercentage(score?: number | null, fallback: string = 'N/A'): string {
  if (score === null || score === undefined || isNaN(score)) return fallback;
  return `${(Number(score) * 100).toFixed(1)}%`;
}

export function formatTimestamp(isoString?: string): string {
  if (!isoString) return 'Unknown';
  try {
    const d = new Date(isoString);
    return d.toISOString().replace('T', ' ').replace('Z', ' UTC');
  } catch {
    return isoString;
  }
}

export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for older or restricted environments
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    return successful;
  }
}
