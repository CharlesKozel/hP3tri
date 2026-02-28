export function getApiBase(): string {
    const params = new URLSearchParams(window.location.search);
    const host = params.get('host');
    if (!host) return '';
    return host.startsWith('http') ? host : `http://${host}`;
}
