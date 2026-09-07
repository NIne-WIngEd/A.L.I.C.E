"""Explicit Magnolia system trust. No insecure context or certificate bypass."""
from pathlib import Path
import os
import ssl
import urllib.request

import contract as c

CA_PATHS = ('/etc/pki/tls/certs/ca-bundle.crt', '/etc/ssl/certs/ca-certificates.crt', '/etc/ssl/cert.pem')


def configure(run, candidates=CA_PATHS):
    selected = next((Path(p) for p in candidates if Path(p).is_file() and os.access(p, os.R_OK)), None)
    c.require(selected is not None, 'No readable system CA bundle; HTTPS preparation stopped')
    context = ssl.create_default_context(cafile=str(selected))
    c.require(context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname, 'TLS verification must remain enabled')
    c.require(context.cert_store_stats()['x509_ca'] > 0, 'Selected bundle contains no CA certificates')
    for name in ('SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'PIP_CERT'):
        os.environ[name] = str(selected)
    receipt = {'schema': 'alice.mc10d.explicit-system-ca.v1', 'ca_path': str(selected), 'ca_sha256': c.file_sha(selected), 'ca_bytes': selected.stat().st_size, 'verify_mode': 'CERT_REQUIRED', 'check_hostname': True, 'openssl': ssl.OPENSSL_VERSION}
    c.immutable(Path(run) / 'tls-trust.json', receipt)
    return receipt


def context():
    name = os.environ.get('SSL_CERT_FILE')
    c.require(name is not None and Path(name).is_file(), 'Explicit CA setup must precede HTTPS')
    result = ssl.create_default_context(cafile=name)
    c.require(result.verify_mode == ssl.CERT_REQUIRED and result.check_hostname, 'TLS context lost verification')
    return result


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        c.require(newurl.startswith('https://'), 'External HTTPS redirect downgrade refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(request, timeout=60):
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    c.require(isinstance(url, str) and url.startswith('https://'), 'External preparation URL must use HTTPS')
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context()), HTTPSOnlyRedirect())
    return opener.open(request, timeout=timeout)
