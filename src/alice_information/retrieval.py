"""Controlled P4.2 retrieval, normalization, and explicit live adapter."""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from .contracts import InformationSourceDocument, canonicalize_public_url
from .http_transport import (
    DeterministicInformationHttpTransport,
    DeterministicInformationNameResolver,
    InformationHttpRequest,
    InformationHttpTransport,
    InformationNameResolver,
    InformationRawHttpResponse,
    http_failure,
    validate_global_address,
)
from .live_http import (
    DirectInformationHttpsTransport,
    StdlibInformationDnsBackend,
    StdlibInformationSocketBackend,
    SystemInformationNameResolver,
)
from .live_policy import InformationLiveHttpPolicy
from .policy import InformationPolicy
from .providers import InformationCancellationToken
from .retrieval_policy import InformationHttpRetrievalPolicy
from .temporal_metadata import (
    InformationTemporalMetadataCandidate,
    temporal_candidate_set_sha256,
    temporal_metadata_failure,
)
from .temporal_metadata_policy import APPROVED_MAX_CANDIDATES

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_SINGLETON_HEADERS = {
    "content-disposition",
    "content-encoding",
    "content-length",
    "content-type",
    "location",
    "last-modified",
}
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _VisibleHtmlParser(HTMLParser):
    """Small deterministic visible-text parser with no script execution."""

    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _HIDDEN_TAGS = {"head", "noscript", "script", "style", "svg", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._metadata_block_depth = 0
        self._hidden_tag_stack: list[str] = []
        self._parts: list[str] = []
        self.title_parts: list[str] = []
        self._title_depth = 0
        self.temporal_candidates: list[InformationTemporalMetadataCandidate] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        attribute_map: dict[str, str] = {}
        if lowered == "meta":
            protected_attributes = {"property", "name", "itemprop", "content"}
        elif lowered == "time":
            protected_attributes = {"itemprop", "datetime"}
        else:
            protected_attributes = set()
        for raw_name, raw_value in attrs:
            name = raw_name.lower()
            if name in attribute_map and name in protected_attributes:
                raise ValueError("Duplicate HTML attribute at metadata boundary.")
            if raw_value is not None and name not in attribute_map:
                attribute_map[name] = raw_value
        if self._metadata_block_depth == 0:
            self._capture_temporal_metadata(lowered, attribute_map)
        if lowered in self._HIDDEN_TAGS:
            self._hidden_tag_stack.append(lowered)
            self._hidden_depth += 1
            if lowered != "head":
                self._metadata_block_depth += 1
        if lowered == "title":
            self._title_depth += 1
        if lowered in self._BLOCK_TAGS and self._hidden_depth == 0:
            self._parts.append("\n")

    def _capture_temporal_metadata(
        self,
        tag: str,
        attributes: dict[str, str],
    ) -> None:
        origin: str | None = None
        raw_value: str | None = None
        if tag == "meta":
            markers = {
                value.casefold()
                for value in (attributes.get("property"), attributes.get("name"))
                if value
            }
            itemprop = attributes.get("itemprop", "").casefold()
            detected: list[str] = []
            if "article:published_time" in markers:
                detected.append("html_meta_article_published_time")
            if "article:modified_time" in markers:
                detected.append("html_meta_article_modified_time")
            if itemprop == "datepublished":
                detected.append("html_meta_date_published")
            elif itemprop == "datemodified":
                detected.append("html_meta_date_modified")
            if len(set(detected)) > 1:
                raise ValueError("Conflicting HTML temporal metadata markers.")
            origin = detected[0] if detected else None
            raw_value = attributes.get("content", "")
        elif tag == "time":
            itemprop = attributes.get("itemprop", "").casefold()
            raw_value = attributes.get("datetime", "")
            if itemprop == "datepublished":
                origin = "html_time_date_published"
            elif itemprop == "datemodified":
                origin = "html_time_date_modified"
        if origin is None:
            return
        self.temporal_candidates.append(
            InformationTemporalMetadataCandidate.create(
                origin=origin,
                raw_value=raw_value,
            )
        )
        if len(self.temporal_candidates) > APPROVED_MAX_CANDIDATES:
            raise ValueError("Temporal metadata candidate limit exceeded.")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title" and self._title_depth:
            self._title_depth -= 1
        if lowered in self._HIDDEN_TAGS and self._hidden_tag_stack:
            if self._hidden_tag_stack[-1] == lowered:
                self._hidden_tag_stack.pop()
                self._hidden_depth -= 1
                if lowered != "head" and self._metadata_block_depth:
                    self._metadata_block_depth -= 1
        if lowered in self._BLOCK_TAGS and self._hidden_depth == 0:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title_parts.append(data)
        if self._hidden_depth:
            return
        self._parts.append(data)

    def normalized_text(self) -> str:
        lines: list[str] = []
        for raw in "".join(self._parts).replace("\r", "\n").split("\n"):
            line = _WHITESPACE.sub(" ", raw).strip()
            if line:
                lines.append(line)
            elif lines and lines[-1] != "":
                lines.append("")
        return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

    def title(self) -> str | None:
        value = _WHITESPACE.sub(" ", " ".join(self.title_parts)).strip()
        return value or None


@dataclass(frozen=True)
class InformationRetrievedResource:
    """One deterministic, digest-bound response after redirects and decoding."""

    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    status_code: int
    content_type: str
    charset: str
    content_encoding: str
    normalized_text: str
    content_sha256: str
    wire_bytes: int
    decoded_bytes: int
    peer_address: str
    title: str | None = None
    temporal_metadata_candidates: tuple[InformationTemporalMetadataCandidate, ...] = ()

    def validate(self) -> None:
        if canonicalize_public_url(self.requested_url) != self.requested_url:
            raise http_failure("invalid_source_url")
        if canonicalize_public_url(self.final_url) != self.final_url:
            raise http_failure("invalid_source_url")
        if not self.redirect_chain or self.redirect_chain[0] != self.requested_url:
            raise http_failure("redirect_blocked")
        if self.redirect_chain[-1] != self.final_url:
            raise http_failure("redirect_blocked")
        if len(self.redirect_chain) > 6:
            raise http_failure("redirect_blocked")
        if len(set(self.redirect_chain)) != len(self.redirect_chain):
            raise http_failure("redirect_blocked")
        for previous, current in zip(
            self.redirect_chain, self.redirect_chain[1:], strict=False
        ):
            if urlsplit(previous).scheme == "https" and urlsplit(current).scheme == "http":
                raise http_failure("redirect_blocked")
        for item in self.redirect_chain:
            if canonicalize_public_url(item) != item:
                raise http_failure("redirect_blocked")
        if self.status_code != 200:
            raise http_failure("http_status_rejected")
        if not self.normalized_text:
            raise http_failure("normalization_failed")
        digest = sha256(self.normalized_text.encode("utf-8")).hexdigest()
        if digest != self.content_sha256:
            raise http_failure("normalization_failed")
        if self.wire_bytes < 0 or self.decoded_bytes < 0:
            raise http_failure("response_too_large")
        validate_global_address(self.peer_address)
        if not isinstance(self.temporal_metadata_candidates, tuple):
            raise http_failure("normalization_failed")
        if len(self.temporal_metadata_candidates) > APPROVED_MAX_CANDIDATES:
            raise http_failure("normalization_failed")
        for candidate in self.temporal_metadata_candidates:
            if type(candidate) is not InformationTemporalMetadataCandidate:
                raise http_failure("normalization_failed")
            candidate.validate()

    def to_source_document(
        self,
        *,
        source_id: str,
        provider: str,
        retrieved_at: str,
        published_at: str | None = None,
        updated_at: str | None = None,
    ) -> InformationSourceDocument:
        """Project an undated resource; P4.4b owns verified date projection."""

        self.validate()
        if published_at is not None or updated_at is not None:
            raise temporal_metadata_failure("temporal_resolution_invalid")
        title = self.title or urlsplit(self.final_url).hostname or "External source"
        return InformationSourceDocument.create(
            source_id=source_id,
            provider=provider,
            url=self.final_url,
            title=title,
            normalized_text=self.normalized_text,
            retrieved_at=retrieved_at,
            published_at=published_at,
            updated_at=updated_at,
        )


@dataclass(frozen=True)
class ControlledInformationHttpRetriever:
    """Point-of-use DNS, redirect, peer, byte, and normalization gate."""

    information_policy: InformationPolicy
    retrieval_policy: InformationHttpRetrievalPolicy
    resolver: InformationNameResolver
    transport: InformationHttpTransport

    def __post_init__(self) -> None:
        self.information_policy.validate()
        self.retrieval_policy.validate(information_policy=self.information_policy)
        self._validate_runtime_components()

    def _validate_runtime_components(self) -> None:
        if self.retrieval_policy.live_network_access_allowed:
            raise ValueError("P4.2a retrieval policy must remain network-free.")
        if type(self.resolver) is not DeterministicInformationNameResolver:
            raise ValueError("P4.2a requires the exact deterministic resolver fixture.")
        if type(self.transport) is not DeterministicInformationHttpTransport:
            raise ValueError("P4.2a requires the exact deterministic transport fixture.")

    def retrieve(
        self,
        url: str,
        *,
        cancellation: InformationCancellationToken | None = None,
    ) -> InformationRetrievedResource:
        self.information_policy.validate()
        self.retrieval_policy.validate(information_policy=self.information_policy)
        self._validate_runtime_components()
        try:
            current = canonicalize_public_url(url)
        except ValueError as exc:
            raise http_failure("invalid_source_url") from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        requested = current
        chain: list[str] = [current]
        redirects = 0
        while True:
            target = self.resolver.resolve(
                current,
                policy=self.retrieval_policy,
                cancellation=cancellation,
            )
            request = InformationHttpRequest(
                canonical_url=current,
                headers=(
                    ("accept", "text/html, application/xhtml+xml, text/plain;q=0.9"),
                    ("accept-encoding", "gzip, deflate"),
                    ("connection", "close"),
                    ("user-agent", "ALICE-Information/0.3"),
                ),
                timeout_seconds=self.retrieval_policy.request_timeout_seconds,
            )
            response = self.transport.get(
                request,
                target=target,
                policy=self.retrieval_policy,
                cancellation=cancellation,
            )
            headers = self._validated_headers(response)
            envelope_wire_bytes = self._validate_wire_envelope(response, headers)
            if response.status_code in _REDIRECT_STATUSES:
                if redirects >= self.retrieval_policy.max_redirects:
                    raise http_failure("redirect_blocked")
                location = headers.get("location")
                if location is None:
                    raise http_failure("redirect_blocked")
                try:
                    redirected = canonicalize_public_url(urljoin(current, location))
                except ValueError as exc:
                    raise http_failure("redirect_blocked") from exc
                if (
                    urlsplit(current).scheme == "https"
                    and urlsplit(redirected).scheme == "http"
                    and not self.retrieval_policy.https_downgrade_allowed
                ):
                    raise http_failure("redirect_blocked")
                if redirected in chain:
                    raise http_failure("redirect_blocked")
                redirects += 1
                current = redirected
                chain.append(current)
                continue
            if response.status_code != 200:
                raise http_failure("http_status_rejected")
            content_type, charset = self._content_type(headers)
            content_encoding = headers.get("content-encoding", "identity").lower()
            if content_encoding not in self.retrieval_policy.allowed_content_encodings:
                raise http_failure("unsupported_content_type")
            disposition = headers.get("content-disposition", "")
            if "attachment" in disposition.lower():
                raise http_failure("unsupported_content_type")
            decoded, wire_bytes = self._decode_body(
                response.body_chunks,
                content_encoding=content_encoding,
                cancellation=cancellation,
            )
            if wire_bytes != envelope_wire_bytes:
                raise http_failure("response_header_invalid")
            try:
                text = decoded.decode(charset, errors="strict")
            except (LookupError, UnicodeDecodeError) as exc:
                raise http_failure("content_decode_failed") from exc
            normalized, title, temporal_candidates = self._normalize(
                text,
                content_type=content_type,
            )
            last_modified = headers.get("last-modified")
            if last_modified is not None:
                temporal_candidates = temporal_candidates + (
                    InformationTemporalMetadataCandidate.create(
                        origin="http_last_modified",
                        raw_value=last_modified,
                    ),
                )
            if len(temporal_candidates) > APPROVED_MAX_CANDIDATES:
                raise http_failure("normalization_failed")
            resource = InformationRetrievedResource(
                requested_url=requested,
                final_url=current,
                redirect_chain=tuple(chain),
                status_code=response.status_code,
                content_type=content_type,
                charset=charset,
                content_encoding=content_encoding,
                normalized_text=normalized,
                content_sha256=sha256(normalized.encode("utf-8")).hexdigest(),
                wire_bytes=wire_bytes,
                decoded_bytes=len(decoded),
                peer_address=validate_global_address(response.peer_address),
                title=title,
                temporal_metadata_candidates=temporal_candidates,
            )
            resource.validate()
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            return resource

    def _validated_headers(self, response: InformationRawHttpResponse) -> dict[str, str]:
        response.validate_shape()
        total = 0
        normalized: dict[str, str] = {}
        for raw_name, raw_value in response.headers:
            name = raw_name.strip().lower()
            value = raw_value.strip()
            if not name or any(ord(char) < 33 or ord(char) > 126 for char in name):
                raise http_failure("response_header_invalid")
            total += len(name.encode("ascii", errors="ignore")) + len(
                value.encode("utf-8")
            ) + 4
            if total > self.retrieval_policy.max_header_bytes:
                raise http_failure("response_header_invalid")
            if name == "transfer-encoding":
                raise http_failure("response_header_invalid")
            if name in _SINGLETON_HEADERS and name in normalized:
                raise http_failure("response_header_invalid")
            normalized[name] = value
        return normalized

    def _validate_wire_envelope(
        self,
        response: InformationRawHttpResponse,
        headers: dict[str, str],
    ) -> int:
        wire_bytes = 0
        for chunk in response.body_chunks:
            wire_bytes += len(chunk)
            if wire_bytes > self.retrieval_policy.max_wire_bytes:
                raise http_failure("response_too_large")
        declared_length = headers.get("content-length")
        if declared_length is not None:
            if not declared_length.isdigit():
                raise http_failure("response_header_invalid")
            if int(declared_length) > self.retrieval_policy.max_wire_bytes:
                raise http_failure("response_too_large")
            if int(declared_length) != wire_bytes:
                raise http_failure("response_header_invalid")
        return wire_bytes

    def _content_type(self, headers: dict[str, str]) -> tuple[str, str]:
        raw = headers.get("content-type")
        if raw is None:
            raise http_failure("unsupported_content_type")
        parts = [part.strip() for part in raw.split(";")]
        media_type = parts[0].lower()
        if media_type not in self.retrieval_policy.allowed_content_types:
            raise http_failure("unsupported_content_type")
        charset = "utf-8"
        charset_seen = False
        for parameter in parts[1:]:
            if not parameter:
                continue
            name, separator, value = parameter.partition("=")
            if name.strip().lower() == "charset" and separator:
                if charset_seen:
                    raise http_failure("response_header_invalid")
                charset_seen = True
                charset = value.strip().strip('"').lower()
        if charset not in self.retrieval_policy.allowed_charsets:
            raise http_failure("unsupported_content_type")
        if charset in {"ascii", "us-ascii"}:
            charset = "ascii"
        return media_type, charset

    def _decode_body(
        self,
        chunks: tuple[bytes, ...],
        *,
        content_encoding: str,
        cancellation: InformationCancellationToken | None = None,
    ) -> tuple[bytes, int]:
        wire = 0
        output = bytearray()
        decompressor: zlib.Decompress | None = None
        if content_encoding == "gzip":
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif content_encoding == "deflate":
            decompressor = zlib.decompressobj(zlib.MAX_WBITS)
        try:
            for chunk in chunks:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                wire += len(chunk)
                if wire > self.retrieval_policy.max_wire_bytes:
                    raise http_failure("response_too_large")
                if decompressor is None:
                    output.extend(chunk)
                else:
                    remaining = self.retrieval_policy.max_decoded_bytes - len(output)
                    output.extend(decompressor.decompress(chunk, remaining + 1))
                    if decompressor.unconsumed_tail:
                        raise http_failure("response_too_large")
                if len(output) > self.retrieval_policy.max_decoded_bytes:
                    raise http_failure("response_too_large")
            if decompressor is not None:
                remaining = self.retrieval_policy.max_decoded_bytes - len(output)
                output.extend(decompressor.flush(remaining + 1))
                if len(output) > self.retrieval_policy.max_decoded_bytes:
                    raise http_failure("response_too_large")
                if not decompressor.eof or decompressor.unused_data:
                    raise http_failure("content_decode_failed")
        except zlib.error as exc:
            raise http_failure("content_decode_failed") from exc
        return bytes(output), wire

    def _normalize(
        self,
        text: str,
        *,
        content_type: str,
    ) -> tuple[
        str,
        str | None,
        tuple[InformationTemporalMetadataCandidate, ...],
    ]:
        try:
            if content_type in {"text/html", "application/xhtml+xml"}:
                parser = _VisibleHtmlParser()
                parser.feed(text)
                parser.close()
                normalized = parser.normalized_text()
                title = parser.title()
                temporal_candidates = tuple(parser.temporal_candidates)
            else:
                normalized = _BLANK_LINES.sub(
                    "\n\n",
                    "\n".join(
                        _WHITESPACE.sub(" ", line).strip()
                        for line in text.replace("\r\n", "\n")
                        .replace("\r", "\n")
                        .split("\n")
                    ),
                ).strip()
                title = None
                temporal_candidates = ()
            normalized = normalized.replace("\x00", "").strip()
        except Exception as exc:  # parser internals fail closed at this boundary
            raise http_failure("normalization_failed") from exc
        if not normalized:
            raise http_failure("normalization_failed")
        return normalized, title, temporal_candidates


@dataclass(frozen=True)
class LiveControlledInformationHttpRetriever(ControlledInformationHttpRetriever):
    """Explicit P4.2b live adapter with exact production components only.

    The adapter is intentionally not registered as a Phase 4 provider and is
    not reachable from the Phase 3 runtime in this milestone.
    """

    live_policy: InformationLiveHttpPolicy

    def __post_init__(self) -> None:
        self.information_policy.validate()
        self.retrieval_policy.validate(information_policy=self.information_policy)
        self.live_policy.validate(
            information_policy=self.information_policy,
            retrieval_policy=self.retrieval_policy,
        )
        self._validate_runtime_components()

    def _validate_runtime_components(self) -> None:
        if type(self.resolver) is not SystemInformationNameResolver:
            raise ValueError("P4.2b requires the exact system resolver adapter.")
        if type(self.transport) is not DirectInformationHttpsTransport:
            raise ValueError("P4.2b requires the exact direct HTTPS transport.")
        if self.resolver.information_policy != self.information_policy:
            raise ValueError("P4.2b resolver information policy does not match.")
        if self.resolver.retrieval_policy != self.retrieval_policy:
            raise ValueError("P4.2b resolver retrieval policy does not match.")
        if self.resolver.live_policy != self.live_policy:
            raise ValueError("P4.2b resolver activation policy does not match.")
        if self.transport.information_policy != self.information_policy:
            raise ValueError("P4.2b transport information policy does not match.")
        if self.transport.retrieval_policy != self.retrieval_policy:
            raise ValueError("P4.2b transport retrieval policy does not match.")
        if self.transport.live_policy != self.live_policy:
            raise ValueError("P4.2b transport activation policy does not match.")
        if type(self.resolver.backend) is not StdlibInformationDnsBackend:
            raise ValueError("P4.2b resolver backend changed after initialization.")
        if type(self.transport.backend) is not StdlibInformationSocketBackend:
            raise ValueError("P4.2b socket backend changed after initialization.")
        self.resolver.validate_live_boundary()
        self.transport.validate_live_boundary()


@dataclass(frozen=True)
class InformationDuplicateObservation:
    """One duplicate body detected across two public source URLs."""

    content_sha256: str
    retained_url: str
    duplicate_url: str


def deduplicate_retrieved_resources(
    resources: tuple[InformationRetrievedResource, ...],
) -> tuple[
    tuple[InformationRetrievedResource, ...],
    tuple[InformationDuplicateObservation, ...],
]:
    """Deduplicate only exact body and temporal-evidence versions."""

    retained: list[InformationRetrievedResource] = []
    duplicates: list[InformationDuplicateObservation] = []
    by_digest: dict[tuple[str, str], InformationRetrievedResource] = {}
    for resource in resources:
        resource.validate()
        version_key = (
            resource.content_sha256,
            temporal_candidate_set_sha256(resource.temporal_metadata_candidates),
        )
        prior = by_digest.get(version_key)
        if prior is None:
            by_digest[version_key] = resource
            retained.append(resource)
            continue
        duplicates.append(
            InformationDuplicateObservation(
                content_sha256=resource.content_sha256,
                retained_url=prior.final_url,
                duplicate_url=resource.final_url,
            )
        )
    return tuple(retained), tuple(duplicates)
