"""Separately reviewed named facts; never rendered into provider inputs.

These documents carry assertions and exact external bindings. Parsing or a
self-digest does not establish review, source acceptance or gradient authority.
Callers must protect private case bytes before reading them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re
from typing import Mapping

from ..identity.codec import CognitiveFrameDocument
from ..identity.contracts import IdentityError
from .contracts import TeachingError, canonical
from .lineage import ConstructionLineage

CASE_SCHEMA = 'alice-personality-reviewed-teaching-case-v1'
CASE_LINEAGE_SCHEMA = 'alice-personality-reviewed-teaching-case-v2'
OUTCOME_CASE_SCHEMA = 'alice-personality-reviewed-teaching-case-v3'
ARTIFACT_ROLES = frozenset({'source_package', 'compiled_receipt', 'annotation_protocol', 'review_record'})
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise TeachingError('case pointers and annotation descriptions require nonempty strings')


def _exact(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise TeachingError('case fields differ from the strict reviewed-case schema')
    return value


def validate_artifact_pins(pins, source_class, *, label='case'):
    roles = ARTIFACT_ROLES if source_class == 'public_mechanical_fixture' else ARTIFACT_ROLES | {'prepared_receipt'}
    if not isinstance(pins, Mapping) or set(pins) != roles \
            or any(not isinstance(v, str) or not _DIGEST.fullmatch(v) for v in pins.values()):
        raise TeachingError(label + ' source/compiler/protocol/review artifacts need exact external SHA bindings')


@dataclass(frozen=True)
class TargetFact:
    pointers: tuple[str, ...]
    value: float | None
    uncertainty: float | None
    available: bool
    support_ids: tuple[str, ...]
    disconfirmation_ids: tuple[str, ...]

    def validate(self, source_ids):
        if type(self.pointers) is not tuple or not self.pointers:
            raise TeachingError('a named target needs an explicit ordered pointer tuple')
        for pointer in self.pointers:
            _text(pointer)
        if type(self.available) is not bool:
            raise TeachingError('target availability must be an actual Boolean')
        if self.available:
            for value in (self.value, self.uncertainty):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise TeachingError('available annotations need finite numeric facts')
            if not 0 <= self.uncertainty <= 1:
                raise TeachingError('annotation uncertainty must be a probability')
        elif self.value is not None or self.uncertainty is not None:
            raise TeachingError('unknown targets cannot carry hidden gold values')
        for pointers in (self.support_ids, self.disconfirmation_ids):
            if type(pointers) is not tuple or len(set(pointers)) != len(pointers) \
                    or any(not isinstance(p, str) or p not in source_ids for p in pointers):
                raise TeachingError('annotation support/disconfirmation must name exact provided source pointers')


@dataclass(frozen=True)
class NamedTarget:
    field: str
    mode: str
    facts: tuple[TargetFact, ...]

    def validate(self, source_ids):
        _text(self.field)
        if self.mode not in {'independent', 'partial_distribution', 'closed_distribution'}:
            raise TeachingError('unknown target comparison semantics')
        if type(self.facts) is not tuple or not self.facts or any(type(fact) is not TargetFact for fact in self.facts):
            raise TeachingError('targets require explicit typed reviewed facts')
        for fact in self.facts:
            fact.validate(source_ids)
        if len({fact.pointers for fact in self.facts}) != len(self.facts):
            raise TeachingError('duplicate target pointers are forbidden')


@dataclass(frozen=True)
class ReviewedTeachingCase:
    case_id: str
    phase: str
    split: str
    source_class: str
    document: CognitiveFrameDocument
    source_family_ids: tuple[str, ...]
    artifact_sha256: Mapping[str, str]
    annotation_origin: str
    review_id: str
    annotation_note: str
    targets: tuple[NamedTarget, ...]
    schema: str = CASE_SCHEMA
    lineage: ConstructionLineage | None = None
    outcome_binding: Mapping | None = None

    def validate(self):
        if self.schema not in {CASE_SCHEMA, CASE_LINEAGE_SCHEMA, OUTCOME_CASE_SCHEMA} or self.phase not in {'n1', 'n2', 'calibration'}:
            raise TeachingError('unsupported reviewed-case schema or teaching phase')
        if (self.schema == CASE_SCHEMA) != (self.lineage is None):
            raise TeachingError('legacy cases retain strict splits; new lineage requires its explicit case schema')
        if (self.schema == OUTCOME_CASE_SCHEMA) != (self.outcome_binding is not None):
            raise TeachingError('actual outcome bindings require their explicit case schema')
        if self.outcome_binding is not None:
            from .outcomes import validate_outcome_binding
            validate_outcome_binding(self.outcome_binding, self.document)
            if self.phase != 'calibration':
                raise TeachingError('actual calibration outcomes cannot replace N1/N2 teaching')
        splits = {'CALIBRATION_TRAIN', 'CALIBRATION_DEV'} if self.phase == 'calibration' else {'TRAIN', 'DEV'}
        if self.split not in splits:
            raise TeachingError('reviewed cases cannot consume FINAL or cross teaching/calibration splits')
        governed = 'governed_calibration_targets' if self.phase == 'calibration' else 'governed_identity_targets'
        if self.source_class not in {'public_mechanical_fixture', governed}:
            raise TeachingError('case source class differs from the selected phase')
        origin = {'mechanical_fixture'} if self.source_class == 'public_mechanical_fixture' else {'reviewed_source_case'}
        if self.annotation_origin not in origin:
            raise TeachingError('annotation origin cannot promote fixtures to governed identity gold')
        for value in (self.case_id, self.review_id, self.annotation_note):
            _text(value)
        if type(self.document) is not CognitiveFrameDocument:
            raise TeachingError('reviewed cases need the complete target-free cognitive document')
        try:
            self.document.validate()
        except IdentityError as exc:
            raise TeachingError('reviewed case has an invalid target-free document') from exc
        actual = {entry.record.source_family_id for entry in self.document.sources}
        if type(self.source_family_ids) is not tuple or len(set(self.source_family_ids)) != len(self.source_family_ids) \
                or set(self.source_family_ids) != actual:
            raise TeachingError('case source families must preserve complete source lineage')
        if self.lineage is not None:
            if type(self.lineage) is not ConstructionLineage:
                raise TeachingError('new cases require typed construction lineage')
            self.lineage.validate(tuple(entry.record for entry in self.document.sources))
        validate_artifact_pins(self.artifact_sha256, self.source_class)
        if type(self.targets) is not tuple or not self.targets or any(type(t) is not NamedTarget for t in self.targets):
            raise TeachingError('source facts alone are not reviewed teaching targets')
        if len({t.field for t in self.targets}) != len(self.targets):
            raise TeachingError('duplicate target fields are forbidden')
        sources = {entry.record.record_id for entry in self.document.sources}
        for target in self.targets:
            target.validate(sources)

    def to_mapping(self):
        self.validate()
        # JSON arrays are explicit at the serialized boundary, including the
        # embedded document; values contain no nonfinite placeholders.
        value = asdict(self)
        if self.lineage is None:
            value.pop('lineage')
        if self.outcome_binding is None:
            value.pop('outcome_binding')
        return json.loads(canonical(value))

    @property
    def sha256(self):
        return sha256(canonical(self.to_mapping())).hexdigest()

    @classmethod
    def from_mapping(cls, raw):
        fields = set(cls.__dataclass_fields__)
        if type(raw) is dict and raw.get('schema') != OUTCOME_CASE_SCHEMA:
            fields.remove('outcome_binding')
        if type(raw) is dict and raw.get('schema') == CASE_SCHEMA:
            fields.remove('lineage')
        _exact(raw, fields)
        if type(raw['source_family_ids']) is not list or type(raw['targets']) is not list:
            raise TeachingError('serialized case collections require JSON lists')
        targets = []
        for target in raw['targets']:
            _exact(target, NamedTarget.__dataclass_fields__)
            if type(target['facts']) is not list:
                raise TeachingError('serialized target facts require JSON lists')
            facts = []
            for fact in target['facts']:
                _exact(fact, TargetFact.__dataclass_fields__)
                for key in ('pointers', 'support_ids', 'disconfirmation_ids'):
                    if type(fact[key]) is not list:
                        raise TeachingError('serialized annotation pointers require JSON lists')
                facts.append(TargetFact(tuple(fact['pointers']), fact['value'], fact['uncertainty'],
                                        fact['available'], tuple(fact['support_ids']), tuple(fact['disconfirmation_ids'])))
            targets.append(NamedTarget(target['field'], target['mode'], tuple(facts)))
        try:
            document = CognitiveFrameDocument.from_mapping(raw['document'])
        except (IdentityError, TypeError, KeyError) as exc:
            raise TeachingError('invalid embedded target-free document') from exc
        case = cls(raw['case_id'], raw['phase'], raw['split'], raw['source_class'], document,
                   tuple(raw['source_family_ids']), raw['artifact_sha256'], raw['annotation_origin'],
                   raw['review_id'], raw['annotation_note'], tuple(targets), raw['schema'],
                   ConstructionLineage.from_mapping(raw['lineage']) if 'lineage' in raw else None,
                   raw.get('outcome_binding'))
        case.validate()
        return case

    @classmethod
    def from_json(cls, payload):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise TeachingError('duplicate JSON case keys are forbidden')
                result[key] = value
            return result
        def invalid(_):
            raise TeachingError('nonfinite JSON case values are forbidden')
        try:
            return cls.from_mapping(json.loads(payload, object_pairs_hook=unique, parse_constant=invalid))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise TeachingError('malformed reviewed teaching case') from exc
